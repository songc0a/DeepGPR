"""CUDA Graphs of whole checkpointed sweeps of the fused subgridded step (:mod:`.fused`).

The arrangement of :mod:`DeepGPR.hybrid.sweep3d` for the subgridded step: every step
writes into a given buffer, so ``k`` forward graphs ``F_j: U_j -> U_(j+1)`` (sampling the
main-grid and subgrid receivers of step ``j``) and ``k`` reverse graphs
``R_j: (U_j, U_(j+1), L_((j+1) mod 2)) -> L_(j mod 2)`` (the hand-written transpose,
coefficient cotangents accumulated in place) are captured once on fixed buffers and
replayed for every segment of the checkpointed adjoint sweep; forward-only runs replay
two ping-pong graphs. The state buffers, coefficient copies, scratch and accumulators
belong to the solver's static (shared by its propagators); the graphs and the per-step
buffers (main-rate and subgrid-rate source samples, receiver data and cotangents of one
segment) to each propagator. The sweep has ``nt + 1`` steps when receivers lie in a
subgrid (they are recorded one step late, :mod:`.adjoint`).
``DEEPGPR_SUBGRID_DISABLE_GRAPHS=1`` switches to the eager fused steps.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch

from ..hybrid.sweep3d import _Capture, _copy, _is_oom, _zero, sweep_k  # noqa: F401
from . import fused

_KEY = "_subgrid_sweep_buffers"


def enabled(st) -> bool:
    """Sweep graphs for ``st``: CUDA, the fused step, graphs not disabled."""
    if os.environ.get("DEEPGPR_SUBGRID_DISABLE_GRAPHS") or not hasattr(torch.cuda, "CUDAGraph"):
        return False
    return fused.available(st)


def fit_k(st, k: int, steps: int) -> int:
    """Largest even ``k' <= k`` whose sweep buffers fit next to the other needs of a
    gradient (the rule of :func:`DeepGPR.hybrid.sweep3d.fit_k`)."""
    dev = torch.device(st.device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    try:
        free, _ = torch.cuda.mem_get_info(idx)
    except Exception:  # pragma: no cover - old PyTorch
        return k
    sb = fused.state_bytes(st)
    old = st.__dict__.get(_KEY)
    if old is not None and old.alive:
        free += (old.k + 3) * sb
    free += torch.cuda.memory_reserved(idx) - torch.cuda.memory_allocated(idx)
    while k > 2 and (k + 13) * sb + (1 << 30) > free:
        k -= 2
    return k


class SweepBuffers:
    """State and cotangent buffers, coefficient copies, scratch, accumulators and the
    graph memory pool of one subgridded static."""

    _versions = 0

    def __init__(self, st, c, k: int):
        self.device, self.k = torch.device(st.device), k
        ref = fused._ref_state(st)
        self.U = [fused.empty_state(st, ref) for _ in range(k + 1)]
        self.L = [fused.empty_state(st, ref), fused.empty_state(st, ref)]
        for b in self.U + self.L:
            _zero(b)
        self.c = c.replaced([t.detach().clone().contiguous() for t in c.tensors()])
        self.pre = fused.prepare(st)
        self.acc = fused.Accumulators()
        self.pool = torch.cuda.graph_pool_handle()
        SweepBuffers._versions += 1
        self.version = SweepBuffers._versions

    @property
    def alive(self) -> bool:
        return self.U is not None

    def compatible(self, c, k: int) -> bool:
        a, b = self.c.tensors(), c.tensors()
        return self.alive and self.k >= k and len(a) == len(b) and all(
            x.shape == y.shape and x.dtype == y.dtype for x, y in zip(a, b))

    def free(self):
        self.U = self.L = self.c = self.pre = self.acc = None

    def load(self, c):
        with torch.no_grad():
            for a, b in zip(self.c.tensors(), c.tensors()):
                a.copy_(b)


def buffers(st, c, k: int) -> SweepBuffers:
    buf = st.__dict__.get(_KEY)
    if buf is None or not buf.compatible(c, k):
        old = st.__dict__.pop(_KEY, None)
        if old is not None:
            k = max(k, old.k) if old.alive else k
            old.free()
        buf = None
        torch.cuda.empty_cache()
        buf = st.__dict__[_KEY] = SweepBuffers(st, c, k)
    return buf


def release(st):
    """Free the sweep buffers of ``st`` (graphs are rebuilt on the next call)."""
    old = st.__dict__.pop(_KEY, None)
    if old is not None:
        old.free()
    torch.cuda.empty_cache()


class SweepGraphs:
    """The graphs of one propagator: sources ``src`` (``nsrc`` samples per step; per
    subgrid ``fine[g]``: subgrid sources, sub-steps ``ratios[g]``), receivers ``rec``."""

    def __init__(self, st, src, rec, nsrc: int, ratios: List[int], fine: List[bool]):
        self.st, self.src, self.rec = st, src, rec
        self.nsrc = max(1, int(nsrc))
        self.ratios, self.fine = list(ratios), list(fine)
        self.buf: Optional[SweepBuffers] = None
        self.version = None
        self.k = 0
        self.F = self.R = self.P = None
        self.rkey = None

    # ------------------------------------------------------------- set-up
    def _bind(self, c, k: int):
        need = max(k, self.k)
        if self.buf is not None and not self.buf.alive:
            self.buf = None
        buf = buffers(self.st, c, need)
        if buf.version != self.version or need != self.k:
            k = need
            self.buf, self.version, self.k = buf, buf.version, k
            kw = dict(dtype=self.st.dtype, device=torch.device(self.st.device))
            S, nrec = self.st.nshot, self.rec.count
            self.amp = torch.zeros((k, self.nsrc), **kw)
            self.gamp = torch.zeros((k, self.nsrc), **kw)
            self.ampf = [torch.zeros((k, r, self.nsrc), **kw) if f else None
                         for r, f in zip(self.ratios, self.fine)]
            self.gampf = [torch.zeros((k, r, self.nsrc), **kw) if f else None
                          for r, f in zip(self.ratios, self.fine)]
            self.rec_m = torch.zeros((k, S, nrec), **kw)
            self.rec_f = torch.zeros((k, S, nrec), **kw)
            self.gd_m = torch.zeros((k, S, nrec), **kw)
            self.gd_f = torch.zeros((k, S, nrec), **kw)
            self.F = self.R = self.P = None
            self.rkey = None
        buf.load(c)
        return buf

    def _af(self, i, bufs):
        if not any(self.fine):
            return None
        return [None if b is None else b[i] for b in bufs]

    def _fwd(self, i, j):
        st, buf, src, rec = self.st, self.buf, self.src, self.rec

        def f():
            fused.step_into(st, buf.c, buf.U[i], buf.U[j], src, self.amp[i] if src is not None else None,
                            self._af(i, self.ampf), rec, self.rec_m[i], self.rec_f[i], buf.pre)
        return f

    def _rev(self, j, need_c, need_w, need_wf):
        st, buf, src, rec = self.st, self.buf, self.src, self.rec
        has_src = src is not None

        def f():
            fused.vjp_into(st, buf.c, buf.U[j], buf.U[j + 1], buf.L[(j + 1) % 2], buf.L[j % 2],
                           src, self.amp[j] if has_src else None, self._af(j, self.ampf),
                           self.gd_m[j], self.gd_f[j], rec, buf.acc, need_c, need_w and has_src,
                           self.gamp[j], need_wf, self._af(j, self.gampf), buf.pre)
        return f

    def _capture_forward(self):
        if self.F is None:
            cap = _Capture(self.buf)
            cap.warm([self._fwd(0, 1)])
            self.F = [cap.capture(self._fwd(j, j + 1)) for j in range(self.k)]

    def _capture_reverse(self, need_c, need_w, need_wf):
        key = (need_c, need_w, need_wf)
        if self.rkey != key:
            cap = _Capture(self.buf)
            self.buf.acc.zero()
            cap.warm([self._rev(0, *key)])
            self.R = [cap.capture(self._rev(j, *key)) for j in range(self.k)]
            self.rkey = key

    # ------------------------------------------------------------- per-segment data
    def _load_amps(self, w, wf, start: int, m: int, nt: int):
        if self.src is None:
            return
        self.amp[:m].zero_()
        mm = min(m, nt - start)
        if mm > 0:
            self.amp[:mm].copy_(w[:, start:start + mm].t())
        for g, (r, x) in enumerate(zip(self.ratios, wf or [None] * len(self.ratios))):
            if self.ampf[g] is None or x is None:
                continue
            idx, ok = _fine_index(start, m, r, x.shape[1], x.device)
            vals = x.index_select(1, idx.clamp(0, x.shape[1] - 1).reshape(-1)).view(-1, m, r)
            self.ampf[g][:m].copy_((vals * ok.to(vals.dtype)).permute(1, 2, 0))

    def _gather(self, data, start: int, m: int, nt: int, extra: int):
        mm = min(m, nt - start)
        if mm > 0:
            data[:, start:start + mm] += self.rec_m[:mm].transpose(0, 1)
        if extra:
            lo = max(start, 1)
            if lo < start + m:
                data[:, lo - 1:start + m - 1] += self.rec_f[lo - start:m].transpose(0, 1)

    # ------------------------------------------------------------- forward only
    def run(self, c, w, wf, steps: int, nt: int, extra: int, data):
        buf = self._bind(c, 2)
        if self.P is None:
            cap = _Capture(buf)
            cap.warm([self._fwd(0, 1)])
            self.P = [cap.capture(self._fwd(0, 1)), cap.capture(self._fwd(1, 0))]
        _zero(buf.U[0])
        cur = 0
        for n in range(steps):
            self._load_slot(w, wf, n, nt, cur)
            self.P[cur].replay()
            if n < nt:
                data[:, n] += self.rec_m[cur]
            if extra and n >= 1:
                data[:, n - 1] += self.rec_f[cur]
            cur = 1 - cur
        return buf.U[cur]

    def _load_slot(self, w, wf, n, nt, slot):
        if self.src is None:
            return
        if n < nt:
            self.amp[slot].copy_(w[:, n])
        else:
            self.amp[slot].zero_()
        for g, (r, x) in enumerate(zip(self.ratios, wf or [None] * len(self.ratios))):
            if self.ampf[g] is None or x is None:
                continue
            idx, ok = _fine_index(n, 1, r, x.shape[1], x.device)
            vals = x.index_select(1, idx.clamp(0, x.shape[1] - 1).reshape(-1)) * ok.reshape(1, -1).to(x.dtype)
            self.ampf[g][slot].copy_(vals.t())

    # ------------------------------------------------------------- adjoint sweep
    def forward(self, c, w, wf, steps: int, nt: int, extra: int, k: int, data, checkpoints):
        buf = self._bind(c, k)
        self._capture_forward()
        _zero(buf.U[0])
        checkpoints[0] = tuple(t.clone() for t in buf.U[0])
        for start in range(0, steps, k):
            m = min(k, steps - start)
            self._load_amps(w, wf, start, m, nt)
            for j in range(m):
                self.F[j].replay()
            self._gather(data, start, m, nt, extra)
            if start + m < steps:
                checkpoints[start + m] = tuple(t.clone() for t in buf.U[m])
                _copy(buf.U[0], buf.U[m])

    def backward(self, c, w, wf, grad_data, steps: int, nt: int, extra: int, k: int, checkpoints,
                 need_c, need_w, need_wf, g_w, g_wf):
        """Reverse sweep; returns the (zeroed, then accumulated) accumulators. ``g_w``
        (nsrc, nt) is written, ``g_wf`` (per subgrid or ``None``) accumulated."""
        buf = self._bind(c, k)
        self._capture_forward()
        self._capture_reverse(need_c, need_w, need_wf)
        buf.acc.zero()
        first = True
        for start in reversed(range(0, steps, k)):
            m = min(k, steps - start)
            _copy(buf.U[0], checkpoints[start])
            self._load_amps(w, wf, start, m, nt)
            self.gd_m[:m].zero_()
            mm = min(m, nt - start)
            if mm > 0:
                self.gd_m[:mm].copy_(grad_data[:, start:start + mm].transpose(0, 1))
            self.gd_f[:m].zero_()
            if extra:
                lo = max(start, 1)
                if lo < start + m:
                    self.gd_f[lo - start:m].copy_(grad_data[:, lo - 1:start + m - 1].transpose(0, 1))
            if first:
                _zero(buf.L[m % 2])
                first = False
            for j in range(m):
                self.F[j].replay()
            for j in reversed(range(m)):
                self.R[j].replay()
            if g_w is not None and self.src is not None and mm > 0:
                g_w[:, start:start + mm] = self.gamp[:mm].t()
            if need_wf and g_wf is not None:
                for g, (r, x) in enumerate(zip(self.ratios, g_wf)):
                    if self.gampf[g] is None or x is None:
                        continue
                    idx, ok = _fine_index(start, m, r, x.shape[1], x.device)
                    vals = self.gampf[g][:m].permute(2, 0, 1)            # (nsrc, m, r)
                    x.index_add_(1, idx[ok], vals[:, ok])
        return buf.acc


def _fine_index(start: int, m: int, r: int, length: int, device):
    """Indices ``(n - 1) r + 1 + j`` of the subgrid-rate samples of main steps
    ``start..start+m-1`` (``(m, r)``) and their validity (:func:`.hsg.fine_step_samples`)."""
    n = torch.arange(start, start + m, device=device)[:, None]
    idx = (n - 1) * r + 1 + torch.arange(r, device=device)[None, :]
    ok = (idx >= 0) & (idx < length)
    return idx, ok


__all__ = ["SweepBuffers", "SweepGraphs", "buffers", "enabled", "fit_k", "release"]
