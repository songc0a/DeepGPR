"""CUDA Graphs of whole checkpointed sweeps of the fused 3D step (no per-step copies).

The older :class:`~DeepGPR.hybrid.graphs.GraphedStep` captures one step on static
input buffers and copies the new state back into them after every replay; for large
states that copy costs as much as the graph saves, so it is only used below 32 MB.
Here every step writes its result into a given buffer (:func:`.fused3d.step_into`,
:func:`.fused3d.vjp_into`), so the steps are captured once on a fixed set of buffers
and replayed without any copy of the state:

* forward only (:meth:`SweepGraphs.run`): two state buffers, graphs ``U_0 -> U_1``
  and ``U_1 -> U_0``;
* the checkpointed adjoint sweep with segments of ``k`` steps (``k`` even): ``k``
  forward graphs ``F_j: U_j -> U_(j+1)`` (also sampling the receivers of step ``j``)
  serve the forward pass and the recomputation of a segment, ``k`` reverse graphs
  ``R_j: (U_j, U_(j+1), L_((j+1) mod 2)) -> L_(j mod 2)`` apply the transpose of step
  ``j`` and accumulate the coefficient cotangents in place. Per segment only the
  checkpoint is copied into ``U_0`` (and saved in the forward pass); with ``k`` even the
  cotangent produced by one segment is where the next one starts.

The state buffers, the static coefficient copies and the accumulators belong to the
solver's step data (:class:`SweepBuffers`, shared by all propagators of a solver, which
run one at a time); each propagator keeps its own graphs (they bake in its source and
receiver operators) and its small per-step buffers (source samples, receiver data and
cotangents of one segment, filled once per segment). The coefficients are refreshed by
copy at every call, so the graphs survive across calls (FWI iterations). All graphs
share one memory pool for their temporaries: they are replayed one at a time on one
stream and have no outputs in the pool. ``DEEPGPR_HYBRID_DISABLE_GRAPHS=1`` switches to
the eager fused steps (same kernels).
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch

from . import functional as fn2
from . import functional3d as fn3
from . import fused2d, fused3d


def module(st):
    """The fused-step module of a 2D or 3D static (:mod:`.fused2d` / :mod:`.fused3d`)."""
    return fused2d if isinstance(st, fn2.StepStatic) else fused3d


def enabled(st) -> bool:
    """Sweep graphs for ``st``: CUDA, a fused step (2D or 3D), graphs not disabled."""
    if os.environ.get("DEEPGPR_HYBRID_DISABLE_GRAPHS") or not hasattr(torch.cuda, "CUDAGraph"):
        return False
    return module(st).any_available(st)


def sweep_k(nt: int, k: Optional[int]) -> int:
    """Even segment length near ``k`` (default ``ceil(sqrt(nt))``), at least 2."""
    k = int(math.ceil(math.sqrt(max(1, nt)))) if k is None else int(k)
    k = max(2, k)
    return k + (k % 2)


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc)


def fit_k(st, k: int, nt: int) -> int:
    """Largest even ``k' <= k`` whose sweep buffers (``k' + 3`` states) fit in the free
    device memory next to the other needs of a gradient (measured on the 3D models:
    about ten states for the coefficients and their autograd graph, the temporaries of
    the step and of its transpose, the graph pool and checkpoint transfers, plus 1 GB);
    the solver's current sweep buffers count as free (they are replaced)."""
    from .graphs import state_bytes
    dev = torch.device(st.device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    try:
        free, _ = torch.cuda.mem_get_info(idx)
    except Exception:  # pragma: no cover - old PyTorch
        return k
    old = st.__dict__.get("_sweep_buffers")
    sb = state_bytes(st)
    if old is not None and old.alive:
        free += (old.k + 3) * sb
    free += torch.cuda.memory_reserved(idx) - torch.cuda.memory_allocated(idx)
    while k > 2 and (k + 13) * sb + (1 << 30) > free:
        k -= 2
    return k


def _zero(ts):
    for t in ts:
        if t.numel():
            t.zero_()


def _copy(dst, src):
    for a, b in zip(dst, src):
        if a.numel():
            a.copy_(b, non_blocking=True)


class SweepBuffers:
    """State and cotangent buffers, static coefficients, accumulators and the graph
    memory pool of one :class:`~DeepGPR.hybrid.functional3d.StepStatic3D`."""

    _versions = 0

    def __init__(self, st, c, k: int):
        # no reference to ``st`` (it owns this object: no reference cycle, freed with it)
        self.device, self.k = torch.device(st.device), k
        self.mod = module(st)
        self.U = [self.mod.empty_state(st) for _ in range(k + 1)]
        self.L = [self.mod.empty_state(st), self.mod.empty_state(st)]
        for b in self.U + self.L:
            _zero(b)
        self.c = c.replaced([t.detach().clone().contiguous() for t in c.tensors()])
        self.pre = self.mod.prepare(st, self.c)          # LTS: coefficient-derived arrays
        self.acc = fused3d.Accumulators()
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
        """Release the buffers now (graphs captured on them must not be replayed again:
        :class:`SweepGraphs` recaptures on new buffers)."""
        self.U = self.L = self.c = self.pre = self.acc = None

    def load(self, st, c):
        with torch.no_grad():
            for a, b in zip(self.c.tensors(), c.tensors()):
                a.copy_(b)
            if self.pre is not None:
                self.mod.prepare(st, self.c, self.pre)


def buffers(st, c, k: int) -> SweepBuffers:
    """The sweep buffers of ``st`` with at least ``k`` segment steps (replaced, which
    invalidates every propagator's graphs, when too small)."""
    buf = st.__dict__.get("_sweep_buffers")
    if buf is None or not buf.compatible(c, k):
        old = st.__dict__.pop("_sweep_buffers", None)
        if old is not None:
            k = max(k, old.k) if old.alive else k
            old.free()                     # also when other propagators still hold it
        buf = None
        torch.cuda.empty_cache()
        buf = st.__dict__["_sweep_buffers"] = SweepBuffers(st, c, k)
    return buf


def release(st):
    """Free the sweep buffers of ``st`` (graphs are rebuilt on the next call)."""
    old = st.__dict__.pop("_sweep_buffers", None)
    if old is not None:
        old.free()
    torch.cuda.empty_cache()


class _Capture:
    def __init__(self, buf: SweepBuffers):
        self.buf = buf
        self.device = buf.device

    def warm(self, fns):
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.no_grad(), torch.cuda.stream(side):
            for f in fns:
                f()
        torch.cuda.current_stream(self.device).wait_stream(side)

    def capture(self, fn):
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g, pool=self.buf.pool):
            fn()
        return g


class SweepGraphs:
    """The graphs of one propagator (sources ``src``, receivers ``rec``, ``nsrc``
    source samples per step) on the sweep buffers of its solver."""

    def __init__(self, st, src, rec, nsrc: int):
        self.st, self.src, self.rec = st, src, rec
        self.nsrc = max(1, int(nsrc))
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
            dev = torch.device(self.st.device)
            kw = dict(dtype=self.st.dtype, device=dev)
            S, nrec = self.st.nshot, self.rec.count
            self.amp = torch.zeros((k, self.nsrc), **kw)
            self.gamp = torch.zeros((k, self.nsrc), **kw)
            self.rec_out = torch.zeros((k, S, nrec), **kw)
            self.gd = torch.zeros((k, S, nrec), **kw)
            self.F = self.R = self.P = None
            self.rkey = None
        buf.load(self.st, c)
        return buf

    def _fwd(self, i, j, sample=True):
        st, buf, src, rec = self.st, self.buf, self.src, self.rec
        amp = self.amp[i] if src is not None else None

        def f():
            buf.mod.step_any(st, buf.c, buf.U[i], buf.U[j], src, amp, rec,
                             self.rec_out[i] if sample else None, buf.pre)
        return f

    def _rev(self, j, need_c, need_w):
        st, buf, src, rec = self.st, self.buf, self.src, self.rec
        has_src = src is not None

        def f():
            buf.mod.vjp_any(st, buf.c, buf.U[j], buf.U[j + 1], buf.L[(j + 1) % 2],
                            buf.L[j % 2], src, self.amp[j] if has_src else None, self.gd[j],
                            rec, buf.acc, need_c, need_w and has_src, self.gamp[j], buf.pre)
        return f

    def _capture_forward(self):
        if self.F is None:
            cap = _Capture(self.buf)
            cap.warm([self._fwd(0, 1)])
            self.F = [cap.capture(self._fwd(j, j + 1)) for j in range(self.k)]

    def _capture_reverse(self, need_c, need_w):
        if self.rkey != (need_c, need_w):
            cap = _Capture(self.buf)
            self.buf.acc.zero()
            cap.warm([self._rev(0, need_c, need_w)])
            self.R = [cap.capture(self._rev(j, need_c, need_w)) for j in range(self.k)]
            self.rkey = (need_c, need_w)

    # ------------------------------------------------------------- forward only
    def run(self, c, w, nt, data):
        """Forward only: ``data`` (S, nt, nrec) from zero initial fields; returns the
        final state (a sweep buffer, valid until the next call)."""
        buf = self._bind(c, 2)
        if self.P is None:
            cap = _Capture(buf)
            cap.warm([self._fwd(0, 1)])
            self.P = [cap.capture(self._fwd(0, 1)), cap.capture(self._fwd(1, 0))]
        _zero(buf.U[0])
        cur = 0
        for n in range(nt):
            if self.src is not None:
                self.amp[cur].copy_(w[:, n])
            self.P[cur].replay()
            data[:, n].copy_(self.rec_out[cur])
            cur = 1 - cur
        return buf.U[cur]

    # ------------------------------------------------------------- adjoint sweep
    def forward(self, c, w, nt, k, data, checkpoints):
        """Forward pass with checkpoints at the segment starts ``0, k, 2k, ...``."""
        buf = self._bind(c, k)
        self._capture_forward()
        _zero(buf.U[0])
        checkpoints[0] = tuple(t.clone() for t in buf.U[0])
        for start in range(0, nt, k):
            m = min(k, nt - start)
            if self.src is not None:
                self.amp[:m].copy_(w[:, start:start + m].t())
            for j in range(m):
                self.F[j].replay()
            data[:, start:start + m].copy_(self.rec_out[:m].transpose(0, 1))
            if start + m < nt:
                checkpoints[start + m] = tuple(t.clone() for t in buf.U[m])
                _copy(buf.U[0], buf.U[m])

    def backward(self, c, w, grad_data, nt, k, checkpoints, need_c, need_w, g_w):
        """Reverse sweep; returns the (zeroed, then accumulated) accumulators."""
        buf = self._bind(c, k)
        self._capture_forward()
        self._capture_reverse(need_c, need_w)
        buf.acc.zero()
        first = True
        for start in reversed(range(0, nt, k)):
            m = min(k, nt - start)
            _copy(buf.U[0], checkpoints[start])
            if self.src is not None:
                self.amp[:m].copy_(w[:, start:start + m].t())
            self.gd[:m].copy_(grad_data[:, start:start + m].transpose(0, 1))
            if first:
                _zero(buf.L[m % 2])            # cotangent of the final state
                first = False
            for j in range(m):
                self.F[j].replay()
            for j in reversed(range(m)):
                self.R[j].replay()
            if g_w is not None and self.src is not None:
                g_w[:, start:start + m] = self.gamp[:m].t()
        return buf.acc


__all__ = ["SweepBuffers", "SweepGraphs", "buffers", "enabled", "fit_k", "release",
           "sweep_k"]
