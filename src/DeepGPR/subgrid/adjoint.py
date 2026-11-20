r"""Discrete adjoint of the subgridded time stepping (:mod:`.hsg`).

Forward model (discrete, exactly as implemented in :func:`DeepGPR.subgrid.hsg.step`)::

    u^0 = 0,   u^(n+1) = S(u^n, c(m), w_n, wf_n),
    d_n = R_main u^(n+1) + R_sub u^(n+2),    J = J(d)

``u`` holds the main and every subgrid Yee state (fields and CPML auxiliaries),
the main-grid E precursors of the previous main level and the subgrid receiver
buffers; ``c(m)`` the ``ca``/``cb`` of the main grid and of every subgrid, ``w_n``
the main-rate source samples and ``wf_n`` the subgrid-rate samples of the step
(band-limited resampling of ``w``). Subgrid receivers record the subgrid field at
main time ``n + 1`` during step ``n + 1``, hence the shift.

Adjoint recursion (``lambda^(nt') = 0``, ``nt' = nt + 1`` with subgrid receivers)::

    lambda^n = (dS/du)^T [lambda^(n+1) + (dR/du)^T g_n],
    dJ/dc   += (dS/dc)^T [...],   dJ/dw_n = (dS/dw)^T [...]

with ``g_n`` the data cotangents mapped to step ``n``. The products are the
reverse-mode derivatives of :func:`~DeepGPR.subgrid.hsg.step` itself (its Yee
updates and coupling kernels have exact hand-written transposes on CUDA,
:mod:`DeepGPR.hybrid.cuda3d` and :mod:`.cuda`), so the gradient is that of the
discrete forward model; no continuous adjoint equations are used. States are
checkpointed every ``k`` steps and each segment is recomputed once in the
reverse sweep (memory ``nt / k + k`` states; default ``k = ceil(sqrt(nt))``).

``dJ/dc`` reaches the model-grid ``eps_r`` and ``sigma`` through
:meth:`~DeepGPR.subgrid.SubgridMaxwell3D.material_fields` (block averages to the
main grid, samples or averages to the subgrids) and the coefficient formulas.

On CUDA the sweeps run on the fused step and its hand-written transpose
(:mod:`.fused`), replayed as CUDA Graphs on fixed buffers (:mod:`.sweep`);
``DEEPGPR_SUBGRID_DISABLE_GRAPHS=1`` runs the same fused steps eagerly (also used when
the error indicator is accumulated), ``DEEPGPR_SUBGRID_AUTOGRAD_VJP=1`` differentiates
:func:`~DeepGPR.subgrid.hsg.step` with autograd step by step (the reference).
"""

from __future__ import annotations

import math
import os
from typing import List, Optional

import torch

from ..hybrid.adjoint import _Checkpoints
from . import fused, graphs, hsg, sweep
from .indicators import DWRIndicator, third_difference


class _SubgridPropagate(torch.autograd.Function):
    """``data = F(c, w, wf)``; backward = checkpointed discrete adjoint."""

    @staticmethod
    def forward(ctx, prop, template, w, n_wf, *tensors):  # type: ignore[override]
        st = prop.static
        ctx.fused = None
        if fused.available(st) and not os.environ.get("DEEPGPR_SUBGRID_AUTOGRAD_VJP"):
            return _SubgridPropagate._forward_fused(ctx, prop, template, w, n_wf, tensors)
        wf = list(tensors[:n_wf])
        coef_tensors = tensors[n_wf:]
        coefs = template.replaced(coef_tensors)
        wf_list = prop._wf_list(wf)
        nt = w.shape[1]
        steps = nt + prop.extra
        k = prop.checkpoint_every or max(1, int(math.ceil(math.sqrt(steps))))
        need = any(ctx.needs_input_grad)
        if not need:
            k = steps + 1
        state = prop.F.zero_state(st, prop.rec)
        checkpoints = {0: state}
        data = torch.zeros((st.nshot, nt, prop.rec.count), dtype=st.dtype, device=st.device)
        graph = None
        if prop.use_graph and st.device.type == "cuda":
            graph = graphs.make(st, coefs, prop.src, prop.rec, w.shape[0], F=prop.F)
            if graph is not None:
                graph.set_state(state)
        with torch.no_grad():
            for n in range(steps):
                amp, af = prop._amps(w, wf_list, n)
                if graph is not None:
                    dm, dfine = graph(amp, af)
                else:
                    state = prop.F.step(st, coefs, state, prop.src, amp, af, prop.rec)
                    dm, dfine = prop.F.sample(st, prop.rec, state)
                if n < nt:
                    data[:, n] += dm
                if n >= 1 and prop.extra:
                    data[:, n - 1] += dfine
                if (n + 1) % k == 0 and n + 1 < steps:
                    checkpoints[n + 1] = graph.snapshot() if graph is not None else state
        ctx.prop, ctx.template, ctx.k, ctx.n_wf = prop, template, k, n_wf
        ctx.checkpoints = checkpoints
        ctx.save_for_backward(w, *tensors)
        return data

    # ------------------------------------------------------------ fused (CUDA) path
    @staticmethod
    def _forward_fused(ctx, prop, template, w, n_wf, tensors):
        st = prop.static
        coefs = template.replaced([t.detach().contiguous() for t in tensors[n_wf:]])
        wd = w.detach().contiguous()
        wf = prop._wf_per_subgrid([t.detach() for t in tensors[:n_wf]])
        nt = wd.shape[1]
        steps, extra = nt + prop.extra, prop.extra
        data = torch.zeros((st.nshot, nt, prop.rec.count), dtype=st.dtype, device=st.device)
        sg = prop._sweep_graphs(wd.shape[0]) if (prop.use_graph and sweep.enabled(st)) else None
        need = any(ctx.needs_input_grad)
        with torch.no_grad():
            if not need:
                if sg is not None:
                    sg.run(coefs, wd, wf, steps, nt, extra, data)
                else:
                    _eager_forward(prop, coefs, wd, wf, steps, nt, extra, None, data, None)
                return data
            if prop.indicator_obj is not None:
                # eager reverse sweep with the indicator, on the segments of the autograd
                # sweep (its time term is skipped at every segment start)
                k = prop.checkpoint_every or max(1, int(math.ceil(math.sqrt(steps))))
            else:
                k = sweep.sweep_k(steps, prop.checkpoint_every)
                if sg is not None:
                    k = sweep.fit_k(st, k, steps)
            checkpoints = _Checkpoints(st.device, _offload(st, steps, k))
            oom = False
            if sg is not None:
                try:
                    sg.forward(coefs, wd, wf, steps, nt, extra, k, data, checkpoints)
                except RuntimeError as exc:
                    if not sweep._is_oom(exc):
                        raise
                    oom = True
            if oom:                       # (outside the handler: frees the frames)
                sg = None
                checkpoints = None
                prop.release()
                checkpoints = _Checkpoints(st.device, _offload(st, steps, k))
                data.zero_()
            if sg is None:
                _eager_forward(prop, coefs, wd, wf, steps, nt, extra, k, data, checkpoints)
        ctx.fused = dict(k=k, sg=sg)
        ctx.prop, ctx.template, ctx.n_wf = prop, template, n_wf
        ctx.checkpoints = checkpoints
        ctx.save_for_backward(w, *tensors)
        return data

    @staticmethod
    def _backward_fused(ctx, grad_data):
        prop, template, n_wf = ctx.prop, ctx.template, ctx.n_wf
        st = prop.static
        w, *tensors = ctx.saved_tensors
        need_w = ctx.needs_input_grad[2]
        need_wf = any(ctx.needs_input_grad[4:4 + n_wf])
        need_c = any(ctx.needs_input_grad[4 + n_wf:])
        coef_tensors = tensors[n_wf:]
        coefs = template.replaced([t.detach().contiguous() for t in coef_tensors])
        wd = w.detach().contiguous()
        wf_slots = [t.detach() for t in tensors[:n_wf]]
        wf = prop._wf_per_subgrid(wf_slots)
        gd = grad_data.detach().contiguous()
        nt = wd.shape[1]
        steps, extra = nt + prop.extra, prop.extra
        k, sg = ctx.fused["k"], ctx.fused["sg"]
        g_w = torch.zeros_like(wd) if need_w else None
        g_wf_slots = [torch.zeros_like(t) for t in wf_slots] if need_wf else None
        g_wf = prop._wf_per_subgrid(g_wf_slots) if need_wf else None
        ind = prop.indicator_obj
        with torch.no_grad():
            acc, oom = None, False
            if sg is not None and ind is None:
                try:
                    acc = sg.backward(coefs, wd, wf, gd, steps, nt, extra, k, ctx.checkpoints,
                                      need_c, need_w, need_wf, g_w, g_wf)
                except RuntimeError as exc:
                    if not sweep._is_oom(exc):
                        raise
                    oom = True
            if oom:
                acc = None
                prop.release()
                if g_w is not None:
                    g_w.zero_()
                for t in g_wf_slots or []:
                    t.zero_()
            if acc is None:
                acc = _eager_backward(prop, coefs, wd, wf, gd, steps, nt, extra, k, ctx.checkpoints,
                                      need_c, need_w, need_wf, g_w, g_wf, ind)
            g_coef = acc.finalize_list(coefs) if need_c else [None] * len(coef_tensors)
        ctx.checkpoints = None
        return (None, None, g_w, None, *(g_wf_slots if g_wf_slots is not None else [None] * n_wf),
                *g_coef)

    @staticmethod
    def backward(ctx, grad_data):  # type: ignore[override]
        if ctx.fused is not None:
            return _SubgridPropagate._backward_fused(ctx, grad_data)
        prop, template = ctx.prop, ctx.template
        st = prop.static
        w, *tensors = ctx.saved_tensors
        n_wf = ctx.n_wf
        wf = tensors[:n_wf]
        coef_tensors = tensors[n_wf:]
        need_w = ctx.needs_input_grad[2]
        need_wf = any(ctx.needs_input_grad[4:4 + n_wf])
        need_c = any(ctx.needs_input_grad[4 + n_wf:])
        leaves = [t.detach().requires_grad_(need_c) for t in coef_tensors]
        coefs_leaf = template.replaced(leaves)
        coefs_plain = template.replaced([t.detach() for t in coef_tensors])
        wf_plain = prop._wf_list([t.detach() for t in wf])
        nt = w.shape[1]
        steps = nt + prop.extra
        k = ctx.k
        g_coef = [torch.zeros_like(t) for t in coef_tensors]
        g_w = torch.zeros_like(w) if need_w else None
        g_wf = [torch.zeros_like(t) for t in wf] if need_wf else None
        lam = list(prop.F.zero_state(st, prop.rec))
        active = [i for i, t in enumerate(lam) if t.numel()]
        ind = prop.indicator_obj
        later = None             # main E/H of u^(n+2) (from the step processed before)
        zero_g = torch.zeros((st.nshot, prop.rec.count), dtype=st.dtype, device=st.device)
        for start in reversed(range(0, steps, k)):
            stop = min(steps, start + k)
            states = [ctx.checkpoints[start]]
            with torch.no_grad():
                for n in range(start, stop - 1):
                    amp, af = prop._amps(w, wf_plain, n)
                    states.append(prop.F.step(st, coefs_plain, states[-1], prop.src, amp, af,
                                              prop.rec))
            for n in reversed(range(start, stop)):
                u = states[n - start]
                with torch.enable_grad():
                    u_leaf = list(u)
                    for i in active:
                        u_leaf[i] = u[i].detach().requires_grad_(True)
                    amp = (w[:, n] if n < nt else torch.zeros_like(w[:, 0])).detach()
                    amp = amp.requires_grad_(need_w and n < nt)
                    af_leaf = None
                    if wf_plain is not None:
                        af_leaf = [None if x is None else hsg.fine_step_samples(x, n, r)
                                   .detach().requires_grad_(need_wf)
                                   for x, r in zip(wf_plain, prop.ratios)]
                    pert = None
                    if ind is not None:
                        pert = ([torch.zeros_like(u[3 + c]).requires_grad_(True) for c in range(3)],
                                [torch.zeros_like(u[c]).requires_grad_(True) for c in range(3)])
                    nxt = prop.F.step(st, coefs_leaf, tuple(u_leaf), prop.src, amp, af_leaf, prop.rec,
                                      pert)
                    dm, dfine = prop.F.sample(st, prop.rec, nxt)
                    gm = grad_data[:, n] if n < nt else zero_g
                    gfi = grad_data[:, n - 1] if (n >= 1 and prop.extra) else zero_g
                    pairs = [(nxt[i], lam[i]) for i in active] + [(dm, gm), (dfine, gfi)]
                    pairs = [(o, g) for o, g in pairs if o.requires_grad]
                    outs = [o for o, _ in pairs]
                    gouts = [g for _, g in pairs]
                    ins: List[torch.Tensor] = [u_leaf[i] for i in active]
                    if need_c:
                        ins += leaves
                    if need_w and n < nt:
                        ins.append(amp)
                    fine_ins = [a for a in (af_leaf or []) if a is not None] if need_wf else []
                    ins += fine_ins
                    if pert is not None:
                        ins += pert[0] + pert[1]
                    grads = torch.autograd.grad(outs, ins, gouts, allow_unused=True)
                if pert is not None:
                    gH, gE = grads[-6:-3], grads[-3:]
                    E_n, H_prev = u[0:3], u[3:6]
                    E_1, H_half = [t.detach() for t in nxt[0:3]], [t.detach() for t in nxt[3:6]]
                    prev = states[n - 1 - start] if n > start else None
                    tt = None
                    if prev is not None and later is not None:
                        tt = ([third_difference(prev[c], E_n[c], E_1[c], later[0][c])
                               for c in range(3)],
                              [third_difference(prev[3 + c], H_prev[c], H_half[c], later[1][c])
                               for c in range(3)])
                    ind.accumulate(coefs_plain.main_cb, gE, gH, E_n, H_half, time_terms=tt)
                    later = (E_1, H_half)
                    del nxt
                for slot, i in enumerate(active):
                    g = grads[slot]
                    lam[i] = torch.zeros_like(u[i]) if g is None else g
                pos = len(active)
                if need_c:
                    for j in range(len(leaves)):
                        if grads[pos + j] is not None:
                            g_coef[j] += grads[pos + j]
                    pos += len(leaves)
                if need_w and n < nt:
                    if grads[pos] is not None:
                        g_w[:, n] = grads[pos]
                    pos += 1
                if fine_ins:
                    gi = 0
                    for g_idx, x in enumerate(wf_plain):
                        if x is None:
                            continue
                        ga = grads[pos + gi]
                        gi += 1
                        if ga is not None:
                            prop._scatter_fine(g_wf[prop._wf_slot[g_idx]], ga, n,
                                               prop.ratios[g_idx])
            del states
        return (None, None, g_w, None, *(g_wf if g_wf is not None else [None] * n_wf), *g_coef)


def _offload(st, steps: int, k: int) -> bool:
    """Checkpoints in host memory when they would not fit on the device next to the
    sweep buffers (``k + 3`` states), ten states of temporaries and 1 GB."""
    dev = torch.device(st.device)
    if dev.type != "cuda":
        return False
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    try:
        free = torch.cuda.mem_get_info(idx)[0]
    except Exception:  # pragma: no cover - old PyTorch
        return False
    free += torch.cuda.memory_reserved(idx) - torch.cuda.memory_allocated(idx)
    sb = fused.state_bytes(st)
    buf = st.__dict__.get(sweep._KEY)
    resident = 0 if (buf is not None and buf.alive and buf.k >= k) else (k + 3) * sb
    return (steps // max(k, 1)) * sb + resident + 10 * sb + (1 << 30) > free


def _fine_samples(wf, n: int, ratios):
    """Per subgrid ``None`` or the subgrid-rate samples ``(r, nsrc)`` of main step ``n``."""
    if wf is None:
        return None
    return [None if x is None else hsg.fine_step_samples(x, n, r).t().contiguous()
            for x, r in zip(wf, ratios)]


def _eager_forward(prop, c, w, wf, steps, nt, extra, k, data, checkpoints):
    """Fused steps without graphs; checkpoints every ``k`` steps."""
    st = prop.static
    pre = fused.prepare(st)
    state = fused.empty_state(st)
    for t in state:
        t.zero_()
    if checkpoints is not None:
        checkpoints[0] = state
    S = st.nshot
    rm = torch.zeros((S, prop.rec.count), dtype=st.dtype, device=st.device)
    rf = torch.zeros_like(rm)
    zero_amp = torch.zeros_like(w[:, 0])
    for n in range(steps):
        out = fused.empty_state(st, state)
        amp = w[:, n].contiguous() if n < nt else zero_amp
        fused.step_into(st, c, state, out, prop.src, amp, _fine_samples(wf, n, prop.ratios),
                        prop.rec, rm, rf, pre)
        if n < nt:
            data[:, n] += rm
        if extra and n >= 1:
            data[:, n - 1] += rf
        state = out
        if checkpoints is not None and (n + 1) % k == 0 and n + 1 < steps:
            checkpoints[n + 1] = state
    return state


def _eager_backward(prop, c, w, wf, gd, steps, nt, extra, k, checkpoints, need_c, need_w,
                    need_wf, g_w, g_wf, ind=None):
    """Reverse sweep with the fused transpose, without graphs; with the error indicator
    ``ind`` (:class:`~DeepGPR.subgrid.indicators.DWRIndicator`) accumulated as in the
    autograd sweep."""
    st = prop.static
    pre = fused.prepare(st)
    acc = fused.Accumulators()
    lam = fused.empty_state(st)
    for t in lam:
        t.zero_()
    S = st.nshot
    zero_g = torch.zeros((S, prop.rec.count), dtype=st.dtype, device=st.device)
    zero_amp = torch.zeros_like(w[:, 0])
    gamp = torch.zeros_like(w[:, 0])
    gampf = None if wf is None else [None if x is None else torch.zeros((r, x.shape[0]),
                                                                        dtype=x.dtype, device=x.device)
                                     for x, r in zip(wf, prop.ratios)]
    bufs = None
    if ind is not None:
        main = fused.empty_state(st)
        bufs = ([torch.empty_like(main[c_]) for c_ in range(3)],
                [torch.empty_like(main[3 + c_]) for c_ in range(3)])
    later = None
    for start in reversed(range(0, steps, k)):
        stop = min(steps, start + k)
        states = [checkpoints[start]]
        for n in range(start, stop):
            out = fused.empty_state(st, states[0])
            amp = w[:, n].contiguous() if n < nt else zero_amp
            fused.step_into(st, c, states[-1], out, prop.src, amp,
                            _fine_samples(wf, n, prop.ratios), None, None, None, pre)
            states.append(out)
        for n in reversed(range(start, stop)):
            amp = w[:, n].contiguous() if n < nt else zero_amp
            gm = gd[:, n].contiguous() if n < nt else zero_g
            gf = gd[:, n - 1].contiguous() if (extra and n >= 1) else zero_g
            lout = fused.empty_state(st, states[0])
            u, nxt = states[n - start], states[n + 1 - start]
            fused.vjp_into(st, c, u, nxt, lam, lout, prop.src, amp,
                           _fine_samples(wf, n, prop.ratios), gm, gf, prop.rec, acc, need_c,
                           need_w and n < nt, gamp, need_wf, gampf, pre, bufs)
            lam = lout
            if need_w and n < nt:
                g_w[:, n] = gamp
            if need_wf and g_wf is not None:
                for x, ga, r in zip(g_wf, gampf, prop.ratios):
                    if x is not None and ga is not None:
                        prop._scatter_fine(x, ga.t(), n, r)
            if ind is not None:
                gE, gH = bufs
                E_n, H_prev = list(u[0:3]), list(u[3:6])
                E_1, H_half = list(nxt[0:3]), list(nxt[3:6])
                prev = states[n - 1 - start] if n > start else None
                tt = None
                if prev is not None and later is not None:
                    tt = ([third_difference(prev[c_], E_n[c_], E_1[c_], later[0][c_])
                           for c_ in range(3)],
                          [third_difference(prev[3 + c_], H_prev[c_], H_half[c_], later[1][c_])
                           for c_ in range(3)])
                ind.accumulate(c.main_cb, gE, gH, E_n, H_half, time_terms=tt)
                later = ([t.clone() for t in E_1], [t.clone() for t in H_half])
            states[n + 1 - start] = None
        del states
    return acc


class SubgridPropagator:
    """Differentiable receiver data of a :class:`~DeepGPR.subgrid.SubgridMaxwell3D`.

    ``data = prop(eps_r, sigma, source_amplitudes)`` with model-grid ``eps_r`` and
    ``sigma``; gradients by the discrete adjoint (:mod:`.adjoint`).

    Args:
        solver: A configured solver (main grid, subgrids, ``dt``, ``nshot``).
        source_positions, receiver_positions: ``(nshot, n, 3)`` physical [m].
        checkpoint_every: Steps between stored states (default ``ceil(sqrt(nt))``).
        use_graph: CUDA Graph replay of the forward pass.
        indicator: Accumulate the adjoint-weighted error indicator of the main grid
            during the reverse sweep (:class:`~DeepGPR.subgrid.indicators.DWRIndicator`,
            available as :attr:`indicator_obj` after ``backward``).
    """

    solver_name = "fdtd_subgrid"

    def __init__(self, solver, source_positions, receiver_positions, *,
                 checkpoint_every: Optional[int] = None, use_graph: bool = True,
                 indicator: bool = False):
        self.solver = solver
        self.static = solver.static
        self.F = getattr(solver, "functional", None) or hsg
        self.src = solver.point_ops(source_positions, source=True)
        self.rec = solver.point_ops(receiver_positions, source=False)
        self.extra = 1 if self.rec.has_sub else 0
        self.dt = solver.dt
        self.checkpoint_every = checkpoint_every
        self.use_graph = use_graph
        self.indicator = bool(indicator)
        self.indicator_obj: Optional[DWRIndicator] = None
        self.device, self.dtype = solver.device, solver.dtype
        self.ratios = [sg.time_ratio for sg in self.static.subs]     # sub-steps per main step
        # one resampled source vector per subgrid with subgrid sources
        self._wf_slot = {}
        for g in range(len(self.static.subs)):
            if self.src.sub[g].shot.numel():
                self._wf_slot[g] = len(self._wf_slot)

    def _wf_list(self, wf):
        if not self._wf_slot:
            return None
        return [wf[self._wf_slot[g]] if g in self._wf_slot else None
                for g in range(len(self.static.subs))]

    def _wf_per_subgrid(self, wf):
        """Per subgrid ``None`` or its resampled source vector (from the slot list)."""
        if not self._wf_slot or wf is None:
            return None
        return [wf[self._wf_slot[g]] if g in self._wf_slot else None
                for g in range(len(self.static.subs))]

    def _sweep_graphs(self, nsrc: int):
        """The CUDA Graphs of this propagator's fused sweeps (:mod:`.sweep`)."""
        sg = self.__dict__.get("_sweep")
        if sg is None or sg.nsrc != max(1, int(nsrc)):
            fine = [g in self._wf_slot for g in range(len(self.static.subs))]
            sg = self._sweep = sweep.SweepGraphs(self.static, self.src, self.rec, nsrc,
                                                 self.ratios, fine)
        return sg

    def release(self):
        """Free the CUDA Graphs of this propagator and the sweep buffers of its solver."""
        self.__dict__.pop("_sweep", None)
        sweep.release(self.static)

    def _amps(self, w, wf_list, n):
        nt = w.shape[1]
        amp = w[:, n] if n < nt else torch.zeros_like(w[:, 0])
        af = None if wf_list is None else [None if x is None else hsg.fine_step_samples(x, n, r)
                                           for x, r in zip(wf_list, self.ratios)]
        return amp, af

    @staticmethod
    def _scatter_fine(g_wf, ga, n, r):
        start = (n - 1) * r + 1
        for j in range(r):
            i = start + j
            if 0 <= i < g_wf.shape[1]:
                g_wf[:, i] += ga[:, j]

    def __call__(self, eps_r, sigma, source_amplitudes):
        w = torch.as_tensor(source_amplitudes, device=self.device).to(self.dtype)
        if w.ndim == 3:
            w = w[:, :, 0]
        if w.ndim == 1:
            w = w[None]
        nsrc = int(max([p.item.max().item() + 1 for p in [self.src.main] + self.src.sub
                        if p.item.numel()] + [1]))
        if w.shape[0] == 1 and nsrc > 1:
            w = w.expand(nsrc, -1)
        fields = self.solver.material_fields(eps_r.to(self.device, self.dtype),
                                             sigma.to(self.device, self.dtype))
        coefs = self.solver.build_coefficients(*fields)
        # a fresh indicator per propagation; filled by the reverse sweep (see :mod:`.indicators`)
        self.indicator_obj = DWRIndicator(self.solver) if self.indicator else None
        wf = [hsg.fine_source_samples(w, self.ratios[g]) for g in sorted(self._wf_slot,
                                                                          key=self._wf_slot.get)]
        return _SubgridPropagate.apply(self, coefs, w, len(wf), *wf, *coefs.tensors())

    def time_axis(self, nt: int) -> torch.Tensor:
        """Times ``(n + 1) dt`` of the receiver samples."""
        return (torch.arange(nt, dtype=torch.float64) + 1) * self.dt


__all__ = ["SubgridPropagator"]
