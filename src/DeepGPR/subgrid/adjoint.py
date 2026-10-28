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
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

from . import graphs, hsg
from .indicators import DWRIndicator, third_difference


class _SubgridPropagate(torch.autograd.Function):
    """``data = F(c, w, wf)``; backward = checkpointed discrete adjoint."""

    @staticmethod
    def forward(ctx, prop, template, w, n_wf, *tensors):  # type: ignore[override]
        st = prop.static
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

    @staticmethod
    def backward(ctx, grad_data):  # type: ignore[override]
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
