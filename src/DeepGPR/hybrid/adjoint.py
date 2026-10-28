r"""Discrete adjoint of the hybrid DGTD-FDTD time stepping.

Forward model (all quantities discrete, exactly as implemented in
:func:`DeepGPR.hybrid.functional.step`)::

    u^0 = 0,   u^(n+1) = S(u^n, c(m), w_n),   d_n = R u^(n+1),   J = J(d)

``u`` is the full coupled state (Yee E/H and CPML auxiliaries, DG E/H and DG
CPML auxiliaries), ``c(m)`` the material coefficients (Yee ``ca``/``cb``, DG
``ca``/``cb`` and face impedances of the upwind penalty), ``w_n`` the source
samples and ``R`` the receiver interpolation. ``S`` is linear in ``u`` and
``w`` and contains every operator of the scheme: DG volume and surface terms,
Yee differences, the interface projection ``C`` and its transpose, the ghost
rows, conductivity, CPML recursions and source injection.

Adjoint recursion (``lambda^nt = 0``)::

    g_n          = lambda^(n+1) + R^T dJ/dd_n
    lambda^n     = (dS/du)^T g_n                     (u^n -> u^(n+1))
    dJ/dc       += (dS/dc)^T g_n
    dJ/dw_n      = (dS/dw)^T g_n

The products ``(dS/du)^T g``, ``(dS/dc)^T g`` and ``(dS/dw)^T g`` are evaluated
by reverse-mode differentiation of ``S`` itself (one vector-Jacobian product
per step), so they are the exact transposes of the implemented step; no
continuous adjoint equations are used. ``u^n`` is needed in the reverse sweep:
the forward pass keeps a checkpoint every ``k`` steps and the backward pass
recomputes each segment once (memory ``(nt/k + k)`` states, about
``2 sqrt(nt)`` for the default ``k``).

``dJ/dc`` is passed on by autograd to the user's model-grid ``eps_r`` and
``sigma`` through :meth:`HybridTM2D.material_fields` (model pixels -> DG
elements and the refined Yee grid, air above the surface, CPML extension) and
:func:`~DeepGPR.hybrid.functional.build_coefficients`.

The same machinery serves the 3D solver :class:`~DeepGPR.hybrid.HybridMaxwell3D`
(:mod:`DeepGPR.hybrid.functional3d`); there the DG operator may run as fused
CUDA kernels whose backward pass is their exact transpose (:mod:`.cuda3d`).
"""

from __future__ import annotations

import math
import os
from typing import List

import torch

from . import functional as fn
from . import functional3d as fn3
from . import graphs, vjp2d, vjp3d


def _fused_vjp(F, st):
    """The hand-written step transpose for this step (``None``: use autograd)."""
    if os.environ.get("DEEPGPR_HYBRID_AUTOGRAD_VJP"):
        return None
    if F is fn3 and vjp3d.supported(st):
        return vjp3d
    if F is fn and vjp2d.supported(st):
        return vjp2d
    return None


class _HybridPropagate(torch.autograd.Function):
    """``data = F(w, c)``; backward = checkpointed discrete adjoint."""

    @staticmethod
    def forward(ctx, prop, template, w, *coef_tensors):  # type: ignore[override]
        st, F = prop.static, prop.fn
        coefs = template.replaced(coef_tensors)
        nt, k = w.shape[1], prop.checkpoint_every
        state = F.zero_state(st)
        checkpoints = {0: state}
        if not any(ctx.needs_input_grad):
            k = nt + 1               # forward only: no checkpoints needed
        data = torch.zeros((st.nshot, nt, prop.rec_ops.count), dtype=st.dtype, device=st.device)
        fused = (vjp3d.supported(st) if F is fn3 else vjp2d.supported(st))
        graph = graphs.make(st, coefs, prop.src_ops, prop.rec_ops, w.shape[0]) if fused else None
        if graph is not None:                       # CUDA Graph replay of the same step
            for n in range(nt):
                data[:, n] = graph(w[:, n])
                if (n + 1) % k == 0 and n + 1 < nt:
                    checkpoints[n + 1] = graph.snapshot()
        else:
            for n in range(nt):
                state = F.step(st, coefs, state, prop.src_ops, w[:, n])
                data[:, n] = F.sample(st, prop.rec_ops, state)
                if (n + 1) % k == 0 and n + 1 < nt:
                    checkpoints[n + 1] = state
        ctx.graph = graph if any(ctx.needs_input_grad) else None
        ctx.prop, ctx.template, ctx.checkpoints = prop, template, checkpoints
        ctx.save_for_backward(w, *coef_tensors)
        return data

    @staticmethod
    def backward(ctx, grad_data):  # type: ignore[override]
        prop, template = ctx.prop, ctx.template
        st, F = prop.static, prop.fn
        w, *coef_tensors = ctx.saved_tensors
        need_w = ctx.needs_input_grad[2]
        need_c = any(ctx.needs_input_grad[3:])
        leaves = [t.detach().requires_grad_(need_c) for t in coef_tensors]
        coefs_leaf = template.replaced(leaves)
        coefs_plain = template.replaced([t.detach() for t in coef_tensors])
        g_w = torch.zeros_like(w) if need_w else None
        nt, k = w.shape[1], prop.checkpoint_every
        V = _fused_vjp(F, st)
        if V is not None:
            return _HybridPropagate._backward_fused(ctx, V, grad_data, w, coefs_plain, g_w,
                                                    need_c, need_w)
        g_coef = [torch.zeros_like(t) for t in coef_tensors]
        lam = list(F.zero_state(st))
        active = [i for i, t in enumerate(lam) if t.numel()]
        for start in reversed(range(0, nt, k)):
            stop = min(nt, start + k)
            states = [ctx.checkpoints[start]]
            with torch.no_grad():
                for n in range(start, stop - 1):
                    states.append(F.step(st, coefs_plain, states[-1], prop.src_ops, w[:, n]))
            for n in reversed(range(start, stop)):
                u = states[n - start]
                with torch.enable_grad():
                    u_leaf = list(u)
                    for i in active:
                        u_leaf[i] = u[i].detach().requires_grad_(True)
                    w_n = w[:, n].detach().requires_grad_(need_w)
                    nxt = F.step(st, coefs_leaf, tuple(u_leaf), prop.src_ops, w_n)
                    rec = F.sample(st, prop.rec_ops, nxt)
                    outs = [nxt[i] for i in active] + [rec]
                    gouts = [lam[i] for i in active] + [grad_data[:, n]]
                    ins: List[torch.Tensor] = [u_leaf[i] for i in active]
                    if need_c:
                        ins += leaves
                    if need_w:
                        ins.append(w_n)
                    grads = torch.autograd.grad(outs, ins, gouts, allow_unused=True)
                for slot, i in enumerate(active):
                    g = grads[slot]
                    lam[i] = torch.zeros_like(u[i]) if g is None else g
                pos = len(active)
                if need_c:
                    for j in range(len(leaves)):
                        if grads[pos + j] is not None:
                            g_coef[j] += grads[pos + j]
                    pos += len(leaves)
                if need_w and grads[pos] is not None:
                    g_w[:, n] = grads[pos]
            del states
        return (None, None, g_w, *g_coef)

    @staticmethod
    def _backward_fused(ctx, V, grad_data, w, coefs, g_w, need_c, need_w):
        """Reverse sweep with the hand-written step transpose ``V`` (:mod:`.vjp3d` or
        :mod:`.vjp2d`, fused CUDA step); same recursion as the autograd sweep above."""
        prop, template = ctx.prop, ctx.template
        st, F = prop.static, prop.fn
        coefs = coefs.replaced([t.contiguous() for t in coefs.tensors()])
        nt, k = w.shape[1], prop.checkpoint_every
        graph, ctx.graph = ctx.graph, None
        acc: dict = {}
        lam = list(F.zero_state(st))
        with torch.no_grad():
            for start in reversed(range(0, nt, k)):
                stop = min(nt, start + k)
                states = [ctx.checkpoints[start]]
                if graph is not None:
                    graph.set_state(states[0])
                for n in range(start, stop):
                    if n + 1 == stop and stop in ctx.checkpoints:
                        states.append(ctx.checkpoints[stop])
                    elif graph is not None:
                        graph(w[:, n])
                        states.append(graph.snapshot())
                    else:
                        states.append(F.step(st, coefs, states[-1], prop.src_ops, w[:, n]))
                for n in reversed(range(start, stop)):
                    lam = V.sample_adjoint(st, prop.rec_ops, grad_data[:, n], lam)
                    lam, g_amp = V.step_vjp(st, coefs, states[n - start], states[n + 1 - start],
                                            lam, prop.src_ops, w[:, n], acc, need_c, need_w)
                    states[n + 1 - start] = None
                    if need_w:
                        g_w[:, n] = g_amp
                del states
        names = [nm for nm in template.NAMES if getattr(template, nm) is not None]
        acc = V.finalize(acc, coefs)
        g_coef = [acc[nm] if nm in acc else torch.zeros_like(t)
                  for nm, t in zip(names, coefs.tensors())]
        return (None, None, g_w, *g_coef)


class HybridPropagator:
    """Differentiable receiver data of a :class:`~DeepGPR.hybrid.HybridTM2D` or
    :class:`~DeepGPR.hybrid.HybridMaxwell3D`.

    ``data = prop(eps_r, sigma, source_amplitudes)`` with model-grid ``eps_r``
    and ``sigma`` (the layout passed to the solver) and gradients by the
    discrete adjoint of the hybrid scheme.

    Args:
        solver: A configured solver (mesh, band, ``dt``, ``nshot``).
        source_positions: ``(nshot, nsrc, 2)`` physical ``(x, y)`` [m] in 2D,
            ``(nshot, nsrc, 3)`` ``(x, y, z)`` in 3D.
        receiver_positions: ``(nshot, nrx, 2|3)``.
        checkpoint_every: Steps between stored states (default ``ceil(sqrt(nt))``).
    """

    solver_name = "hybrid_dgtd"

    def __init__(self, solver, source_positions, receiver_positions, *,
                 checkpoint_every: int = None):
        self.solver = solver
        self.static = solver.static
        self.fn = fn3 if isinstance(self.static, fn3.StepStatic3D) else fn
        self.src_ops = solver.point_ops(source_positions, source=True)
        self.rec_ops = solver.point_ops(receiver_positions, source=False)
        self.dt = solver.dt
        self.checkpoint_every = checkpoint_every
        self.device, self.dtype = solver.device, solver.dtype

    def __call__(self, eps_r, sigma, source_amplitudes):
        w = torch.as_tensor(source_amplitudes, device=self.device).to(self.dtype)
        if w.ndim == 3:
            w = w[:, :, 0]
        if w.ndim == 1:
            w = w[None]
        nsrc = int(max(self.src_ops.fd_item.max().item() if self.src_ops.fd_item.numel() else 0,
                       self.src_ops.dg_item.max().item() if self.src_ops.dg_item.numel() else 0)) + 1
        if w.shape[0] == 1 and nsrc > 1:
            w = w.expand(nsrc, -1)
        if self.checkpoint_every is None:
            self.checkpoint_every = max(1, int(math.ceil(math.sqrt(w.shape[1]))))
        fields = self.solver.material_fields(eps_r.to(self.device, self.dtype),
                                             sigma.to(self.device, self.dtype))
        coefs = self.fn.build_coefficients(self.static, fields, self.dt)
        return _HybridPropagate.apply(self, coefs, w, *coefs.tensors())

    def time_axis(self, nt: int) -> torch.Tensor:
        """Times ``(n + 1) dt`` of the receiver samples."""
        return (torch.arange(nt, dtype=torch.float64) + 1) * self.dt


__all__ = ["HybridPropagator"]
