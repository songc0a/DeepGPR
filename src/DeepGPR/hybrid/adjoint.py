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
(:mod:`DeepGPR.hybrid.functional3d`). On CUDA its step and the step's transpose run
entirely as fused kernels with explicit outputs (:mod:`.fused3d`: global step, local
time stepping and pure Yee), and the forward pass, the recomputation and the reverse
sweep are replayed as CUDA Graphs on fixed buffers (:mod:`.sweep3d`, kept on the
propagator and its solver between calls; :meth:`HybridPropagator.release` frees them).
``DEEPGPR_HYBRID_DISABLE_GRAPHS=1`` runs the same fused steps eagerly;
``DEEPGPR_HYBRID_AUTOGRAD_VJP=1`` differentiates the PyTorch-glue step with autograd.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional

import torch

from . import functional as fn
from . import functional3d as fn3
from . import fused2d, fused3d, graphs, sweep3d, vjp2d, vjp3d, vjp3d_lts


class _Checkpoints(dict):
    """Checkpoint states of the forward pass, kept on the device or (``offload``) in
    host memory; ``ckpts[n]`` returns the state on the device either way."""

    def __init__(self, device, offload: bool):
        super().__init__()
        self.device, self.offload = device, offload

    def __setitem__(self, n, state):
        if self.offload and n != 0:
            state = tuple(t.detach().to("cpu") for t in state)
        super().__setitem__(n, state)

    def __getitem__(self, n):
        state = super().__getitem__(n)
        if self.offload and n != 0:
            state = tuple(t.to(self.device, non_blocking=True) for t in state)
        return state


def _offload(prop, st, nt: int, k: int) -> bool:
    """Keep the checkpoints in host memory when they would take more than
    ``DEEPGPR_HYBRID_CKPT_GPU_MB`` (default 4096) of device memory (``prop.offload``
    True/False forces it)."""
    if prop.offload is not None:
        return bool(prop.offload)
    if torch.device(st.device).type != "cuda" or k > nt:
        return False
    limit = float(os.environ.get("DEEPGPR_HYBRID_CKPT_GPU_MB", "4096")) * 2 ** 20
    return graphs.state_bytes(st) * (nt // max(k, 1)) > limit


def _offload_fused(prop, st, nt: int, k: int) -> bool:
    """Checkpoint placement of the fused 3D path: on the device whenever they fit in the
    free memory next to the sweep buffers (``k + 3`` states, unless already allocated) and
    about ten states of other needs plus 1 GB; ``prop.offload`` or an explicit
    ``DEEPGPR_HYBRID_CKPT_GPU_MB`` select the fixed rule of :func:`_offload`."""
    if prop.offload is not None or os.environ.get("DEEPGPR_HYBRID_CKPT_GPU_MB") is not None:
        return _offload(prop, st, nt, k)
    if torch.device(st.device).type != "cuda" or k > nt:
        return False
    dev = torch.device(st.device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    try:
        free = torch.cuda.mem_get_info(idx)[0]
    except Exception:  # pragma: no cover - old PyTorch
        return _offload(prop, st, nt, k)
    free += torch.cuda.memory_reserved(idx) - torch.cuda.memory_allocated(idx)
    sb = graphs.state_bytes(st)
    buf = st.__dict__.get("_sweep_buffers")
    sweep = 0 if (buf is not None and buf.alive and buf.k >= k) else (k + 3) * sb
    return (nt // max(k, 1)) * sb + sweep + 10 * sb + (1 << 30) > free


def _lts(st) -> bool:
    """Local time stepping of the band (the step is sub-cycled)."""
    return st.dg is not None and int(getattr(st.dg, "lts", 1)) > 1


def _fused_vjp(F, st):
    """The hand-written step transpose for this step (``None``: use autograd)."""
    if os.environ.get("DEEPGPR_HYBRID_AUTOGRAD_VJP"):
        return None
    if F is fn3 and vjp3d.supported(st):
        return vjp3d
    if F is fn3 and vjp3d_lts.supported(st) and not os.environ.get("DEEPGPR_HYBRID_LTS_AUTOGRAD"):
        return vjp3d_lts
    if F is fn and vjp2d.supported(st):
        return vjp2d
    return None


class _HybridPropagate(torch.autograd.Function):
    """``data = F(w, c)``; backward = checkpointed discrete adjoint."""

    @staticmethod
    def forward(ctx, prop, template, w, *coef_tensors):  # type: ignore[override]
        st, F = prop.static, prop.fn
        ctx.fused3d = None
        mod = fused3d if F is fn3 else fused2d
        if (mod.any_available(st) and not os.environ.get("DEEPGPR_HYBRID_AUTOGRAD_VJP")
                and not (mod.lts_available(st) and os.environ.get("DEEPGPR_HYBRID_LTS_AUTOGRAD"))):
            return _HybridPropagate._forward_fused3d(ctx, prop, template, w, coef_tensors)
        coefs = template.replaced(coef_tensors)
        nt, k = w.shape[1], prop.checkpoint_every
        state = F.zero_state(st)
        if not any(ctx.needs_input_grad):
            k = nt + 1               # forward only: no checkpoints needed
        checkpoints = _Checkpoints(st.device, _offload(prop, st, nt, k))
        checkpoints[0] = state
        data = torch.zeros((st.nshot, nt, prop.rec_ops.count), dtype=st.dtype, device=st.device)
        fused = (vjp3d.supported(st) if F is fn3 else vjp2d.supported(st))
        # graph replay of the fused step, and of the sub-cycled LTS step (verified to
        # capture; the autograd sweep below replays it for the segment recomputation)
        graph = (graphs.make(st, coefs, prop.src_ops, prop.rec_ops, w.shape[0])
                 if fused or _lts(st) else None)
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
    def _forward_fused3d(ctx, prop, template, w, coef_tensors):
        """Forward pass on the fully fused global step (:mod:`.fused3d`), replayed as
        CUDA Graphs on fixed buffers (:mod:`.sweep3d`) unless graphs are disabled."""
        st = prop.static
        coefs = template.replaced([t.detach().contiguous() for t in coef_tensors])
        wd = w.detach().contiguous()
        nt = wd.shape[1]
        data = torch.zeros((st.nshot, nt, prop.rec_ops.count), dtype=st.dtype, device=st.device)
        sg = prop._sweep_graphs(wd.shape[0]) if sweep3d.enabled(st) else None
        if not any(ctx.needs_input_grad):
            with torch.no_grad():
                if sg is not None:
                    sg.run(coefs, wd, nt, data)
                else:
                    _eager_forward(prop, coefs, wd, nt, None, data, None)
            return data
        k = sweep3d.sweep_k(nt, prop.checkpoint_every)
        if sg is not None:
            k = sweep3d.fit_k(st, k, nt)     # fewer resident states when memory is short
        checkpoints = _Checkpoints(st.device, _offload_fused(prop, st, nt, k))
        with torch.no_grad():
            oom = False
            if sg is not None:
                try:
                    sg.forward(coefs, wd, nt, k, data, checkpoints)
                except RuntimeError as exc:          # graph buffers do not fit: eager steps
                    if not sweep3d._is_oom(exc):
                        raise
                    oom = True
            if oom:                                  # (outside the handler: frees the frames)
                sg = None
                checkpoints = None
                _release_after_oom(prop)
                checkpoints = _Checkpoints(st.device, _offload_fused(prop, st, nt, k))
            if sg is None:
                _eager_forward(prop, coefs, wd, nt, k, data, checkpoints)
        ctx.fused3d = dict(k=k, sg=sg)
        ctx.prop, ctx.template, ctx.checkpoints = prop, template, checkpoints
        ctx.save_for_backward(w, *coef_tensors)
        return data

    @staticmethod
    def _backward_fused3d(ctx, grad_data):
        prop, template = ctx.prop, ctx.template
        w, *coef_tensors = ctx.saved_tensors
        need_w = ctx.needs_input_grad[2]
        need_c = any(ctx.needs_input_grad[3:])
        coefs = template.replaced([t.detach().contiguous() for t in coef_tensors])
        wd = w.detach().contiguous()
        gd = grad_data.detach().contiguous()
        nt, k, sg = wd.shape[1], ctx.fused3d["k"], ctx.fused3d["sg"]
        g_w = torch.zeros_like(wd) if need_w else None
        with torch.no_grad():
            acc, oom = None, False
            if sg is not None:
                try:
                    acc = sg.backward(coefs, wd, gd, nt, k, ctx.checkpoints, need_c, need_w, g_w)
                except RuntimeError as exc:          # reverse graphs do not fit: eager steps
                    if not sweep3d._is_oom(exc):
                        raise
                    oom = True
            if oom:                                  # (outside the handler: frees the frames)
                acc = None
                _release_after_oom(prop)
                if g_w is not None:
                    g_w.zero_()
            if acc is None:
                acc = _eager_backward(prop, coefs, wd, gd, nt, k, ctx.checkpoints, need_c,
                                      need_w, g_w)
            g = acc.finalize(coefs) if need_c else {}
        ctx.checkpoints = None
        g_coef = [g[nm] if nm in g else (torch.zeros_like(t) if need_c else None)
                  for nm, t in zip([nm for nm in template.NAMES if getattr(template, nm) is not None],
                                   coef_tensors)]
        return (None, None, g_w, *g_coef)

    @staticmethod
    def backward(ctx, grad_data):  # type: ignore[override]
        if ctx.fused3d is not None:
            return _HybridPropagate._backward_fused3d(ctx, grad_data)
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
        graph, ctx.graph = ctx.graph, None
        vjp = _HybridPropagate._vjp_graph(prop, template, coef_tensors, w, need_c, need_w)
        if vjp is not None:
            return _HybridPropagate._backward_graphed(ctx, vjp, graph, grad_data, w, g_w,
                                                      need_c, need_w)
        for start in reversed(range(0, nt, k)):
            stop = min(nt, start + k)
            states = [ctx.checkpoints[start]]
            with torch.no_grad():
                if graph is not None:              # replay for the segment recomputation
                    graph.set_state(states[0])
                for n in range(start, stop - 1):
                    if graph is not None:
                        graph(w[:, n])
                        states.append(graph.snapshot())
                    else:
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
    def _vjp_graph(prop, template, coef_tensors, w, need_c, need_w):
        """The graphed reverse step of ``prop`` (cached on it, coefficients refreshed),
        for the sub-cycled LTS step on CUDA; ``None`` otherwise."""
        st = prop.static
        if not (_lts(st) and torch.device(st.device).type == "cuda"):
            return None
        key = (need_c, need_w, w.shape[0])
        cache = prop.__dict__.setdefault("_vjp_graphs", {})
        vjp = cache.get(key)
        if vjp is None:
            vjp = graphs.make_vjp(st, template, [t.detach() for t in coef_tensors], prop.src_ops,
                                  prop.rec_ops, w.shape[0], need_c, need_w)
            if vjp is None:
                return None
            cache.clear()                      # one graph per propagator (memory)
            cache[key] = vjp
        vjp.reset([t.detach() for t in coef_tensors])
        return vjp

    @staticmethod
    def _backward_graphed(ctx, vjp, graph, grad_data, w, g_w, need_c, need_w):
        """Reverse sweep with the graphed autograd step transpose (same recursion as
        the eager autograd sweep)."""
        prop = ctx.prop
        st, F = prop.static, prop.fn
        nt, k = w.shape[1], prop.checkpoint_every
        lam = list(F.zero_state(st))
        with torch.no_grad():
            for start in reversed(range(0, nt, k)):
                stop = min(nt, start + k)
                states = [ctx.checkpoints[start]]
                if graph is not None:
                    graph.set_state(states[0])
                for n in range(start, stop - 1):
                    if graph is not None:
                        graph(w[:, n])
                        states.append(graph.snapshot())
                    else:
                        states.append(F.step(st, vjp._args[1], states[-1], prop.src_ops,
                                             w[:, n]))
                for n in reversed(range(start, stop)):
                    lam = vjp(states[n - start], lam, grad_data[:, n], w[:, n])
                    states[n - start] = None
                    if need_w:
                        g_w[:, n] = vjp.g_w
                del states
        names = len(vjp.g_coef)
        g_coef = [t.clone() for t in vjp.g_coef] if need_c else [None] * names
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


def _release_after_oom(prop):
    """Drop the graphs and sweep buffers after an out-of-memory error and return the
    memory to the device before the eager fallback."""
    import gc
    prop.release()
    gc.collect()
    torch.cuda.empty_cache()


def _eager_forward(prop, c, w, nt, k, data, checkpoints):
    """Fused steps without graphs (fresh output buffers); checkpoints every ``k``."""
    st = prop.static
    mod = sweep3d.module(st)
    pre = mod.prepare(st, c)
    state = prop.fn.zero_state(st)
    if checkpoints is not None:
        checkpoints[0] = state
    tmp = torch.empty((st.nshot, prop.rec_ops.count), dtype=st.dtype, device=st.device)
    for n in range(nt):
        out = mod.empty_state(st, state)
        mod.step_any(st, c, state, out, prop.src_ops, w[:, n].contiguous(), prop.rec_ops,
                     tmp, pre)
        data[:, n] = tmp
        state = out
        if checkpoints is not None and (n + 1) % k == 0 and n + 1 < nt:
            checkpoints[n + 1] = state
    return state


def _eager_backward(prop, c, w, gd, nt, k, checkpoints, need_c, need_w, g_w):
    """Reverse sweep with the fused step transpose, without graphs."""
    st = prop.static
    mod = sweep3d.module(st)
    pre = mod.prepare(st, c)
    acc = fused3d.Accumulators()
    lam = list(prop.fn.zero_state(st))
    gamp = torch.zeros(w.shape[0], dtype=w.dtype, device=w.device)
    for start in reversed(range(0, nt, k)):
        stop = min(nt, start + k)
        states = [checkpoints[start]]
        for n in range(start, stop):
            out = mod.empty_state(st, states[0])
            mod.step_any(st, c, states[-1], out, prop.src_ops, w[:, n].contiguous(), pre=pre)
            states.append(out)
        for n in reversed(range(start, stop)):
            lam_out = mod.empty_state(st, states[0])
            mod.vjp_any(st, c, states[n - start], states[n + 1 - start], lam, lam_out,
                        prop.src_ops, w[:, n].contiguous(), gd[:, n].contiguous(),
                        prop.rec_ops, acc, need_c, need_w, gamp, pre)
            lam = lam_out
            states[n + 1 - start] = None
            if need_w:
                g_w[:, n] = gamp
        del states
    return acc


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
        offload_checkpoints: Keep the checkpoint states in host memory (``None``:
            automatic, when they would exceed ``DEEPGPR_HYBRID_CKPT_GPU_MB`` = 4096 MB
            on the device; the reverse sweep copies one back per segment).
    """

    solver_name = "hybrid_dgtd"

    def __init__(self, solver, source_positions, receiver_positions, *,
                 checkpoint_every: int = None, offload_checkpoints: Optional[bool] = None):
        self.solver = solver
        self.static = solver.static
        self.fn = fn3 if isinstance(self.static, fn3.StepStatic3D) else fn
        self.src_ops = solver.point_ops(source_positions, source=True)
        self.rec_ops = solver.point_ops(receiver_positions, source=False)
        self.dt = solver.dt
        self.checkpoint_every = checkpoint_every
        self._auto_checkpoint = checkpoint_every is None
        self.offload = offload_checkpoints
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
        if self._auto_checkpoint:          # sqrt(nt) of *this* call (nt may change between calls)
            self.checkpoint_every = max(1, int(math.ceil(math.sqrt(w.shape[1]))))
        fields = self.solver.material_fields(eps_r.to(self.device, self.dtype),
                                             sigma.to(self.device, self.dtype))
        coefs = self.fn.build_coefficients(self.static, fields, self.dt)
        return _HybridPropagate.apply(self, coefs, w, *coefs.tensors())

    def time_axis(self, nt: int) -> torch.Tensor:
        """Times ``(n + 1) dt`` of the receiver samples."""
        return (torch.arange(nt, dtype=torch.float64) + 1) * self.dt

    def release(self):
        """Free the CUDA Graphs of this propagator and the sweep buffers of its solver
        (they are kept between calls; rebuilt on the next call)."""
        self.__dict__.pop("_sweep", None)
        sweep3d.release(self.static)

    def _sweep_graphs(self, nsrc: int):
        """The CUDA Graphs of this propagator's fused 3D sweeps (:mod:`.sweep3d`)."""
        sg = self.__dict__.get("_sweep")
        if sg is None or sg.nsrc != max(1, int(nsrc)):
            sg = self._sweep = sweep3d.SweepGraphs(self.static, self.src_ops, self.rec_ops, nsrc)
        return sg


__all__ = ["HybridPropagator"]
