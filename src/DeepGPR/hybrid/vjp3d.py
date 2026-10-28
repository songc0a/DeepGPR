r"""Hand-written vector-Jacobian product of the fused CUDA 3D hybrid step.

:func:`DeepGPR.hybrid.functional3d.step` with the fused kernels of
:mod:`.cuda3d` is (in order, ``u = u^n``, primes for ``u^(n+1)``)::

    1  (fh1, psh', Bfd')   = YeeH(fe, fh, psh, Bfd)
    2  E*                  = P fe|rows                     (interface projection)
    3  T_h                 = Terms(E, H, E*; e0..e3)
    4  (H', B')            = DGUpdH(T_h, H, B)
    5  fh'                 = fh1, ghost rows 2 C^T H'_t - fh1|inner
    6  pen                 = s [C^T(g E_t) - C^T(g) fe|rows]  (interface penalty)
    7  (fe1, pse', Dfd')   = YeeE(fh', fe, pse, Dfd; ca, cb, UPML, pen)
    8  X_h                 = g n_x (E*_z, -E*_y)
    9  T_e                 = Terms(H', E, X_h; h0..h3)
    10 (E1, D')            = DGUpdE(T_e, E, D; ca, cb, UPML)
    11 (fe', E')           = inject(fe1, E1; cb, w)

:func:`step_vjp` applies the transpose of exactly these operations in reverse
order: the transpose kernels of :mod:`.cuda3d` for 1, 3, 4, 7, 9 and 10 and
explicit transposes of the (sparse, linear) coupling operations 2, 5, 6, 8 and
11. It needs ``u^n`` and ``u^(n+1)`` (both available in the checkpointed
reverse sweep) and recomputes only ``E*``, ``X_h``, ``pen`` and ``T_e``;
compared with reverse-mode differentiation of the step it avoids the second
forward evaluation and the autograd bookkeeping (zero-filled cotangents of
every gather, slice backward, accumulation). It is checked against the
autograd VJP of the same step in ``tests/test_hybrid_3d.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from . import functional3d as fn3


def supported(st: fn3.StepStatic3D) -> bool:
    """``True`` when the step runs fully on the fused kernels (Yee and DG)."""
    y, d = st.yee, st.dg
    return (y is not None and y.cuda is not None and d is not None and d.cuda is not None
            and torch.device(st.device).type == "cuda")


def _ranges(st: fn3.StepStatic3D):
    """Range of each coupling in the compact interface-slot layout."""
    out, o = [], 0
    for cp in st.couplings:
        n = int(cp.slots.numel())
        out.append(slice(o, o + n))
        o += n
    return out


def _acc(acc: dict, key: str, g: Optional[torch.Tensor], convert=None):
    """Accumulate the fresh tensor ``g`` under ``key`` (``"<coefficient>:<source>"``)
    in its own (kernel, per-shot) layout; :func:`finalize` applies ``convert`` (sum
    over shots, transposition) once at the end of the sweep."""
    if g is None:
        return
    ent = acc.get(key)
    if ent is None:
        acc[key] = [g, convert]
    else:
        ent[0].add_(g)


def finalize(acc: dict, c: fn3.StepCoefs3D) -> Dict[str, torch.Tensor]:
    """Coefficient cotangents (name -> tensor shaped like the coefficient)."""
    out: Dict[str, torch.Tensor] = {}
    for key, (t, convert) in acc.items():
        name = key.split(":")[0]
        v = (convert(t) if convert is not None else t).reshape(getattr(c, name).shape)
        out[name] = v if name not in out else out[name] + v
    return out


_SHOTS = lambda t: t.sum(0)  # noqa: E731
_FACE = lambda t: t.sum(0).t()  # noqa: E731
_FACE2 = lambda t: t.sum(0).transpose(1, 2)  # noqa: E731


def sample_adjoint(st: fn3.StepStatic3D, ops, g: torch.Tensor, lam: List[torch.Tensor]):
    """``lam += R^T g`` for :func:`~DeepGPR.hybrid.functional3d.sample` (out of place
    on the touched components)."""
    S, n = st.nshot, ops.count
    lam = list(lam)
    flat = g.reshape(-1)
    if ops.fd_shot.numel():
        f = lam[ops.comp]
        vals = flat.index_select(0, ops.fd_shot * n + ops.fd_item) * ops.fd_w
        lam[ops.comp] = f.reshape(-1).index_add(0, ops.fd_shot * f[0].numel() + ops.fd_node,
                                                vals).view(f.shape)
    if ops.dg_shot.numel():
        d = st.dg
        E = lam[18]
        vals = flat.index_select(0, ops.dg_shot * n + ops.dg_item)[:, None] * ops.dg_w
        comp = E[:, ops.comp].reshape(S * d.K, d.Np).index_add(
            0, ops.dg_shot * d.K + ops.dg_elem, vals).view(S, 1, d.K, d.Np)
        lam[18] = torch.cat([E[:, :ops.comp], comp, E[:, ops.comp + 1:]], dim=1)
    return lam


def _inject_adjoint(st, c, ops, amp, lam_fe, lam_E, acc, need_c, need_w):
    """Transpose of :func:`~DeepGPR.hybrid.functional3d.inject` with respect to the
    source samples and ``cb`` (the fields pass through unchanged)."""
    S = st.nshot
    g_amp = torch.zeros_like(amp) if need_w else None
    if ops.fd_shot.numel():
        f = lam_fe[ops.comp]
        lf = f.reshape(S, -1)[ops.fd_shot, ops.fd_node]
        cbn = c.fd_cb.reshape(-1).index_select(0, ops.fd_node)
        if need_w:
            g_amp.index_add_(0, ops.fd_item, -lf * cbn * ops.fd_w)
        if need_c:
            g = torch.zeros_like(c.fd_cb).reshape(-1).index_add_(
                0, ops.fd_node, -lf * ops.fd_w * amp.index_select(0, ops.fd_item))
            _acc(acc, "fd_cb:src", g)
    if ops.dg_shot.numel():
        d = st.dg
        le = lam_E[:, ops.comp].reshape(S * d.K, d.Np).index_select(
            0, ops.dg_shot * d.K + ops.dg_elem)
        s = (le * ops.dg_w).sum(-1)
        if need_w:
            g_amp.index_add_(0, ops.dg_item, -s * c.dg_cb.reshape(-1).index_select(0, ops.dg_elem))
        if need_c:
            g = torch.zeros_like(c.dg_cb).reshape(-1).index_add_(
                0, ops.dg_elem, -s * amp.index_select(0, ops.dg_item))
            _acc(acc, "dg_cb:src", g)
    return g_amp


def step_vjp(st: fn3.StepStatic3D, c: fn3.StepCoefs3D, u, u_next, lam, src=None, amp=None,
             acc: Optional[Dict[str, torch.Tensor]] = None, need_c: bool = True,
             need_w: bool = False):
    """Transpose of one fused step.

    Args:
        st, c: Static data and coefficients (plain tensors).
        u, u_next: States ``u^n`` and ``u^(n+1) = step(u^n)`` (24 tensors each).
        lam: Cotangent of ``u^(n+1)`` (receiver terms included).
        src, amp: Source operator and samples of this step (as in ``step``).
        acc: Coefficient cotangents, accumulated in place in kernel layouts (a
            dict, initially empty; :func:`finalize` converts it); required when
            ``need_c``.

    Returns:
        ``(lam_n, g_amp)``: cotangent of ``u^n`` (24 tensors) and of ``amp``
        (``None`` unless ``need_w``).
    """
    y, d = st.yee, st.dg
    yc, dc = y.cuda, d.cuda
    S = st.nshot
    NY, NZ = y.shape[1], y.shape[2]
    fe, fh = list(u[0:3]), list(u[3:6])
    pse, psh = list(u[6:12]), list(u[12:18])
    E, H, Daux, Baux, Dfd, Bfd = u[18:24]
    fh2, H1 = list(u_next[3:6]), u_next[19]
    lfe, lfh = list(lam[0:3]), list(lam[3:6])
    lpse, lpsh = list(lam[6:12]), list(lam[12:18])
    lE, lH, lD, lB, lDfd, lBfd = lam[18:24]
    cs = st.couplings
    ranges = _ranges(st)
    g_if = c.g_if
    z = torch.zeros(0, dtype=E.dtype, device=E.device)

    # ---------------------------------------------------------------- recompute
    Eext = fn3.interface_trace(st, fe, True) if cs else None          # compact (S, 2, nX)
    Xh = fn3.interface_x(st, c, Eext, True) if (cs and g_if is not None) else None
    face, n0 = fn3.interface_slots(st) if cs else (None, None)
    pen = fn3.interface_penalty_rows(st, c, fe, E) if g_if is not None else None
    T_e = dc.terms_fwd(H1, E, Xh, c.h0, c.h1, c.h2, c.h3)

    # ---------------------------------------------------------------- 11 inject
    g_amp = None
    if src is not None:
        g_amp = _inject_adjoint(st, c, src, amp, lfe, lE, acc, need_c, need_w)

    # ---------------------------------------------------------------- 10 DG E update
    hist = c.u_hist
    gT, gE_a, gD, gca, gcb, gh, gnw, gold = dc.update_bwd(
        1, T_e, E, Daux, lE.contiguous(), lD.contiguous(), c.dg_ca.reshape(-1),
        c.dg_cb.reshape(-1), hist, c.u_new, c.u_old)
    del T_e
    if need_c:
        _acc(acc, "dg_ca:upd", gca, _SHOTS)
        _acc(acc, "dg_cb:upd", gcb, _SHOTS)
        if gh is not None and hist is not None:
            _acc(acc, "u_hist:upd", gh, _SHOTS)
            _acc(acc, "u_new:upd", gnw, _SHOTS)
            _acc(acc, "u_old:upd", gold, _SHOTS)

    # ---------------------------------------------------------------- 9 Terms (H', E, X_h)
    # cotangents of H' (+ lam_H) and of E (+ the update's), summed in the kernel
    lH1, lE_n, gXh, g0, g1, g2, g3 = dc.terms_bwd(H1, E, gT, c.h0, c.h1, c.h2, c.h3,
                                                  Xh is not None, addF=lH.contiguous(),
                                                  addG=gE_a)
    del gT
    if need_c:
        for name, g in (("h0", g0), ("h1", g1), ("h3", g3)):
            _acc(acc, name + ":flux", g, _FACE)
        _acc(acc, "h2:flux", g2, _FACE2)

    # ---------------------------------------------------------------- 8 X_h
    gEext = None                              # cotangent of E* (S, 2, nX)
    if Xh is not None:
        x0, x1 = gXh[:, 0], gXh[:, 1]
        gs = g_if.reshape(-1).index_select(0, face) * n0
        gEext = torch.stack([-gs * x1, gs * x0], dim=1)
        if need_c:
            gacc = torch.zeros_like(g_if.reshape(-1)).index_add_(
                0, face, (x0 * Eext[:, 1] - x1 * Eext[:, 0]).sum(0) * n0)
            _acc(acc, "g_if:x", gacc)
    del gXh

    # ---------------------------------------------------------------- 7 Yee E update
    fu = [t if t is not None else z for t in (c.fu_hist, c.fu_new, c.fu_old)]
    gsrc, gfe_e, gDfd, gpse, gca, gcb, gh, gnw, gold, gpen = yc.e_bwd(
        fh2, fe, Dfd, c.fd_ca, c.fd_cb, fu[0], fu[1], fu[2], z if pen is None else pen, pse,
        [t.contiguous() for t in lfe], lDfd.contiguous(), [t.contiguous() for t in lpse])
    if need_c:
        _acc(acc, "fd_ca:upd", gca, _SHOTS)
        _acc(acc, "fd_cb:upd", gcb, _SHOTS)
        if c.fu_hist is not None:
            _acc(acc, "fu_hist:upd", gh, _SHOTS)
            _acc(acc, "fu_new:upd", gnw, _SHOTS)
            _acc(acc, "fu_old:upd", gold, _SHOTS)
    lfh2 = [a + b for a, b in zip(lfh, gsrc)]
    lfe_n = gfe_e                              # fresh tensors, updated in place below

    # ---------------------------------------------------------------- 6 penalty
    if gpen is not None:
        scale = 2.0 / y.spacing[0]
        gfl = g_if.reshape(-1)
        lEf = lE_n.view(S, 3, -1)
        Ef = E.reshape(S, 3, -1)
        gacc = torch.zeros_like(gfl) if need_c else None
        for r, (cp, rg) in enumerate(zip(cs, ranges)):
            g = gfl.index_select(0, face[rg])
            for comp, lat in ((1, 0), (2, 1)):
                op = cp.to_fd[lat]
                gp = gpen[:, r, lat].reshape(S, -1)
                q = scale * op.transpose_apply(gp)                 # (S, nif)
                lEf[:, comp].index_add_(1, cp.nodes, g * q)
                row = fe[comp][:, cp.row].reshape(S, -1)
                lfe_n[comp][:, cp.row] -= (scale * op(g[None]) * gp).view(S, NY, NZ)
                if need_c:
                    em = Ef[:, comp].index_select(1, cp.nodes)
                    gacc.index_add_(0, face[rg], (em * q).sum(0)
                                    - scale * op.transpose_apply((gp * row).sum(0, keepdim=True))[0])
        if need_c:
            _acc(acc, "g_if:pen", gacc)

    # ---------------------------------------------------------------- 5 ghost rows
    lfh1 = lfh2
    for cp in cs:
        for comp, lat in ((2, 0), (1, 1)):
            gg = lfh1[comp][:, cp.ghost].clone()
            lH1.view(S, 3, -1)[:, comp].index_add_(
                1, cp.nodes, 2.0 * cp.to_fd[lat].transpose_apply(gg.reshape(S, -1)))
            lfh1[comp][:, cp.inner] -= gg
            lfh1[comp][:, cp.ghost] = 0.0

    # ---------------------------------------------------------------- 4 DG H update
    gT, gH_a, gB, *_ = dc.update_bwd(0, None, H, Baux, lH1.contiguous(), lB.contiguous())

    # ---------------------------------------------------------------- 3 Terms (E, H, E*)
    lE_n, lH_n, gX, g0, g1, g2, g3 = dc.terms_bwd(E, H, gT, c.e0, c.e1, c.e2, c.e3,
                                                  Eext is not None, addF=lE_n, addG=gH_a)
    del gT
    if need_c:
        for name, g in (("e0", g0), ("e1", g1), ("e3", g3)):
            _acc(acc, name + ":flux", g, _FACE)
        _acc(acc, "e2:flux", g2, _FACE2)

    # ---------------------------------------------------------------- 2 projection E*
    if cs:
        gEext = gX if gEext is None else gEext + gX
        for cp, rg in zip(cs, ranges):
            for comp, k in ((1, 0), (2, 1)):
                lfe_n[comp][:, cp.row] += cp.to_dg[k].transpose_apply(
                    gEext[:, k, rg]).view(S, NY, NZ)
    del gX

    # ---------------------------------------------------------------- 1 Yee H update
    gsrc, gfh, gBfd, gpsh = yc.h_bwd(fe, fh, Bfd, psh, [t.contiguous() for t in lfh1],
                                     lBfd.contiguous(), [t.contiguous() for t in lpsh])
    for k in range(3):
        lfe_n[k] += gsrc[k]
    out = lfe_n + gfh + gpse + gpsh + [lE_n, lH_n, gD, gB, gDfd, gBfd]
    return out, g_amp
