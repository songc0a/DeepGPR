r"""Hand-written vector-Jacobian product of the fused CUDA 2D (TM) hybrid step.

:func:`DeepGPR.hybrid.functional.step` on the fused kernels of :mod:`.cuda2d`
is (``u = u^n``, primes for ``u^(n+1)``)::

    1  (Hx1, Hy1, phy', phx') = YeeH(Ez, Hx, Hy, phy, phx)
    2  X                      = P Ez|rows                      (interface projection)
    3  T_h                    = Terms_H(E, Hx_d, Hy_d, X; Ym, Yp)
    4  (Hx_d', Hy_d', qh')    = DGUpdH(T_h, Hx_d, Hy_d, qh)
    5  Hy'                    = Hy1, ghost rows 2 C^T Hy_d' - Hy1|inner   (Hx' = Hx1)
    6  (Ez1, pex', pey')      = YeeE(Hx', Hy', Ez, pex, pey; ca, cb)
    7  T_e                    = Terms_E(E, Hx_d', Hy_d'; Zm, Zp)
    8  (E1, qe')              = DGUpdE(T_e, E, qe; ca, cb)
    9  (Ez', E')              = inject(Ez1, E1; cb, w)

:func:`step_vjp` applies the transposes in reverse order: the transpose
kernels for 1, 3, 4, 6, 7 and 8, explicit transposes of the linear coupling
operations 2, 5 and 9. It uses ``u^n`` and ``u^(n+1)`` and recomputes only ``X``
and ``T_e`` (for the ``cb`` cotangent of the DG update). Checked against the
autograd VJP of the same step in ``tests/test_hybrid_adjoint.py``.
"""

from __future__ import annotations

from typing import Optional

import torch

from . import functional as fn
from .vjp3d import _SHOTS, _acc, finalize  # noqa: F401  (finalize re-exported)


def supported(st: fn.StepStatic) -> bool:
    """``True`` when every present part (Yee, DG) runs on the fused kernels."""
    y, d = st.yee, st.dg
    if torch.device(st.device).type != "cuda" or (y is None and d is None):
        return False
    return (y is None or y.cuda is not None) and (d is None or d.cuda is not None)


def _ranges(st):
    out, o = [], 0
    for cp in st.couplings:
        n = int(cp.slots.numel())
        out.append(slice(o, o + n))
        o += n
    return out


def sample_adjoint(st: fn.StepStatic, ops, g: torch.Tensor, lam):
    """``lam += R^T g`` for :func:`~DeepGPR.hybrid.functional.sample` (out of place)."""
    S, n = st.nshot, ops.count
    lam = list(lam)
    flat = g.reshape(-1)
    if ops.fd_shot.numel():
        Ez = lam[0]
        vals = flat.index_select(0, ops.fd_shot * n + ops.fd_item) * ops.fd_w
        lam[0] = Ez.reshape(-1).index_add(0, ops.fd_shot * Ez[0].numel() + ops.fd_node,
                                          vals).view(Ez.shape)
    if ops.dg_shot.numel():
        d = st.dg
        E = lam[7]
        vals = flat.index_select(0, ops.dg_shot * n + ops.dg_item)[:, None] * ops.dg_w
        lam[7] = E.reshape(S * d.K, d.Np).index_add(
            0, ops.dg_shot * d.K + ops.dg_elem, vals).view(E.shape)
    return lam


def _inject_adjoint(st, c, ops, amp, lEz, lE, acc, need_c, need_w):
    S = st.nshot
    g_amp = torch.zeros_like(amp) if need_w else None
    if ops.fd_shot.numel():
        lf = lEz.reshape(-1).index_select(0, ops.fd_shot * lEz[0].numel() + ops.fd_node)
        if need_w:
            g_amp.index_add_(0, ops.fd_item,
                             -lf * c.fd_cb.reshape(-1).index_select(0, ops.fd_node) * ops.fd_w)
        if need_c:
            _acc(acc, "fd_cb:src", torch.zeros_like(c.fd_cb).reshape(-1).index_add_(
                0, ops.fd_node, -lf * ops.fd_w * amp.index_select(0, ops.fd_item)))
    if ops.dg_shot.numel():
        d = st.dg
        le = lE.reshape(S * d.K, d.Np).index_select(0, ops.dg_shot * d.K + ops.dg_elem)
        s = (le * ops.dg_w).sum(-1)
        if need_w:
            g_amp.index_add_(0, ops.dg_item,
                             -s * c.dg_cb.reshape(-1).index_select(0, ops.dg_elem))
        if need_c:
            _acc(acc, "dg_cb:src", torch.zeros_like(c.dg_cb).reshape(-1).index_add_(
                0, ops.dg_elem, -s * amp.index_select(0, ops.dg_item)))
    return g_amp


def step_vjp(st: fn.StepStatic, c: fn.StepCoefs, u, u_next, lam, src=None, amp=None,
             acc: Optional[dict] = None, need_c: bool = True, need_w: bool = False):
    """Transpose of one fused 2D step (arguments as :func:`.vjp3d.step_vjp`)."""
    y, d = st.yee, st.dg
    yc = None if y is None else y.cuda
    dc = None if d is None else d.cuda
    S = st.nshot
    Ez, Hx, Hy, pex, pey, phx, phy, E, Hxd, Hyd, qex, qey, qhx, qhy = u
    Hx1, Hy1, Hxd1, Hyd1 = u_next[1], u_next[2], u_next[8], u_next[9]
    lEz, lHx, lHy, lpex, lpey, lphx, lphy, lE, lHxd, lHyd, lqex, lqey, lqhx, lqhy = (
        t.contiguous() for t in lam)
    cs = st.couplings
    sc = getattr(st, "staircase", None)
    ranges = _ranges(st)
    out = [None] * 14

    # ---------------------------------------------------------------- recompute
    if sc is not None:
        X = sc.trace(Ez, True)
        Th, Tv = sc.traces(Hxd1, Hyd1)
        gTh, gTv = torch.zeros_like(Th), torch.zeros_like(Tv)
    else:
        X = fn.interface_trace(st, Ez, True) if (d is not None and cs) else None
    T_e = dc.terms_fwd(1, E, Hxd1, Hyd1, None, c.Zm, c.Zp) if d is not None else None

    # ---------------------------------------------------------------- 9 inject
    g_amp = None
    if src is not None:
        g_amp = _inject_adjoint(st, c, src, amp, lEz, lE, acc, need_c, need_w)

    # ---------------------------------------------------------------- 8, 7 DG E update and terms
    if d is not None:
        ca, cb = c.dg_ca.reshape(-1), c.dg_cb.reshape(-1)
        gT, gE_a, _, gqex, gqey, gca, gcb = dc.update_bwd(1, T_e, E, qex, qey, ca, cb, lE,
                                                          None, lqex, lqey)
        del T_e
        if need_c:
            _acc(acc, "dg_ca:upd", gca, _SHOTS)
            _acc(acc, "dg_cb:upd", gcb, _SHOTS)
        lE_n, lHxd1, lHyd1, _, gzm, gzp = dc.terms_bwd(1, E, Hxd1, Hyd1, gT, c.Zm, c.Zp, False,
                                                       addE=gE_a, addHx=lHxd, addHy=lHyd)
        del gT
        if need_c:
            _acc(acc, "Zm:flux", gzm, _SHOTS)
            _acc(acc, "Zp:flux", gzp, _SHOTS)
        out[10], out[11] = gqex, gqey

    # ---------------------------------------------------------------- 6' concave corners (staircase)
    cv = None
    if sc is not None and sc.n_concave:
        lEz, gEold, cvx, cvy, gca_c, gcb_c = sc.concave_transpose(lEz, Ez, Hx1, Hy1, c.fd_ca,
                                                                  c.fd_cb, Th, Tv, gTh, gTv)
        cv = (gEold, cvx, cvy)
        if need_c:
            node = sc.cv["node"]
            _acc(acc, "fd_ca:cv", torch.zeros_like(c.fd_ca).reshape(-1).index_add_(0, node, gca_c))
            _acc(acc, "fd_cb:cv", torch.zeros_like(c.fd_cb).reshape(-1).index_add_(0, node, gcb_c))

    # ---------------------------------------------------------------- 6 Yee E update
    lHy2 = lHy
    lHx1 = lHx
    if y is not None:
        gHx1, gHy1, gEz_a, gpex, gpey, gfca, gfcb = yc.e_bwd(Hx1, Hy1, Ez, pex, pey, c.fd_ca,
                                                             c.fd_cb, lEz, lpex, lpey)
        if need_c:
            _acc(acc, "fd_ca:upd", gfca, _SHOTS)
            _acc(acc, "fd_cb:upd", gfcb, _SHOTS)
        lHx1 = lHx + gHx1
        lHy2 = lHy + gHy1
        lEz_n = gEz_a
        out[3], out[4] = gpex, gpey

    # ---------------------------------------------------------------- 5 ghost values
    if sc is not None:
        if cv is not None:
            gEold, cvx, cvy = cv
            lEz_n.view(S, -1).index_add_(1, gEold[0], gEold[1])
            for idx, v in cvx:
                lHx1.view(S, -1).index_add_(1, idx, v)
            for idx, v in cvy:
                lHy2.view(S, -1).index_add_(1, idx, v)
        lHx1, lHy2 = sc.ghosts_transpose(lHx1, lHy2, gTh, gTv)
        sc.traces_transpose(gTh, gTv, lHxd1, lHyd1)
    elif d is not None and cs:
        hflat = lHyd1.view(S, -1)
        for cp in cs:
            gg = lHy2[:, cp.ghost, :].clone()
            hflat.index_add_(1, cp.nodes, 2.0 * cp.to_fd.transpose_apply(gg))
            lHy2[:, cp.inner, :] -= gg
            lHy2[:, cp.ghost, :] = 0.0

    # ---------------------------------------------------------------- 4, 3 DG H update and terms
    if d is not None:
        gT, gHxd_a, gHyd_a, gqhx, gqhy, _, _ = dc.update_bwd(0, None, None, qhx, qhy, None,
                                                             None, lHxd1, lHyd1, lqhx, lqhy)
        lE_n, lHxd_n, lHyd_n, gX, gym, gyp = dc.terms_bwd(0, E, Hxd, Hyd, gT, c.Ym, c.Yp,
                                                          X is not None, addE=lE_n,
                                                          addHx=gHxd_a, addHy=gHyd_a)
        del gT
        if need_c:
            _acc(acc, "Ym:flux", gym, _SHOTS)
            _acc(acc, "Yp:flux", gyp, _SHOTS)
        out[7], out[8], out[9], out[12], out[13] = lE_n, lHxd_n, lHyd_n, gqhx, gqhy
        # ------------------------------------------------------------ 2 projection
        if sc is not None:
            sc.trace_transpose(gX, lEz_n)
        elif X is not None:
            for cp, rg in zip(cs, ranges):
                lEz_n[:, cp.row, :] += cp.to_dg.transpose_apply(gX[:, rg])

    # ---------------------------------------------------------------- 1 Yee H update
    if y is not None:
        gEz_b, gHx0, gHy0, gphy, gphx = yc.h_bwd(lHx1.contiguous(), lHy2.contiguous(), lphy,
                                                 lphx)
        lEz_n += gEz_b
        out[0], out[1], out[2], out[5], out[6] = lEz_n, gHx0, gHy0, gphx, gphy
    for i, t in enumerate(out):
        if t is None:
            out[i] = lam[i]       # empty parts (pure Yee / pure DG) pass through
    return out, g_amp


__all__ = ["finalize", "sample_adjoint", "step_vjp", "supported"]
