"""Fully fused 2D (TM) hybrid step and its transpose with explicit outputs.

The 2D counterpart of :mod:`.fused3d`: :func:`step_into` / :func:`vjp_into` (one
global step, also pure Yee) and :func:`lts_step_into` / :func:`lts_vjp_into` (local
time stepping of the band, :func:`DeepGPR.hybrid.functional.lts_h_half_step`) write the
new state, or the cotangent of the old state, into given buffers; every operation is a
CUDA kernel (:mod:`.cuda2d`, the 2D coupling / source / receiver kernels of
:mod:`.glue3d`). The LTS transpose is hand-written here (before, the sub-cycled step was
differentiated with autograd). The transposes consume their input cotangent in place.
Arithmetic as :func:`DeepGPR.hybrid.functional.step`; the transposes are those of
:func:`DeepGPR.hybrid.vjp2d.step_vjp` and of reverse-mode differentiation of the
sub-cycled step. The staircase interface (``st.staircase``) keeps the PyTorch path.

State: ``(Ez, Hx, Hy, pex, pey, phx, phy, E, Hxd, Hyd, qex, qey, qhx, qhy)``.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from . import functional as fn
from .cuda3d import _check, _ptr, _ptrs, _stream
from .fused3d import Accumulators, _SHOTS
from .glue3d import combine, coupling_glue2d, point_glue  # noqa: F401

__all__ = ["Accumulators", "LTSPre2D", "any_available", "available", "empty_state",
           "lts_available", "lts_step_into", "lts_vjp_into", "prepare", "step_any", "step_into",
           "vjp_any", "vjp_into"]


def _base(st) -> bool:
    y, d = st.yee, st.dg
    if torch.device(st.device).type != "cuda" or y is None or y.cuda is None:
        return False
    if getattr(st, "staircase", None) is not None:
        return False
    return d is None or (d.cuda is not None and bool(st.couplings))


def available(st) -> bool:
    """The fused global step applies (pure Yee or band with couplings, no LTS)."""
    return _base(st) and (st.dg is None or int(getattr(st.dg, "lts", 1)) == 1)


def lts_available(st) -> bool:
    """The fused sub-cycled step applies."""
    d = st.dg
    return (_base(st) and d is not None and int(getattr(d, "lts", 1)) > 1
            and hasattr(d.cuda, "lts_z") and hasattr(st.yee.cuda, "slab_r"))


def any_available(st) -> bool:
    return available(st) or lts_available(st)


def empty_state(st, like=None) -> List[torch.Tensor]:
    ref = like if like is not None else fn.zero_state(st)
    return [torch.empty_like(t) for t in ref]


def _z(ref):
    return torch.zeros(0, dtype=ref.dtype, device=ref.device)


def _yee_coef_acc(acc, c, S):
    """Running sums of the Yee ``ca``/``cb`` cotangents (summed over the shots in the kernel)."""
    return (acc.buf("fd_ca:upd", tuple(c.fd_ca.shape), c.fd_ca),
            acc.buf("fd_cb:upd", tuple(c.fd_cb.shape), c.fd_cb))


def _flux_acc(acc, c, S, names):
    """Running sums ``(S, K * 3 Nfp)`` of the face-coefficient cotangents (``Ym``/``Yp`` or
    ``Zm``/``Zp``)."""
    return tuple(acc.buf(nm + ":flux", (S, getattr(c, nm).numel()), getattr(c, nm), _SHOTS)
                 for nm in names)


# =============================================================================== global step
def step_into(st, c, u, out, src=None, amp=None, rec=None, rec_out=None):
    """One step ``u -> out`` (14 tensors each)."""
    y, d = st.yee, st.dg
    yc, dc = y.cuda, (d.cuda if d is not None else None)
    Ez, Hx, Hy, pex, pey, phx, phy, E, Hxd, Hyd, qex, qey, qhx, qhy = u
    oEz, oHx, oHy, opex, opey, ophx, ophy, oE, oHxd, oHyd, oqex, oqey, oqhx, oqhy = out
    yc.h_fwd(Ez, Hx, Hy, phy, phx, out=(oHx, oHy, ophy, ophx))
    if d is not None:
        g = coupling_glue2d(st)
        X = torch.empty((st.nshot, g.nX), dtype=Ez.dtype, device=Ez.device)
        g.trace(g.rows_planes(Ez), X)
        T = dc.terms_fwd(0, E, Hxd, Hyd, X, c.Ym, c.Yp)
        dc.update_fwd(0, T, Hxd, Hyd, qhx, qhy, out=(oHxd, oHyd, oqhx, oqhy))
        g.ghost(g.ghost_planes(oHy), oHyd)
    yc.e_fwd(oHx, oHy, Ez, pex, pey, c.fd_ca, c.fd_cb, out=(oEz, opex, opey))
    if d is not None:
        T = dc.terms_fwd(1, E, oHxd, oHyd, None, c.Zm, c.Zp)
        dc.update_fwd(1, T, E, E, qex, qey, c.dg_ca.reshape(-1), c.dg_cb.reshape(-1),
                      out=(oE, oqex, oqey))
    if src is not None:
        point_glue(st, src).inject(oEz, oE, c.fd_cb, c.dg_cb if d is not None else None, amp)
    if rec is not None and rec_out is not None:
        point_glue(st, rec).sample(oEz, oE, rec_out)


def _sources_bwd(st, c, src, amp, lEz, lE, acc, need_c, need_w, gamp):
    if src is None or not (need_w or need_c):
        return
    if need_w:
        gamp.zero_()
    gcf = gcd = None
    if need_c:
        gcf = acc.buf("fd_cb:src", c.fd_cb.shape, c.fd_cb)
        if st.dg is not None:
            gcd = acc.buf("dg_cb:src", c.dg_cb.shape, c.dg_cb)
    point_glue(st, src).inject_bwd(lEz, lE, c.fd_cb, c.dg_cb if st.dg is not None else None,
                                   amp, gamp if need_w else None, gcf, gcd)


def vjp_into(st, c, u, u_next, lam, out, src=None, amp=None, gd=None, rec=None,
             acc: Optional[Accumulators] = None, need_c: bool = True, need_w: bool = False,
             gamp=None):
    """Transpose of :func:`step_into` (``lam`` destroyed; contract of
    :func:`DeepGPR.hybrid.fused3d.vjp_into`)."""
    y, d = st.yee, st.dg
    yc, dc = y.cuda, (d.cuda if d is not None else None)
    S = st.nshot
    Ez, Hx, Hy, pex, pey, phx, phy, E, Hxd, Hyd, qex, qey, qhx, qhy = u
    Hx1, Hy1, Hxd1, Hyd1 = u_next[1], u_next[2], u_next[8], u_next[9]
    lEz, lHx, lHy, lpex, lpey, lphx, lphy, lE, lHxd, lHyd, lqex, lqey, lqhx, lqhy = lam
    oEz, oHx, oHy, opex, opey, ophx, ophy, oE, oHxd, oHyd, oqex, oqey, oqhx, oqhy = out
    if rec is not None and gd is not None:
        point_glue(st, rec).sample_bwd(gd, lEz, lE)
    X = T_e = None
    if d is not None:
        g = coupling_glue2d(st)
        X = torch.empty((S, g.nX), dtype=Ez.dtype, device=Ez.device)
        g.trace(g.rows_planes(Ez), X)
        T_e = dc.terms_fwd(1, E, Hxd1, Hyd1, None, c.Zm, c.Zp)
    _sources_bwd(st, c, src, amp, lEz, lE, acc, need_c, need_w, gamp)
    lE_n = None
    if d is not None:
        ca, cb = c.dg_ca.reshape(-1), c.dg_cb.reshape(-1)
        gT, gE_a, _, _, _, gca, gcb = dc.update_bwd(1, T_e, E, qex, qey, ca, cb, lE, None, lqex,
                                                    lqey, out_q=(oqex, oqey))
        del T_e
        if need_c:
            acc.add("dg_ca:upd", gca, _SHOTS)
            acc.add("dg_cb:upd", gcb, _SHOTS)
        lE_n, lHxd1, lHyd1, *_ = dc.terms_bwd(
            1, E, Hxd1, Hyd1, gT, c.Zm, c.Zp, False, addE=gE_a, addHx=lHxd, addHy=lHyd,
            acc_coef=_flux_acc(acc, c, S, ("Zm", "Zp")) if need_c else None)
        del gT, gE_a
    yc.e_bwd(Hx1, Hy1, Ez, pex, pey, c.fd_ca, c.fd_cb, lEz, lpex, lpey, out=(oEz, opex, opey),
             src_acc=(lHx, lHy), coef_acc=_yee_coef_acc(acc, c, S) if need_c else None)
    if d is not None:
        g.ghost_bwd(g.ghost_planes(lHy), lHyd1)
        gT, gHxd_a, gHyd_a, *_ = dc.update_bwd(0, None, None, qhx, qhy, None, None, lHxd1, lHyd1,
                                               lqhx, lqhy, out_q=(oqhx, oqhy))
        _, _, _, gX, *_ = dc.terms_bwd(
            0, E, Hxd, Hyd, gT, c.Ym, c.Yp, True, addE=lE_n, addHx=gHxd_a, addHy=gHyd_a,
            out=(oE, oHxd, oHyd), acc_coef=_flux_acc(acc, c, S, ("Ym", "Yp")) if need_c else None)
        del gT
        g.trace_bwd(g.rows_planes(oEz), gX)
    yc.h_bwd(lHx, lHy, lphy, lphx, out=(oHx, oHy, ophy, ophx), gEz_acc=oEz)


# =============================================================================== local time stepping
def _lts2d_z(dc, Te, z, cb0, a):
    out = torch.empty_like(z)
    _check(dc.lib.lts2d_z_fwd(dc.code, z.shape[0], dc.K, dc.Np, _ptr(Te), _ptr(z), _ptr(cb0), a,
                              _ptr(out), _stream(z)), "2D LTS z update")
    return out


def _lts2d_z_bwd(dc, Te, g, cb0, a, gcb0_acc):
    """``(gTe, gcb0)``; the cotangent of ``z`` is ``g`` itself; ``gcb0`` per element summed
    over shots (accumulated into ``gcb0_acc`` when given)."""
    gTe = torch.empty_like(Te)
    gcb0 = torch.empty_like(cb0)
    _check(dc.lib.lts2d_z_bwd(dc.code, g.shape[0], dc.K, dc.Np, _ptr(Te), _ptr(g), _ptr(cb0), a,
                              _ptr(gTe), _ptr(gcb0), _stream(g)), "2D LTS z transpose")
    if gcb0_acc is not None:
        gcb0_acc.add_(gcb0)
    return gTe


def _lts2d_eta(dc, T, ex, ey, tavg, beta, b, w):
    S = T.shape[0]
    first = ex is None
    shape = (S, dc.K, dc.Np)
    exo = torch.empty(shape, dtype=T.dtype, device=T.device)
    eyo = torch.empty_like(exo)
    tavgo = torch.empty_like(T)
    _check(dc.lib.lts2d_eta_fwd(dc.code, S, dc.K, dc.Np, _ptr(T), _ptr(ex), _ptr(ey), _ptr(tavg),
                                _ptr(dc.ch), beta, b, w, int(first), _ptr(exo), _ptr(eyo),
                                _ptr(tavgo), _stream(T)), "2D LTS eta update")
    return exo, eyo, tavgo


def _lts2d_eta_bwd(dc, gex, gey, gtavg, beta, b, w, first, like_T):
    """``(gT, gex_in, gey_in)`` (``None`` inputs: zero cotangents); the cotangent of the
    input average is ``gtavg`` itself."""
    S = like_T.shape[0]
    gT = torch.empty_like(like_T)
    shape = (S, dc.K, dc.Np)
    gxi = None if first else torch.empty(shape, dtype=like_T.dtype, device=like_T.device)
    gyi = None if first else torch.empty_like(gxi)
    _check(dc.lib.lts2d_eta_bwd(dc.code, S, dc.K, dc.Np, _ptr(gex), _ptr(gey), _ptr(gtavg),
                                _ptr(dc.ch), beta, b, w, _ptr(gT), _ptr(gxi), _ptr(gyi),
                                _stream(like_T)), "2D LTS eta transpose")
    return gT, gxi, gyi


def _q2_fwd(yc, sd, ex, ey, z, cb0, a):
    out = torch.empty_like(z)
    q = _ptrs([ex, ey, z, cb0, sd["masks"][0], out])
    _check(yc.lib.lts2d_slab_q(yc.code, 0, z.shape[0], sd["n"], yc.NY, yc.gd, q, a, _stream(z)),
           "LTS slab E rates (2D)")
    return out


def _q2_bwd(yc, sd, ex, ey, g, cb0, a, gex=None, gey=None, gcb0=None):
    """Transpose of :func:`_q2_fwd` w.r.t. ``(ex, ey)`` (accumulated into ``gex``/``gey`` when
    given) and ``cb0`` (accumulated into ``gcb0`` when given); ``g`` passes to ``z``."""
    acc = gex is not None
    gex = torch.empty_like(ex) if gex is None else gex
    gey = torch.empty_like(ey) if gey is None else gey
    q = _ptrs([ex, ey, g, cb0, sd["masks"][0], gex, gey, gcb0])
    flags = 1 | (2 if acc else 0) | (4 if gcb0 is not None else 0)
    _check(yc.lib.lts2d_slab_q(yc.code, flags, g.shape[0], sd["n"], yc.NY, yc.gd, q, a, _stream(g)),
           "LTS slab E rates transpose (2D)")
    return gex, gey


def _r2_fwd(yc, sd, z, eta, ravg, first, last, beta, b, w, r0, corr_into=None):
    """Slab rates (:class:`DeepGPR.hybrid.cuda2d._SlabR2`): ``(r | None, eta_new | None,
    ravg_new, corr | None)``; with ``corr_into`` (the fine rows' slab views ``(Hx, phy, Hy,
    phx)``) the last sub-step's corrections are added there."""
    new = lambda: torch.empty_like(z)  # noqa: E731
    e = list(eta) if (eta is not None and not first and not last) else [None, None]
    ra = list(ravg) if not first else [None, None]
    cm = r0 is not None
    q0 = list(r0) if cm else [None, None]
    r_out = [new(), new()] if first else [None, None]
    eta_out = [new(), new()] if not last else [None, None]
    ravg_out = [new(), new()]
    acc = cm and corr_into is not None
    corr = (list(corr_into) if acc else [new() for _ in range(4)]) if cm else [None] * 4
    stc = sd["stat"] + sd["masks"][1:]
    q = _ptrs([z] + e + ra + stc + r_out + eta_out + ravg_out + q0 + corr)
    _check(yc.lib.lts2d_slab_r(yc.code, 0, z.shape[0], sd["n"], yc.NY, yc.gd, q, beta, b, w,
                               int(first) | (2 if acc else 0), _stream(z)), "LTS slab H rates (2D)")
    return ((r_out if first else None), (None if last else eta_out), ravg_out,
            (corr if (cm and not acc) else None))


def _r2_bwd(yc, sd, like, first, last, cm, beta, b, w, gr, ge, ga, gq, gz=None):
    """Transpose of :func:`_r2_fwd`: ``(gz, g_eta | None, g_ravg | None, g_r0 | None)``; ``gz``
    accumulated into the given buffer."""
    new = lambda: torch.empty_like(like)  # noqa: E731
    gr_ = list(gr) if first else [None, None]
    ge_ = list(ge) if (not last and ge is not None) else [None, None]
    ga_ = list(ga) if ga is not None else [torch.zeros_like(like), torch.zeros_like(like)]
    gq_ = list(gq) if (cm and gq is not None) else [None] * 4
    acc = gz is not None
    gz = new() if gz is None else gz
    gein = [new(), new()] if (not first and not last) else [None, None]
    gain = [new(), new()] if (cm and not first) else [None, None]
    gr0 = [new(), new()] if cm else [None, None]
    stc = sd["stat"] + sd["masks"][1:]
    q = _ptrs(gr_ + ge_ + ga_ + stc + gq_ + [gz] + gein + gain + gr0)
    _check(yc.lib.lts2d_slab_r(yc.code, 1, like.shape[0], sd["n"], yc.NY, yc.gd, q, beta, b, w,
                               int(first) | (2 if acc else 0), _stream(like)),
           "LTS slab H rates transpose (2D)")
    g_ravg = None if first else (gain if cm else ga_)
    return gz, (gein if gein[0] is not None else None), g_ravg, (gr0 if cm else None)


class LTSPre2D:
    """Coefficient-derived arrays of the 2D LTS step (``dt/eps`` on the slabs and per
    element) and the zero DG field, computed into fixed buffers by :meth:`update`."""

    def __init__(self, st, c):
        y, d = st.yee, st.dg
        self.slabs = fn.lts_slabs(st)
        kw = dict(dtype=st.dtype, device=st.device)
        self.cb0_s = [torch.empty((1, sl.hi - sl.lo + 1, y.shape[1]), **kw) for sl in self.slabs]
        self.cb0_dg = torch.empty((d.K,), **kw)
        self.zero = torch.zeros((st.nshot, d.K, d.Np), **kw)
        self.zero_s = [torch.zeros((st.nshot, sl.hi - sl.lo + 1, y.shape[1]), **kw)
                       for sl in self.slabs]
        self.update(c)

    def update(self, c):
        with torch.no_grad():
            for sl, buf in zip(self.slabs, self.cb0_s):
                ca, cb = c.fd_ca[sl.lo:sl.hi + 1], c.fd_cb[sl.lo:sl.hi + 1]
                buf.copy_((2.0 * cb / (1.0 + ca))[None])
            self.cb0_dg.copy_((2.0 * c.dg_cb / (1.0 + c.dg_ca)).reshape(-1))


def prepare(st, c, pre=None):
    if not lts_available(st):
        return None
    if pre is None:
        return LTSPre2D(st, c)
    pre.update(c)
    return pre


def step_any(st, c, u, out, src=None, amp=None, rec=None, rec_out=None, pre=None):
    if pre is not None:
        lts_step_into(st, c, u, out, src, amp, rec, rec_out, pre)
    else:
        step_into(st, c, u, out, src, amp, rec, rec_out)


def vjp_any(st, c, u, u_next, lam, out, src=None, amp=None, gd=None, rec=None, acc=None,
            need_c=True, need_w=False, gamp=None, pre=None):
    if pre is not None:
        lts_vjp_into(st, c, u, u_next, lam, out, src, amp, gd, rec, acc, need_c, need_w, gamp, pre)
    else:
        vjp_into(st, c, u, u_next, lam, out, src, amp, gd, rec, acc, need_c, need_w, gamp)


def _slab_rows(g, slabs, arrs):
    return g.rows_planes([a for a in arrs], [(sl.hi - sl.lo + 1) * g.NY for sl in slabs],
                         [sl.cp.row - sl.lo for sl in slabs])


def _slab_ghosts(g, slabs, arrs):
    return g.ghost_planes([a for a in arrs], [(sl.hi - sl.lo + 1) * g.NY for sl in slabs],
                          [sl.cp.ghost - sl.lo for sl in slabs], [sl.cp.inner - sl.lo for sl in slabs])


def _cb0_slab_fn(lo, hi):
    def fn_(t, c):
        ca, cb = c.fd_ca[lo:hi + 1], c.fd_cb[lo:hi + 1]
        gcb, gca = torch.zeros_like(c.fd_cb), torch.zeros_like(c.fd_ca)
        gcb[lo:hi + 1] = t[0] * 2.0 / (1.0 + ca)
        gca[lo:hi + 1] = -t[0] * 2.0 * cb / (1.0 + ca) ** 2
        return [("fd_cb", gcb), ("fd_ca", gca)]
    return fn_


def _cb0_dg_fn(t, c):
    ca, cb = c.dg_ca.reshape(-1), c.dg_cb.reshape(-1)
    return [("dg_cb", t * 2.0 / (1.0 + ca)), ("dg_ca", -t * 2.0 * cb / (1.0 + ca) ** 2)]


def lts_step_into(st, c, u, out, src=None, amp=None, rec=None, rec_out=None, pre=None):
    """One sub-cycled step ``u -> out`` (:func:`DeepGPR.hybrid.functional.lts_h_half_step`
    and the rest of :func:`DeepGPR.hybrid.functional.step`)."""
    y, d = st.yee, st.dg
    yc, dc = y.cuda, d.cuda
    g = coupling_glue2d(st)
    pre = pre if pre is not None else LTSPre2D(st, c)
    S = st.nshot
    p = int(d.lts)
    lc = d.lts_coef if d.lts_coef is not None else fn.lts_coefficients(p)
    w, c0, ck, beta = lc["w"], lc["c0"], lc["c"], lc["beta"]
    inv_p = 1.0 / p
    slabs = pre.slabs
    ns = len(slabs)
    sds = [yc.slab_data(sl) for sl in slabs]
    Ez, Hx, Hy, pex, pey, phx, phy, E, Hxd, Hyd, qex, qey, qhx, qhy = u
    oEz, oHx, oHy, opex, opey, ophx, ophy, oE, oHxd, oHyd, oqex, oqey, oqhx, oqhy = out
    kw = dict(dtype=Ez.dtype, device=Ez.device)
    yc.h_fwd(Ez, Hx, Hy, phy, phx, out=(oHx, oHy, ophy, ophx))
    z_s = [Ez[:, sl.lo:sl.hi + 1].contiguous() for sl in slabs]
    X = torch.empty((S, g.nX), **kw)
    g.trace(_slab_rows(g, slabs, z_s), X)
    T = dc.terms_fwd(0, E, Hxd, Hyd, X, c.Ym, c.Yp)
    ex, ey, T_avg = _lts2d_eta(dc, T, None, None, None, 0.0, c0 * inv_p, w[0])
    dd0, eta_s, dd_avg, corr = [], [], [], [None] * ns
    for i in range(ns):
        r_i, e_i, a_i, _ = _r2_fwd(yc, sds[i], z_s[i], None, None, True, False, 0.0, c0 * inv_p,
                                   w[0], None)
        dd0.append(r_i)
        eta_s.append(e_i)
        dd_avg.append(a_i)
    z_dg = E
    for m in range(1, p):
        g.ghost(_slab_ghosts(g, slabs, [eta_s[i][1] for i in range(ns)]), ey)
        Te = dc.terms_fwd(1, pre.zero, ex, ey, None, c.Zm, c.Zp)
        z_dg = _lts2d_z(dc, Te, z_dg, pre.cb0_dg, inv_p)
        for i in range(ns):
            z_s[i] = _q2_fwd(yc, sds[i], eta_s[i][0], eta_s[i][1], z_s[i], pre.cb0_s[i], inv_p)
        X = torch.empty((S, g.nX), **kw)
        g.trace(_slab_rows(g, slabs, z_s), X)
        T = dc.terms_fwd(0, z_dg, Hxd, Hyd, X, c.Ym, c.Yp)
        ex, ey, T_avg = _lts2d_eta(dc, T, ex, ey, T_avg, beta[m], ck[m] * inv_p, w[m])
        last = m == p - 1
        for i, sl in enumerate(slabs):
            into = ([t[:, sl.lo:sl.hi + 1] for t in (oHx, ophy, oHy, ophx)]
                    if (last and S == 1) else None)
            _, eta_s[i], dd_avg[i], corr[i] = _r2_fwd(
                yc, sds[i], z_s[i], eta_s[i], dd_avg[i], False, last, beta[m], ck[m] * inv_p, w[m],
                dd0[i] if last else None, into)
    dc.update_fwd(0, T_avg, Hxd, Hyd, qhx, qhy, out=(oHxd, oHyd, oqhx, oqhy))
    for i, sl in enumerate(slabs):
        if corr[i] is not None:                 # several shots: corrections returned
            for t, cr in zip((oHx, ophy, oHy, ophx), corr[i]):
                t.index_add_(1, sl.rows, cr)
    g.ghost(g.ghost_planes(oHy), oHyd)
    yc.e_fwd(oHx, oHy, Ez, pex, pey, c.fd_ca, c.fd_cb, out=(oEz, opex, opey))
    T = dc.terms_fwd(1, E, oHxd, oHyd, None, c.Zm, c.Zp)
    dc.update_fwd(1, T, E, E, qex, qey, c.dg_ca.reshape(-1), c.dg_cb.reshape(-1),
                  out=(oE, oqex, oqey))
    if src is not None:
        point_glue(st, src).inject(oEz, oE, c.fd_cb, c.dg_cb, amp)
    if rec is not None and rec_out is not None:
        point_glue(st, rec).sample(oEz, oE, rec_out)


def lts_vjp_into(st, c, u, u_next, lam, out, src=None, amp=None, gd=None, rec=None,
                 acc: Optional[Accumulators] = None, need_c: bool = True, need_w: bool = False,
                 gamp=None, pre=None):
    """Hand-written transpose of :func:`lts_step_into` (``lam`` destroyed)."""
    y, d = st.yee, st.dg
    yc, dc = y.cuda, d.cuda
    g = coupling_glue2d(st)
    pre = pre if pre is not None else LTSPre2D(st, c)
    S = st.nshot
    p = int(d.lts)
    lc = d.lts_coef if d.lts_coef is not None else fn.lts_coefficients(p)
    w, c0, ck, beta = lc["w"], lc["c0"], lc["c"], lc["beta"]
    inv_p = 1.0 / p
    slabs = pre.slabs
    ns = len(slabs)
    sds = [yc.slab_data(sl) for sl in slabs]
    Ez, Hx, Hy, pex, pey, phx, phy, E, Hxd, Hyd, qex, qey, qhx, qhy = u
    Hx1, Hy1, Hxd1, Hyd1 = u_next[1], u_next[2], u_next[8], u_next[9]
    lEz, lHx, lHy, lpex, lpey, lphx, lphy, lE, lHxd, lHyd, lqex, lqey, lqhx, lqhy = lam
    oEz, oHx, oHy, opex, opey, ophx, ophy, oE, oHxd, oHyd, oqex, oqey, oqhx, oqhy = out
    kw = dict(dtype=Ez.dtype, device=Ez.device)
    one = S == 1
    raw_dg = acc.raw("lts_cb0_dg", (d.K,), E, _cb0_dg_fn) if need_c else None
    raw_s = ([acc.raw(f"lts_cb0_s:{i}", pre.cb0_s[i].shape, E, _cb0_slab_fn(sl.lo, sl.hi))
              for i, sl in enumerate(slabs)] if need_c else [None] * ns)
    yflux = (lambda: _flux_acc(acc, c, S, ("Ym", "Yp"))) if need_c else (lambda: None)  # noqa: E731
    zflux = (lambda: _flux_acc(acc, c, S, ("Zm", "Zp"))) if need_c else (lambda: None)  # noqa: E731
    if rec is not None and gd is not None:
        point_glue(st, rec).sample_bwd(gd, lEz, lE)
    # ================================================================ recompute
    zs = [[Ez[:, sl.lo:sl.hi + 1].contiguous()] for sl in slabs]
    Xs = [torch.empty((S, g.nX), **kw)]
    g.trace(_slab_rows(g, slabs, [zs[i][0] for i in range(ns)]), Xs[0])
    zdg = [E]
    T0 = dc.terms_fwd(0, E, Hxd, Hyd, Xs[0], c.Ym, c.Yp)
    exs, eys = [None] * p, [None] * p
    exs[0], eys[0], T_avg = _lts2d_eta(dc, T0, None, None, None, 0.0, c0 * inv_p, w[0])
    del T0
    dd0, eta_s, dd_avg = [], [None] * ns, []
    etl = [[None] * p for _ in range(ns)]
    for i in range(ns):
        r_i, e_i, a_i, _ = _r2_fwd(yc, sds[i], zs[i][0], None, None, True, False, 0.0, c0 * inv_p,
                                   w[0], None)
        dd0.append(r_i)
        eta_s[i] = e_i
        dd_avg.append(a_i)
    Tes = [None] * p
    for m in range(1, p):
        g.ghost(_slab_ghosts(g, slabs, [eta_s[i][1] for i in range(ns)]), eys[m - 1])
        for i in range(ns):
            etl[i][m - 1] = eta_s[i]
        Tes[m] = dc.terms_fwd(1, pre.zero, exs[m - 1], eys[m - 1], None, c.Zm, c.Zp)
        zdg.append(_lts2d_z(dc, Tes[m], zdg[m - 1], pre.cb0_dg, inv_p))
        for i in range(ns):
            zs[i].append(_q2_fwd(yc, sds[i], etl[i][m - 1][0], etl[i][m - 1][1], zs[i][m - 1],
                                 pre.cb0_s[i], inv_p))
        Xm = torch.empty((S, g.nX), **kw)
        g.trace(_slab_rows(g, slabs, [zs[i][m] for i in range(ns)]), Xm)
        Xs.append(Xm)
        Tm = dc.terms_fwd(0, zdg[m], Hxd, Hyd, Xm, c.Ym, c.Yp)
        exs[m], eys[m], T_avg = _lts2d_eta(dc, Tm, exs[m - 1], eys[m - 1], T_avg, beta[m],
                                           ck[m] * inv_p, w[m])
        del Tm
        last = m == p - 1
        for i in range(ns):
            _, eta_s[i], dd_avg[i], _ = _r2_fwd(yc, sds[i], zs[i][m], etl[i][m - 1], dd_avg[i],
                                                False, last, beta[m], ck[m] * inv_p, w[m], None)
    T_e = dc.terms_fwd(1, E, Hxd1, Hyd1, None, c.Zm, c.Zp)
    # ================================================================ reverse
    _sources_bwd(st, c, src, amp, lEz, lE, acc, need_c, need_w, gamp)
    # ---- DG E update and its terms
    ca, cb = c.dg_ca.reshape(-1), c.dg_cb.reshape(-1)
    gT, gE_a, _, _, _, gca, gcb = dc.update_bwd(1, T_e, E, qex, qey, ca, cb, lE, None, lqex, lqey,
                                                out_q=(oqex, oqey))
    del T_e
    if need_c:
        acc.add("dg_ca:upd", gca, _SHOTS)
        acc.add("dg_cb:upd", gcb, _SHOTS)
    lE_n, lHxd1, lHyd1, *_ = dc.terms_bwd(1, E, Hxd1, Hyd1, gT, c.Zm, c.Zp, False, addE=gE_a,
                                          addHx=lHxd, addHy=lHyd, acc_coef=zflux())
    del gT, gE_a
    # ---- Yee E update (cotangents of Hx', Hy' accumulated into lHx, lHy)
    yc.e_bwd(Hx1, Hy1, Ez, pex, pey, c.fd_ca, c.fd_cb, lEz, lpex, lpey, out=(oEz, opex, opey),
             src_acc=(lHx, lHy), coef_acc=_yee_coef_acc(acc, c, S) if need_c else None)
    # ---- ghost rows of Hy'
    g.ghost_bwd(g.ghost_planes(lHy), lHyd1)
    # ---- the corrections of the fine rows: their cotangents are those of Hx1 .. phx1
    g_corr = [[t[:, sl.lo:sl.hi + 1].contiguous() for t in (lHx, lphy, lHy, lphx)] for sl in slabs]
    # ---- DG H update with the averaged terms
    gTavg, gHxd_a, gHyd_a, *_ = dc.update_bwd(0, None, None, qhx, qhy, None, None, lHxd1, lHyd1,
                                              lqhx, lqhy, out_q=(oqhx, oqhy))
    lHxd_n, lHyd_n = gHxd_a, gHyd_a
    # ---- the recursion, m = p-1 .. 1
    lex = ley = None                     # cotangents of ex^m, ey^m (carried)
    lz = None                            # cotangent of z^m (DG), carried
    lzs = [None] * ns                    # cotangents of z_s^m, carried
    leta_s = [None] * ns
    lravg = [None] * ns
    lr0 = [None] * ns
    for m in range(p - 1, 0, -1):
        last = m == p - 1
        letl = [None] * ns
        for i in range(ns):
            gz, gein, gav, gr0 = _r2_bwd(yc, sds[i], zs[i][m], False, last, last, beta[m],
                                         ck[m] * inv_p, w[m], None, None if last else leta_s[i],
                                         lravg[i], g_corr[i] if last else None, gz=lzs[i])
            lzs[i] = gz
            letl[i] = gein
            lravg[i] = gav
            if gr0 is not None:
                lr0[i] = gr0 if lr0[i] is None else [a + b for a, b in zip(lr0[i], gr0)]
        # eta update of step m: (ex^m, ey^m, Tavg^m) <- T^m, ex^(m-1), ey^(m-1), Tavg^(m-1)
        gT_m, gxi, gyi = _lts2d_eta_bwd(dc, lex, ley, gTavg, beta[m], ck[m] * inv_p, w[m], False,
                                        gTavg)
        # T^m = Terms0(z^m, Hxd, Hyd, X^m)
        gzm, gHx_m, gHy_m, gX, *_ = dc.terms_bwd(0, zdg[m], Hxd, Hyd, gT_m, c.Ym, c.Yp, True,
                                                  addE=lz, acc_coef=yflux())
        lHxd_n.add_(gHx_m)
        lHyd_n.add_(gHy_m)
        g.trace_bwd(_slab_rows(g, slabs, lzs), gX)
        # z_s^m = SlabQ2(eta~^(m-1), z_s^(m-1)): z passes, eta~ and cb0 get the transposes
        for i in range(ns):
            gex, gey = _q2_bwd(yc, sds[i], etl[i][m - 1][0], etl[i][m - 1][1], lzs[i],
                               pre.cb0_s[i], inv_p,
                               *(letl[i] if letl[i] is not None else (None, None)), gcb0=raw_s[i])
            letl[i] = [gex, gey]
        # z^m = Z(Te^m, z^(m-1)): z passes (gzm is now the cotangent of z^(m-1))
        gTe = _lts2d_z_bwd(dc, Tes[m], gzm, pre.cb0_dg, inv_p, raw_dg)
        # Te^m = Terms1(0, ex^(m-1), ey^(m-1))
        _, gexm, geym, *_ = dc.terms_bwd(1, pre.zero, exs[m - 1], eys[m - 1], gTe, c.Zm, c.Zp,
                                         False, addHx=gxi, addHy=gyi, acc_coef=zflux())
        # ghost rows of eta~ (ey) from the DG ey^(m-1)
        g.ghost_bwd(_slab_ghosts(g, slabs, [letl[i][1] for i in range(ns)]), geym)
        for i in range(ns):
            leta_s[i] = letl[i]
        lex, ley, lz = gexm, geym, gzm
    # ---- first sub-step
    for i in range(ns):
        z0 = pre.zero_s[i]
        lzs[i], _, _, _ = _r2_bwd(yc, sds[i], zs[i][0], True, False, False, 0.0, c0 * inv_p, w[0],
                                  lr0[i] if lr0[i] is not None else [z0, z0], leta_s[i], lravg[i],
                                  None, gz=lzs[i])
    gT0, _, _ = _lts2d_eta_bwd(dc, lex, ley, gTavg, 0.0, c0 * inv_p, w[0], True, gTavg)
    _, _, _, gX, *_ = dc.terms_bwd(0, E, Hxd, Hyd, gT0, c.Ym, c.Yp, True, addE=lz,
                                   acc_coef=yflux(), out=(oE, oHxd, oHyd))
    # (the volume part of gE, plus lz = cotangent of z^0 = E, went to oE; add the rest)
    oE.add_(lE_n)
    oHxd.add_(lHxd_n)
    oHyd.add_(lHyd_n)
    g.trace_bwd(_slab_rows(g, slabs, lzs), gX)
    for i, sl in enumerate(slabs):
        oEz[:, sl.lo:sl.hi + 1] += lzs[i]
    # ---- Yee H update
    yc.h_bwd(lHx, lHy, lphy, lphx, out=(oHx, oHy, ophy, ophx), gEz_acc=oEz)
