r"""Hand-written vector-Jacobian product of the fused CUDA 3D hybrid step with local
time stepping (``dg.lts = p > 1``, :func:`~DeepGPR.hybrid.functional3d.lts_h_half_step`).

The step is (fused path, ``u = u^n``; ``i`` runs over the couplings/slabs)::

    1   (fh1, psh', Bfd1)   = YeeH(fe, fh, psh, Bfd)
    2   z_s^0 = fe|slabs,  E*^0 = P(z_s^0|rows),  z^0 = E
    3   (eta^0, cavg^0)     = TermsEta(E, H, E*^0; first)          (operator + epilogue)
    4   (r0, eta_s^0, ravg^0) = SlabR(z_s^0; first)
    5   for m = 1 .. p-1:
          eta~_s^(m-1) = eta_s^(m-1) with ghost planes 2 G(eta^(m-1)) - eta_s^(m-1)|inner
          z^m      = TermsZ(eta^(m-1), 0; z^(m-1))                    (operator + epilogue)
          z_s^m    = SlabQ(eta~_s^(m-1), z_s^(m-1)),   E*^m = P(z_s^m|rows)
          (eta^m, cavg^m) = TermsEta(z^m, H, E*^m; eta^(m-1), cavg^(m-1))
          (eta_s^m, ravg^m, corr) = SlabR(z_s^m, eta~_s^(m-1), ravg^(m-1); r0 on the last)
    6   (H_c, Baux') = LTSUpdate_H(cavg^(p-1), H, Baux)            (p-fold UPML in PML elements)
    7   H' = damp_H(H, H_c)                                        (split penalty, if any)
    8   fh1|planes += corr;  fh1, Bfd1 at the fine-plane UPML nodes = Fold(fh, Bfd, ravg)
    9   fh' = fh1 with ghost planes 2 G(H') - fh1|inner
    10  pen = interface penalty rows (fe, E);  (fe1, pse', Dfd') = YeeE(fh', fe, pse, Dfd; pen)
    11  fe1, Dfd' at the fine-plane UPML nodes = Fold(fe, Dfd, curl_bwd(fh'|slabs) + pen)
    12  X_h = g n_x (E*^0_z, -E*^0_y);  T_e = Terms(H', E, X_h)
    13  (E_c, Daux') = LTSUpdate_E(T_e, E, Daux);  E' = damp_E(E, E_c);  inject

:func:`step_vjp` recomputes the recursion intermediates (``z^m``, ``eta^m``,
``z_s^m``, ``eta~_s^m``, ``E*^m``, ``ravg``, ``cavg``, the damping stages, ``T_e``)
from ``u^n`` and ``u^(n+1)`` and applies the transposes in reverse order: the
transpose kernels of every fused piece (operator with the epilogue cotangent formed in
registers, slab recursion, folds, the LTS updates, Yee H/E) and explicit transposes
of the sparse coupling operations (trace, ghost planes, interface penalty, X_h,
sources). Compared with reverse-mode differentiation of the sub-cycled step it avoids
the autograd bookkeeping on full Yee arrays (zero-filled slice and gather
cotangents, clones of in-place updates, accumulations). It is checked against the
autograd VJP of the PyTorch step in ``tests/test_hybrid_lts.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from . import functional3d as fn3
from . import vjp3d
from .cuda3d import _check, _ptrs, _stream

sample_adjoint = vjp3d.sample_adjoint
finalize = vjp3d.finalize
_acc = vjp3d._acc
_SHOTS, _FACE, _FACE2 = vjp3d._SHOTS, vjp3d._FACE, vjp3d._FACE2


def supported(st: fn3.StepStatic3D) -> bool:
    """``True`` for the fused LTS step (Yee, DG, slab and epilogue kernels on CUDA)."""
    y, d = st.yee, st.dg
    return (y is not None and y.cuda is not None and d is not None and d.cuda is not None
            and torch.device(st.device).type == "cuda" and int(getattr(d, "lts", 1)) > 1
            and bool(st.couplings) and hasattr(y.cuda, "slab_r")
            and hasattr(d.cuda, "terms_eta") and getattr(d.cuda, "uf", "x") != "x")


# --------------------------------------------------------------------------- raw kernels
def _c(ts):
    return [None if t is None else t.contiguous() for t in ts]


def _slab_q_fwd(yc, sd, f, z, cb0, a):
    """``where(fine_e, z + a cb0 curl_bwd(f), z)`` (``z`` given) or ``curl_bwd(f)``."""
    f = _c(f)
    zmode = z is not None
    zz = _c(z) if zmode else [None] * 3
    out = [torch.empty_like(t) for t in f]
    m = sd["mask_e"] if zmode else [None] * 3
    q = _ptrs(f + zz + [cb0 if zmode else None] + list(m) + out)
    _check(yc.lib.lts3d_slab_q(yc.code, 0, f[0].shape[0], sd["n"], yc.shape[1], yc.shape[2], yc.gd,
                               q, float(a), int(zmode), _stream(f[0])), "LTS slab E rates")
    return out


def _slab_q_bwd(yc, sd, f, g, cb0, a, want_cb0):
    """Transpose of :func:`_slab_q_fwd`: ``(g_f, g_cb0)`` (``g_z = g`` in the z mode)."""
    f, g = _c(f), _c(g)
    zmode = cb0 is not None
    gf = [torch.empty_like(t) for t in f]
    gcb0 = torch.empty_like(cb0) if (zmode and want_cb0) else None
    m = sd["mask_e"] if zmode else [None] * 3
    q = _ptrs(f + g + [cb0 if zmode else None] + list(m) + gf + [gcb0])
    _check(yc.lib.lts3d_slab_q(yc.code, 1, f[0].shape[0], sd["n"], yc.shape[1], yc.shape[2], yc.gd,
                               q, float(a), int(zmode), _stream(f[0])), "LTS slab E rates transpose")
    return gf, gcb0


def _slab_r_fwd(yc, sd, z, eta, ravg, first, last, beta, b, w, r0):
    """Forward of the slab H rates (see :class:`~DeepGPR.hybrid.cuda3d._SlabR`):
    ``(r | None, eta_new | None, ravg_new, corr | None)``."""
    z = _c(z)
    eta_ = _c(eta) if (eta is not None and not first and not last) else [None] * 3
    ravg_ = _c(ravg) if not first else [None] * 3
    cm = r0 is not None
    r0_ = _c(r0) if cm else [None] * 3
    r_out = [torch.empty_like(t) for t in z] if first else [None] * 3
    eta_out = [torch.empty_like(t) for t in z] if not last else [None] * 3
    ravg_out = [torch.empty_like(t) for t in z]
    corr = [torch.empty_like(t) for t in z] if cm else [None] * 3
    mh = sd["mask_h"] if cm else [None] * 3
    q = _ptrs(z + eta_ + ravg_ + sd["chr"] + r_out + eta_out + ravg_out + r0_ + list(mh) + corr)
    _check(yc.lib.lts3d_slab_r(yc.code, 0, z[0].shape[0], sd["n"], yc.shape[1], yc.shape[2], yc.gd,
                               q, float(beta), float(b), float(w), int(first), _stream(z[0])),
           "LTS slab H rates")
    return ((r_out if first else None), (None if last else eta_out), ravg_out,
            (corr if cm else None))


def _slab_r_bwd(yc, sd, like, first, last, cm, beta, b, w, gr, ge, ga, gq):
    """Transpose of :func:`_slab_r_fwd`: ``(g_z, g_eta | None, g_ravg | None, g_r0 | None)``
    (``g_ravg`` is ``ga`` itself when there is no correction output)."""
    zz = lambda: torch.zeros_like(like)  # noqa: E731
    gr_ = _c(gr) if first else [None] * 3
    ge_ = _c(ge) if (not last and ge is not None) else [None] * 3
    ga_ = _c(ga) if ga is not None else [zz() for _ in range(3)]
    gq_ = _c(gq) if (cm and gq is not None) else [None] * 3
    new = lambda: torch.empty_like(like)  # noqa: E731
    gz = [new() for _ in range(3)]
    gein = [new() for _ in range(3)] if (not first and not last) else [None] * 3
    gain = [new() for _ in range(3)] if (cm and not first) else [None] * 3
    gr0 = [new() for _ in range(3)] if cm else [None] * 3
    mh = sd["mask_h"] if cm else [None] * 3
    q = _ptrs(gr_ + ge_ + ga_ + sd["chr"] + gz + gein + gq_ + list(mh) + gain + gr0)
    _check(yc.lib.lts3d_slab_r(yc.code, 1, like.shape[0], sd["n"], yc.shape[1], yc.shape[2], yc.gd,
                               q, float(beta), float(b), float(w), int(first), _stream(like)),
           "LTS slab H rates transpose")
    g_ravg = None if first else (gain if cm else ga_)
    return gz, (gein if gein[0] is not None else None), g_ravg, (gr0 if cm else None)


def _fold_fwd(yc, p, sign, idx, f, aux, rate, fac):
    f, aux, rate = f.contiguous(), aux.contiguous(), rate.contiguous()
    fac = _c(fac)
    S, n = f.shape[0], fac[0].numel()
    fval = torch.empty((S, n), dtype=f.dtype, device=f.device)
    auxn = torch.empty_like(fval)
    q = _ptrs(list(idx) + [f, aux, rate] + fac + [fval, auxn])
    _check(yc.lib.lts3d_fold(yc.code, 0, S, n, int(p), float(sign), f.shape[1], aux.shape[1],
                             rate.shape[1], q, _stream(f)), "LTS UPML fold")
    return fval, auxn


def _fold_bwd(yc, p, sign, idx, f, aux, rate, fac, gfval, gauxn, gf, gaux, grate, want_fac):
    """Transpose of :func:`_fold_fwd`; ``gf``/``gaux``/``grate`` are written at the entries
    (zero there beforehand: they may be accumulators of other, disjoint entries)."""
    f, aux, rate = f.contiguous(), aux.contiguous(), rate.contiguous()
    fac = _c(fac)
    S, n = f.shape[0], fac[0].numel()
    gfac = [torch.empty_like(t) for t in fac[2:]] if want_fac else [None] * 3
    q = _ptrs(list(idx) + [f, aux, rate] + fac + [gfval.contiguous(), None if gauxn is None
                                                    else gauxn.contiguous(), gf, gaux, grate] + gfac)
    _check(yc.lib.lts3d_fold(yc.code, 1, S, n, int(p), float(sign), f.shape[1], aux.shape[1],
                             rate.shape[1], q, _stream(f)), "LTS UPML fold transpose")
    return gfac if want_fac else None


def _ltsupd(dc, mode, p, rate, f, aux, ca, cb, hist, nw, old):
    z = torch.zeros(0, dtype=f.dtype, device=f.device)
    ca, cb = (z if t is None else t.contiguous() for t in (ca, cb))
    hist, nw, old = (z if t is None else t.contiguous() for t in (hist, nw, old))
    rate, f, aux = rate.contiguous(), f.contiguous(), aux.contiguous()
    fnew, auxn = torch.empty_like(f), torch.empty_like(aux)
    u = dc.uf or {}
    q = _ptrs([dc.pml, rate, f, aux, dc.ch, ca, cb, u.get("a1"), u.get("b1"), hist, nw, old, fnew,
               auxn])
    _check(dc.lib.lts3d_update(dc.code, 0, mode, f.shape[0], dc.K, dc.Np, dc.Kp, int(p), q,
                               _stream(f)), "LTS DG update")
    return fnew, auxn


def _ltsupd_bwd(dc, mode, p, rate, f, aux, ca, cb, hist, nw, old, gfn, gauxn, want_cacb,
                want_fac):
    z = torch.zeros(0, dtype=f.dtype, device=f.device)
    ca, cb = (z if t is None else t.contiguous() for t in (ca, cb))
    hist, nw, old = (z if t is None else t.contiguous() for t in (hist, nw, old))
    rate, f, aux = rate.contiguous(), f.contiguous(), aux.contiguous()
    gfn = torch.zeros_like(f) if gfn is None else gfn.contiguous()
    gauxn = None if gauxn is None else gauxn.contiguous()
    grate, gf, gaux = torch.empty_like(rate), torch.empty_like(f), torch.empty_like(aux)
    e = mode == 1 and want_cacb
    gca = torch.empty_like(ca) if e else None
    gcb = torch.empty_like(cb) if e else None
    want = dc.Kp > 0 and want_fac and hist.numel() > 0
    gh, gn, go = ((torch.empty_like(hist), torch.empty_like(nw), torch.empty_like(old)) if want
                  else (None, None, None))
    u = dc.uf or {}
    q = _ptrs([dc.pml, rate, f, aux, dc.ch, ca, cb, u.get("a1"), u.get("b1"), hist, nw, old, gfn,
               gauxn, grate, gf, gaux, gca, gcb, gh, gn, go])
    _check(dc.lib.lts3d_update(dc.code, 1, mode, f.shape[0], dc.K, dc.Np, dc.Kp, int(p), q,
                               _stream(f)), "LTS DG update transpose")
    return grate, gf, gaux, gca, gcb, gh, gn, go


def _curl(T):
    return T[:, 0::2] - T[:, 1::2]


# --------------------------------------------------------------------------- transpose
def step_vjp(st: fn3.StepStatic3D, c: fn3.StepCoefs3D, u, u_next, lam, src=None, amp=None,
             acc: Optional[Dict[str, torch.Tensor]] = None, need_c: bool = True,
             need_w: bool = False):
    """Transpose of one fused LTS step; same contract as :func:`vjp3d.step_vjp`."""
    y, d = st.yee, st.dg
    yc, dc = y.cuda, d.cuda
    S = st.nshot
    NY, NZ = y.shape[1], y.shape[2]
    p = int(d.lts)
    lc = d.lts_coef if d.lts_coef is not None else fn3.lts_coefficients(p)
    w, c0, ck, beta = lc["w"], lc["c0"], lc["c"], lc["beta"]
    inv_p = 1.0 / p
    slabs = fn3.lts_slabs(st)
    ns = len(slabs)
    sds = [yc.slab_data(sl) for sl in slabs]
    fe, fh = list(u[0:3]), list(u[3:6])
    pse, psh = list(u[6:12]), list(u[12:18])
    E, H, Daux, Baux, Dfd, Bfd = u[18:24]
    fh2, H1 = list(u_next[3:6]), u_next[19]
    lfe, lfh = list(lam[0:3]), list(lam[3:6])
    lpse, lpsh = list(lam[6:12]), list(lam[12:18])
    lE, lH, lD, lB, lDfd, lBfd = lam[18:24]
    cs = st.couplings
    ranges = vjp3d._ranges(st)
    g_if = c.g_if
    eb = (c.e0, c.e1, c.e2, c.e3)
    hb = (c.h0, c.h1, c.h2, c.h3)
    split = bool(getattr(d, "lts_split", False))
    kd = max(1, int(getattr(d, "lts_pen_k", 1)))
    sdmp = 1.0 / kd
    zero = torch.zeros(0, dtype=E.dtype, device=E.device)
    zeroE = torch.zeros_like(E)
    gop, top = fn3.ghost_op(st), fn3.trace_op(st)
    rows_l = [sl.cp.row - sl.lo for sl in slabs]
    if acc is None:
        acc = {}

    # ================================================================ recompute
    cb0_dg = (2.0 * c.dg_cb / (1.0 + c.dg_ca)).reshape(-1).contiguous()          # (K,)
    cb0f = 2.0 * c.fd_cb / (1.0 + c.fd_ca)
    cb0_s = [cb0f[sl.lo:sl.hi + 1][None].contiguous() for sl in slabs]

    def trace(zl):
        x = torch.stack([t.reshape(S, -1) for i in range(ns)
                         for t in (zl[i][1][:, rows_l[i]], zl[i][2][:, rows_l[i]])], dim=1)
        return top(x.reshape(S, -1)).view(S, 2, -1)

    zs = [[[fe[k][:, sl.lo:sl.hi + 1].contiguous() for k in range(3)]] for sl in slabs]
    Eext = [trace([zs[i][0] for i in range(ns)])]
    zdg = [E]
    eta = []
    o1, cav = torch.empty_like(E), torch.empty_like(E)
    dc.terms_fwd(E.contiguous(), H.contiguous(), Eext[0].contiguous(), *eb,
                 epi=(1, [None, None, None, dc.ch, None, o1, cav], 1, 0.0, c0 * inv_p, w[0], 0.0))
    eta.append(o1)
    r0, ravg = [], []
    etl = [[None] * p for _ in range(ns)]          # eta~_s^(m) (ghost planes written), m < p - 1
    eta_s = [None] * ns
    for i in range(ns):
        r_i, e_i, a_i, _ = _slab_r_fwd(yc, sds[i], zs[i][0], None, None, True, False, 0.0,
                                       c0 * inv_p, w[0], None)
        r0.append(r_i)
        eta_s[i] = e_i
        ravg.append(a_i)
    ravg_hist = [[a] for a in ravg]                  # ravg^m of every slab (inputs of SlabR)
    zcurl = [None] * p
    cav_last = cav
    for m in range(1, p):
        g = gop(eta[m - 1].reshape(S, -1)).view(S, ns, 2, NY, NZ)
        for i, sl in enumerate(slabs):
            et = list(eta_s[i])
            inner = sl.cp.inner - sl.lo
            hz = 2.0 * g[:, i, 0] - et[2][:, inner]
            hy = 2.0 * g[:, i, 1] - et[1][:, inner]
            et[2] = et[2].index_copy(1, sl.ghost, hz[:, None])
            et[1] = et[1].index_copy(1, sl.ghost, hy[:, None])
            etl[i][m - 1] = et
        zn = torch.empty_like(E)
        cu = torch.empty_like(E) if need_c else None
        dc.terms_fwd(eta[m - 1], zeroE, None, *hb,
                     epi=(2, [None, None, zdg[m - 1], None, cb0_dg, zn, cu], 0, 0.0, 0.0, 0.0, inv_p))
        zdg.append(zn)
        zcurl[m] = cu
        for i in range(ns):
            zs[i].append(_slab_q_fwd(yc, sds[i], etl[i][m - 1], zs[i][m - 1], cb0_s[i], inv_p))
        Eext.append(trace([zs[i][m] for i in range(ns)]))
        o1, o2 = torch.empty_like(E), torch.empty_like(E)
        dc.terms_fwd(zdg[m], H.contiguous(), Eext[m].contiguous(), *eb,
                     epi=(1, [eta[m - 1], cav_last, None, dc.ch, None, o1, o2], 0, beta[m],
                          ck[m] * inv_p, w[m], 0.0))
        eta.append(o1)
        cav_last = o2
        last = m == p - 1
        for i in range(ns):
            _, e_new, a_new, _ = _slab_r_fwd(yc, sds[i], zs[i][m], etl[i][m - 1], ravg[i], False,
                                             last, beta[m], ck[m] * inv_p, w[m],
                                             r0[i] if last else None)
            eta_s[i] = e_new
            ravg[i] = a_new
            ravg_hist[i].append(a_new)
    uf = d.upml_fine or {}
    Hc, _ = _ltsupd(dc, 0, p, cav_last, H, Baux, None, None, uf.get("h_hist"), uf.get("h_new"),
                    uf.get("h_old"))
    pen_h = fn3._pen_coefs(c, "h") if split else None
    pen_e = fn3._pen_coefs(c, "e") if split else None
    mH = []
    if split:
        mH = [0.5 * (H + Hc)]
        for _ in range(kd):
            T = dc.terms_fwd(zeroE, mH[-1].contiguous(), None, *pen_h)
            mH.append(mH[-1] - sdmp * d.ch[None, None] * _curl(T))
    # E side
    pen = fn3.interface_penalty_rows(st, c, fe, E) if g_if is not None else None
    face, n0 = fn3.interface_slots(st)
    Xh = fn3.interface_x(st, c, Eext[0], True) if g_if is not None else None
    T_e = dc.terms_fwd(H1.contiguous(), E.contiguous(), None if Xh is None else Xh.contiguous(), *hb)
    ca_f, cb_f = c.dg_ca.reshape(-1).contiguous(), c.dg_cb.reshape(-1).contiguous()
    Ec, _ = _ltsupd(dc, 1, p, T_e, E, Daux, ca_f, cb_f, c.u_hist_f, c.u_new_f, c.u_old_f)
    mE, curlE = [], []
    if split:
        mE = [0.5 * (E + Ec)]
        for _ in range(kd):
            T = dc.terms_fwd(zeroE, mE[-1].contiguous(), None, *pen_e)
            cu = _curl(T).contiguous()
            curlE.append(cu)
            mE.append(mE[-1] + sdmp * cb0_dg.view(1, 1, -1, 1) * cu)
    # fine-plane E UPML rates
    rhs_s = [None] * ns
    if y.upml is not None:
        for i, sl in enumerate(slabs):
            if not sl.upml_e:
                continue
            r_ = _slab_q_fwd(yc, sds[i], [fh2[k][:, sl.lo:sl.hi + 1] for k in range(3)], None,
                             None, 0.0)
            if pen is not None:
                for comp in (1, 2):
                    r_[comp] = r_[comp].index_add(1, sl.row, pen[:, i, comp - 1][:, None])
            rhs_s[i] = r_

    # ================================================================ reverse
    # ---- 13 inject
    g_amp = None
    if src is not None:
        g_amp = vjp3d._inject_adjoint(st, c, src, amp, lfe, lE, acc, need_c, need_w)
    g_cb0_dg = torch.zeros_like(cb0_dg) if need_c else None
    # ---- 13 damp_E
    if split:
        lm = lE
        lEc = 0.5 * lm
        lE_old = -0.5 * lm
        for j in reversed(range(kd)):
            gcb = torch.empty((S, d.K), dtype=E.dtype, device=E.device) if need_c else None
            _, gG, _, gb0, gb1, gb2, gb3 = dc.terms_bwd(
                zeroE, mE[j].contiguous(), None, *pen_e, False,
                gepi=(2, [lm.contiguous(), None, None, cb0_dg, curlE[j] if need_c else None, gcb,
                          None], 0.0, 0.0, 0.0, sdmp))
            if need_c:
                _acc(acc, "ph1:pen", gb1, _FACE)
                _acc(acc, "ph3:pen", gb3, _FACE)
                g_cb0_dg += gcb.sum(0)
            lm = lm + gG
        lEc = lEc + 0.5 * lm
        lE_old = lE_old + 0.5 * lm
    else:
        lEc, lE_old = lE, None
    # ---- 13 LTSUpdate_E
    want_u = need_c and c.u_hist_f is not None
    grate, gf, gaux, gca, gcb, gh, gnw, gold = _ltsupd_bwd(
        dc, 1, p, T_e, E, Daux, ca_f, cb_f, c.u_hist_f, c.u_new_f, c.u_old_f, lEc, lD, need_c, want_u)
    lD_old = gaux
    lE_n = gf if lE_old is None else gf + lE_old
    if need_c:
        _acc(acc, "dg_ca:lts", gca)
        _acc(acc, "dg_cb:lts", gcb)
        if gh is not None:
            _acc(acc, "u_hist_f:lts", gh)
            _acc(acc, "u_new_f:lts", gnw)
            _acc(acc, "u_old_f:lts", gold)
    # ---- 12 T_e = Terms(H', E, X_h)
    lH1, lE_n, gXh, g0, g1, g2, g3 = dc.terms_bwd(H1.contiguous(), E.contiguous(), grate, *hb,
                                                  Xh is not None, addF=lH.contiguous(),
                                                  addG=lE_n.contiguous())
    del grate
    if need_c:
        for name, gq in (("h0", g0), ("h1", g1), ("h3", g3)):
            _acc(acc, name + ":flux_e", gq, _FACE)
        _acc(acc, "h2:flux_e", g2, _FACE2)
    gEext0 = None
    if Xh is not None:
        x0, x1 = gXh[:, 0], gXh[:, 1]
        gs = g_if.reshape(-1).index_select(0, face) * n0
        gEext0 = torch.stack([-gs * x1, gs * x0], dim=1)
        if need_c:
            gacc = torch.zeros_like(g_if.reshape(-1)).index_add_(
                0, face, (x0 * Eext[0][:, 1] - x1 * Eext[0][:, 0]).sum(0) * n0)
            _acc(acc, "g_if:x", gacc)
    # ---- 11 fine-plane UPML (E): replaced entries of fe1 / Dfd'
    lfe1 = [t.clone() for t in lfe]
    lDfd1 = lDfd.clone()
    lfe_extra = [None] * 3
    lDfd_extra = None
    lfh2 = [t for t in lfh]
    g_pen_f = torch.zeros_like(pen) if (pen is not None and y.upml is not None) else None
    if y.upml is not None:
        want_f = need_c and c.fu_hist_f is not None
        gfu = [torch.zeros_like(c.fu_hist_f) for _ in range(3)] if want_f else None
        for i, sl in enumerate(slabs):
            if not sl.upml_e:
                continue
            g_rhs = [torch.zeros_like(rhs_s[i][k]) for k in range(3)]
            for j, (k, pos, idx_full, local, fac) in enumerate(sl.upml_e):
                idx = sds[i]["e"][j]
                hist, nw, old = (t.index_select(0, pos) for t in (c.fu_hist_f, c.fu_new_f, c.fu_old_f))
                lv = lfe1[k].reshape(S, -1)
                gfval = lv.index_select(1, idx_full)
                gauxn = lDfd1.index_select(1, pos)
                lv.index_fill_(1, idx_full, 0.0)
                lDfd1.index_fill_(1, pos, 0.0)
                if lfe_extra[k] is None:
                    lfe_extra[k] = torch.zeros_like(fe[k])
                if lDfd_extra is None:
                    lDfd_extra = torch.zeros_like(Dfd)
                grate_s = torch.zeros_like(rhs_s[i][k])
                gfac = _fold_bwd(yc, p, 1.0, idx, fe[k].reshape(S, -1), Dfd,
                                 rhs_s[i][k].reshape(S, -1), (fac["a1"], fac["b1"], hist, nw, old),
                                 gfval, gauxn, lfe_extra[k].view(S, -1), lDfd_extra,
                                 grate_s.view(S, -1), want_f)
                g_rhs[k] = g_rhs[k] + grate_s
                if want_f:
                    for q_, gq in zip(gfu, gfac):
                        q_.index_add_(0, pos, gq)
            if g_pen_f is not None:
                for comp in (1, 2):
                    g_pen_f[:, i, comp - 1] += g_rhs[comp][:, rows_l[i]]
            gfh_s, _ = _slab_q_bwd(yc, sds[i], [fh2[k][:, sl.lo:sl.hi + 1] for k in range(3)], g_rhs,
                                   None, 0.0, False)
            for k in range(3):
                t = lfh2[k].clone() if lfh2[k] is lfh[k] else lfh2[k]
                t[:, sl.lo:sl.hi + 1] += gfh_s[k]
                lfh2[k] = t
        if want_f:
            _acc(acc, "fu_hist_f:fold", gfu[0])
            _acc(acc, "fu_new_f:fold", gfu[1])
            _acc(acc, "fu_old_f:fold", gfu[2])
    # ---- 10 Yee E update
    fu = [t if t is not None else zero for t in (c.fu_hist, c.fu_new, c.fu_old)]
    gsrc, gfe_e, gDfd, gpse, gca, gcb, gh, gnw, gold, gpen = yc.e_bwd(
        [t.contiguous() for t in fh2], [t.contiguous() for t in fe], Dfd, c.fd_ca, c.fd_cb, fu[0],
        fu[1], fu[2], zero if pen is None else pen, [t.contiguous() for t in pse],
        [t.contiguous() for t in lfe1], lDfd1.contiguous(), [t.contiguous() for t in lpse])
    if need_c:
        _acc(acc, "fd_ca:upd", gca, _SHOTS)
        _acc(acc, "fd_cb:upd", gcb, _SHOTS)
        if c.fu_hist is not None:
            _acc(acc, "fu_hist:upd", gh, _SHOTS)
            _acc(acc, "fu_new:upd", gnw, _SHOTS)
            _acc(acc, "fu_old:upd", gold, _SHOTS)
    lfh2 = [a + b for a, b in zip(lfh2, gsrc)]
    lfe_n = gfe_e
    for k in range(3):
        if lfe_extra[k] is not None:
            lfe_n[k] = lfe_n[k] + lfe_extra[k]
    lDfd_old = gDfd if lDfd_extra is None else gDfd + lDfd_extra
    # ---- 10 interface penalty rows
    if gpen is not None or g_pen_f is not None:
        gp_all = gpen if g_pen_f is None else (g_pen_f if gpen is None else gpen + g_pen_f)
        scale = 2.0 / y.spacing[0]
        gfl = g_if.reshape(-1)
        lEf = lE_n.view(S, 3, -1)
        Ef = E.reshape(S, 3, -1)
        gacc = torch.zeros_like(gfl) if need_c else None
        for r, (cp, rg) in enumerate(zip(cs, ranges)):
            gq = gfl.index_select(0, face[rg])
            for comp, lat in ((1, 0), (2, 1)):
                op = cp.to_fd[lat]
                gp = gp_all[:, r, lat].reshape(S, -1)
                qv = scale * op.transpose_apply(gp)
                lEf[:, comp].index_add_(1, cp.nodes, gq * qv)
                row = fe[comp][:, cp.row].reshape(S, -1)
                lfe_n[comp][:, cp.row] -= (scale * op(gq[None]) * gp).view(S, NY, NZ)
                if need_c:
                    em = Ef[:, comp].index_select(1, cp.nodes)
                    gacc.index_add_(0, face[rg], (em * qv).sum(0)
                                    - scale * op.transpose_apply((gp * row).sum(0, keepdim=True))[0])
        if need_c:
            _acc(acc, "g_if:pen", gacc)
    # ---- 9 ghost planes of fh'
    lfh1 = lfh2
    gy = torch.zeros((S, ns, 2, NY, NZ), dtype=E.dtype, device=E.device)
    for r, cp in enumerate(cs):
        for comp, j in ((2, 0), (1, 1)):
            gg = lfh1[comp][:, cp.ghost].clone()
            gy[:, r, j] = gg
            lfh1[comp][:, cp.inner] -= gg
            lfh1[comp][:, cp.ghost] = 0.0
    lH1 = lH1 + 2.0 * gop.transpose_apply(gy.reshape(S, -1)).view_as(H1)
    # ---- 8 fine-plane UPML (H) and the corrections
    lBfd1 = lBfd.clone()
    lfh_extra = [None] * 3
    lBfd_extra = None
    g_ravg = [[torch.zeros_like(ravg[i][k]) for k in range(3)] for i in range(ns)]
    g_corr = []
    for i, sl in enumerate(slabs):
        for j, (k, pos, idx_full, local, fac) in enumerate(sl.upml_h):
            idx = sds[i]["h"][j]
            lv = lfh1[k].reshape(S, -1)
            gfval = lv.index_select(1, idx_full)
            gauxn = lBfd1.index_select(1, pos)
            lv.index_fill_(1, idx_full, 0.0)
            lBfd1.index_fill_(1, pos, 0.0)
            if lfh_extra[k] is None:
                lfh_extra[k] = torch.zeros_like(fh[k])
            if lBfd_extra is None:
                lBfd_extra = torch.zeros_like(Bfd)
            _fold_bwd(yc, p, -1.0, idx, fh[k].reshape(S, -1), Bfd, ravg[i][k].reshape(S, -1),
                      (fac["a1"], fac["b1"], fac["hist"], fac["new"], fac["old"]), gfval, gauxn,
                      lfh_extra[k].view(S, -1), lBfd_extra, g_ravg[i][k].view(S, -1), False)
        g_corr.append([lfh1[k][:, sl.lo:sl.hi + 1].contiguous() for k in range(3)])
    # ---- 7 damp_H
    if split:
        lm = lH1
        lHc = 0.5 * lm
        lH_old = -0.5 * lm
        for j in reversed(range(kd)):
            _, gG, _, gb0, gb1, gb2, gb3 = dc.terms_bwd(
                zeroE, mH[j].contiguous(), None, *pen_h, False,
                gepi=(1, [lm.contiguous(), None, dc.ch, None, None, None, None], 0.0, sdmp, 0.0, 0.0))
            if need_c:
                _acc(acc, "pe1:pen", gb1, _FACE)
            lm = lm + gG
        lHc = lHc + 0.5 * lm
        lH_old = lH_old + 0.5 * lm
    else:
        lHc, lH_old = lH1, None
    # ---- 6 LTSUpdate_H
    lcav, gfH, gauxB, *_ = _ltsupd_bwd(dc, 0, p, cav_last, H, Baux, None, None, uf.get("h_hist"),
                                       uf.get("h_new"), uf.get("h_old"), lHc, lB, False, False)
    lB_old = gauxB
    lH_n = gfH if lH_old is None else gfH + lH_old
    # ---- 5 the recursion, m = p-1 .. 1
    leta = [None] * p                    # cotangents of eta^m (DG)
    lzdg = None                          # cotangent of z^m (DG), carried
    lzs = [None] * ns                    # cotangents of z_s^m, carried
    leta_s = [None] * ns                 # cotangents of the outputs eta_s^m of SlabR(m)
    lravg = g_ravg                       # cotangents of ravg^m
    lr0 = [None] * ns
    g_cb0_s = [None] * ns
    for m in range(p - 1, 0, -1):
        last = m == p - 1
        # SlabR(m)
        letl = [None] * ns               # cotangent of eta~_s^(m-1)
        for i in range(ns):
            gz, gein, gav, gr0 = _slab_r_bwd(yc, sds[i], zs[i][m][0], False, last, last, beta[m],
                                             ck[m] * inv_p, w[m], None,
                                             None if last else leta_s[i], lravg[i],
                                             g_corr[i] if last else None)
            lzs[i] = gz if lzs[i] is None else [a + b for a, b in zip(lzs[i], gz)]
            letl[i] = gein
            lravg[i] = gav
            if gr0 is not None:
                lr0[i] = gr0 if lr0[i] is None else [a + b for a, b in zip(lr0[i], gr0)]
        # TermsEta(m)
        geta_in = torch.empty_like(E) if leta[m] is not None else None
        gF, gG, gX, g0, g1, g2, g3 = dc.terms_bwd(
            zdg[m], H.contiguous(), None, *eb, True,
            gepi=(1, [None if leta[m] is None else leta[m].contiguous(), lcav.contiguous(), dc.ch, None,
                      None, None, geta_in], beta[m], ck[m] * inv_p, w[m], 0.0))
        lzdg = gF if lzdg is None else lzdg + gF
        lH_n = lH_n + gG
        if need_c:
            for name, gq in (("e0", g0), ("e1", g1), ("e3", g3)):
                _acc(acc, name + ":flux_h", gq, _FACE)
            _acc(acc, "e2:flux_h", g2, _FACE2)
        tt = top.transpose_apply(gX.reshape(S, -1)).view(S, ns, 2, NY, NZ)
        for i in range(ns):
            lzs[i][1][:, rows_l[i]] += tt[:, i, 0]
            lzs[i][2][:, rows_l[i]] += tt[:, i, 1]
        # SlabQ(m)
        for i in range(ns):
            gfq, gcb0 = _slab_q_bwd(yc, sds[i], etl[i][m - 1], lzs[i], cb0_s[i], inv_p, need_c)
            letl[i] = gfq if letl[i] is None else [a + b for a, b in zip(letl[i], gfq)]
            if need_c:
                g_cb0_s[i] = gcb0 if g_cb0_s[i] is None else g_cb0_s[i] + gcb0
        # TermsZ(m)
        gcb = torch.empty((S, d.K), dtype=E.dtype, device=E.device) if need_c else None
        gF, _, _, g0, g1, g2, g3 = dc.terms_bwd(
            eta[m - 1], zeroE, None, *hb, False,
            gepi=(2, [lzdg.contiguous(), None, None, cb0_dg, zcurl[m], gcb, None], 0.0, 0.0, 0.0,
                  inv_p))
        if need_c:
            for name, gq in (("h0", g0), ("h1", g1), ("h3", g3)):
                _acc(acc, name + ":flux_z", gq, _FACE)
            _acc(acc, "h2:flux_z", g2, _FACE2)
            g_cb0_dg += gcb.sum(0)
        le = gF if geta_in is None else gF + geta_in
        # ghost planes of eta~_s^(m-1)
        gy = torch.zeros((S, ns, 2, NY, NZ), dtype=E.dtype, device=E.device)
        for i, sl in enumerate(slabs):
            lt = [t.clone() for t in letl[i]]
            inner = sl.cp.inner - sl.lo
            gs_ = int(sl.ghost)
            for comp, j in ((2, 0), (1, 1)):
                gg = lt[comp][:, gs_].clone()
                gy[:, i, j] = 2.0 * gg
                lt[comp][:, inner] -= gg
                lt[comp][:, gs_] = 0.0
            leta_s[i] = lt
        le = le + gop.transpose_apply(gy.reshape(S, -1)).view_as(E)
        leta[m - 1] = le if leta[m - 1] is None else leta[m - 1] + le
    # ---- 4 SlabR(first)
    for i in range(ns):
        gz, _, _, _ = _slab_r_bwd(yc, sds[i], zs[i][0][0], True, False, False, 0.0, c0 * inv_p, w[0],
                                  lr0[i] if lr0[i] is not None else [torch.zeros_like(zs[i][0][0])] * 3,
                                  leta_s[i], lravg[i], None)
        lzs[i] = gz if lzs[i] is None else [a + b for a, b in zip(lzs[i], gz)]
    # ---- 3 TermsEta(first)
    gF, gG, gX, g0, g1, g2, g3 = dc.terms_bwd(
        E.contiguous(), H.contiguous(), None, *eb, True,
        gepi=(1, [leta[0].contiguous(), lcav.contiguous(), dc.ch, None, None, None, None], 0.0,
              c0 * inv_p, w[0], 0.0))
    if need_c:
        for name, gq in (("e0", g0), ("e1", g1), ("e3", g3)):
            _acc(acc, name + ":flux_h0", gq, _FACE)
        _acc(acc, "e2:flux_h0", g2, _FACE2)
    lE_n = lE_n + gF + (lzdg if lzdg is not None else 0.0)
    lH_n = lH_n + gG
    gE0 = gX if gEext0 is None else gX + gEext0
    # ---- 2 trace E*^0 and the slab views of fe
    tt = top.transpose_apply(gE0.reshape(S, -1)).view(S, ns, 2, NY, NZ)
    for i, sl in enumerate(slabs):
        lzs[i][1][:, rows_l[i]] += tt[:, i, 0]
        lzs[i][2][:, rows_l[i]] += tt[:, i, 1]
        for k in range(3):
            lfe_n[k][:, sl.lo:sl.hi + 1] += lzs[i][k]
    # cb0 chain rule (Yee slabs and DG band)
    if need_c:
        gca_f = torch.zeros_like(c.fd_ca)
        gcb_f = torch.zeros_like(c.fd_cb)
        for i, sl in enumerate(slabs):
            if g_cb0_s[i] is None:
                continue
            ca_s, cb_s = c.fd_ca[sl.lo:sl.hi + 1], c.fd_cb[sl.lo:sl.hi + 1]
            gsl = g_cb0_s[i][0]
            gcb_f[sl.lo:sl.hi + 1] += gsl * 2.0 / (1.0 + ca_s)
            gca_f[sl.lo:sl.hi + 1] += -gsl * 2.0 * cb_s / (1.0 + ca_s) ** 2
        _acc(acc, "fd_ca:lts", gca_f)
        _acc(acc, "fd_cb:lts", gcb_f)
        ca_d, cb_d = c.dg_ca.reshape(-1), c.dg_cb.reshape(-1)
        _acc(acc, "dg_cb:lts_cb0", g_cb0_dg * 2.0 / (1.0 + ca_d))
        _acc(acc, "dg_ca:lts_cb0", -g_cb0_dg * 2.0 * cb_d / (1.0 + ca_d) ** 2)
    # ---- 1 Yee H update
    gsrc, gfh, gBfd, gpsh = yc.h_bwd([t.contiguous() for t in fe], [t.contiguous() for t in fh], Bfd,
                                     [t.contiguous() for t in psh], [t.contiguous() for t in lfh1],
                                     lBfd1.contiguous(), [t.contiguous() for t in lpsh])
    for k in range(3):
        lfe_n[k] = lfe_n[k] + gsrc[k]
        if lfh_extra[k] is not None:
            gfh[k] = gfh[k] + lfh_extra[k]
    lBfd_old = gBfd if lBfd_extra is None else gBfd + lBfd_extra
    out = list(lfe_n) + list(gfh) + list(gpse) + list(gpsh) + [lE_n, lH_n, lD_old, lB_old,
                                                               lDfd_old, lBfd_old]
    return out, g_amp


__all__ = ["finalize", "sample_adjoint", "step_vjp", "supported"]
