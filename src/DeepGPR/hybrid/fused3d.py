"""Fully fused 3D step (global time step) and its transpose with explicit outputs.

:func:`step_into` advances the coupled DGTD-FDTD state (or the pure Yee state) by one
step and writes ``u^(n+1)`` into given buffers; :func:`vjp_into` applies the transpose
of that step and writes the cotangent of ``u^n`` into given buffers. Every operation
is a CUDA kernel: the fused Yee and DG kernels of :mod:`.cuda3d` and the coupling,
source and receiver kernels of :mod:`.glue3d`; there are no PyTorch gathers, scatters
or elementwise passes over full arrays left (the Yee transpose accumulates its
cotangents and coefficient gradients in place). The arithmetic is that of
:func:`DeepGPR.hybrid.functional3d.step` and of :func:`DeepGPR.hybrid.vjp3d.step_vjp`
(checked against both in ``tests/test_hybrid_fused.py``).

Because outputs go to given buffers, a sequence of steps can be captured once as
CUDA Graphs on fixed buffers (:class:`DeepGPR.hybrid.graphs.SweepGraphs`) without
copying the state in and out of static buffers at every step.

:func:`vjp_into` destroys its input cotangent ``lam`` (it accumulates into it in
place); callers keep two cotangent buffers and alternate them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from . import functional3d as fn3
from .glue3d import coupling_glue, point_glue

_SHOTS = lambda t: t.sum(0)  # noqa: E731
_FACE = lambda t: t.sum(0).t()  # noqa: E731
_FACE2 = lambda t: t.sum(0).transpose(1, 2)  # noqa: E731


def available(st: fn3.StepStatic3D) -> bool:
    """``True`` when the global step of ``st`` runs fully on fused kernels: CUDA, the Yee
    kernels attached, and either no DG band (pure Yee) or the DG kernels with couplings
    and one global step (no local time stepping)."""
    y, d = st.yee, st.dg
    if torch.device(st.device).type != "cuda" or y is None or y.cuda is None:
        return False
    if d is None:
        return True
    return d.cuda is not None and bool(st.couplings) and int(getattr(d, "lts", 1)) == 1


def empty_state(st: fn3.StepStatic3D, like=None) -> List[torch.Tensor]:
    """Uninitialised state buffers (shapes of :func:`~DeepGPR.hybrid.functional3d.zero_state`)."""
    ref = like if like is not None else fn3.zero_state(st)
    return [torch.empty_like(t) for t in ref]


class Accumulators:
    """Coefficient cotangents of a reverse sweep, accumulated in place in kernel layouts
    (per-shot arrays, face layouts); :meth:`finalize` converts them once. Buffers are
    created on first use and kept, so that a captured reverse step adds into fixed
    memory; :meth:`zero` resets them."""

    def __init__(self):
        self.items: Dict[str, list] = {}
        self.deferred: Dict[str, list] = {}

    def buf(self, key: str, shape, ref: torch.Tensor, convert=None) -> torch.Tensor:
        ent = self.items.get(key)
        if ent is None:
            ent = self.items[key] = [torch.zeros(shape, dtype=ref.dtype, device=ref.device),
                                     convert]
        return ent[0]

    def add(self, key: str, g: Optional[torch.Tensor], convert=None):
        if g is None:
            return
        self.buf(key, g.shape, g, convert).add_(g)

    def face(self, c, S: int, names):
        """Running sums of the face-coefficient cotangents ``(b0, b1, b2, b3)`` of a DG
        transpose (:meth:`~DeepGPR.hybrid.cuda3d.DGCudaOps.terms_bwd` with ``acc_face``),
        one per coefficient name (``None``: not wanted), in the kernel layouts
        ``(S, 4, K)`` / ``(S, 2, 4, K)``."""
        out = []
        for j, nm in enumerate(names):
            if nm is None:
                out.append(None)
                continue
            t = getattr(c, nm)
            K = t.shape[-2]
            out.append(self.buf(nm + ":face", (S, 2, 4, K) if j == 2 else (S, 4, K), t,
                                _FACE2 if j == 2 else _FACE))
        return tuple(out)

    def raw(self, key: str, shape, ref: torch.Tensor, fn) -> torch.Tensor:
        """A running sum converted only once, at :meth:`finalize`, by the linear map
        ``fn(t, c) -> [(name, value), ...]`` (a chain rule with constant factors)."""
        ent = self.deferred.get(key)
        if ent is None:
            ent = self.deferred[key] = [torch.zeros(shape, dtype=ref.dtype, device=ref.device), fn]
        return ent[0]

    def zero(self):
        for ent in list(self.items.values()) + list(self.deferred.values()):
            ent[0].zero_()

    def finalize(self, c: fn3.StepCoefs3D) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        def put(name, v):
            v = v.reshape(getattr(c, name).shape)
            out[name] = v.clone() if name not in out else out[name] + v
        for key, (t, convert) in self.items.items():
            put(key.split(":")[0], convert(t) if convert is not None else t)
        for t, fn in self.deferred.values():
            for name, v in fn(t, c):
                put(name, v)
        return out


def _unpack(u):
    return (list(u[0:3]), list(u[3:6]), list(u[6:12]), list(u[12:18])) + tuple(u[18:24])


def _z(ref):
    return torch.zeros(0, dtype=ref.dtype, device=ref.device)


def _fu(c, ref):
    z = _z(ref)
    return [t if t is not None else z for t in (c.fu_hist, c.fu_new, c.fu_old)]


def step_into(st: fn3.StepStatic3D, c: fn3.StepCoefs3D, u: Sequence[torch.Tensor],
              out: Sequence[torch.Tensor], src=None, amp=None, rec=None, rec_out=None):
    """One step ``u -> out`` (24 tensors each, ``out`` written, ``u`` unchanged); with
    ``rec``/``rec_out`` the receiver samples of the new state go to ``rec_out`` (S, n)."""
    ctx = step_h_part(st, c, u, out)
    step_e_part(st, c, u, out, ctx, src, amp, rec, rec_out)


def step_h_part(st: fn3.StepStatic3D, c: fn3.StepCoefs3D, u, out):
    """H half of :func:`step_into` (Yee H, DG H, ghost rows; the interface penalty of the
    E half is prepared here): writes the H parts of ``out``; returns the context of
    :func:`step_e_part` (the Yee H of ``out`` may be changed in between, as the
    subgrids' outer-surface corrections do)."""
    y, d = st.yee, st.dg
    yc, dc = y.cuda, (d.cuda if d is not None else None)
    fe, fh, pse, psh, E, H, Daux, Baux, Dfd, Bfd = _unpack(u)
    ofe, ofh, opse, opsh, oE, oH, oD, oB, oDfd, oBfd = _unpack(out)
    z = _z(fe[0])
    hybrid = d is not None
    yc.h_fwd(fe, fh, Bfd, psh, out=(ofh, oBfd, opsh))
    Xh = None
    pen = z
    if hybrid:
        g = coupling_glue(st)
        S = st.nshot
        Ex = torch.empty((S, 2, g.nX), dtype=fe[0].dtype, device=fe[0].device)
        if c.g_if is not None:
            Xh = torch.empty_like(Ex)
        planes = g.fe_planes(fe[1], fe[2])
        g.trace(planes, Ex, Xh, c.g_if)
        T = dc.terms_fwd(E, H, Ex, c.e0, c.e1, c.e2, c.e3)
        dc.update_fwd(0, T, H, Baux, out=(oH, oB))
        del T
        g.ghost(g.ghost_planes(ofh[2], ofh[1]), oH)
        if c.g_if is not None:
            pen = torch.empty((S, g.ncp, 2, g.NY, g.NZ), dtype=fe[0].dtype, device=fe[0].device)
            g.penalty(planes, E, c.g_if, pen)
    return (Xh, pen)


def step_e_part(st: fn3.StepStatic3D, c: fn3.StepCoefs3D, u, out, ctx, src=None, amp=None,
                rec=None, rec_out=None):
    """E half of :func:`step_into` (Yee E from the Yee H of ``out``, DG E, sources,
    receivers)."""
    y, d = st.yee, st.dg
    yc, dc = y.cuda, (d.cuda if d is not None else None)
    fe, fh, pse, psh, E, H, Daux, Baux, Dfd, Bfd = _unpack(u)
    ofe, ofh, opse, opsh, oE, oH, oD, oB, oDfd, oBfd = _unpack(out)
    hybrid = d is not None
    Xh, pen = ctx
    fu = _fu(c, fe[0])
    yc.e_fwd(ofh, fe, Dfd, c.fd_ca, c.fd_cb, fu[0], fu[1], fu[2], pen, pse,
             out=(ofe, oDfd, opse))
    if hybrid:
        T = dc.terms_fwd(oH, E, Xh, c.h0, c.h1, c.h2, c.h3)
        dc.update_fwd(1, T, E, Daux, c.dg_ca.reshape(-1), c.dg_cb.reshape(-1), c.u_hist,
                      c.u_new, c.u_old, out=(oE, oD))
        del T
    # ---------------------------------------------------------------- sources / receivers
    if src is not None:
        pg = point_glue(st, src)
        pg.inject(ofe[src.comp], oE, c.fd_cb, c.dg_cb if hybrid else None, amp)
    if rec is not None and rec_out is not None:
        point_glue(st, rec).sample(ofe[rec.comp], oE, rec_out)


def vjp_into(st: fn3.StepStatic3D, c: fn3.StepCoefs3D, u, u_next, lam, out, src=None,
             amp=None, gd=None, rec=None, acc: Optional[Accumulators] = None,
             need_c: bool = True, need_w: bool = False, gamp=None):
    """Transpose of :func:`step_into` (``u^n -> u^(n+1)``, receivers sampled from
    ``u^(n+1)``): ``out`` = cotangent of ``u^n`` given the cotangent ``lam`` of
    ``u^(n+1)`` and the receiver cotangent ``gd`` (S, n) of this step. ``lam`` is
    destroyed. Coefficient cotangents accumulate into ``acc`` (``need_c``), the source
    sample cotangent goes to ``gamp`` (zeroed here; ``need_w``)."""
    ctx = vjp_e_part(st, c, u, u_next, lam, out, src, amp, gd, rec, acc, need_c, need_w, gamp)
    vjp_h_part(st, c, u, u_next, lam, out, acc, need_c, ctx)


def vjp_e_part(st: fn3.StepStatic3D, c: fn3.StepCoefs3D, u, u_next, lam, out, src=None,
               amp=None, gd=None, rec=None, acc: Optional[Accumulators] = None,
               need_c: bool = True, need_w: bool = False, gamp=None):
    """Transpose of :func:`step_e_part` (and of the receivers): writes the cotangents of
    the E-half inputs into ``out``, accumulates the cotangent of the Yee H it read into
    ``lam`` (the Yee H of ``u^(n+1)``); returns the context of :func:`vjp_h_part`."""
    y, d = st.yee, st.dg
    yc, dc = y.cuda, (d.cuda if d is not None else None)
    hybrid = d is not None
    fe, fh, pse, psh, E, H, Daux, Baux, Dfd, Bfd = _unpack(u)
    fh2, H1 = list(u_next[3:6]), u_next[19]
    lfe, lfh, lpse, lpsh, lE, lH, lD, lB, lDfd, lBfd = _unpack(lam)
    ofe, ofh, opse, opsh, oE, oH, oD, oB, oDfd, oBfd = _unpack(out)
    z = _z(fe[0])
    S = st.nshot
    # ---------------------------------------------------------------- receivers
    if rec is not None and gd is not None:
        point_glue(st, rec).sample_bwd(gd, lam[rec.comp], lE)
    # ---------------------------------------------------------------- recompute
    Xh, pen, Ex = None, z, None
    if hybrid:
        g = coupling_glue(st)
        Ex = torch.empty((S, 2, g.nX), dtype=fe[0].dtype, device=fe[0].device)
        if c.g_if is not None:
            Xh = torch.empty_like(Ex)
        planes = g.fe_planes(fe[1], fe[2])
        g.trace(planes, Ex, Xh, c.g_if)
        if c.g_if is not None:
            pen = torch.empty((S, g.ncp, 2, g.NY, g.NZ), dtype=fe[0].dtype, device=fe[0].device)
            g.penalty(planes, E, c.g_if, pen)
        T_e = dc.terms_fwd(H1, E, Xh, c.h0, c.h1, c.h2, c.h3)
    # ---------------------------------------------------------------- sources
    if src is not None and (need_w or need_c):
        if need_w:
            gamp.zero_()
        gcf = gcd = None
        if need_c:
            gcf = acc.buf("fd_cb:src", c.fd_cb.shape, c.fd_cb)
            if hybrid:
                gcd = acc.buf("dg_cb:src", c.dg_cb.shape, c.dg_cb)
        point_glue(st, src).inject_bwd(lam[src.comp], lE, c.fd_cb, c.dg_cb if hybrid else None,
                                       amp, gamp if need_w else None, gcf, gcd)
    if hybrid:
        # ------------------------------------------------------------ DG E update
        gT, gE_a, _, gca, gcb, gh, gnw, gold = dc.update_bwd(
            1, T_e, E, Daux, lE, lD, c.dg_ca.reshape(-1), c.dg_cb.reshape(-1), c.u_hist, c.u_new,
            c.u_old, out_gaux=oD)
        del T_e
        if need_c:
            acc.add("dg_ca:upd", gca, _SHOTS)
            acc.add("dg_cb:upd", gcb, _SHOTS)
            if gh is not None and c.u_hist is not None:
                acc.add("u_hist:upd", gh, _SHOTS)
                acc.add("u_new:upd", gnw, _SHOTS)
                acc.add("u_old:upd", gold, _SHOTS)
        # ------------------------------------------------------------ Terms(H', E, X_h)
        lH1, lE_n, gXh, *_ = dc.terms_bwd(
            H1, E, gT, c.h0, c.h1, c.h2, c.h3, Xh is not None, addF=lH, addG=gE_a,
            acc_face=acc.face(c, S, ("h0", "h1", "h2", "h3")) if need_c else None)
        del gT, gE_a
        if need_c and gXh is not None:
            g.xh_gif(gXh, Ex, acc.buf("g_if:x", (c.g_if.numel(),), c.g_if))
    # ---------------------------------------------------------------- Yee E update
    fu = _fu(c, fe[0])
    coef = None
    if need_c:
        coef = (acc.buf("fd_ca:upd", tuple(c.fd_ca.shape), c.fd_ca),     # summed over the shots
                acc.buf("fd_cb:upd", tuple(c.fd_cb.shape), c.fd_cb), None, None, None)
        if c.fu_hist is not None:
            ntot = yc.side["e"]["ntot"]
            coef = coef[:2] + tuple(acc.buf(nm + ":upd", (ntot,), c.fd_ca)
                                    for nm in ("fu_hist", "fu_new", "fu_old"))
    res = yc.e_bwd(fh2, fe, Dfd, c.fd_ca, c.fd_cb, fu[0], fu[1], fu[2], pen, pse, lfe, lDfd,
                   lpse, out=(ofe, oDfd, opse), gsrc_acc=lfh, coef_acc=coef)
    gpen = res[9]
    if not hybrid:
        return None
    # ---------------------------------------------------------------- penalty rows
    if gpen is not None:
        g.penalty_bwd(planes, g.fe_planes(ofe[1], ofe[2]), E, c.g_if, gpen, None, lE_n,
                      acc.buf("g_if:pen", (c.g_if.numel(),), c.g_if) if need_c else None)
    return (g, lE_n, gXh, lH1)


def vjp_h_part(st: fn3.StepStatic3D, c: fn3.StepCoefs3D, u, u_next, lam, out,
               acc: Optional[Accumulators] = None, need_c: bool = True, ctx=None):
    """Transpose of :func:`step_h_part` given the complete cotangent of its Yee H output
    in ``lam``: writes the remaining input cotangents into ``out``."""
    y, d = st.yee, st.dg
    yc, dc = y.cuda, (d.cuda if d is not None else None)
    hybrid = d is not None
    fe, fh, pse, psh, E, H, Daux, Baux, Dfd, Bfd = _unpack(u)
    lfe, lfh, lpse, lpsh, lE, lH, lD, lB, lDfd, lBfd = _unpack(lam)
    ofe, ofh, opse, opsh, oE, oH, oD, oB, oDfd, oBfd = _unpack(out)
    S = st.nshot
    if hybrid:
        g, lE_n, gXh, lH1 = ctx
        # ------------------------------------------------------------ ghost planes
        g.ghost_bwd(g.ghost_planes(lfh[2], lfh[1]), lH1)
        # ------------------------------------------------------------ DG H update
        gT, gH_a, _, *_ = dc.update_bwd(0, None, H, Baux, lH1, lB, out_gaux=oB)
        del lH1
        # ------------------------------------------------------------ Terms(E, H, E*)
        _, _, gX, *_ = dc.terms_bwd(
            E, H, gT, c.e0, c.e1, c.e2, c.e3, True, addF=lE_n, addG=gH_a, out=(oE, oH),
            acc_face=acc.face(c, S, ("e0", "e1", "e2", "e3")) if need_c else None)
        del gT, gH_a, lE_n
        # ------------------------------------------------------------ projection E*
        g.trace_bwd(g.fe_planes(ofe[1], ofe[2]), gX, gXh, c.g_if)
    # ---------------------------------------------------------------- Yee H update
    yc.h_bwd(fe, fh, Bfd, psh, lfh, lBfd, lpsh, out=(ofh, oBfd, opsh), gsrc_acc=ofe)


# =============================================================================== local time stepping
# The sub-cycled step of functional3d.lts_h_half_step / e_half_step and its transpose
# (vjp3d_lts.step_vjp) with the coupling glue as kernels, the split-penalty damping as
# operator epilogues, the UPML folds of the fine Yee planes scattered / gathered in the
# fold kernels, coefficient-derived arrays prepared once per coefficient set
# (:class:`LTSPre`) and the cotangent destroyed in place instead of cloned.

from .cuda3d import _check, _ptrs, _stream  # noqa: E402
from .glue3d import combine  # noqa: E402
from .vjp3d_lts import _slab_q_bwd, _slab_q_fwd, _slab_r_bwd, _slab_r_fwd  # noqa: E402


def lts_available(st: fn3.StepStatic3D) -> bool:
    """The fused LTS step applies (the kernels of :func:`DeepGPR.hybrid.vjp3d_lts.supported`)."""
    y, d = st.yee, st.dg
    return (torch.device(st.device).type == "cuda" and y is not None and y.cuda is not None
            and d is not None and d.cuda is not None and int(getattr(d, "lts", 1)) > 1
            and bool(st.couplings) and hasattr(y.cuda, "slab_r") and hasattr(d.cuda, "terms_eta")
            and getattr(d.cuda, "uf", "x") != "x")


def any_available(st: fn3.StepStatic3D) -> bool:
    """A fused step (global or LTS) applies."""
    return available(st) or lts_available(st)


class LTSPre:
    """Coefficient-derived arrays of the LTS step (``dt/eps`` on the slabs and in the
    band, the gathered fine-plane UPML factors, the zero DG field and the penalty face
    coefficients), computed into fixed buffers by :meth:`update`."""

    def __init__(self, st, c):
        y, d = st.yee, st.dg
        self.slabs = fn3.lts_slabs(st)
        S = st.nshot
        kw = dict(dtype=st.dtype, device=st.device)
        self.cb0_s = [torch.empty((1, sl.hi - sl.lo + 1) + tuple(y.shape[1:]), **kw)
                      for sl in self.slabs]
        self.cb0_dg = torch.empty((d.K,), **kw)
        self.zero = torch.zeros((S, 3, d.K, d.Np), **kw)
        z, z2 = torch.zeros_like(c.e0), torch.zeros_like(c.e2)
        self.pen_h = (z, c.pe1, z2, z) if c.pe1 is not None else None
        self.pen_e = (z, c.ph1, z2, c.ph3) if c.ph1 is not None else None
        self.fu_f = [[tuple(torch.empty(pos.numel(), **kw) for _ in range(3))
                      for (_, pos, _, _, _) in sl.upml_e] for sl in self.slabs]
        # slab-shaped scratch of the transpose that is written at fixed entries only (the
        # fine-plane UPML folds): zeroed once, the same entries are overwritten every step
        shp = [(S, sl.hi - sl.lo + 1) + tuple(y.shape[1:]) for sl in self.slabs]
        self.g_rhs = [[torch.zeros(sh, **kw) for _ in range(3)] if sl.upml_e else None
                      for sl, sh in zip(self.slabs, shp)]
        self.g_ravg = [[torch.zeros(sh, **kw) for _ in range(3)] for sh in shp]
        self.zero_s = [torch.zeros(sh, **kw) for sh in shp]
        self.c = c
        self.update(c)

    def update(self, c):
        """Recompute from ``c`` (the coefficient set the step will use)."""
        with torch.no_grad():
            for sl, buf in zip(self.slabs, self.cb0_s):
                ca, cb = c.fd_ca[sl.lo:sl.hi + 1], c.fd_cb[sl.lo:sl.hi + 1]
                buf.copy_((2.0 * cb / (1.0 + ca))[None])
            self.cb0_dg.copy_((2.0 * c.dg_cb / (1.0 + c.dg_ca)).reshape(-1))
            for sl, sets in zip(self.slabs, self.fu_f):
                for (_, pos, _, _, _), bufs in zip(sl.upml_e, sets):
                    for src, dst in zip((c.fu_hist_f, c.fu_new_f, c.fu_old_f), bufs):
                        torch.index_select(src, 0, pos, out=dst)


def prepare(st, c, pre=None):
    """The per-coefficient-set data of the fused step of ``st`` (``None`` for the global
    step); with ``pre`` given it is updated in place."""
    if not lts_available(st):
        return None
    if pre is None:
        return LTSPre(st, c)
    pre.update(c)
    return pre


def step_any(st, c, u, out, src=None, amp=None, rec=None, rec_out=None, pre=None):
    """:func:`step_into` or :func:`lts_step_into` (``pre`` from :func:`prepare`)."""
    if pre is not None:
        lts_step_into(st, c, u, out, src, amp, rec, rec_out, pre)
    else:
        step_into(st, c, u, out, src, amp, rec, rec_out)


def vjp_any(st, c, u, u_next, lam, out, src=None, amp=None, gd=None, rec=None, acc=None,
            need_c=True, need_w=False, gamp=None, pre=None):
    """:func:`vjp_into` or :func:`lts_vjp_into`."""
    if pre is not None:
        lts_vjp_into(st, c, u, u_next, lam, out, src, amp, gd, rec, acc, need_c, need_w, gamp, pre)
    else:
        vjp_into(st, c, u, u_next, lam, out, src, amp, gd, rec, acc, need_c, need_w, gamp)


def _fold_scatter(yc, p, sign, idx, f, aux, rate, fac, df, daux):
    """``p`` UPML sub-steps of the entries ``idx`` written into ``df`` / ``daux``."""
    S, n = f.shape[0], fac[0].numel()
    q = _ptrs(list(idx) + [f, aux, rate] + list(fac) + [df, daux])
    _check(yc.lib.lts3d_fold(yc.code, 2, S, n, int(p), float(sign), f.shape[1], aux.shape[1],
                             rate.shape[1], q, _stream(f)), "LTS UPML fold (scatter)")


def _fold_gather(yc, p, sign, idx, f, aux, rate, fac, lf, laux, cf, caux, grate, gfac=None):
    """Transpose of :func:`_fold_scatter`: output cotangents gathered from ``lf`` / ``laux``
    (zeroed there), the cotangents of the old ``f`` / ``aux`` into compact ``cf`` / ``caux``
    (S, n), the rate cotangent written at the entries of ``grate``, the factor cotangents
    into ``gfac`` (three ``(n,)`` arrays) when given."""
    S, n = f.shape[0], fac[0].numel()
    g3 = list(gfac) if gfac is not None else [None] * 3
    q = _ptrs(list(idx) + [f, aux, rate] + list(fac) + [lf, laux, cf, caux, grate] + g3)
    _check(yc.lib.lts3d_fold(yc.code, 7, S, n, int(p), float(sign), f.shape[1], aux.shape[1],
                             rate.shape[1], q, _stream(f)), "LTS UPML fold transpose (gather)")


def _ltsupd_into(dc, mode, p, rate, f, aux, ca, cb, hist, nw, old, fnew, auxn):
    z = torch.zeros(0, dtype=f.dtype, device=f.device)
    u = dc.uf or {}
    q = _ptrs([dc.pml, rate, f, aux, dc.ch, z if ca is None else ca, z if cb is None else cb,
               u.get("a1"), u.get("b1"), z if hist is None else hist, z if nw is None else nw,
               z if old is None else old, fnew, auxn])
    _check(dc.lib.lts3d_update(dc.code, 0, mode, f.shape[0], dc.K, dc.Np, dc.Kp, int(p), q,
                               _stream(f)), "LTS DG update")


def _ltsupd_bwd_into(dc, mode, p, rate, f, aux, ca, cb, hist, nw, old, gfn, gauxn, gaux,
                     want_cacb, want_fac):
    """Transpose of :func:`_ltsupd_into`; ``gaux`` written into the given buffer; returns
    ``(grate, gf, gca, gcb, gh, gn, go)``."""
    z = torch.zeros(0, dtype=f.dtype, device=f.device)
    ca, cb = (z if t is None else t for t in (ca, cb))
    hist, nw, old = (z if t is None else t for t in (hist, nw, old))
    grate, gf = torch.empty_like(rate), torch.empty_like(f)
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
    return grate, gf, gca, gcb, gh, gn, go


def _cc(ts):
    return [None if t is None else t.contiguous() for t in ts]


def _sr_fwd(yc, sd, z, eta, ravg, first, last, beta, b, w, r0, corr_into=None):
    """Slab H rates and recursion (:func:`DeepGPR.hybrid.vjp3d_lts._slab_r_fwd`); with
    ``corr_into`` (the fine H planes' slab views, contiguous) the last sub-step's
    correction is added there instead of returned."""
    z = _cc(z)
    eta_ = _cc(eta) if (eta is not None and not first and not last) else [None] * 3
    ravg_ = _cc(ravg) if not first else [None] * 3
    cm = r0 is not None
    r0_ = _cc(r0) if cm else [None] * 3
    r_out = [torch.empty_like(t) for t in z] if first else [None] * 3
    eta_out = [torch.empty_like(t) for t in z] if not last else [None] * 3
    ravg_out = [torch.empty_like(t) for t in z]
    acc = cm and corr_into is not None
    corr = (list(corr_into) if acc else [torch.empty_like(t) for t in z]) if cm else [None] * 3
    mh = sd["mask_h"] if cm else [None] * 3
    q = _ptrs(z + eta_ + ravg_ + sd["chr"] + r_out + eta_out + ravg_out + r0_ + list(mh) + corr)
    _check(yc.lib.lts3d_slab_r(yc.code, 0, z[0].shape[0], sd["n"], yc.shape[1], yc.shape[2], yc.gd,
                               q, float(beta), float(b), float(w), int(first) | (2 if acc else 0),
                               _stream(z[0])), "LTS slab H rates")
    return ((r_out if first else None), (None if last else eta_out), ravg_out,
            (corr if (cm and not acc) else None))


def _sr_bwd(yc, sd, like, first, last, cm, beta, b, w, gr, ge, ga, gq, gz=None):
    """Transpose of :func:`_sr_fwd` (:func:`DeepGPR.hybrid.vjp3d_lts._slab_r_bwd`); with
    ``gz`` given the cotangent of ``z`` is accumulated into it."""
    gr_ = _cc(gr) if first else [None] * 3
    ge_ = _cc(ge) if (not last and ge is not None) else [None] * 3
    ga_ = _cc(ga) if ga is not None else [torch.zeros_like(like) for _ in range(3)]
    gq_ = _cc(gq) if (cm and gq is not None) else [None] * 3
    new = lambda: torch.empty_like(like)  # noqa: E731
    acc = gz is not None
    gz = list(gz) if acc else [new() for _ in range(3)]
    gein = [new() for _ in range(3)] if (not first and not last) else [None] * 3
    gain = [new() for _ in range(3)] if (cm and not first) else [None] * 3
    gr0 = [new() for _ in range(3)] if cm else [None] * 3
    mh = sd["mask_h"] if cm else [None] * 3
    q = _ptrs(gr_ + ge_ + ga_ + sd["chr"] + gz + gein + gq_ + list(mh) + gain + gr0)
    _check(yc.lib.lts3d_slab_r(yc.code, 1, like.shape[0], sd["n"], yc.shape[1], yc.shape[2], yc.gd,
                               q, float(beta), float(b), float(w), int(first) | (2 if acc else 0),
                               _stream(like)), "LTS slab H rates transpose")
    g_ravg = None if first else (gain if cm else ga_)
    return gz, (gein if gein[0] is not None else None), g_ravg, (gr0 if cm else None)


def _sq_bwd(yc, sd, f, g, cb0, a, gf=None, gcb0=None):
    """Transpose of the slab E rates (:func:`DeepGPR.hybrid.vjp3d_lts._slab_q_bwd`): the
    cotangent of ``f`` into ``gf`` (accumulated when given), that of ``cb0`` accumulated
    into ``gcb0`` (z mode, when given)."""
    f, g = _cc(f), _cc(g)
    zmode = cb0 is not None
    acc = gf is not None
    gf = list(gf) if acc else [torch.empty_like(t) for t in f]
    m = sd["mask_e"] if zmode else [None] * 3
    flags = (1 if zmode else 0) | (2 if acc else 0) | (4 if gcb0 is not None else 0)
    q = _ptrs(f + g + [cb0 if zmode else None] + list(m) + gf + [gcb0])
    _check(yc.lib.lts3d_slab_q(yc.code, 1, f[0].shape[0], sd["n"], yc.shape[1], yc.shape[2], yc.gd,
                               q, float(a), flags, _stream(f[0])), "LTS slab E rates transpose")
    return gf


def _cb0_slab_fn(lo, hi):
    def fn(t, c):          # dt/eps = 2 cb / (1 + ca) on the slab planes lo..hi
        ca, cb = c.fd_ca[lo:hi + 1], c.fd_cb[lo:hi + 1]
        gcb, gca = torch.zeros_like(c.fd_cb), torch.zeros_like(c.fd_ca)
        gcb[lo:hi + 1] = t[0] * 2.0 / (1.0 + ca)
        gca[lo:hi + 1] = -t[0] * 2.0 * cb / (1.0 + ca) ** 2
        return [("fd_cb", gcb), ("fd_ca", gca)]
    return fn


def _cb0_dg_fn(t, c):      # dt/eps = 2 cb / (1 + ca) per element, t per shot (S, K)
    g = t.sum(0)
    ca, cb = c.dg_ca.reshape(-1), c.dg_cb.reshape(-1)
    return [("dg_cb", g * 2.0 / (1.0 + ca)), ("dg_ca", -g * 2.0 * cb / (1.0 + ca) ** 2)]


def _slab_planes(g, slabs, arrs):
    """Interface rows of slab arrays ``arrs[i] = (x, y, z)`` (S, n_i, NY, NZ)."""
    return [(a[1], a[2], (sl.hi - sl.lo + 1) * g.NYZ, (sl.cp.row - sl.lo) * g.NYZ, 0)
            for a, sl in zip(arrs, slabs)]


def _slab_ghost_planes(g, slabs, arrs):
    """Ghost (a) and inner (b) rows of the slab H-like arrays (lattice 0: z, 1: y)."""
    return [(a[2], a[1], (sl.hi - sl.lo + 1) * g.NYZ, (sl.cp.ghost - sl.lo) * g.NYZ,
             (sl.cp.inner - sl.lo) * g.NYZ) for a, sl in zip(arrs, slabs)]


def _damp(g, dc, pre, kind, F_old, F_cons, out, kd):
    """Split penalty damping (:func:`~DeepGPR.hybrid.functional3d.lts_damp`) into ``out``:
    ``m = (F_old + F_cons)/2``, ``kd`` explicit penalty sub-steps as operator epilogues,
    ``out = m + (F_cons - F_old)/2``."""
    m = torch.empty_like(F_old)
    combine(g.k, 0, 0.5, F_old, F_cons, None, m)
    coefs = pre.pen_h if kind == "h" else pre.pen_e
    cb0 = dc.ch if kind == "h" else pre.cb0_dg
    a = (-1.0 if kind == "h" else 1.0) / kd
    for _ in range(kd):
        mn = torch.empty_like(m)
        dc.terms_fwd(pre.zero, m, None, *coefs,
                     epi=(2, [None, None, m, None, cb0, mn, None], 0, 0.0, 0.0, 0.0, a))
        m = mn
    combine(g.k, 1, 0.5, m, F_cons, F_old, out)


def lts_step_into(st, c, u, out, src=None, amp=None, rec=None, rec_out=None, pre=None):
    """One sub-cycled step ``u -> out`` (as :func:`step_into`)."""
    y, d = st.yee, st.dg
    yc, dc = y.cuda, d.cuda
    g = coupling_glue(st)
    pre = pre if pre is not None else LTSPre(st, c)
    S = st.nshot
    p = int(d.lts)
    lc = d.lts_coef if d.lts_coef is not None else fn3.lts_coefficients(p)
    w, c0, ck, beta = lc["w"], lc["c0"], lc["c"], lc["beta"]
    inv_p = 1.0 / p
    slabs = pre.slabs
    ns = len(slabs)
    sds = [yc.slab_data(sl) for sl in slabs]
    split = bool(getattr(d, "lts_split", False))
    kd = max(1, int(getattr(d, "lts_pen_k", 1)))
    fe, fh, pse, psh, E, H, Daux, Baux, Dfd, Bfd = _unpack(u)
    ofe, ofh, opse, opsh, oE, oH, oD, oB, oDfd, oBfd = _unpack(out)
    eb, hb = (c.e0, c.e1, c.e2, c.e3), (c.h0, c.h1, c.h2, c.h3)
    kwE = dict(dtype=E.dtype, device=E.device)
    # ---------------------------------------------------------------- Yee H (plain macro step)
    yc.h_fwd(fe, fh, Bfd, psh, out=(ofh, oBfd, opsh))
    # ---------------------------------------------------------------- fine-set recursion
    zs = [[fe[k][:, sl.lo:sl.hi + 1].contiguous() for k in range(3)] for sl in slabs]
    Ex0 = torch.empty((S, 2, g.nX), **kwE)
    Xh = torch.empty_like(Ex0) if c.g_if is not None else None
    g.trace(_slab_planes(g, slabs, zs), Ex0, Xh, c.g_if)
    eta, cav = torch.empty_like(E), torch.empty_like(E)
    dc.terms_fwd(E, H, Ex0, *eb, epi=(1, [None, None, None, dc.ch, None, eta, cav], 1, 0.0,
                                      c0 * inv_p, w[0], 0.0))
    r0, eta_s, ravg, corr = [], [], [], [None] * ns
    for i in range(ns):
        r_i, e_i, a_i, _ = _sr_fwd(yc, sds[i], zs[i], None, None, True, False, 0.0, c0 * inv_p,
                                   w[0], None)
        r0.append(r_i)
        eta_s.append(e_i)
        ravg.append(a_i)
    zdg = E
    for m in range(1, p):
        g.ghost(_slab_ghost_planes(g, slabs, eta_s), eta)
        zn = torch.empty_like(E)
        dc.terms_fwd(eta, pre.zero, None, *hb,
                     epi=(2, [None, None, zdg, None, pre.cb0_dg, zn, None], 0, 0.0, 0.0, 0.0, inv_p))
        zdg = zn
        for i in range(ns):
            zs[i] = _slab_q_fwd(yc, sds[i], eta_s[i], zs[i], pre.cb0_s[i], inv_p)
        Exm = torch.empty_like(Ex0)
        g.trace(_slab_planes(g, slabs, zs), Exm)
        o1, o2 = torch.empty_like(E), torch.empty_like(E)
        dc.terms_fwd(zdg, H, Exm, *eb, epi=(1, [eta, cav, None, dc.ch, None, o1, o2], 0, beta[m],
                                            ck[m] * inv_p, w[m], 0.0))
        eta, cav = o1, o2
        last = m == p - 1
        for i, sl in enumerate(slabs):
            # last sub-step: the correction of the fine H planes is added in the kernel
            into = ([ofh[k][:, sl.lo:sl.hi + 1] for k in range(3)] if (last and S == 1) else None)
            _, eta_s[i], ravg[i], corr[i] = _sr_fwd(
                yc, sds[i], zs[i], eta_s[i], ravg[i], False, last, beta[m], ck[m] * inv_p, w[m],
                r0[i] if last else None, into)
    # ---------------------------------------------------------------- DG H
    uf = d.upml_fine or {}
    if split:
        Hc = torch.empty_like(H)
        _ltsupd_into(dc, 0, p, cav, H, Baux, None, None, uf.get("h_hist"), uf.get("h_new"),
                     uf.get("h_old"), Hc, oB)
        _damp(g, dc, pre, "h", H, Hc, oH, kd)
    else:
        _ltsupd_into(dc, 0, p, cav, H, Baux, None, None, uf.get("h_hist"), uf.get("h_new"),
                     uf.get("h_old"), oH, oB)
    # ---------------------------------------------------------------- fine Yee H planes
    for i, sl in enumerate(slabs):
        if corr[i] is not None:            # several shots: correction returned, added here
            for k in range(3):
                ofh[k].narrow(1, sl.lo, sl.hi - sl.lo + 1).add_(corr[i][k])
        for j, (k, pos, idx_full, local, fac) in enumerate(sl.upml_h):
            _fold_scatter(yc, p, -1.0, sds[i]["h"][j], fh[k].reshape(S, -1), Bfd,
                          ravg[i][k].reshape(S, -1),
                          (fac["a1"], fac["b1"], fac["hist"], fac["new"], fac["old"]),
                          ofh[k].view(S, -1), oBfd)
    g.ghost(g.ghost_planes(ofh[2], ofh[1]), oH)
    # ---------------------------------------------------------------- Yee E
    pen = _z(fe[0])
    if c.g_if is not None:
        pen = torch.empty((S, g.ncp, 2, g.NY, g.NZ), **kwE)
        g.penalty(g.fe_planes(fe[1], fe[2]), E, c.g_if, pen)
    fu = _fu(c, fe[0])
    yc.e_fwd(ofh, fe, Dfd, c.fd_ca, c.fd_cb, fu[0], fu[1], fu[2], pen, pse,
             out=(ofe, oDfd, opse))
    if y.upml is not None:
        for i, sl in enumerate(slabs):
            if not sl.upml_e:
                continue
            rhs = _slab_q_fwd(yc, sds[i], [ofh[k][:, sl.lo:sl.hi + 1] for k in range(3)], None,
                              None, 0.0)
            if pen.numel():
                for comp in (1, 2):
                    rhs[comp].index_add_(1, sl.row, pen[:, i, comp - 1][:, None])
            for j, (k, pos, idx_full, local, fac) in enumerate(sl.upml_e):
                hist, nw, old = pre.fu_f[i][j]
                _fold_scatter(yc, p, 1.0, sds[i]["e"][j], fe[k].reshape(S, -1), Dfd,
                              rhs[k].reshape(S, -1), (fac["a1"], fac["b1"], hist, nw, old),
                              ofe[k].view(S, -1), oDfd)
    # ---------------------------------------------------------------- DG E
    T = dc.terms_fwd(oH, E, Xh, *hb)
    ca_f, cb_f = c.dg_ca.reshape(-1), c.dg_cb.reshape(-1)
    if split:
        Ec = torch.empty_like(E)
        _ltsupd_into(dc, 1, p, T, E, Daux, ca_f, cb_f, c.u_hist_f, c.u_new_f, c.u_old_f, Ec, oD)
        _damp(g, dc, pre, "e", E, Ec, oE, kd)
    else:
        _ltsupd_into(dc, 1, p, T, E, Daux, ca_f, cb_f, c.u_hist_f, c.u_new_f, c.u_old_f, oE, oD)
    del T
    # ---------------------------------------------------------------- sources / receivers
    if src is not None:
        point_glue(st, src).inject(ofe[src.comp], oE, c.fd_cb, c.dg_cb, amp)
    if rec is not None and rec_out is not None:
        point_glue(st, rec).sample(ofe[rec.comp], oE, rec_out)


def lts_vjp_into(st, c, u, u_next, lam, out, src=None, amp=None, gd=None, rec=None,
                 acc: Optional[Accumulators] = None, need_c: bool = True, need_w: bool = False,
                 gamp=None, pre=None):
    """Transpose of :func:`lts_step_into` (contract of :func:`vjp_into`; ``lam`` destroyed).
    Coefficient cotangents accumulate in the kernels (face coefficients, ``dt/eps``); the
    linear chain rule of ``dt/eps = 2 cb / (1 + ca)`` is applied once, at finalize."""
    y, d = st.yee, st.dg
    yc, dc = y.cuda, d.cuda
    g = coupling_glue(st)
    pre = pre if pre is not None else LTSPre(st, c)
    S = st.nshot
    p = int(d.lts)
    lc = d.lts_coef if d.lts_coef is not None else fn3.lts_coefficients(p)
    w, c0, ck, beta = lc["w"], lc["c0"], lc["c"], lc["beta"]
    inv_p = 1.0 / p
    slabs = pre.slabs
    ns = len(slabs)
    sds = [yc.slab_data(sl) for sl in slabs]
    fe, fh, pse, psh, E, H, Daux, Baux, Dfd, Bfd = _unpack(u)
    fh2, H1 = list(u_next[3:6]), u_next[19]
    lfe, lfh, lpse, lpsh, lE, lH, lD, lB, lDfd, lBfd = _unpack(lam)
    ofe, ofh, opse, opsh, oE, oH, oD, oB, oDfd, oBfd = _unpack(out)
    g_if = c.g_if
    eb, hb = (c.e0, c.e1, c.e2, c.e3), (c.h0, c.h1, c.h2, c.h3)
    split = bool(getattr(d, "lts_split", False))
    kd = max(1, int(getattr(d, "lts_pen_k", 1)))
    sdmp = 1.0 / kd
    zeroE = pre.zero
    cb0_dg, cb0_s = pre.cb0_dg, pre.cb0_s
    rows_l = [sl.cp.row - sl.lo for sl in slabs]
    kwE = dict(dtype=E.dtype, device=E.device)
    one = S == 1                           # slab views of the full arrays are contiguous
    face = (lambda names: acc.face(c, S, names)) if need_c else (lambda names: None)  # noqa: E731
    raw_dg = acc.raw("lts_cb0_dg", (S, d.K), E, _cb0_dg_fn) if need_c else None
    raw_s = ([acc.raw(f"lts_cb0_s:{i}", cb0_s[i].shape, E, _cb0_slab_fn(sl.lo, sl.hi))
              for i, sl in enumerate(slabs)] if need_c else [None] * ns)
    if rec is not None and gd is not None:
        point_glue(st, rec).sample_bwd(gd, lam[rec.comp], lE)
    # ================================================================ recompute
    zs = [[[fe[k][:, sl.lo:sl.hi + 1].contiguous() for k in range(3)]] for sl in slabs]
    Ex = [torch.empty((S, 2, g.nX), **kwE)]
    Xh = torch.empty_like(Ex[0]) if g_if is not None else None
    g.trace(_slab_planes(g, slabs, [zs[i][0] for i in range(ns)]), Ex[0], Xh, g_if)
    zdg = [E]
    eta = []
    o1, cav = torch.empty_like(E), torch.empty_like(E)
    dc.terms_fwd(E, H, Ex[0], *eb, epi=(1, [None, None, None, dc.ch, None, o1, cav], 1, 0.0,
                                        c0 * inv_p, w[0], 0.0))
    eta.append(o1)
    ravg = []
    etl = [[None] * p for _ in range(ns)]
    eta_s = [None] * ns
    for i in range(ns):
        _, e_i, a_i, _ = _sr_fwd(yc, sds[i], zs[i][0], None, None, True, False, 0.0, c0 * inv_p,
                                 w[0], None)
        eta_s[i] = e_i
        ravg.append(a_i)
    zcurl = [None] * p
    cav_last = cav
    for m in range(1, p):
        g.ghost(_slab_ghost_planes(g, slabs, eta_s), eta[m - 1])
        for i in range(ns):
            etl[i][m - 1] = eta_s[i]
        zn = torch.empty_like(E)
        cu = torch.empty_like(E) if need_c else None
        dc.terms_fwd(eta[m - 1], zeroE, None, *hb,
                     epi=(2, [None, None, zdg[m - 1], None, cb0_dg, zn, cu], 0, 0.0, 0.0, 0.0, inv_p))
        zdg.append(zn)
        zcurl[m] = cu
        for i in range(ns):
            zs[i].append(_slab_q_fwd(yc, sds[i], etl[i][m - 1], zs[i][m - 1], cb0_s[i], inv_p))
        Exm = torch.empty_like(Ex[0])
        g.trace(_slab_planes(g, slabs, [zs[i][m] for i in range(ns)]), Exm)
        Ex.append(Exm)
        o1, o2 = torch.empty_like(E), torch.empty_like(E)
        dc.terms_fwd(zdg[m], H, Ex[m], *eb, epi=(1, [eta[m - 1], cav_last, None, dc.ch, None, o1, o2],
                                                 0, beta[m], ck[m] * inv_p, w[m], 0.0))
        eta.append(o1)
        cav_last = o2
        last = m == p - 1
        for i in range(ns):        # (the correction output is not needed here)
            _, eta_s[i], ravg[i], _ = _sr_fwd(yc, sds[i], zs[i][m], etl[i][m - 1], ravg[i], False,
                                              last, beta[m], ck[m] * inv_p, w[m], None)
    uf = d.upml_fine or {}
    Hc, scratch_B = torch.empty_like(H), torch.empty_like(Baux)
    _ltsupd_into(dc, 0, p, cav_last, H, Baux, None, None, uf.get("h_hist"), uf.get("h_new"),
                 uf.get("h_old"), Hc, scratch_B)
    del scratch_B
    mH = []
    if split:
        m0 = torch.empty_like(H)
        combine(g.k, 0, 0.5, H, Hc, None, m0)
        mH = [m0]
        for _ in range(kd):
            mn = torch.empty_like(H)
            dc.terms_fwd(zeroE, mH[-1], None, *pre.pen_h,
                         epi=(2, [None, None, mH[-1], None, dc.ch, mn, None], 0, 0.0, 0.0, 0.0,
                              -sdmp))
            mH.append(mn)
    pen = _z(fe[0])
    if g_if is not None:
        pen = torch.empty((S, g.ncp, 2, g.NY, g.NZ), **kwE)
        g.penalty(g.fe_planes(fe[1], fe[2]), E, g_if, pen)
    T_e = dc.terms_fwd(H1, E, Xh, *hb)
    ca_f, cb_f = c.dg_ca.reshape(-1), c.dg_cb.reshape(-1)
    Ec, scratch_D = torch.empty_like(E), torch.empty_like(Daux)
    _ltsupd_into(dc, 1, p, T_e, E, Daux, ca_f, cb_f, c.u_hist_f, c.u_new_f, c.u_old_f, Ec,
                 scratch_D)
    del scratch_D
    mE, curlE = [], []
    if split:
        m0 = torch.empty_like(E)
        combine(g.k, 0, 0.5, E, Ec, None, m0)
        mE = [m0]
        for _ in range(kd):
            mn, cu = torch.empty_like(E), torch.empty_like(E)
            dc.terms_fwd(zeroE, mE[-1], None, *pre.pen_e,
                         epi=(2, [None, None, mE[-1], None, cb0_dg, mn, cu], 0, 0.0, 0.0, 0.0, sdmp))
            curlE.append(cu)
            mE.append(mn)
    rhs_s = [None] * ns
    if y.upml is not None:
        for i, sl in enumerate(slabs):
            if not sl.upml_e:
                continue
            r_ = _slab_q_fwd(yc, sds[i], [fh2[k][:, sl.lo:sl.hi + 1] for k in range(3)], None,
                             None, 0.0)
            if pen.numel():
                for comp in (1, 2):
                    r_[comp].index_add_(1, sl.row, pen[:, i, comp - 1][:, None])
            rhs_s[i] = r_
    # ================================================================ reverse
    # ---- 13 sources
    if src is not None and (need_w or need_c):
        if need_w:
            gamp.zero_()
        gcf = gcd = None
        if need_c:
            gcf = acc.buf("fd_cb:src", c.fd_cb.shape, c.fd_cb)
            gcd = acc.buf("dg_cb:src", c.dg_cb.shape, c.dg_cb)
        point_glue(st, src).inject_bwd(lam[src.comp], lE, c.fd_cb, c.dg_cb, amp,
                                       gamp if need_w else None, gcf, gcd)
    # ---- 13 damp_E: out = m_kd + (Ec - E)/2, m_0 = (E + Ec)/2  ->  lEc = (lE + lm_0)/2,
    #      lE_old = (lm_0 - lE)/2 (added to the update's cotangent below)
    lm = lE
    if split:
        for j in reversed(range(kd)):
            _, lm, *_ = dc.terms_bwd(
                zeroE, mE[j], None, *pre.pen_e, False,
                gepi=(2, [lm, None, None, cb0_dg, curlE[j] if need_c else None, raw_dg, None], 0.0,
                      0.0, 0.0, sdmp), addG=lm, acc_face=face((None, "ph1", None, "ph3")),
                acc_cb0=need_c)
        lEc = torch.empty_like(E)
        combine(g.k, 0, 0.5, lE, lm, None, lEc)
    else:
        lEc = lE
    # ---- 13 LTSUpdate_E
    want_u = need_c and c.u_hist_f is not None
    grate, lE_n, gca, gcb, gh, gnw, gold = _ltsupd_bwd_into(
        dc, 1, p, T_e, E, Daux, ca_f, cb_f, c.u_hist_f, c.u_new_f, c.u_old_f, lEc, lD, oD, need_c,
        want_u)
    if split:
        combine(g.k, 1, 0.5, lE_n, lm, lE, lE_n)          # + lE_old
    if need_c:
        acc.add("dg_ca:lts", gca)
        acc.add("dg_cb:lts", gcb)
        if gh is not None:
            acc.add("u_hist_f:lts", gh)
            acc.add("u_new_f:lts", gnw)
            acc.add("u_old_f:lts", gold)
    # ---- 12 T_e = Terms(H', E, X_h)
    lH1, lE_n, gXh, *_ = dc.terms_bwd(H1, E, grate, *hb, Xh is not None, addF=lH, addG=lE_n,
                                      acc_face=face(("h0", "h1", "h2", "h3")))
    del grate, T_e
    if need_c and gXh is not None:
        g.xh_gif(gXh, Ex[0], acc.buf("g_if:x", (g_if.numel(),), g_if))
    # ---- 11 fine-plane UPML (E): fold transposes (cotangents gathered from lfe / lDfd)
    g_pen_f = torch.zeros_like(pen) if (pen.numel() and y.upml is not None) else None
    efold = []
    if y.upml is not None:
        want_f = need_c and c.fu_hist_f is not None
        gfu = ([acc.buf(nm + ":fold", t.shape, t) for nm, t in
                (("fu_hist_f", c.fu_hist_f), ("fu_new_f", c.fu_new_f), ("fu_old_f", c.fu_old_f))]
               if want_f else None)
        for i, sl in enumerate(slabs):
            if not sl.upml_e:
                continue
            g_rhs = pre.g_rhs[i]           # zero except at the fold entries (rewritten each step)
            for j, (k, pos, idx_full, local, fac) in enumerate(sl.upml_e):
                idx = sds[i]["e"][j]
                n_e = int(idx_full.numel())
                cf, ca_ = torch.empty((S, n_e), **kwE), torch.empty((S, n_e), **kwE)
                gfac = [torch.empty(n_e, **kwE) for _ in range(3)] if want_f else None
                hist, nw, old = pre.fu_f[i][j]
                _fold_gather(yc, p, 1.0, idx, fe[k].reshape(S, -1), Dfd, rhs_s[i][k].reshape(S, -1),
                             (fac["a1"], fac["b1"], hist, nw, old), lfe[k].view(S, -1), lDfd, cf,
                             ca_, g_rhs[k].view(S, -1), gfac)
                efold.append((k, idx[0], idx[1], cf, ca_))
                if want_f:
                    for q_, gq in zip(gfu, gfac):
                        q_.index_add_(0, pos, gq)
            if g_pen_f is not None:
                for comp in (1, 2):
                    g_pen_f[:, i, comp - 1] += g_rhs[comp][:, rows_l[i]]
            fh_s = [fh2[k][:, sl.lo:sl.hi + 1] for k in range(3)]
            if one:
                _sq_bwd(yc, sds[i], fh_s, g_rhs, None, 0.0,
                        gf=[lfh[k][:, sl.lo:sl.hi + 1] for k in range(3)])
            else:
                gfh_s = _sq_bwd(yc, sds[i], fh_s, g_rhs, None, 0.0)
                for k in range(3):
                    lfh[k][:, sl.lo:sl.hi + 1] += gfh_s[k]
    # ---- 10 Yee E update
    fu = _fu(c, fe[0])
    coef = None
    if need_c:
        coef = (acc.buf("fd_ca:upd", tuple(c.fd_ca.shape), c.fd_ca),     # summed over the shots
                acc.buf("fd_cb:upd", tuple(c.fd_cb.shape), c.fd_cb), None, None, None)
        if c.fu_hist is not None:
            ntot = yc.side["e"]["ntot"]
            coef = coef[:2] + tuple(acc.buf(nm + ":upd", (ntot,), c.fd_ca)
                                    for nm in ("fu_hist", "fu_new", "fu_old"))
    res = yc.e_bwd(fh2, fe, Dfd, c.fd_ca, c.fd_cb, fu[0], fu[1], fu[2], pen, pse, lfe, lDfd, lpse,
                   out=(ofe, oDfd, opse), gsrc_acc=lfh, coef_acc=coef)
    gpen = res[9]
    for k, ifull, ipos, cf, ca_ in efold:   # fold cotangents of the old fe / Dfd (zero there now)
        g.k.call(13, [S, cf.shape[1], ofe[k][0].numel(), oDfd.shape[1]],
                 [ifull, ipos, cf, ca_, ofe[k], oDfd], [], cf, "glue scatter")
    # ---- 10 interface penalty rows
    if gpen is not None or g_pen_f is not None:
        g.penalty_bwd(g.fe_planes(fe[1], fe[2]), g.fe_planes(ofe[1], ofe[2]), E, g_if, gpen,
                      g_pen_f, lE_n, acc.buf("g_if:pen", (g_if.numel(),), g_if) if need_c else None)
    # ---- 9 ghost planes of fh'
    g.ghost_bwd(g.ghost_planes(lfh[2], lfh[1]), lH1)
    # ---- 8 fine-plane UPML (H) and the corrections
    g_ravg = pre.g_ravg                    # zero except at the fold entries (rewritten each step)
    hfold, g_corr = [], []
    for i, sl in enumerate(slabs):
        for j, (k, pos, idx_full, local, fac) in enumerate(sl.upml_h):
            idx = sds[i]["h"][j]
            n_e = int(idx_full.numel())
            cf, ca_ = torch.empty((S, n_e), **kwE), torch.empty((S, n_e), **kwE)
            _fold_gather(yc, p, -1.0, idx, fh[k].reshape(S, -1), Bfd, ravg[i][k].reshape(S, -1),
                         (fac["a1"], fac["b1"], fac["hist"], fac["new"], fac["old"]),
                         lfh[k].view(S, -1), lBfd, cf, ca_, g_ravg[i][k].view(S, -1), None)
            hfold.append((k, idx[0], idx[1], cf, ca_))
        g_corr.append([lfh[k][:, sl.lo:sl.hi + 1].contiguous() for k in range(3)])
    # ---- 7 damp_H (as damp_E)
    lm_h = lH1
    if split:
        for j in reversed(range(kd)):
            _, lm_h, *_ = dc.terms_bwd(
                zeroE, mH[j], None, *pre.pen_h, False,
                gepi=(1, [lm_h, None, dc.ch, None, None, None, None], 0.0, sdmp, 0.0, 0.0),
                addG=lm_h, acc_face=face((None, "pe1", None, None)))
        lHc = torch.empty_like(H)
        combine(g.k, 0, 0.5, lH1, lm_h, None, lHc)
    else:
        lHc = lH1
    # ---- 6 LTSUpdate_H
    lcav, lH_n, *_ = _ltsupd_bwd_into(dc, 0, p, cav_last, H, Baux, None, None, uf.get("h_hist"),
                                      uf.get("h_new"), uf.get("h_old"), lHc, lB, oB, False, False)
    if split:
        combine(g.k, 1, 0.5, lH_n, lm_h, lH1, lH_n)       # + lH_old
    # ---- 5 the recursion, m = p-1 .. 1
    leta = [None] * p
    lzdg = None
    lzs = [None] * ns
    leta_s = [None] * ns
    lravg = list(g_ravg)
    lr0 = [None] * ns
    for m in range(p - 1, 0, -1):
        last = m == p - 1
        letl = [None] * ns
        for i in range(ns):
            gz, gein, gav, gr0 = _sr_bwd(yc, sds[i], zs[i][m][0], False, last, last, beta[m],
                                         ck[m] * inv_p, w[m], None, None if last else leta_s[i],
                                         lravg[i], g_corr[i] if last else None, gz=lzs[i])
            lzs[i] = gz
            letl[i] = gein
            lravg[i] = gav
            if gr0 is not None:
                lr0[i] = gr0 if lr0[i] is None else [a + b for a, b in zip(lr0[i], gr0)]
        geta_in = torch.empty_like(E) if leta[m] is not None else None
        gF, gG, gX, *_ = dc.terms_bwd(
            zdg[m], H, None, *eb, True,
            gepi=(1, [leta[m], lcav, dc.ch, None, None, None, geta_in], beta[m], ck[m] * inv_p,
                  w[m], 0.0), acc_face=face(("e0", "e1", "e2", "e3")))
        lzdg = gF if lzdg is None else lzdg.add_(gF)
        lH_n.add_(gG)
        g.trace_bwd(_slab_planes(g, slabs, lzs), gX)
        for i in range(ns):
            letl[i] = _sq_bwd(yc, sds[i], etl[i][m - 1], lzs[i], cb0_s[i], inv_p, gf=letl[i],
                              gcb0=raw_s[i])
        gF, *_ = dc.terms_bwd(
            eta[m - 1], zeroE, None, *hb, False,
            gepi=(2, [lzdg, None, None, cb0_dg, zcurl[m], raw_dg, None], 0.0, 0.0, 0.0, inv_p),
            acc_face=face(("h0", "h1", "h2", "h3")), acc_cb0=need_c)
        le = gF if geta_in is None else gF.add_(geta_in)
        g.ghost_bwd(_slab_ghost_planes(g, slabs, letl), le)
        for i in range(ns):
            leta_s[i] = letl[i]
        leta[m - 1] = le if leta[m - 1] is None else leta[m - 1].add_(le)
    # ---- 4 SlabR(first)
    for i in range(ns):
        z0 = pre.zero_s[i]
        lzs[i], _, _, _ = _sr_bwd(yc, sds[i], zs[i][0][0], True, False, False, 0.0, c0 * inv_p,
                                  w[0], lr0[i] if lr0[i] is not None else [z0, z0, z0],
                                  leta_s[i], lravg[i], None, gz=lzs[i])
    # ---- 3 TermsEta(first)
    gF, gG, gX, *_ = dc.terms_bwd(
        E, H, None, *eb, True,
        gepi=(1, [leta[0], lcav, dc.ch, None, None, None, None], 0.0, c0 * inv_p, w[0], 0.0),
        acc_face=face(("e0", "e1", "e2", "e3")))
    torch.add(lE_n, gF, out=oE)
    if lzdg is not None:
        oE.add_(lzdg)
    torch.add(lH_n, gG, out=oH)
    # ---- 2 trace E*^0 (and the X_h transpose) into the slabs, then the slab views of fe
    g.trace_bwd(_slab_planes(g, slabs, lzs), gX, gXh, g_if)
    for i, sl in enumerate(slabs):
        for k in range(3):
            ofe[k][:, sl.lo:sl.hi + 1] += lzs[i][k]
    # ---- 1 Yee H update
    yc.h_bwd(fe, fh, Bfd, psh, lfh, lBfd, lpsh, out=(ofh, oBfd, opsh), gsrc_acc=ofe)
    for k, ifull, ipos, cf, ca_ in hfold:   # fold cotangents of the old fh / Bfd (zero there now)
        g.k.call(13, [S, cf.shape[1], ofh[k][0].numel(), oBfd.shape[1]],
                 [ifull, ipos, cf, ca_, ofh[k], oBfd], [], cf, "glue scatter")


__all__ = ["Accumulators", "LTSPre", "any_available", "available", "empty_state", "lts_available",
           "lts_step_into", "lts_vjp_into", "prepare", "step_any", "step_into", "vjp_any",
           "vjp_into"]
