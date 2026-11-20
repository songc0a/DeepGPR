r"""Fused subgridded step and its hand-written transpose with explicit outputs (CUDA).

:func:`step_into` advances the state of a :class:`~DeepGPR.subgrid.SubgridMaxwell3D`
(Yee main grid, :func:`DeepGPR.subgrid.hsg.step`) or of a
:class:`~DeepGPR.subgrid.hybrid.HybridSubgridMaxwell3D` (DGTD-FDTD main step,
:func:`DeepGPR.subgrid.hybrid.step`) by one main step and writes ``u^(n+1)`` into given
buffers; :func:`vjp_into` applies the transpose of that step and writes the cotangent
of ``u^n`` into given buffers. The operations are the CUDA kernels of the forward step:
the fused Yee updates of every grid (:mod:`DeepGPR.hybrid.cuda3d`), the main step of
:mod:`DeepGPR.hybrid.fused3d` (split into its H and E halves) for the hybrid, the
precursor and surface-correction kernels (:mod:`.cuda`) and the point kernels of
:mod:`DeepGPR.hybrid.glue3d`, each with its exact transpose. The arithmetic is that of
:func:`~DeepGPR.subgrid.hsg.step`, whose reverse-mode derivative is the reference.

Transpose of one main step (``r`` sub-steps per subgrid, ``h = (r - 1) / 2``)::

    receivers^T, then per subgrid: OS E^T, (sources, IS E, E update)^T of the hsg_1
    sub-steps with their H updates and IS H corrections, then pe_next^T;
    main E half^T (sources, Yee/DG E);
    per subgrid: OS H^T, subgrid receivers^T, the hsg_2 sub-steps^T; then ph_cur^T;
    main H half^T; pe_cur^T and ph_prev^T into the cotangent of u^n.

The E updates' transposes need their inputs for the coefficient cotangents, so the
reverse step first recomputes the ``r - 1`` intermediate subgrid states of the step
(fields and CPML auxiliaries, from ``u^n`` and the precursors) into scratch buffers;
the main grid needs only ``u^n`` and ``u^(n+1)``. The precursors of the new main fields
(``ph_cur``, ``pe_next``) are taken from ``u^(n+1)``: the precursor stencils never
reach a node changed by an outer-surface correction (:func:`available` checks this
once per static), so the values are those of the forward step.

Fine receivers are sampled directly into an ``(S, n)`` output (the receiver buffer of
the state is not written on this path). :func:`vjp_into` destroys its input cotangent.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch

from ..hybrid import functional3d as fn3
from ..hybrid import fused3d
from ..hybrid.glue3d import PointGlue, point_glue
from . import hsg

_SHOTS = lambda t: t.sum(0)  # noqa: E731
SUB = hsg.SUB_STATE


def _hybrid(st) -> bool:
    return hasattr(st, "hyb")


def _base(st) -> int:
    return fn3.STATE_SIZE if _hybrid(st) else hsg.MAIN_STATE


def _main_yee(st):
    return st.hyb.yee if _hybrid(st) else st.main


def _z(ref):
    return torch.zeros(0, dtype=ref.dtype, device=ref.device)


def _taps_clear(st) -> bool:
    """``True`` when no precursor stencil reads a main node that an outer-surface
    correction changes (then the precursors of the corrected and of the uncorrected
    fields agree)."""
    key = "_fused_taps_clear"
    if key in st.__dict__:
        return st.__dict__[key]
    ok = True
    for kind in ("e", "h"):
        targets = {c: [] for c in range(3)}
        for sg in st.subs:
            corr = sg.os_e if kind == "e" else sg.os_h
            for c, idx in zip(corr.comp, corr.idx):
                targets[c].append(idx.cpu().numpy())
        tg = {c: (np.concatenate(v) if v else np.zeros(0, np.int64)) for c, v in targets.items()}
        for sg in st.subs:
            ops, comps = (sg.pe, sg.pe_comp) if kind == "e" else (sg.ph, sg.ph_comp)
            for op, c in zip(ops, comps):
                if not op.n:
                    continue
                idx = op.idx.view(op.n, op.T).cpu().numpy()
                used = idx[op.w.cpu().numpy() != 0]
                if np.isin(used, tg[c]).any():
                    ok = False
    st.__dict__[key] = ok
    return ok


def available(st) -> bool:
    """The fused step applies: CUDA, the Yee kernels attached to every grid, the
    coupling kernels to every subgrid, a fused global main step for the hybrid (no
    local time stepping), and precursor stencils clear of the outer surfaces."""
    if torch.device(st.device).type != "cuda":
        return False
    y = _main_yee(st)
    if y is None or y.cuda is None:
        return False
    if _hybrid(st):
        if not fused3d.available(st.hyb):
            return False
    elif y.upml is not None:
        return False
    for sg in st.subs:
        if sg.cuda is None or sg.yee.cuda is None or sg.yee.upml is not None:
            return False
    return _taps_clear(st)


def _ref_state(st):
    if _hybrid(st):
        from . import hybrid as hyb_mod
        return hyb_mod.zero_state(st)
    return hsg.zero_state(st)


def empty_state(st, like=None) -> List[torch.Tensor]:
    """Uninitialised state buffers (shapes of the solver's ``zero_state`` without
    subgrid receivers)."""
    ref = like if like is not None else _ref_state(st)
    return [torch.empty_like(t) for t in ref]


def state_bytes(st) -> int:
    return int(sum(t.numel() * t.element_size() for t in _ref_state(st)))


def _parts(t, base, g):
    b = base + SUB * g
    return SimpleNamespace(fe=list(t[b:b + 3]), fh=list(t[b + 3:b + 6]), pse=list(t[b + 6:b + 12]),
                           psh=list(t[b + 12:b + 18]), pe=t[b + 18], rec=t[b + 19])


class Accumulators(fused3d.Accumulators):
    """Coefficient cotangents of a subgridded reverse sweep (keys ``main_ca``/``main_cb``
    or the hybrid's names, ``sub_ca{g}``/``sub_cb{g}``)."""

    def finalize_list(self, c) -> List[torch.Tensor]:
        """Cotangents in the order of ``c.tensors()``."""
        hyb = hasattr(c, "hyb")
        named: Dict[str, torch.Tensor] = {}

        def ref_of(name):
            if name.startswith("sub_ca"):
                return c.sub_ca[int(name[6:])]
            if name.startswith("sub_cb"):
                return c.sub_cb[int(name[6:])]
            return getattr(c.hyb if hyb else c, name)

        def put(name, v):
            v = v.reshape(ref_of(name).shape)
            named[name] = v.clone() if name not in named else named[name] + v

        for key, (t, convert) in self.items.items():
            put(key.split(":")[0], convert(t) if convert is not None else t)
        for t, fn in self.deferred.values():
            for name, v in fn(t, c.hyb if hyb else c):
                put(name, v)
        if hyb:
            names = [nm for nm in c.hyb.NAMES if getattr(c.hyb, nm) is not None]
        else:
            names = ["main_ca", "main_cb"]
        n = len(c.sub_ca)
        names = names + [f"sub_ca{g}" for g in range(n)] + [f"sub_cb{g}" for g in range(n)]
        return [named[nm] if nm in named else torch.zeros_like(t)
                for nm, t in zip(names, c.tensors())]


# --------------------------------------------------------------------------- scratch
class Pre:
    """Scratch of the fused step: precursor vectors and their cotangents, and per
    subgrid ``r - 1`` intermediate sub-step states (the forward step's ping-pong
    buffer, the reverse step's recomputed intermediates)."""

    def __init__(self, st):
        S = st.nshot
        kw = dict(dtype=st.dtype, device=torch.device(st.device))
        ref = _ref_state(st)
        base = _base(st)
        self.ph_prev, self.ph_cur, self.pe_next = [], [], []
        self.g_ph_prev, self.g_ph_cur, self.g_pe_next = [], [], []
        self.store = []
        for g, sg in enumerate(st.subs):
            n_pe, n_ph = sg.cuda.pe.n, sg.cuda.ph.n
            self.ph_prev.append(torch.zeros((S, n_ph), **kw))
            self.ph_cur.append(torch.zeros((S, n_ph), **kw))
            self.pe_next.append(torch.zeros((S, n_pe), **kw))
            self.g_ph_prev.append(torch.zeros((S, n_ph), **kw))
            self.g_ph_cur.append(torch.zeros((S, n_ph), **kw))
            self.g_pe_next.append(torch.zeros((S, n_pe), **kw))
            p = _parts(ref, base, g)
            self.store.append([SimpleNamespace(fe=[torch.zeros_like(t) for t in p.fe],
                                               fh=[torch.zeros_like(t) for t in p.fh],
                                               pse=[torch.zeros_like(t) for t in p.pse],
                                               psh=[torch.zeros_like(t) for t in p.psh])
                               for _ in range(max(1, sg.time_ratio - 1))])
        self.rec_tmp = None


def prepare(st, c=None, pre: Optional[Pre] = None) -> Pre:
    """Scratch buffers of the fused step (coefficient independent; ``pre`` reused)."""
    return pre if pre is not None else Pre(st)


def _glue(yee, st, pts: hsg.Points, count: int, comp: int) -> Optional[PointGlue]:
    """Point kernels of ``pts`` on the Yee grid ``yee`` (cached on the points)."""
    if pts is None or not pts.shot.numel():
        return None
    cache = pts.__dict__.setdefault("_glue", {})
    key = (id(yee), int(count))
    g = cache.get(key)
    if g is None:
        dev = torch.device(st.device)
        zi = torch.zeros(0, dtype=torch.long, device=dev)
        ops = SimpleNamespace(fd_shot=pts.shot, fd_node=pts.node, fd_item=pts.item, fd_w=pts.w,
                              dg_shot=zi, dg_elem=zi, dg_item=zi,
                              dg_w=torch.zeros((0, 1), dtype=st.dtype, device=dev),
                              count=int(count), comp=int(comp))
        ns = SimpleNamespace(yee=yee, dg=None, nshot=st.nshot, device=dev, dtype=st.dtype)
        g = cache[key] = PointGlue(yee.cuda.lib, ns, ops)
    return g


# --------------------------------------------------------------------------- sub-steps
class _Cur:
    """Current subgrid fields of a sub-step sequence (and the update counts)."""

    def __init__(self, p):
        self.fe, self.pse, self.fh, self.psh = p.fe, p.pse, p.fh, p.psh
        self.ne = self.nh = 0


def _h_upd(sg, cur, dst):
    z = _z(cur.fe[0])
    sg.yee.cuda.h_fwd(cur.fe, cur.fh, z, cur.psh, out=(dst.fh, z, dst.psh))
    cur.fh, cur.psh = dst.fh, dst.psh
    cur.nh += 1


def _e_upd(sg, ca, cb, cur, dst):
    z = _z(cur.fe[0])
    sg.yee.cuda.e_fwd(cur.fh, cur.fe, z, ca, cb, z, z, z, z, cur.pse, out=(dst.fe, z, dst.pse))
    cur.fe, cur.pse = dst.fe, dst.pse
    cur.ne += 1


def _src(pg, cur, comp, cb, ampf, j):
    if pg is not None and ampf is not None:
        pg.inject(cur.fe[comp], _z(cb), cb, None, ampf[j])


def _first_half(sg, ca, cb, cur, pv, dst_h, dst_e, pg, comp, ampf):
    """hsg_2 sub-steps of one subgrid (no main-grid operations)."""
    r = sg.time_ratio
    h = (r - 1) // 2
    for k in range(1, h + 1):
        _h_upd(sg, cur, dst_h(cur.nh))
        w = (k + h) / r
        sg.cuda.is_h.fwd(cur.fh, 1.0 - w, w, pv.pe_prev, pv.pe_cur)
        _e_upd(sg, ca, cb, cur, dst_e(cur.ne))
        sg.cuda.is_e.fwd(cur.fe, 1.0 - k / r, k / r, pv.ph_prev, pv.ph_cur, None, cb)
        _src(pg, cur, comp, cb, ampf, k - 1)
    _h_upd(sg, cur, dst_h(cur.nh))
    sg.cuda.is_h.fwd(cur.fh, 1.0, 0.0, pv.pe_cur, None)


def _second_half(sg, ca, cb, cur, pv, dst_h, dst_e, pg, comp, ampf, recompute=False):
    """hsg_1 sub-steps of one subgrid; ``recompute``: stop after the last stored
    intermediate (the last H and E updates produce ``u^(n+1)``)."""
    r = sg.time_ratio
    h = (r - 1) // 2
    for k in range(1, h + 1):
        _e_upd(sg, ca, cb, cur, dst_e(cur.ne))
        w = (k + h) / r
        sg.cuda.is_e.fwd(cur.fe, 1.0 - w, w, pv.ph_prev, pv.ph_cur, None, cb)
        _src(pg, cur, comp, cb, ampf, h + k - 1)
        if recompute and cur.nh == r - 1:
            return
        _h_upd(sg, cur, dst_h(cur.nh))
        sg.cuda.is_h.fwd(cur.fh, 1.0 - k / r, k / r, pv.pe_cur, pv.pe_next)
    if recompute:
        return
    _e_upd(sg, ca, cb, cur, dst_e(cur.ne))
    sg.cuda.is_e.fwd(cur.fe, 1.0, 0.0, pv.ph_cur, None, None, cb)
    _src(pg, cur, comp, cb, ampf, r - 1)


def _coefs(st, c):
    """``(main_ca, main_cb, [sub_ca], [sub_cb])`` of either solver."""
    if _hybrid(st):
        return c.hyb.fd_ca, c.hyb.fd_cb, c.sub_ca, c.sub_cb
    return c.main_ca, c.main_cb, c.sub_ca, c.sub_cb


def _sub_points(st, pset, g):
    if pset is None or _hybrid(st):
        return None
    return _glue(st.subs[g].yee, st, pset.sub[g], pset.count, pset.comp)


def _main_points(st, pset):
    if pset is None:
        return None
    if _hybrid(st):
        return point_glue(st.hyb, pset.hybrid)
    return _glue(st.main, st, pset.main, pset.count, pset.comp)


def _pv(st, g, u, u_next, pre, base):
    """Precursor vectors of subgrid ``g``: ``pe_prev`` (state), ``pe_cur`` (the new
    state's slot), ``pe_next``/``ph_prev``/``ph_cur`` (scratch)."""
    return SimpleNamespace(pe_prev=u[base + SUB * g + 18], pe_cur=u_next[base + SUB * g + 18],
                           pe_next=pre.pe_next[g], ph_prev=pre.ph_prev[g], ph_cur=pre.ph_cur[g])


# --------------------------------------------------------------------------- forward
def step_into(st, c, u, out, src=None, amp=None, ampf=None, rec=None, rec_m=None, rec_f=None,
              pre: Optional[Pre] = None):
    """One main step ``u -> out`` (``out`` written, ``u`` unchanged). ``amp`` (nsrc,):
    main-rate source samples; ``ampf``: per subgrid ``None`` or the subgrid-rate samples
    ``(r, nsrc)`` of the step; receivers: main grid (state ``u^(n+1)``) into ``rec_m``,
    subgrid receivers (recorded within the step) into ``rec_f`` (both (S, n))."""
    hyb = _hybrid(st)
    base = _base(st)
    pre = prepare(st, c, pre)
    z = _z(u[0])
    fe, fh = list(u[0:3]), list(u[3:6])
    ofe, ofh = list(out[0:3]), list(out[3:6])
    main_ca, main_cb, sub_ca, sub_cb = _coefs(st, c)
    subs = st.subs
    for g, sg in enumerate(subs):
        sg.cuda.pe.fwd(fe, out=out[base + SUB * g + 18])          # pe_cur
        sg.cuda.ph.fwd(fh, out=pre.ph_prev[g])
    # ------------------------------------------------------------------ main H half
    if hyb:
        hctx = fused3d.step_h_part(st.hyb, c.hyb, u[:base], out[:base])
    else:
        st.main.cuda.h_fwd(fe, fh, z, list(u[12:18]), out=(ofh, z, list(out[12:18])))
    for g, sg in enumerate(subs):
        sg.cuda.ph.fwd(ofh, out=pre.ph_cur[g])
    # ------------------------------------------------------------------ subgrids, hsg_2
    curs = []
    nrf = sum(1 for g in range(len(subs)) if _sub_points(st, rec, g) is not None)
    rf_first = True
    for g, sg in enumerate(subs):
        pin, po = _parts(u, base, g), _parts(out, base, g)
        tmp = pre.store[g][0]
        dst = lambda i, po=po, tmp=tmp: po if i % 2 == 0 else tmp  # noqa: E731
        cur = _Cur(pin)
        pv = _pv(st, g, u, out, pre, base)
        pv.pe_next = None
        pg = _sub_points(st, src, g)
        af = ampf[g] if (ampf is not None and pg is not None) else None
        _first_half(sg, sub_ca[g], sub_cb[g], cur, pv, dst, dst, pg,
                    src.comp if src is not None else 0, af)
        rg = _sub_points(st, rec, g)
        if rg is not None and rec_f is not None:
            if nrf == 1:
                rg.sample(cur.fe[rec.comp], z, rec_f)
            else:
                if pre.rec_tmp is None or pre.rec_tmp.shape != rec_f.shape:
                    pre.rec_tmp = torch.empty_like(rec_f)
                if rf_first:
                    rg.sample(cur.fe[rec.comp], z, rec_f)
                else:
                    rg.sample(cur.fe[rec.comp], z, pre.rec_tmp)
                    rec_f.add_(pre.rec_tmp)
            rf_first = False
        sg.cuda.os_h.fwd(ofh, 1.0, 0.0, None, None, cur.fe)
        curs.append(cur)
    if rec_f is not None and nrf == 0:
        rec_f.zero_()
    # ------------------------------------------------------------------ main E half
    if hyb:
        fused3d.step_e_part(st.hyb, c.hyb, u[:base], out[:base], hctx,
                            None if src is None else src.hybrid, amp)
    else:
        st.main.cuda.e_fwd(ofh, fe, z, main_ca, main_cb, z, z, z, z, list(u[6:12]),
                           out=(ofe, z, list(out[6:12])))
        pg = _main_points(st, src)
        if pg is not None and amp is not None:
            pg.inject(ofe[src.comp], z, main_cb, None, amp)
    for g, sg in enumerate(subs):
        sg.cuda.pe.fwd(ofe, out=pre.pe_next[g])
    # ------------------------------------------------------------------ subgrids, hsg_1
    for g, sg in enumerate(subs):
        po = _parts(out, base, g)
        tmp = pre.store[g][0]
        dst = lambda i, po=po, tmp=tmp: po if i % 2 == 0 else tmp  # noqa: E731
        pv = _pv(st, g, u, out, pre, base)
        pg = _sub_points(st, src, g)
        af = ampf[g] if (ampf is not None and pg is not None) else None
        _second_half(sg, sub_ca[g], sub_cb[g], curs[g], pv, dst, dst, pg,
                     src.comp if src is not None else 0, af)
        sg.cuda.os_e.fwd(ofe, 1.0, 0.0, None, None, curs[g].fh, main_cb)
    # ------------------------------------------------------------------ receivers
    if rec is not None and rec_m is not None:
        rg = _main_points(st, rec)
        if rg is not None:
            rg.sample(ofe[rec.comp], out[18] if hyb else z, rec_m)
        else:
            rec_m.zero_()


# --------------------------------------------------------------------------- transpose
class _LCur:
    """Current cotangents of one subgrid in the reverse step (E chain: ``(fe, pse)``,
    H chain: ``(fh, psh)``), alternating between the cotangent buffers ``lam`` (input,
    destroyed) and ``out`` so that the ``r``-th transpose of each chain lands in ``out``."""

    def __init__(self, lam_p, out_p, r):
        self.lam, self.out, self.r = lam_p, out_p, r
        self.e = (lam_p.fe, lam_p.pse)
        self.h = (lam_p.fh, lam_p.psh)
        self.ne = self.nh = 0

    def next_e(self):
        self.ne += 1
        p = self.out if (self.r - self.ne) % 2 == 0 else self.lam
        return (p.fe, p.pse)

    def next_h(self):
        self.nh += 1
        p = self.out if (self.r - self.nh) % 2 == 0 else self.lam
        return (p.fh, p.psh)


class _SubBwd:
    """Transposes of the sub-steps of one subgrid (values from the recomputation)."""

    def __init__(self, st, c, g, u, u_next, lam, out, pre, base, acc, need_c, src, ampf,
                 need_wf, gampf):
        self.sg = sg = st.subs[g]
        self.r = sg.time_ratio
        self.h = (self.r - 1) // 2
        _, _, sub_ca, sub_cb = _coefs(st, c)
        self.ca, self.cb = sub_ca[g], sub_cb[g]
        self.pin = _parts(u, base, g)
        self.pnext = _parts(u_next, base, g)
        self.store = pre.store[g]
        self.pv = _pv(st, g, u, u_next, pre, base)
        self.gv = SimpleNamespace(pe_prev=out[base + SUB * g + 18], pe_cur=lam[base + SUB * g + 18],
                                  pe_next=pre.g_pe_next[g], ph_prev=pre.g_ph_prev[g],
                                  ph_cur=pre.g_ph_cur[g])
        self.lc = _LCur(_parts(lam, base, g), _parts(out, base, g), self.r)
        self.z = _z(u[0])
        self.coef = None
        self.gcb_is = self.gcb_src = None
        if need_c:
            shp = tuple(self.ca.shape)                         # summed over the shots
            self.coef = (acc.buf(f"sub_ca{g}:upd", shp, self.ca),
                         acc.buf(f"sub_cb{g}:upd", shp, self.cb), self.z, self.z, self.z)
            self.gcb_is = acc.buf(f"sub_cb{g}:is", (self.cb.numel(),), self.cb)
        self.pg = _sub_points(st, src, g)
        self.comp = src.comp if src is not None else 0
        self.ampf = ampf[g] if (ampf is not None and self.pg is not None) else None
        self.gampf = gampf[g] if (gampf is not None and self.pg is not None and need_wf) else None
        if need_c and self.pg is not None and self.ampf is not None:
            self.gcb_src = acc.buf(f"sub_cb{g}:src", (self.cb.numel(),), self.cb)

    # ---------------------------------------------------------------- values
    def e_in(self, i):
        """Inputs ``(fe, pse, fh)`` of the E update ``i`` of the step."""
        p = self.pin if i == 0 else self.store[i - 1]
        fh = self.pnext.fh if i == self.r - 1 else self.store[i].fh
        return p.fe, p.pse, fh

    def first_half_fe(self):
        """The subgrid E at the end of the hsg_2 half (read by the OS H correction)."""
        return self.pin.fe if self.h == 0 else self.store[self.h - 1].fe

    # ---------------------------------------------------------------- transposes
    def e_bwd(self, i):
        fe, pse, fh = self.e_in(i)
        lc, z = self.lc, self.z
        dst = lc.next_e()
        self.sg.yee.cuda.e_bwd(fh, fe, z, self.ca, self.cb, z, z, z, z, pse, lc.e[0], z, lc.e[1],
                               out=(dst[0], z, dst[1]), gsrc_acc=lc.h[0], coef_acc=self.coef)
        lc.e = dst

    def h_bwd(self):
        lc, z, p = self.lc, self.z, self.pin
        dst = lc.next_h()
        self.sg.yee.cuda.h_bwd(p.fe, p.fh, z, p.psh, lc.h[0], z, lc.h[1], out=(dst[0], z, dst[1]),
                               gsrc_acc=lc.e[0])
        lc.h = dst

    def is_e_bwd(self, a, b, V0, V1, g0, g1):
        self.sg.cuda.is_e.bwd(self.lc.e[0], a, b, V0, V1, None, self.cb, g0, g1, None, self.gcb_is)

    def is_h_bwd(self, a, b, V0, V1, g0, g1):
        self.sg.cuda.is_h.bwd(self.lc.h[0], a, b, V0, V1, None, None, g0, g1, None, None)

    def src_bwd(self, j):
        if self.pg is None or self.ampf is None or (self.gampf is None and self.gcb_src is None):
            return
        ga = None
        if self.gampf is not None:
            ga = self.gampf[j]
            ga.zero_()
        self.pg.inject_bwd(self.lc.e[0][self.comp], self.z, self.cb, None, self.ampf[j], ga,
                           self.gcb_src, None)

    # ---------------------------------------------------------------- halves
    def second_half(self):
        r, h, pv, gv = self.r, self.h, self.pv, self.gv
        self.src_bwd(r - 1)
        self.is_e_bwd(1.0, 0.0, pv.ph_cur, None, gv.ph_cur, None)
        self.e_bwd(r - 1)
        for k in range(h, 0, -1):
            self.is_h_bwd(1.0 - k / r, k / r, pv.pe_cur, pv.pe_next, gv.pe_cur, gv.pe_next)
            self.h_bwd()
            self.src_bwd(h + k - 1)
            w = (k + h) / r
            self.is_e_bwd(1.0 - w, w, pv.ph_prev, pv.ph_cur, gv.ph_prev, gv.ph_cur)
            self.e_bwd(h + k - 1)

    def first_half(self):
        r, h, pv, gv = self.r, self.h, self.pv, self.gv
        self.is_h_bwd(1.0, 0.0, pv.pe_cur, None, gv.pe_cur, None)
        self.h_bwd()
        for k in range(h, 0, -1):
            self.src_bwd(k - 1)
            self.is_e_bwd(1.0 - k / r, k / r, pv.ph_prev, pv.ph_cur, gv.ph_prev, gv.ph_cur)
            self.e_bwd(k - 1)
            w = (k + h) / r
            self.is_h_bwd(1.0 - w, w, pv.pe_prev, pv.pe_cur, gv.pe_prev, gv.pe_cur)
            self.h_bwd()


def _recompute(st, c, g, u, u_next, pre, base, src, ampf):
    """The ``r - 1`` intermediate states of subgrid ``g`` (into ``pre.store[g]``)."""
    sg = st.subs[g]
    if sg.time_ratio < 3:
        return
    _, _, sub_ca, sub_cb = _coefs(st, c)
    store = pre.store[g]
    dst = lambda i: store[i]  # noqa: E731
    cur = _Cur(_parts(u, base, g))
    pv = _pv(st, g, u, u_next, pre, base)
    pg = _sub_points(st, src, g)
    af = ampf[g] if (ampf is not None and pg is not None) else None
    comp = src.comp if src is not None else 0
    _first_half(sg, sub_ca[g], sub_cb[g], cur, pv, dst, dst, pg, comp, af)
    _second_half(sg, sub_ca[g], sub_cb[g], cur, pv, dst, dst, pg, comp, af, recompute=True)


def vjp_into(st, c, u, u_next, lam, out, src=None, amp=None, ampf=None, gd_m=None, gd_f=None,
             rec=None, acc: Optional[Accumulators] = None, need_c: bool = True,
             need_w: bool = False, gamp=None, need_wf: bool = False, gampf=None,
             pre: Optional[Pre] = None, ind=None):
    """Transpose of :func:`step_into`: ``out`` = cotangent of ``u^n`` given the cotangent
    ``lam`` of ``u^(n+1)`` (destroyed) and the receiver cotangents ``gd_m``/``gd_f``
    (S, n) of this step. Coefficient cotangents accumulate into ``acc`` (``need_c``); the
    source sample cotangents go to ``gamp`` (nsrc,) and, per subgrid, ``gampf`` (r, nsrc)
    (zeroed here). ``ind = (gE, gH)``: copies of the cotangents of the main E and H
    updates' outputs (the error indicator's weights)."""
    hyb = _hybrid(st)
    base = _base(st)
    pre = prepare(st, c, pre)
    z = _z(u[0])
    fe_n, fh_n = list(u[0:3]), list(u[3:6])
    fe2, fh2 = list(u_next[0:3]), list(u_next[3:6])
    lfe, lfh = list(lam[0:3]), list(lam[3:6])
    ofe, ofh = list(out[0:3]), list(out[3:6])
    main_ca, main_cb, _, _ = _coefs(st, c)
    subs = st.subs
    # ------------------------------------------------------------------ main receivers
    if rec is not None and gd_m is not None:
        rg = _main_points(st, rec)
        if rg is not None:
            rg.sample_bwd(gd_m, lfe[rec.comp], lam[18] if hyb else z)
    # ------------------------------------------------------------------ precursors
    for g, sg in enumerate(subs):
        sg.cuda.ph.fwd(fh_n, out=pre.ph_prev[g])
        sg.cuda.ph.fwd(fh2, out=pre.ph_cur[g])
        sg.cuda.pe.fwd(fe2, out=pre.pe_next[g])
        pre.g_ph_prev[g].zero_()
        pre.g_ph_cur[g].zero_()
        pre.g_pe_next[g].zero_()
        out[base + SUB * g + 18].zero_()
        if out[base + SUB * g + 19].numel():
            out[base + SUB * g + 19].zero_()
    for g in range(len(subs)):
        _recompute(st, c, g, u, u_next, pre, base, src, ampf)
    sb = [_SubBwd(st, c, g, u, u_next, lam, out, pre, base, acc, need_c, src, ampf, need_wf,
                  gampf) for g in range(len(subs))]
    gcb_os = acc.buf(("fd_cb" if hyb else "main_cb") + ":os", (main_cb.numel(),), main_cb) \
        if need_c else None
    # ------------------------------------------------------------------ subgrids, hsg_1^T
    for g, sg in enumerate(subs):
        b = sb[g]
        sg.cuda.os_e.bwd(lfe, 1.0, 0.0, None, None, b.pnext.fh, main_cb, None, None, b.lc.h[0],
                         gcb_os)
        b.second_half()
    for g, sg in enumerate(subs):
        sg.cuda.pe.bwd(pre.g_pe_next[g], lfe)
    if ind is not None:
        for a, t in zip(ind[0], lfe):
            a.copy_(t)
    # ------------------------------------------------------------------ main E half^T
    if hyb:
        hctx = fused3d.vjp_e_part(st.hyb, c.hyb, u[:base], u_next[:base], lam[:base], out[:base],
                                  None if src is None else src.hybrid, amp, None, None, acc, need_c,
                                  need_w, gamp)
    else:
        pg = _main_points(st, src)
        if need_w and gamp is not None:
            gamp.zero_()
        if pg is not None and amp is not None and (need_w or need_c):
            gcf = acc.buf("main_cb:src", (main_cb.numel(),), main_cb) if need_c else None
            pg.inject_bwd(lfe[src.comp], z, main_cb, None, amp, gamp if need_w else None, gcf, None)
        coef = None
        if need_c:
            shp = tuple(main_ca.shape)                             # summed over the shots
            coef = (acc.buf("main_ca:upd", shp, main_ca), acc.buf("main_cb:upd", shp, main_cb),
                    z, z, z)
        st.main.cuda.e_bwd(fh2, fe_n, z, main_ca, main_cb, z, z, z, z, list(u[6:12]), lfe, z,
                           list(lam[6:12]), out=(ofe, z, list(out[6:12])), gsrc_acc=lfh,
                           coef_acc=coef)
    # ------------------------------------------------------------------ subgrids, hsg_2^T
    for g, sg in enumerate(subs):
        b = sb[g]
        sg.cuda.os_h.bwd(lfh, 1.0, 0.0, None, None, b.first_half_fe(), None, None, None,
                         b.lc.e[0], None)
        rg = _sub_points(st, rec, g)
        if rg is not None and gd_f is not None:
            rg.sample_bwd(gd_f, b.lc.e[0][rec.comp], z)
        b.first_half()
    for g, sg in enumerate(subs):
        sg.cuda.ph.bwd(pre.g_ph_cur[g], lfh)
    if ind is not None:
        for a, t in zip(ind[1], lfh):
            a.copy_(t)
    # ------------------------------------------------------------------ main H half^T
    if hyb:
        fused3d.vjp_h_part(st.hyb, c.hyb, u[:base], u_next[:base], lam[:base], out[:base], acc,
                           need_c, hctx)
    else:
        st.main.cuda.h_bwd(fe_n, fh_n, z, list(u[12:18]), lfh, z, list(lam[12:18]),
                           out=(ofh, z, list(out[12:18])), gsrc_acc=ofe)
    for g, sg in enumerate(subs):
        sg.cuda.pe.bwd(lam[base + SUB * g + 18], ofe)          # pe_cur
        sg.cuda.ph.bwd(pre.g_ph_prev[g], ofh)


__all__ = ["Accumulators", "Pre", "available", "empty_state", "prepare", "state_bytes",
           "step_into", "vjp_into"]
