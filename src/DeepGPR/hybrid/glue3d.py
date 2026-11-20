"""Fused CUDA "glue" of the 3D hybrid step (``csrc/glue3d.cu``).

The DG and Yee updates of :mod:`.functional3d` already run as fused kernels
(:mod:`.cuda3d`); the operations between them were PyTorch gathers, scatters and
elementwise kernels: the mortar coupling (projected Yee trace ``E*`` of the DG
H-flux, the interface input ``X_h`` of the E-flux, the ghost planes of the Yee
H update, the interface penalty rows of the Yee E update), source injection and
receiver sampling, and the transposes of all of these in the hand-written
adjoint. :class:`CouplingGlue` and :class:`PointGlue` run each of them as one
or two kernel launches with the same arithmetic. Sparse operators are kept in
CSR form by output and by input, so that the kernels gather (a thread owns its
output) and use atomics only where several entries accumulate into one
coefficient or point (face penalty, source coefficients, shared nodes).

All functions operate in place on the given output tensors (contiguous) and
issue no host synchronisation, so they can be captured in CUDA Graphs.
"""

from __future__ import annotations

import ctypes
from typing import List, Optional, Sequence

import numpy as np
import torch

from .cuda3d import _check, _stream

_MAXCP = 4


def _csr(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, nrows: int):
    """CSR arrays (row pointer, columns, values) sorted by row (then column)."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    order = np.lexsort((cols, rows))
    rows, cols, vals = rows[order], cols[order], vals[order]
    rp = np.zeros(nrows + 1, dtype=np.int64)
    np.add.at(rp, rows + 1, 1)
    return np.cumsum(rp), cols, vals


class _Lib:
    """Kernel launcher: ``call(op, gi, ptrs, gd, ref)`` on the stream of ``ref``."""

    def __init__(self, lib, dtype):
        self.lib = lib
        self.code = {torch.float32: 0, torch.float64: 1}[dtype]

    def call(self, op: int, gi: Sequence[int], ptrs: Sequence, gd: Sequence[float], ref, what):
        gia = (ctypes.c_longlong * max(1, len(gi)))(*[int(v) for v in gi])
        pa = (ctypes.c_void_p * max(1, len(ptrs)))()
        for k, t in enumerate(ptrs):
            pa[k] = None if t is None else (t.data_ptr() if torch.is_tensor(t) else int(t))
        gda = (ctypes.c_double * max(1, len(gd)))(*[float(v) for v in gd])
        _check(self.lib.glue3d_call(self.code, op, gia, pa, gda, _stream(ref)), what)


def plane_args(items):
    """Kernel arguments of a plane set: ``items`` = per coupling ``(a0, a1, stride,
    off_a, off_b)`` (two arrays, per-shot stride, plane offsets in elements)."""
    if len(items) > _MAXCP:
        raise ValueError(f"at most {_MAXCP} couplings")
    gi, ptrs = [], []
    for a0, a1, stride, oa, ob in items:
        gi += [int(stride), int(oa), int(ob)]
        ptrs += [a0, a1]
    return gi, ptrs


class CouplingGlue:
    """Mortar coupling of a :class:`~DeepGPR.hybrid.functional3d.StepStatic3D` in
    the kernel layouts (built once, cached as ``st.glue``)."""

    def __init__(self, lib, st):
        y, d = st.yee, st.dg
        dev, dtype = torch.device(st.device), st.dtype
        self.k = _Lib(lib, dtype)
        self.S = st.nshot
        self.NX, self.NY, self.NZ = y.shape
        self.N = self.NX * self.NY * self.NZ
        self.NYZ = self.NY * self.NZ
        self.K, self.Np, self.Nfp = d.K, d.Np, d.Nfp
        self.KN = d.K * d.Np
        cps = st.couplings
        self.ncp = len(cps)
        if self.ncp > _MAXCP:
            raise ValueError("too many couplings for the fused glue")
        self.rows = [int(cp.row) for cp in cps]
        self.ghosts = [int(cp.ghost) for cp in cps]
        self.inners = [int(cp.inner) for cp in cps]
        nif = [int(cp.slots.numel()) for cp in cps]
        self.off = np.concatenate([[0], np.cumsum(nif)]).astype(np.int64)
        self.nX = int(self.off[-1])
        NYZ, nX = self.NYZ, self.nX
        np_ = lambda t: t.detach().cpu().numpy()  # noqa: E731
        slots = [np_(cp.slots).astype(np.int64) for cp in cps]
        nodes = [np_(cp.nodes).astype(np.int64) for cp in cps]
        tr, tc, tv = [], [], []           # trace, by slot: rows c nX + r, cols j
        ur, uc, uv = [], [], []           # trace transposed: rows (i 2 + c) NYZ + j, cols r
        gr, gn, gf, gv = [], [], [], []   # to_fd by plane node: rows (i 2 + l) NYZ + j
        hr, hc, hv = [], [], []           # to_fd by slot: rows l nX + r, cols j
        for i, cp in enumerate(cps):
            o = int(self.off[i])
            for c in (0, 1):
                op = cp.to_dg[c]
                r, j, v = np_(op.rows), np_(op.cols), np_(op.vals)
                tr.append(c * nX + o + r)
                tc.append(j)
                tv.append(v)
                ur.append((i * 2 + c) * NYZ + j)
                uc.append(o + r)
                uv.append(v)
            for lat in (0, 1):
                op = cp.to_fd[lat]
                j, e, v = np_(op.rows), np_(op.cols), np_(op.vals)
                gr.append((i * 2 + lat) * NYZ + j)
                gn.append(nodes[i][e])
                gf.append(slots[i][e] // self.Nfp)
                gv.append(v)
                hr.append(lat * nX + o + e)
                hc.append(j)
                hv.append(v)
        cat = np.concatenate
        i32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.int32, device=dev)  # noqa: E731
        T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype, device=dev)  # noqa: E731
        rp, cc, vv = _csr(cat(tr), cat(tc), cat(tv), 2 * nX)
        self.t_rp, self.t_col, self.t_val = i32(rp), i32(cc), T(vv)
        rp, cc, vv = _csr(cat(ur), cat(uc), cat(uv), self.ncp * 2 * NYZ)
        self.u_rp, self.u_col, self.u_val = i32(rp), i32(cc), T(vv)
        # to_fd by plane node with the DG node and the face of every entry
        rows, nn, ff, vv = cat(gr), cat(gn), cat(gf), cat(gv)
        order = np.lexsort((nn, rows))
        rows, nn, ff, vv = rows[order], nn[order], ff[order], vv[order]
        rp = np.zeros(self.ncp * 2 * NYZ + 1, dtype=np.int64)
        np.add.at(rp, rows + 1, 1)
        self.g_rp, self.g_node, self.g_face, self.g_val = (i32(np.cumsum(rp)), i32(nn), i32(ff),
                                                           T(vv))
        rp, cc, vv = _csr(cat(hr), cat(hc), cat(hv), 2 * nX)
        self.h_rp, self.h_col, self.h_val = i32(rp), i32(cc), T(vv)
        cp_of = cat([np.full(n, i) for i, n in enumerate(nif)])
        self.cp_of = i32(cp_of)
        self.node_of = i32(cat(nodes))
        self.face_of = i32(cat([s // self.Nfp for s in slots]))
        n0 = np_(d.n[0].reshape(-1))
        self.n0 = T(cat([n0[s] for s in slots]))
        self.scale = 2.0 / float(y.spacing[0])

    # ------------------------------------------------------------- plane sets
    def fe_planes(self, fy, fz):
        """Interface rows of two Yee E components (``fy``, ``fz``: (S, NX, NY, NZ))."""
        return [(fy, fz, self.N, r * self.NYZ, 0) for r in self.rows]

    def ghost_planes(self, hz, hy):
        """Ghost (a) and inner (b) rows of the Yee ``Hz`` / ``Hy`` (lattice 0 / 1)."""
        return [(hz, hy, self.N, g * self.NYZ, n * self.NYZ)
                for g, n in zip(self.ghosts, self.inners)]

    # ------------------------------------------------------------- forward
    def trace(self, planes, Ex, Xh=None, g_if=None):
        """``E* = P fe|rows`` into ``Ex`` (S, 2, nX) and, with ``Xh``, the H-flux input
        ``X_h = g n_x (E*_z, -E*_y)`` (``g_if``: (K, 4))."""
        gi, pp = plane_args(planes)
        self.k.call(1, [self.S, self.nX, self.ncp, 0, 0, 0, 0, 0] + gi,
                    [self.t_rp, self.t_col, self.t_val, self.cp_of, self.face_of, self.n0,
                     g_if, Ex, Xh] + [None] * 7 + pp, [], Ex, "glue trace")

    def ghost(self, planes, F):
        """Ghost planes ``2 C^T F_t - inner`` written in place (``F``: DG (S, 3, K, Np))."""
        gi, pp = plane_args(planes)
        self.k.call(4, [self.S, self.ncp, self.NYZ, self.KN, 0, 0, 0, 0] + gi,
                    [self.g_rp, self.g_node, self.g_val, F] + [None] * 12 + pp, [], F,
                    "glue ghost planes")

    def penalty(self, planes, E, g_if, pen):
        """Yee side of the interface penalty into ``pen`` (S, ncp, 2, NY, NZ)."""
        gi, pp = plane_args(planes)
        self.k.call(6, [self.S, self.ncp, self.NYZ, self.KN, 0, 0, 0, 0] + gi,
                    [self.g_rp, self.g_node, self.g_face, self.g_val, g_if, E, pen]
                    + [None] * 9 + pp, [self.scale], E, "glue penalty")

    # ------------------------------------------------------------- transposes
    def trace_bwd(self, planes, gX=None, gXh=None, g_if=None):
        """``planes += P^T gE`` with ``gE = gX`` plus the transpose of ``X_h``."""
        if gX is None and gXh is None:
            return
        ref = gX if gX is not None else gXh
        gi, pp = plane_args(planes)
        self.k.call(2, [self.S, self.nX, self.ncp, self.NYZ, 0, 0, 0, 0] + gi,
                    [self.u_rp, self.u_col, self.u_val, self.face_of, self.n0, g_if, gX, gXh]
                    + [None] * 8 + pp, [], ref, "glue trace transpose")

    def xh_gif(self, gXh, Ex, gacc):
        """``g_if`` cotangent of ``X_h`` accumulated into ``gacc`` (K * 4)."""
        self.k.call(3, [self.S, self.nX, 0, 0, 0, 0, 0, 0],
                    [self.face_of, self.n0, gXh, Ex, gacc], [], gXh, "glue X_h transpose")

    def ghost_bwd(self, planes, lF):
        """Transpose of :meth:`ghost` (in place on the cotangent planes; ``lF`` += DG part)."""
        gi, pp = plane_args(planes)
        self.k.call(5, [self.S, self.ncp, self.NYZ, self.KN, self.nX, 0, 0, 0] + gi,
                    [self.h_rp, self.h_col, self.h_val, self.cp_of, self.node_of, lF]
                    + [None] * 10 + pp, [], lF, "glue ghost transpose")

    def penalty_bwd(self, fe_planes, lfe_planes, E, g_if, gp1, gp2, lE, gacc=None):
        """Transpose of :meth:`penalty` for the cotangents ``gp1 (+ gp2)`` of the rows:
        ``lE`` and the cotangent planes ``lfe_planes`` in place, ``g_if`` cotangent into
        ``gacc`` (K * 4) when given; ``fe_planes``: the forward input rows."""
        gi1, pp1 = plane_args(fe_planes)
        gi2, pp2 = plane_args(lfe_planes)
        ref = gp1 if gp1 is not None else gp2
        self.k.call(7, [self.S, self.ncp, self.NYZ, self.KN, self.nX, 0, 0, 0] + gi1 + gi2,
                    [self.h_rp, self.h_col, self.h_val, self.cp_of, self.node_of, self.face_of,
                     g_if, E, gp1, gp2, lE, gacc, self.g_rp, self.g_face, self.g_val, None]
                    + pp1 + pp2, [self.scale], ref, "glue penalty transpose")


class PointGlue:
    """Sources / receivers of a :class:`~DeepGPR.hybrid.functional3d.PointOps3D` in the
    kernel layouts (cached on the operator as ``ops.glue``)."""

    def __init__(self, lib, st, ops):
        dev, dtype = torch.device(st.device), st.dtype
        self.k = _Lib(lib, dtype)
        y, d = st.yee, st.dg
        self.S = st.nshot
        self.N = int(np.prod(y.shape))
        self.Np = d.Np if d is not None else 1
        self.KN = d.K * d.Np if d is not None else 0
        # 3D: DG fields (S, 3, K, Np), component ``ops.comp``; 2D (TM): (S, K, Np), Ez
        self.ncomp = 3 if len(y.shape) == 3 else 1
        self.comp = int(getattr(ops, "comp", 0)) if self.ncomp == 3 else 0
        self.n = int(ops.count)
        i32 = lambda t: t.to(device=dev, dtype=torch.int32).contiguous()  # noqa: E731
        self.nfd = int(ops.fd_shot.numel())
        self.ndg = int(ops.dg_shot.numel())
        self.f_shot, self.f_node, self.f_item = i32(ops.fd_shot), i32(ops.fd_node), i32(ops.fd_item)
        self.f_w = ops.fd_w.to(dev, dtype).contiguous()
        self.d_shot, self.d_elem, self.d_item = i32(ops.dg_shot), i32(ops.dg_elem), i32(ops.dg_item)
        self.d_w = ops.dg_w.to(dev, dtype).reshape(-1).contiguous()
        nout = self.S * self.n
        self.nout = nout
        cpu = lambda t: t.detach().cpu().numpy().astype(np.int64)  # noqa: E731
        self.o_fd = self.s_fnode = self.s_fw = None
        if self.nfd:
            o = cpu(ops.fd_shot) * self.n + cpu(ops.fd_item)
            order = np.argsort(o, kind="stable")
            rp = np.zeros(nout + 1, dtype=np.int64)
            np.add.at(rp, o + 1, 1)
            self.o_fd = torch.as_tensor(np.cumsum(rp), dtype=torch.int32, device=dev)
            self.s_fnode = self.f_node[torch.as_tensor(order, device=dev)].contiguous()
            self.s_fw = self.f_w[torch.as_tensor(order, device=dev)].contiguous()
        self.o_dg = self.s_delem = self.s_dw = None
        if self.ndg:
            o = cpu(ops.dg_shot) * self.n + cpu(ops.dg_item)
            order = np.argsort(o, kind="stable")
            rp = np.zeros(nout + 1, dtype=np.int64)
            np.add.at(rp, o + 1, 1)
            self.o_dg = torch.as_tensor(np.cumsum(rp), dtype=torch.int32, device=dev)
            idx = torch.as_tensor(order, device=dev)
            self.s_delem = self.d_elem[idx].contiguous()
            self.s_dw = ops.dg_w.to(dev, dtype)[idx].reshape(-1).contiguous()

    def inject(self, f, E, cb_fd, cb_dg, amp):
        """In place: ``f -= cb w amp`` on the Yee entries (``f``: the source component,
        (S, NX, NY, NZ)), ``E[:, comp] -= cb w amp`` on the DG entries."""
        self.k.call(8, [self.nfd, self.N, self.ndg, self.Np, self.KN, self.comp, self.ncomp],
                    [self.f_shot, self.f_node, self.f_item, self.f_w, cb_fd, amp, f,
                     self.d_shot, self.d_elem, self.d_item, self.d_w, cb_dg, E], [], amp,
                    "glue inject")

    def inject_bwd(self, lam_f, lam_E, cb_fd, cb_dg, amp, gamp=None, gcb_fd=None, gcb_dg=None):
        """Transpose of :meth:`inject` w.r.t. ``amp`` (into ``gamp``, zeroed by the caller)
        and the coefficients (accumulated into ``gcb_fd`` (N,) / ``gcb_dg`` (K,))."""
        if gamp is None and gcb_fd is None and gcb_dg is None:
            return
        self.k.call(9, [self.nfd, self.N, self.ndg, self.Np, self.KN, self.comp, self.ncomp],
                    [self.f_shot, self.f_node, self.f_item, self.f_w, cb_fd, amp, gcb_fd,
                     self.d_shot, self.d_elem, self.d_item, self.d_w, cb_dg, gcb_dg, lam_f, lam_E,
                     gamp], [], amp, "glue inject transpose")

    def sample(self, f, E, out):
        """Receiver samples into ``out`` (S, n) (``f``: the receiver component)."""
        self.k.call(10, [self.nout, self.n, self.N, self.Np, self.KN, self.comp, self.ncomp],
                    [self.o_fd, self.s_fnode, self.s_fw, self.o_dg, self.s_delem, self.s_dw, f, E,
                     out], [], out, "glue sample")

    def sample_bwd(self, g, lam_f, lam_E):
        """In place: ``lam += R^T g`` (``g``: (S, n))."""
        self.k.call(11, [self.nfd, self.n, self.N, self.ndg, self.Np, self.KN, self.comp,
                         self.ncomp],
                    [self.f_shot, self.f_node, self.f_item, self.f_w, g, lam_f,
                     self.d_shot, self.d_elem, self.d_item, self.d_w, lam_E], [], g,
                    "glue sample transpose")


class CouplingGlue2D:
    """Mortar coupling of a 2D (TM) :class:`~DeepGPR.hybrid.functional.StepStatic` (Ez
    interface rows, Hy ghost rows) in the kernel layouts (cached as ``st._glue2d``)."""

    def __init__(self, lib, st):
        y, d = st.yee, st.dg
        dev, dtype = torch.device(st.device), st.dtype
        self.k = _Lib(lib, dtype)
        self.S = st.nshot
        self.NX, self.NY = y.shape
        self.N = self.NX * self.NY
        self.KN = d.K * d.Np
        cps = st.couplings
        self.ncp = len(cps)
        if self.ncp > _MAXCP:
            raise ValueError("too many couplings for the fused glue")
        self.rows = [int(cp.row) for cp in cps]
        self.ghosts = [int(cp.ghost) for cp in cps]
        self.inners = [int(cp.inner) for cp in cps]
        nif = [int(cp.slots.numel()) for cp in cps]
        off = np.concatenate([[0], np.cumsum(nif)]).astype(np.int64)
        self.nX = int(off[-1])
        NY, nX = self.NY, self.nX
        np_ = lambda t: t.detach().cpu().numpy()  # noqa: E731
        nodes = [np_(cp.nodes).astype(np.int64) for cp in cps]
        tr, tc, tv, ur, uc, uv = [], [], [], [], [], []
        gr, gn, gv, hr, hc, hv = [], [], [], [], [], []
        for i, cp in enumerate(cps):
            o = int(off[i])
            r, j, v = np_(cp.to_dg.rows), np_(cp.to_dg.cols), np_(cp.to_dg.vals)
            tr.append(o + r)
            tc.append(j)
            tv.append(v)
            ur.append(i * NY + j)
            uc.append(o + r)
            uv.append(v)
            j, e, v = np_(cp.to_fd.rows), np_(cp.to_fd.cols), np_(cp.to_fd.vals)
            gr.append(i * NY + j)
            gn.append(nodes[i][e])
            gv.append(v)
            hr.append(o + e)
            hc.append(j)
            hv.append(v)
        cat = np.concatenate
        i32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.int32, device=dev)  # noqa: E731
        T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype, device=dev)  # noqa: E731
        rp, cc, vv = _csr(cat(tr), cat(tc), cat(tv), nX)
        self.t_rp, self.t_col, self.t_val = i32(rp), i32(cc), T(vv)
        rp, cc, vv = _csr(cat(ur), cat(uc), cat(uv), self.ncp * NY)
        self.u_rp, self.u_col, self.u_val = i32(rp), i32(cc), T(vv)
        rp, cc, vv = _csr(cat(gr), cat(gn), cat(gv), self.ncp * NY)
        self.g_rp, self.g_node, self.g_val = i32(rp), i32(cc), T(vv)
        rp, cc, vv = _csr(cat(hr), cat(hc), cat(hv), nX)
        self.h_rp, self.h_col, self.h_val = i32(rp), i32(cc), T(vv)
        self.cp_of = i32(cat([np.full(n, i) for i, n in enumerate(nif)]))
        self.node_of = i32(cat(nodes))

    def rows_planes(self, ez, stride=None, rows=None):
        """Interface rows of an Ez-like array ``(S, NX, NY)`` (or of slab arrays: one
        array per coupling with its own stride and local row)."""
        if isinstance(ez, (list, tuple)):
            return [(a, a, st_, r * self.NY, 0) for a, st_, r in zip(ez, stride, rows)]
        return [(ez, ez, self.N, r * self.NY, 0) for r in self.rows]

    def ghost_planes(self, hy, stride=None, ghosts=None, inners=None):
        """Ghost (a) and inner (b) rows of an Hy-like array (or of slab arrays)."""
        if isinstance(hy, (list, tuple)):
            return [(a, a, st_, g * self.NY, n * self.NY)
                    for a, st_, g, n in zip(hy, stride, ghosts, inners)]
        return [(hy, hy, self.N, g * self.NY, n * self.NY)
                for g, n in zip(self.ghosts, self.inners)]

    def trace(self, planes, X):
        gi, pp = plane_args(planes)
        self.k.call(14, [self.S, self.nX, self.ncp, 0, 0, 0, 0, 0] + gi,
                    [self.t_rp, self.t_col, self.t_val, self.cp_of, X] + [None] * 11 + pp, [], X,
                    "glue trace (2D)")

    def trace_bwd(self, planes, gX):
        gi, pp = plane_args(planes)
        self.k.call(15, [self.S, self.nX, self.ncp, self.NY, 0, 0, 0, 0] + gi,
                    [self.u_rp, self.u_col, self.u_val, gX] + [None] * 12 + pp, [], gX,
                    "glue trace transpose (2D)")

    def ghost(self, planes, F):
        gi, pp = plane_args(planes)
        self.k.call(16, [self.S, self.ncp, self.NY, self.KN, 0, 0, 0, 0] + gi,
                    [self.g_rp, self.g_node, self.g_val, F] + [None] * 12 + pp, [], F,
                    "glue ghost rows (2D)")

    def ghost_bwd(self, planes, lF):
        gi, pp = plane_args(planes)
        self.k.call(17, [self.S, self.ncp, self.NY, self.KN, self.nX, 0, 0, 0] + gi,
                    [self.h_rp, self.h_col, self.h_val, self.cp_of, self.node_of, lF]
                    + [None] * 10 + pp, [], lF, "glue ghost transpose (2D)")


def coupling_glue2d(st) -> Optional[CouplingGlue2D]:
    """The 2D coupling glue of ``st`` (built once), ``None`` without couplings or CUDA."""
    g = st.__dict__.get("_glue2d", False)
    if g is False:
        g = None
        y, d = st.yee, st.dg
        if (torch.device(st.device).type == "cuda" and y is not None and d is not None
                and st.couplings and y.cuda is not None and d.cuda is not None):
            g = CouplingGlue2D(y.cuda.lib, st)
        st.__dict__["_glue2d"] = g
    return g


def combine(lib_ops: _Lib, mode: int, a: float, x, y, z, out):
    """``mode 0``: ``out = a (x + y)``; ``mode 1``: ``out = x + a (y - z)`` (flat, contiguous)."""
    lib_ops.call(12, [out.numel(), mode], [x, y, z, out], [a], out, "glue combine")


def coupling_glue(st) -> Optional[CouplingGlue]:
    """The coupling glue of ``st`` (built once), ``None`` without couplings or CUDA."""
    g = st.__dict__.get("_glue", False)
    if g is False:
        g = None
        y, d = st.yee, st.dg
        if (torch.device(st.device).type == "cuda" and y is not None and d is not None
                and st.couplings and y.cuda is not None and d.cuda is not None):
            g = CouplingGlue(y.cuda.lib, st)
        st.__dict__["_glue"] = g
    return g


def point_glue(st, ops) -> Optional[PointGlue]:
    """The point glue of ``ops`` for ``st`` (built once per operator)."""
    if ops is None:
        return None
    key = id(st)
    cache = ops.__dict__.setdefault("_glue", {})
    g = cache.get(key)
    if g is None:
        y = st.yee
        if torch.device(st.device).type != "cuda" or y is None or y.cuda is None:
            return None
        g = cache[key] = PointGlue(y.cuda.lib, st, ops)
    return g


__all__ = ["CouplingGlue", "CouplingGlue2D", "PointGlue", "combine", "coupling_glue",
           "coupling_glue2d", "plane_args", "point_glue"]
