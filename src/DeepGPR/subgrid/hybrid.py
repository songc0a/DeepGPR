r"""DGTD rugged surface + Yee FDTD below + Huygens subgrids in the FDTD part.

The main solver is :class:`DeepGPR.hybrid.HybridMaxwell3D` (body-fitted DGTD
band around the rugged surface, Yee grid above and below it); Huygens subgrids
(:mod:`.hsg`) are attached to its Yee grid below the band, where targets need a
finer grid than the background. One main step::

    hybrid H half   Yee H (+ side UPML), DG H, ghost rows       (as functional3d.step)
    subgrids        hsg_2 + OS H correction of the main Yee H
    hybrid E half   Yee E (+ interface penalty), DG E, sources
    subgrids        hsg_1 + OS E correction of the main Yee E

The hybrid halves reuse the operators of :mod:`DeepGPR.hybrid.functional3d`
block by block (the same arithmetic as its ``step``; that module is not
changed). The subgrids must lie below the band and away from the side UPML.

Models live on a fine model grid of spacing ``h``; the hybrid (its Yee grid and
the DG element materials) sees the ``coarsen^3`` block averages (spacing
``coarsen * h``), every subgrid the model at its own spacing, so one set of
model-grid parameters drives all three discretisations and the exact discrete
adjoint (:class:`~DeepGPR.subgrid.adjoint.SubgridPropagator`) reaches it.
Sources and receivers act on the hybrid part (Yee or DG), as in
:class:`~DeepGPR.hybrid.HybridMaxwell3D`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Sequence

import numpy as np
import torch

from ..hybrid import functional3d as fn3
from ..hybrid.solver3d import HybridMaxwell3D
from . import hsg
from .solver import Subgrid, block_average

HYB_STATE = fn3.STATE_SIZE


@dataclass
class HybridSubgridStatic:
    nshot: int
    hyb: fn3.StepStatic3D
    subs: List[hsg.SubgridStatic]
    dtype: torch.dtype = torch.float64
    device: torch.device = torch.device("cpu")

    @property
    def main(self) -> fn3.YeeStatic3D:
        return self.hyb.yee


@dataclass
class HybridSubgridCoefs:
    hyb: fn3.StepCoefs3D
    sub_ca: List[torch.Tensor]
    sub_cb: List[torch.Tensor]

    def tensors(self) -> List[torch.Tensor]:
        return self.hyb.tensors() + list(self.sub_ca) + list(self.sub_cb)

    def replaced(self, tensors: Sequence[torch.Tensor]) -> HybridSubgridCoefs:
        t = list(tensors)
        k = len(self.hyb.tensors())
        n = len(self.sub_ca)
        return HybridSubgridCoefs(self.hyb.replaced(t[:k]), t[k:k + n], t[k + n:k + 2 * n])

    @property
    def main_cb(self):
        return self.hyb.fd_cb


@dataclass
class HybridPointSet:
    """Points of the hybrid part (``fn3.PointOps3D``); no subgrid points."""

    hybrid: fn3.PointOps3D
    sub: List[hsg.Points]
    count: int
    comp: int

    @property
    def has_sub(self) -> bool:
        return False

    @property
    def main(self):
        return SimpleNamespace(item=torch.cat([self.hybrid.fd_item, self.hybrid.dg_item]))


# --------------------------------------------------------------------------- functional
def zero_state(st: HybridSubgridStatic, rec=None):
    out = list(fn3.zero_state(st.hyb))
    S = st.nshot
    kw = dict(dtype=st.dtype, device=st.device)
    for sg in st.subs:
        out += list(fn3.zero_state(fn3.StepStatic3D(S, sg.yee, None, [], st.dtype, st.device))[:18])
        out.append(torch.zeros((S, sg.is_h.size), **kw))
        out.append(torch.zeros((1, 0), **kw))
    return tuple(out)


def _hybrid_h(st: fn3.StepStatic3D, c: fn3.StepCoefs3D, state, pert_h=None):
    """First half of :func:`DeepGPR.hybrid.functional3d.step` (to the ghost rows)."""
    fe, fh = list(state[0:3]), list(state[3:6])
    pse, psh = list(state[6:12]), list(state[12:18])
    E, H, Daux, Baux = state[18], state[19], state[20], state[21]
    Dfd, Bfd = state[22], state[23]
    y, d = st.yee, st.dg
    S = st.nshot
    fused = y.cuda is not None and fe[0].is_cuda
    if fused:
        fh, psh, Bfd = y.cuda.h_update(fe, fh, psh, Bfd)
    else:
        rhs, psh = fn3._yee_curl(y, fe, y.ph, psh, fn3._fwd)
        fh_new = [torch.where(y.act[3 + k], fh[k] - y.chr * rhs[k], fh[k]) for k in range(3)]
        if y.upml is not None:
            fh_new, Bfd = fn3._yee_upml(y.upml, "h", Bfd, rhs, fh, fh_new)
        fh = fh_new
    Eext, compact = None, False
    if d is not None:
        compact = d.cuda is not None and E.is_cuda
        Eext = fn3.interface_trace(st, fe, compact) if st.couplings else None
        if compact:
            H, Baux = d.cuda.update_h(fn3.dg_operator(d, (c.e0, c.e1, c.e2, c.e3), E, H, Eext), H,
                                      Baux)
        else:
            curl = fn3._dg_curl(fn3.dg_operator(d, (c.e0, c.e1, c.e2, c.e3), E, H, Eext))
            H_new = H - d.ch * curl
            if d.pml_idx is not None:
                u = d.upml
                H_new, Baux = fn3._upml(d.pml_idx, Baux, -curl, H, H_new, u["a1"], u["b1"],
                                        u["h_hist"], u["h_new"], u["h_old"])
            H = H_new
        if st.couplings:
            hflat = H.reshape(S, 3, -1)
            NY, NZ = y.shape[1], y.shape[2]
            rows = fn3._cached(st, "ghosts", lambda: torch.as_tensor(
                [cp.ghost for cp in st.couplings], device=H.device))
            hz = torch.stack([2.0 * cp.to_fd[0](hflat[:, 2].index_select(1, cp.nodes)).view(S, NY, NZ)
                              - fh[2][:, cp.inner] for cp in st.couplings], dim=1)
            hy = torch.stack([2.0 * cp.to_fd[1](hflat[:, 1].index_select(1, cp.nodes)).view(S, NY, NZ)
                              - fh[1][:, cp.inner] for cp in st.couplings], dim=1)
            fh[2] = fh[2].index_copy(1, rows, hz)
            fh[1] = fh[1].index_copy(1, rows, hy)
    if pert_h is not None:
        fh = [f + p for f, p in zip(fh, pert_h)]
    fh = [f if f._base is None else f.clone() for f in fh]
    return [fe, fh, pse, psh, E, H, Daux, Baux, Dfd, Bfd, Eext, compact]


def _hybrid_e(st: fn3.StepStatic3D, c: fn3.StepCoefs3D, parts, src=None, amp=None, pert_e=None):
    """Second half of :func:`DeepGPR.hybrid.functional3d.step` (Yee E, DG E, sources)."""
    fe, fh, pse, psh, E, H, Daux, Baux, Dfd, Bfd, Eext, compact = parts
    y, d = st.yee, st.dg
    fused = y.cuda is not None and fe[0].is_cuda
    if fused:
        pen = fn3.interface_penalty_rows(st, c, fe, E) if c.g_if is not None else None
        fe, pse, Dfd = y.cuda.e_update(fh, fe, pse, Dfd, c, pen)
    else:
        rhs, pse = fn3._yee_curl(y, fh, y.pe, pse, fn3._bwd)
        if c.g_if is not None:
            rhs = fn3._yee_interface_penalty(st, c, rhs, fe, E)
        fe_new = [torch.where(y.act[k], c.fd_ca * fe[k] + c.fd_cb * rhs[k], fe[k]) for k in range(3)]
        if y.upml is not None:
            fe_new, Dfd = fn3._yee_upml(y.upml, "e", Dfd, rhs, fe, fe_new,
                                        (c.fu_hist, c.fu_new, c.fu_old))
        fe = fe_new
    if d is not None:
        Xh = fn3.interface_x(st, c, Eext, compact) if c.g_if is not None else None
        if compact:
            E, Daux = d.cuda.update_e(fn3.dg_operator(d, (c.h0, c.h1, c.h2, c.h3), H, E, Xh), E,
                                      Daux, c.dg_ca, c.dg_cb, c.u_hist, c.u_new, c.u_old)
        else:
            curl = fn3._dg_curl(fn3.dg_operator(d, (c.h0, c.h1, c.h2, c.h3), H, E, Xh))
            E_new = c.dg_ca[None, None] * E + c.dg_cb[None, None] * curl
            if d.pml_idx is not None:
                u = d.upml
                E_new, Daux = fn3._upml(d.pml_idx, Daux, curl, E, E_new, u["a1"], u["b1"],
                                        c.u_hist, c.u_new, c.u_old)
            E = E_new
    if pert_e is not None:
        fe = [f + p for f, p in zip(fe, pert_e)]
    if src is not None and amp is not None:
        fe, E = fn3.inject(st, c, src, amp, fe, E)
    fe = [f if f._base is None else f.clone() for f in fe]
    return fe, pse, E, Daux, Dfd


def step(st: HybridSubgridStatic, c: HybridSubgridCoefs, state, src: Optional[HybridPointSet] = None,
         amp=None, amp_fine=None, rec: Optional[HybridPointSet] = None, pert=None):
    """One main step of the hybrid with subgrids (see the module docstring)."""
    S = st.nshot
    subs = hsg.unpack_subgrids(state, len(st.subs), base=HYB_STATE)
    fe0, fh0 = list(state[0:3]), list(state[3:6])
    pe_cur = [hsg.pre(sg, "e", fe0) for sg in st.subs]
    ph_prev = [hsg.pre(sg, "h", fh0) for sg in st.subs]
    parts = _hybrid_h(st.hyb, c.hyb, state, None if pert is None else pert[0])
    fh = parts[1]
    ph_cur = [hsg.pre(sg, "h", fh) for sg in st.subs]
    fh, new_rec = hsg.subgrids_first_half(st.subs, c, subs, fh, pe_cur, ph_prev, ph_cur, None,
                                          None, None, S)
    parts[1] = fh
    fe, pse, E, Daux, Dfd = _hybrid_e(st.hyb, c.hyb, parts, None if src is None else src.hybrid,
                                      amp, None if pert is None else pert[1])
    pe_next = [hsg.pre(sg, "e", fe) for sg in st.subs]
    fe = hsg.subgrids_second_half(st.subs, c, subs, fe, c.hyb.fd_cb, pe_cur, pe_next, ph_prev,
                                  ph_cur, None, None)
    _, _, _, psh, _, H, _, Baux, _, Bfd, _, _ = parts
    out = list(fe) + list(fh) + list(pse) + list(psh) + [E, H, Daux, Baux, Dfd, Bfd]
    return tuple(out + hsg.pack_subgrids(subs, pe_cur, new_rec))


def sample(st: HybridSubgridStatic, rec: HybridPointSet, state):
    d = fn3.sample(st.hyb, rec.hybrid, state)
    return d, torch.zeros_like(d)


# --------------------------------------------------------------------------- solver
class HybridSubgridMaxwell3D:
    """Hybrid DGTD-FDTD (rugged surface) with Huygens subgrids below the DG band.

    Args:
        eps_r, sigma: Ground model ``(nx, ny, nz)`` on the model grid (spacing ``h``;
            ``n - 1`` divisible by ``coarsen``); above the surface everything is air.
        h: Model spacing [m].
        coarsen: Odd ratio of the hybrid's spacing (Yee grid, DG materials) to ``h``.
        surface: :class:`~DeepGPR.hybrid.topography.Surface3D` (physical frame).
        dg_band: ``(top, bottom)`` depths [m] of the DG band (see
            :class:`~DeepGPR.hybrid.HybridMaxwell3D`).
        subgrids: :class:`~DeepGPR.subgrid.Subgrid` boxes below the band.
        **hybrid_kw: Further arguments of :class:`~DeepGPR.hybrid.HybridMaxwell3D`.
    """

    functional = None      # set below (this module)

    def __init__(self, eps_r, sigma, h: float, *, coarsen: int = 3, surface, dg_band,
                 subgrids: Sequence[Subgrid] = (), pmlthick=10, nshot: int = 1,
                 component: int = 2, device="cpu", dtype=torch.float64, cuda_kernels=True,
                 **hybrid_kw):
        t0 = time.perf_counter()
        self.device, self.dtype = torch.device(device), dtype
        R = int(coarsen)
        if R < 1 or R % 2 == 0:
            raise ValueError("coarsen must be a positive odd integer.")
        eps = torch.as_tensor(eps_r, dtype=dtype, device=self.device)
        sig = torch.as_tensor(sigma, dtype=dtype, device=self.device)
        if any((n - 1) % R for n in eps.shape):
            raise ValueError("Every model dimension minus one must be divisible by coarsen.")
        self.coarsen, self.h = R, float(h)
        self.model_shape = tuple(eps.shape)
        self.nshot = int(nshot)
        self.hyb = HybridMaxwell3D(block_average(eps, R), block_average(sig, R), R * self.h,
                                   surface=surface, dg_band=dg_band, pmlthick=pmlthick,
                                   nshot=nshot, component=component, device=device, dtype=dtype,
                                   cuda_kernels=cuda_kernels, **hybrid_kw)
        if self.hyb.fdtd_refinement != 1:
            raise ValueError("Use coarsen instead of fdtd_refinement.")
        self.dt = self.hyb.dt
        self.pml = list(self.hyb.pml)
        self.dx = R * self.h
        main = self.hyb.static.yee
        NX, NY, NZ = main.shape
        i_b = self.hyb.band[1]
        self.bounds = ((i_b + 2, NX - 3 - self.pml[1]), (self.pml[2] + 1, NY - 2 - self.pml[3]),
                       (self.pml[4] + 1, NZ - 2 - self.pml[5]))
        use_cuda = bool(cuda_kernels) and self.device.type == "cuda"
        subs = []
        self.specs = list(subgrids)
        for spec in self.specs:
            r = int(spec.ratio)
            if R % r:
                raise ValueError(f"Subgrid ratio {r} must divide coarsen = {R}.")
            lo, hi = [], []
            box = np.asarray(spec.box, dtype=np.float64).reshape(3, 2)
            for a in range(3):
                lo.append(int(round(box[a, 0] / self.dx)) + self.pml[2 * a])
                hi.append(int(round(box[a, 1] / self.dx)) + self.pml[2 * a])
            subs.append(hsg.build_subgrid(main, lo, hi, r, spec.is_os_sep, spec.pml_cells,
                                          spec.pml_sep, self.dt, spec.filter, dtype, self.device,
                                          self.nshot, use_cuda, bounds=self.bounds,
                                          time_ratio=spec.time_ratio))
        self.static = HybridSubgridStatic(self.nshot, self.hyb.static, subs, dtype, self.device)
        self._fine_index = [self._fine_model_index(sg) for sg in subs]
        self.model = (eps.detach(), sig.detach())
        self.setup_seconds = time.perf_counter() - t0

    def _fine_model_index(self, sg):
        q = self.coarsen // sg.ratio
        out = []
        for a in range(3):
            n = sg.yee.shape[a]
            base = (sg.lo[a] - self.pml[2 * a]) * self.coarsen
            idx = base + (np.arange(n) - sg.nb) * q
            out.append(torch.as_tensor(np.clip(idx, 0, self.model_shape[a] - 1), device=self.device))
        return out

    def box_bounds(self, g: int, surface: str = "inner"):
        sg = self.static.subs[g]
        lo, hi = (sg.lo, sg.hi) if surface == "inner" else (sg.os_lo, sg.os_hi)
        return (np.array([(lo[a] - self.pml[2 * a]) * self.dx for a in range(3)]),
                np.array([(hi[a] - self.pml[2 * a]) * self.dx for a in range(3)]))

    # ------------------------------------------------------------ materials
    def material_fields(self, eps_r, sigma):
        """Differentiable ``(hybrid fields, [sub_eps], [sub_sig])`` from the model grid."""
        R = self.coarsen
        hyb = self.hyb.material_fields(block_average(eps_r, R), block_average(sigma, R))

        def fine(t, g):
            sg = self.static.subs[g]
            q = R // sg.ratio
            sm = block_average(t, q, stride=1) if q > 1 else t
            i, j, k = self._fine_index[g]
            return sm.index_select(0, i).index_select(1, j).index_select(2, k)

        n = len(self.static.subs)
        return hyb, [fine(eps_r, g) for g in range(n)], [fine(sigma, g) for g in range(n)]

    def build_coefficients(self, hyb_fields, sub_eps, sub_sig) -> HybridSubgridCoefs:
        c = fn3.build_coefficients(self.static.hyb, hyb_fields, self.dt)
        sca, scb = [], []
        for g, sg in enumerate(self.static.subs):
            a, b = fn3.yee_coefficients_3d(sub_eps[g], sub_sig[g], sg.dt)
            sca.append(a)
            scb.append(b)
        return HybridSubgridCoefs(c, sca, scb)

    # ------------------------------------------------------------ points / run
    def point_ops(self, positions, *, source: bool) -> HybridPointSet:
        pos = np.asarray(torch.as_tensor(positions, dtype=torch.float64).cpu())
        flat = pos.reshape(-1, 3)
        for g in range(len(self.static.subs)):
            lo, hi = self.box_bounds(g, "outer")
            inside = np.all((flat > lo - self.dx) & (flat < hi + self.dx), axis=1)
            if inside.any():
                raise ValueError("Sources/receivers must lie outside the subgrids' outer "
                                 "surfaces (subgrid antennas are not supported with the DG band).")
        ops = self.hyb.point_ops(pos, source=source)
        empty = hsg.Points(*(torch.zeros(0, dtype=torch.long, device=self.device),) * 3,
                           torch.zeros(0, dtype=self.dtype, device=self.device))
        return HybridPointSet(ops, [empty] * len(self.static.subs), ops.count, ops.comp)

    def run(self, source_amplitudes, source_positions, receiver_positions, *, coefs=None) -> dict:
        st = self.static
        if coefs is None:
            coefs = self.build_coefficients(*self.material_fields(*self.model))
        w = torch.as_tensor(source_amplitudes, dtype=self.dtype, device=self.device)
        if w.ndim == 1:
            w = w[None]
        nt = w.shape[1]
        src = self.point_ops(source_positions, source=True)
        rec = self.point_ops(receiver_positions, source=False)
        nsrc = int(np.asarray(source_positions).shape[-2])
        if w.shape[0] == 1 and nsrc > 1:
            w = w.expand(nsrc, -1)
        state = zero_state(st)
        data = torch.zeros((self.nshot, nt, rec.count), dtype=self.dtype, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            for n in range(nt):
                state = step(st, coefs, state, src, w[:, n])
                data[:, n] = sample(st, rec, state)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.state = state
        return {"receivers": data, "seconds": time.perf_counter() - t0, "dt": self.dt}

    def summary(self) -> dict:
        info = self.hyb.summary()
        info["coarsen"] = self.coarsen
        info["model_spacing"] = self.h
        info["subgrids"] = [dict(ratio=sg.ratio, shape=sg.yee.shape, lo=sg.lo, hi=sg.hi)
                            for sg in self.static.subs]
        info["sub_nodes"] = [int(np.prod(sg.yee.shape)) for sg in self.static.subs]
        return info


import sys as _sys  # noqa: E402

HybridSubgridMaxwell3D.functional = _sys.modules[__name__]

__all__ = ["HybridPointSet", "HybridSubgridCoefs", "HybridSubgridMaxwell3D", "HybridSubgridStatic",
           "sample", "step", "zero_state"]
