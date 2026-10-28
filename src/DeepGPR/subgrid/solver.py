r"""3D Yee-FDTD with Huygens subgrids (:mod:`.hsg`): model handling and driver.

The model (``eps_r``, ``sigma``) lives on a *model grid* of spacing ``h``. The
main grid has spacing ``dx = coarsen * h`` (``coarsen`` odd); its node values
are averages of the model over the ``coarsen^3`` block centred on the node. A
subgrid with ratio ``r`` (a divisor of ``coarsen``) has spacing ``dx / r`` and
takes block averages over ``(coarsen / r)^3`` model cells (the model values
themselves for ``r = coarsen``). Both maps are linear, so gradients reach the
model grid.

Sources and receivers (component ``component``) are trilinear point operators
on the grid that owns them: inside a subgrid's inner surface they act on the
subgrid (its field is the total field there), outside every outer surface on
the main grid. Points between the two surfaces are rejected. Source samples are
given at the main ``dt``; subgrid sources use their band-limited resampling at
``dt / r`` (:func:`~DeepGPR.subgrid.hsg.fine_source_samples`) with the same
half-step delay as the main grid, so that a source carries the same physical
current wherever it is placed.

Receiver samples are those of DeepGPR's convention: ``data[:, n]`` is the field
at ``(n + 1) dt``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..config.constants import SPEED_OF_LIGHT
from ..hybrid import functional3d as fn3
from ..preprocessing.grid import pmlthick_revert
from ..preprocessing.model_setup import _ExtendModel
from . import graphs, hsg


@dataclass
class Subgrid:
    """A refined box.

    Args:
        box: ``((x0, x1), (y0, y1), (z0, z1))`` [m] in the model frame (model node
            ``(0, 0, 0)`` at the origin), rounded to main-grid nodes. This is the
            working region (inner Huygens surface).
        ratio: Odd refinement ratio ``r`` (main spacing / subgrid spacing).
        is_os_sep: Main cells between the inner and the outer surface (gprMax
            default 3).
        pml_cells: Subgrid CPML thickness [subgrid cells] (gprMax default 6).
        pml_sep: Subgrid cells between the outer surface and the CPML (default
            ``r // 2 + 2``, as gprMax).
        filter: Normal ``(1/4, 1/2, 1/4)`` smoothing of the precursors (gprMax's
            stabilising filter, default on).
        time_ratio: Sub-steps per main step: ``ratio`` (default, ``dt / r``) or 1 (the
            main ``dt``; only when it satisfies the subgrid's CFL limit, as under a
            DGTD band whose ``dt`` is far below the Yee limit).
    """

    box: Sequence[Sequence[float]]
    ratio: int = 3
    is_os_sep: int = 3
    pml_cells: int = 6
    pml_sep: Optional[int] = None
    filter: bool = True
    time_ratio: Optional[int] = None


def block_average(t: torch.Tensor, R: int, stride: Optional[int] = None) -> torch.Tensor:
    """Average over the ``R^3`` block centred on every ``stride``-th node (``R`` odd,
    replicated edges; ``stride`` defaults to ``R``)."""
    if R == 1:
        return t
    p = (R - 1) // 2
    x = F.pad(t[None, None], (p, p, p, p, p, p), mode="replicate")
    return F.avg_pool3d(x, R, stride=R if stride is None else stride)[0, 0]


class SubgridMaxwell3D:
    """3D Yee-FDTD with Huygens subgrids.

    Args:
        eps_r, sigma: Model ``(nx, ny, nz)`` on the model grid (DeepGPR layout,
            axis 0 = depth); ``(n - 1)`` must be divisible by ``coarsen``.
        dx: Model spacing ``h`` [m] (scalar; cubic cells).
        coarsen: Odd ratio of the main spacing to ``h``.
        subgrids: :class:`Subgrid` boxes (``ratio`` must divide ``coarsen``).
        pmlthick: Main-grid CPML thickness [main cells] (int or six values).
        dt: Main time step; default ``cfl`` times the Yee limit of the main grid.
        component: Source/receiver component (0, 1, 2).
        dipole_length: Length ``dl`` of the Hertzian source (default: main spacing),
            ``J = amp dl delta``.
        cuda_kernels: Fused CUDA Yee kernels on GPUs.
    """

    functional = hsg     # zero_state / step / sample of this solver (used by the adjoint)

    def __init__(self, eps_r, sigma, dx: float, *, coarsen: int = 3,
                 subgrids: Sequence[Subgrid] = (), pmlthick=10, dt: Optional[float] = None,
                 cfl: float = 0.95, component: int = 2, nshot: int = 1,
                 dipole_length: Optional[float] = None, cuda_kernels: bool = True,
                 device="cpu", dtype=torch.float64):
        t0 = time.perf_counter()
        self.device, self.dtype = torch.device(device), dtype
        R = int(coarsen)
        if R < 1 or R % 2 == 0 or coarsen != R:
            raise ValueError("coarsen must be a positive odd integer.")
        self.coarsen = R
        eps = torch.as_tensor(eps_r, dtype=dtype, device=self.device)
        sig = torch.as_tensor(sigma, dtype=dtype, device=self.device)
        if eps.ndim != 3 or eps.shape != sig.shape:
            raise ValueError("eps_r and sigma must be 3D arrays of the same shape.")
        if any((n - 1) % R for n in eps.shape):
            raise ValueError("Every model dimension minus one must be divisible by coarsen.")
        self.h = float(dx)
        self.model_shape = tuple(eps.shape)
        self.spacing = (R * self.h,) * 3
        self.dx = self.spacing[0]
        self.main_interior = tuple((n - 1) // R + 1 for n in eps.shape)
        self.pml = [int(v) for v in pmlthick_revert(pmlthick, eps).tolist()]
        self.nshot = int(nshot)
        self.component = int(component)
        self.dipole_length = self.dx if dipole_length is None else float(dipole_length)
        self.cuda_kernels = bool(cuda_kernels)
        self.shape = tuple(n + self.pml[2 * a] + self.pml[2 * a + 1] + 1
                           for a, n in enumerate(self.main_interior))
        if dt is None:
            dt = cfl / (SPEED_OF_LIGHT * math.sqrt(3.0) / self.dx)
        self.dt = float(dt)
        use_cuda = self.cuda_kernels and self.device.type == "cuda"
        main = hsg.yee_grid(self.shape, self.spacing, self.pml, self.dt, dtype, self.device,
                            self.nshot, use_cuda)
        self.specs = list(subgrids)
        subs = []
        for spec in self.specs:
            r = int(spec.ratio)
            if R % r:
                raise ValueError(f"Subgrid ratio {r} must divide coarsen = {R}.")
            lo, hi = self._box_indices(spec.box)
            subs.append(hsg.build_subgrid(main, lo, hi, r, spec.is_os_sep, spec.pml_cells,
                                          spec.pml_sep, self.dt, spec.filter, dtype, self.device,
                                          self.nshot, use_cuda, time_ratio=spec.time_ratio))
        self._check_disjoint(subs)
        self.static = hsg.SubgridStepStatic(self.nshot, main, subs, dtype, self.device)
        self._fine_index = [self._fine_model_index(sg) for sg in subs]
        self.setup_seconds = time.perf_counter() - t0
        self.model = (eps.detach(), sig.detach())
        self.coefs = self.build_coefficients(*self.material_fields(eps, sig))

    # ------------------------------------------------------------ geometry
    def _box_indices(self, box):
        box = np.asarray(box, dtype=np.float64).reshape(3, 2)
        lo, hi = [], []
        for a in range(3):
            i0 = int(round(box[a, 0] / self.dx))
            i1 = int(round(box[a, 1] / self.dx))
            if i1 <= i0:
                raise ValueError("Every subgrid box axis needs x1 > x0 (at least one main cell).")
            lo.append(i0 + self.pml[2 * a])
            hi.append(i1 + self.pml[2 * a])
        return tuple(lo), tuple(hi)

    @staticmethod
    def _check_disjoint(subs):
        for i, a in enumerate(subs):
            for b in subs[i + 1:]:
                sep = [max(b.os_lo[k] - a.os_hi[k], a.os_lo[k] - b.os_hi[k]) for k in range(3)]
                if max(sep) < 2:
                    raise ValueError("Subgrid outer surfaces must be at least two main cells apart.")

    def _fine_model_index(self, sg: hsg.SubgridStatic):
        """Model-grid index of every subgrid node along each axis (clamped)."""
        q = self.coarsen // sg.ratio
        out = []
        for a in range(3):
            n = sg.yee.shape[a]
            base = (sg.lo[a] - self.pml[2 * a]) * self.coarsen
            idx = base + (np.arange(n) - sg.nb) * q
            out.append(torch.as_tensor(np.clip(idx, 0, self.model_shape[a] - 1),
                                       device=self.device))
        return out

    def box_bounds(self, g: int, surface: str = "inner"):
        """Physical ``(lo, hi)`` corners [m] of subgrid ``g``'s inner/outer surface."""
        sg = self.static.subs[g]
        lo, hi = (sg.lo, sg.hi) if surface == "inner" else (sg.os_lo, sg.os_hi)
        return (np.array([(lo[a] - self.pml[2 * a]) * self.dx for a in range(3)]),
                np.array([(hi[a] - self.pml[2 * a]) * self.dx for a in range(3)]))

    # ------------------------------------------------------------ materials
    def material_fields(self, eps_r, sigma):
        """Differentiable ``(main_eps_pad, main_sig_pad, [sub_eps], [sub_sig])``."""
        R = self.coarsen

        def main(t):
            t = block_average(t, R)
            return F.pad(_ExtendModel.apply(t, self.pml)[None, None], (0, 1, 0, 1, 0, 1),
                         mode="replicate")[0, 0]

        def fine(t, g):
            sg = self.static.subs[g]
            q = R // sg.ratio
            sm = block_average(t, q, stride=1) if q > 1 else t
            i, j, k = self._fine_index[g]
            return sm.index_select(0, i).index_select(1, j).index_select(2, k)

        n = len(self.static.subs)
        return (main(eps_r), main(sigma), [fine(eps_r, g) for g in range(n)],
                [fine(sigma, g) for g in range(n)])

    def build_coefficients(self, main_eps, main_sig, sub_eps, sub_sig) -> hsg.SubgridCoefs:
        ca, cb = fn3.yee_coefficients_3d(main_eps, main_sig, self.dt)
        sca, scb = [], []
        for g, sg in enumerate(self.static.subs):
            a, b = fn3.yee_coefficients_3d(sub_eps[g], sub_sig[g], sg.dt)
            sca.append(a)
            scb.append(b)
        return hsg.SubgridCoefs(ca, cb, sca, scb)

    # ------------------------------------------------------------ points
    def classify(self, x) -> int:
        """Owner of a physical point: subgrid index, ``-1`` for the main grid."""
        x = np.asarray(x, dtype=np.float64)
        for g, sg in enumerate(self.static.subs):
            lo, hi = self.box_bounds(g, "inner")
            m = self.dx / sg.ratio
            if np.all(x > lo + m) and np.all(x < hi - m):
                return g
        for g in range(len(self.static.subs)):
            lo, hi = self.box_bounds(g, "outer")
            if np.all(x > lo - self.dx) and np.all(x < hi + self.dx):
                raise ValueError(f"Point {x} lies between the inner and outer surface of "
                                 f"subgrid {g} (or too close to them).")
        return -1

    def point_ops(self, positions, *, source: bool) -> hsg.PointSet:
        """Operators for physical ``(x, y, z)`` positions ``(nshot, n, 3)`` [m]."""
        pos = np.asarray(torch.as_tensor(positions, dtype=torch.float64).cpu())
        if pos.ndim == 2:
            pos = np.broadcast_to(pos[None], (self.nshot,) + pos.shape)
        if pos.shape[0] != self.nshot or pos.shape[2] != 3:
            raise ValueError("positions must have shape (nshot, n, 3) in metres.")
        c = self.component
        off = np.array(hsg.offsets("e", c))
        n_sub = len(self.static.subs)
        lists = [{"shot": [], "node": [], "item": [], "w": []} for _ in range(n_sub + 1)]
        for s in range(pos.shape[0]):
            for item in range(pos.shape[1]):
                x = pos[s, item]
                g = self.classify(x)
                if g < 0:
                    y, sp = self.static.main, self.dx
                    u = x / self.dx + np.array(self.pml[0::2]) - off
                else:
                    sg = self.static.subs[g]
                    y, sp = sg.yee, self.dx / sg.ratio
                    u = (x / self.dx + np.array(self.pml[0::2]) - np.array(sg.lo)) * sg.ratio \
                        + sg.nb - off
                act = y.act[c]
                i0 = np.floor(u + 1e-9).astype(int)
                t = u - i0
                cell = sp ** 3
                L = lists[g + 1]
                for corner in np.ndindex(2, 2, 2):
                    w = np.prod([t[a] if corner[a] else 1 - t[a] for a in range(3)])
                    if abs(w) < 1e-12:
                        continue
                    ijk = tuple(int(i0[a] + corner[a]) for a in range(3))
                    if not bool(act[ijk]):
                        raise ValueError(f"Point {x} touches an inactive node.")
                    if source:
                        w = w * self.dipole_length / cell
                    L["shot"].append(s)
                    L["node"].append((ijk[0] * y.shape[1] + ijk[1]) * y.shape[2] + ijk[2])
                    L["item"].append(item)
                    L["w"].append(w)

        def pts(L):
            Li = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=self.device)  # noqa: E731
            return hsg.Points(Li(L["shot"]), Li(L["node"]), Li(L["item"]),
                              torch.as_tensor(np.asarray(L["w"], dtype=np.float64),
                                              dtype=self.dtype, device=self.device))
        return hsg.PointSet(pts(lists[0]), [pts(L) for L in lists[1:]], int(pos.shape[1]), c)

    # ------------------------------------------------------------ run
    def fine_amplitudes(self, w: torch.Tensor, src: hsg.PointSet):
        """Subgrid-rate source samples per subgrid (``None`` without subgrid sources)."""
        if not src.has_sub:
            return None
        return [hsg.fine_source_samples(w, sg.time_ratio) if src.sub[g].shot.numel() else None
                for g, sg in enumerate(self.static.subs)]

    def run(self, source_amplitudes, source_positions, receiver_positions, *, coefs=None,
            callback=None, callback_every: int = 0, use_graph: bool = True) -> dict:
        """Run ``nt`` main steps; ``source_amplitudes`` ``(nsrc, nt)`` at the main ``dt``
        (CUDA Graph replay of the step on GPUs unless ``use_graph=False`` or a
        ``callback`` is given)."""
        st = self.static
        c = self.coefs if coefs is None else coefs
        w = torch.as_tensor(source_amplitudes, dtype=self.dtype, device=self.device)
        if w.ndim == 1:
            w = w[None]
        nt = w.shape[1]
        src = self.point_ops(source_positions, source=True)
        rec = self.point_ops(receiver_positions, source=False)
        nsrc = int(np.asarray(source_positions).shape[-2])
        if w.shape[0] == 1 and nsrc > 1:
            w = w.expand(nsrc, -1)
        wf = self.fine_amplitudes(w, src)
        data = torch.zeros((self.nshot, nt, rec.count), dtype=self.dtype, device=self.device)
        extra = 1 if rec.has_sub else 0
        zero = torch.zeros_like(w[:, 0])
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        graph = None
        if use_graph and self.cuda_kernels and callback is None:
            graph = graphs.make(st, c, src, rec, w.shape[0])
        state = hsg.zero_state(st, rec) if graph is None else None
        with torch.no_grad():
            for n in range(nt + extra):
                amp = w[:, n] if n < nt else zero
                af = None if wf is None else [None if x is None else hsg.fine_step_samples(
                    x, n, sg.time_ratio) for x, sg in zip(wf, st.subs)]
                if graph is not None:
                    dm, dfine = graph(amp, af)
                else:
                    state = hsg.step(st, c, state, src, amp, af, rec)
                    dm, dfine = hsg.sample(st, rec, state)
                if n < nt:
                    data[:, n] += dm
                if n >= 1 and extra:
                    data[:, n - 1] += dfine
                if callback is not None and callback_every and n % callback_every == 0:
                    callback(n, state)
        if graph is not None:
            state = graph.snapshot()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.state = state
        return {"receivers": data, "seconds": time.perf_counter() - t0, "dt": self.dt,
                "steps": nt + extra}

    def summary(self) -> dict:
        st = self.static
        n_main = int(np.prod(st.main.shape))
        n_sub = [int(np.prod(sg.yee.shape)) for sg in st.subs]
        return {"dt": self.dt, "main_shape": st.main.shape, "main_spacing": self.dx,
                "model_spacing": self.h, "coarsen": self.coarsen,
                "subgrids": [dict(ratio=sg.ratio, shape=sg.yee.shape, lo=sg.lo, hi=sg.hi,
                                  os_lo=sg.os_lo, os_hi=sg.os_hi, nb=sg.nb) for sg in st.subs],
                "main_nodes": n_main, "sub_nodes": n_sub,
                "node_updates_per_step": n_main + sum(n * sg.time_ratio for n, sg in zip(n_sub, st.subs)),
                "setup_seconds": self.setup_seconds}


__all__ = ["Subgrid", "SubgridMaxwell3D", "block_average"]
