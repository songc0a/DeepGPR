r"""Huygens subgridding (HSG) of DeepGPR's 3D Yee scheme.

A *subgrid* is a complete Yee grid with spacing ``dx / r`` (``r`` odd) and time
step ``dt / r``, terminated by its own CPML, placed over a box of the main grid.
The two grids are coupled through two Huygens (total-field/scattered-field)
surfaces: Berenger's Huygens subgridding (IEEE Trans. Antennas Propag. 54(12),
3797-3804, 2006, doi:10.1109/TAP.2006.886519) in the arrangement of gprMax's
``gprMax/subgrids`` code (subgrid CPML, inner/outer surfaces ``is_os_sep`` main
cells apart, normal precursor filter, odd ratios, ``hsg_2``/``hsg_1`` order).
The *switched* Huygens subgridding of Hartley, Giannopoulos and Davidson (IEEE
Trans. Antennas Propag. 70(8), 6872-6882, 2022, doi:10.1109/TAP.2022.3161371),
whose stabilisation needs no subgrid PML, is **not** implemented here; the
stabilisation is the subgrid CPML and the filter. The two surfaces are:

* **Inner surface (IS)**, the working-region box ``[i0, i1] x [j0, j1] x [k0, k1]``
  (main E-node planes). In the subgrid the IS encloses the *total* field; outside
  it (up to the subgrid CPML) the subgrid carries the *scattered* field, the
  difference between its own field and the main-grid field. The main field
  interpolated to the subgrid (the *precursors*) is injected as the incident
  field of a TF/SF surface.
* **Outer surface (OS)**, the box enlarged by ``is_os_sep`` main cells. Inside the
  OS the main grid carries only the incident field (the subgrid's response is
  absent); the subgrid's scattered field, sampled at the coinciding fine nodes
  (``r`` odd), is injected through a SF/TF surface, so outside the OS the main
  grid carries the total field.

Both surfaces use the same rule. For a Yee difference ``d = f[q+] - f[q-]`` in
the update of a node ``u`` whose two differenced nodes lie on different sides of
a surface, the differenced value is converted to the field type of ``u``::

    IS (fine grid):   d += s_face * inc(q)      inc = interpolated main field
    OS (main grid):   d -= s_face * scat(q)     scat = fine (scattered) field

(``s_face = -1`` on the lower, ``+1`` on the upper face of the box, ``q`` the
differenced node across the surface), and the update gets
``rhs_c += sign_t d_corr / h_a`` for its curl term ``t = (c, a, f, sign_t)``
(:data:`~DeepGPR.hybrid.functional3d.TERMS`). Edge and corner nodes are corrected
once per crossing difference, which is the face-by-face rule of gprMax.

Precursors (main -> fine): tangential ``E`` on the IS planes and tangential
``H`` half a fine cell outside them, by tensor-product linear interpolation of
the main lattice of the same component (the main nodes just outside the box are
used, so the fine nodes near the box edges are interpolated rather than
extrapolated); with ``filter=True`` the main field is first smoothed along the
surface normal with ``(1/4, 1/2, 1/4)`` (gprMax's stabilising filter). In time
they are linear between the two main levels that bracket each fine level.

Time stepping (main E at ``n``, H at ``n - 1/2``; subgrid E at
``n - (r - 1)/(2 r)``, H at ``n - 1/2``; ``h = (r - 1)/2``)::

    main H update                        -> H^(n+1/2)
    subgrid: h x (H, E) + H              (H^(n+1/2) - 1/(2r)... -> subgrid E at n, H at n + 1/(2r))
    OS H correction (subgrid E at n)
    main E update, main sources          -> E^(n+1)
    subgrid: h x (E, H) + E              (subgrid E at n + 1 - h/r, H at n + 1/2)
    OS E correction (subgrid H at n + 1/2)

which is gprMax's ``hsg_2`` / ``hsg_1`` choreography. Every operation is linear
in the fields and differentiable in the material coefficients of both grids, so
reverse-mode differentiation of :func:`step` is the exact discrete adjoint.

The Yee updates are DeepGPR's (:mod:`DeepGPR.hybrid.functional3d`, pure Yee
mode with CPML on every face), on the fused CUDA kernels of
:mod:`DeepGPR.hybrid.cuda3d` when available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config.constants import SPEED_OF_LIGHT, VACUUM_PERMEABILITY
from ..hybrid import cuda3d
from ..hybrid import functional3d as fn3
from ..solver.pml import build_pml_coeffs

TERMS = fn3.TERMS
#: Order of the curl terms in every correction and precursor vector: the two
#: terms of one component (which share targets at box edges for E) are in
#: different halves.
TERM_ORDER = (0, 2, 4, 1, 3, 5)
#: Normal-direction smoothing of the precursors (gprMax ``filter=True``).
FILTER = (0.25, 0.5, 0.25)
#: Main-grid state tensors: fe[3], fh[3], psi_e[6], psi_h[6].
MAIN_STATE = 18
#: Subgrid state tensors: fe[3], fh[3], psi_e[6], psi_h[6], E precursors of the
#: previous main level, receiver buffer.
SUB_STATE = 20


def offsets(kind: str, comp: int) -> Tuple[float, float, float]:
    """Half-cell offsets of component ``comp`` of ``kind`` (``"e"``/``"h"``)."""
    if kind == "e":
        return tuple(0.5 if a == comp else 0.0 for a in range(3))
    return tuple(0.0 if a == comp else 0.5 for a in range(3))


def _rng(lo: int, hi: int, off: float) -> np.ndarray:
    """Indices of the nodes (offset ``off``) inside the closed box ``[lo, hi]``."""
    return np.arange(lo, hi + 1) if off == 0.0 else np.arange(lo, hi)


def face_entries(lo, hi, kind: str, term: int, side: int):
    """Nodes of one surface correction.

    For curl term ``term = (c, a, f, sign)`` of the update of ``kind`` and the face
    of the box ``[lo, hi]`` normal to ``a`` (``side`` -1 lower, +1 upper): the
    updated nodes ``u`` (component ``c`` of ``kind``) and the differenced nodes
    ``q`` across the face (component ``f`` of the other kind), both as
    ``(n, 3)`` index arrays (same in-plane indices)."""
    c, a, f, _ = TERMS[term]
    off_u = offsets(kind, c)
    axes = [None, None, None]
    for b in (c, f):
        axes[b] = _rng(lo[b], hi[b], off_u[b])
    if kind == "h":           # u: H_c half a cell outside the face; q: E_f on the face
        nu = lo[a] - 1 if side < 0 else hi[a]
        nq = lo[a] if side < 0 else hi[a]
    else:                     # u: E_c on the face; q: H_f half a cell outside
        nu = lo[a] if side < 0 else hi[a]
        nq = lo[a] - 1 if side < 0 else hi[a]
    axes_u, axes_q = list(axes), list(axes)
    axes_u[a] = np.array([nu])
    axes_q[a] = np.array([nq])
    grid = lambda ax: np.stack([g.reshape(-1) for g in np.meshgrid(*ax, indexing="ij")], 1)  # noqa: E731
    return grid(axes_u), grid(axes_q)


def flat(ijk: np.ndarray, shape) -> np.ndarray:
    return (ijk[:, 0] * shape[1] + ijk[:, 1]) * shape[2] + ijk[:, 2]


def _lin_taps(u: np.ndarray, filt: bool):
    """1D linear interpolation taps (``index``, ``weight``), ``(n, T)``, at lattice
    coordinates ``u``; ``filt``: convolved with :data:`FILTER`."""
    i = np.floor(u + 1e-9).astype(np.int64)
    t = u - i
    t[t < 1e-9] = 0.0
    if not filt:
        return np.stack([i, i + 1], 1), np.stack([1.0 - t, t], 1)
    w = np.stack([0.25 * (1 - t), 0.5 * (1 - t) + 0.25 * t, 0.25 * (1 - t) + 0.5 * t, 0.25 * t], 1)
    return np.stack([i - 1, i, i + 1, i + 2], 1), w


class TapOp:
    """``y[:, n] = sum_T w[n, T] x[:, idx[n, T]]`` (fixed tap pattern, batched)."""

    def __init__(self, idx: np.ndarray, w: np.ndarray, n_in: int, dtype, device):
        keep = np.abs(w).max(axis=0) > 0 if w.size else np.zeros(0, bool)
        idx, w = idx[:, keep], w[:, keep]
        self.n, self.T = idx.shape
        self.n_in = int(n_in)
        if self.n and (idx.min() < 0 or idx.max() >= n_in):
            raise ValueError("Precursor stencil leaves the main grid.")
        self.idx = torch.as_tensor(idx.reshape(-1), dtype=torch.long, device=device)
        self.w = torch.as_tensor(w, dtype=dtype, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        S = x.shape[0]
        if self.n == 0:
            return x.new_zeros((S, 0))
        return (x.reshape(S, -1).index_select(1, self.idx).view(S, self.n, self.T) * self.w).sum(-1)

    def dense(self) -> np.ndarray:
        a = np.zeros((self.n, self.n_in))
        rows = np.repeat(np.arange(self.n), self.T)
        np.add.at(a, (rows, self.idx.cpu().numpy()), self.w.cpu().numpy().reshape(-1))
        return a


@dataclass
class Correction:
    """Surface corrections of one field kind of one grid, grouped by curl term.

    For term ``t``: ``field[c_t].flat[idx_t] += coef_t * (cb[idx_t]) * values[:, sl_t]``
    (``cb`` only for E corrections), with ``values`` the precursor vector (IS) or
    the gathered subgrid values (OS: component ``src_comp[t]`` at ``src_idx[t]``)."""

    comp: List[int]
    idx: List[torch.Tensor]
    coef: List[torch.Tensor]
    slices: List[Tuple[int, int]]
    use_cb: bool
    src_comp: List[int] = field(default_factory=list)
    src_idx: List[torch.Tensor] = field(default_factory=list)

    @property
    def size(self) -> int:
        return self.slices[-1][1] if self.slices else 0


@dataclass
class SubgridStatic:
    """Static data of one subgrid (fine Yee grid, its surfaces and precursors)."""

    ratio: int
    yee: fn3.YeeStatic3D
    dt: float
    lo: Tuple[int, int, int]          # IS box in main indices
    hi: Tuple[int, int, int]
    os_lo: Tuple[int, int, int]       # OS box in main indices
    os_hi: Tuple[int, int, int]
    nb: int                           # fine index of the lower IS planes
    is_h: Correction                  # fine H from incident E
    is_e: Correction                  # fine E from incident H
    os_h: Correction                  # main H from fine (scattered) E
    os_e: Correction                  # main E from fine (scattered) H
    pe: List[TapOp]                   # per term: main E_f -> incident E (is_h order)
    ph: List[TapOp]                   # per term: main H_f -> incident H (is_e order)
    pe_comp: List[int]
    ph_comp: List[int]
    filter: bool = True
    cuda: Optional[object] = None     # :class:`~DeepGPR.subgrid.cuda.SubgridCudaOps`
    time_ratio: int = 0               # sub-steps per main step (``ratio`` or 1; 0 = ratio)

    def __post_init__(self):
        if not self.time_ratio:
            self.time_ratio = self.ratio


@dataclass
class Points:
    """Point sources/receivers of one grid: ``(shot, node, item, w)``."""

    shot: torch.Tensor
    node: torch.Tensor
    item: torch.Tensor
    w: torch.Tensor


@dataclass
class PointSet:
    """Points on the main grid and on every subgrid (component ``comp``)."""

    main: Points
    sub: List[Points]
    count: int
    comp: int

    @property
    def has_sub(self) -> bool:
        return any(p.shot.numel() for p in self.sub)


@dataclass
class SubgridStepStatic:
    nshot: int
    main: fn3.YeeStatic3D
    subs: List[SubgridStatic]
    dtype: torch.dtype = torch.float64
    device: torch.device = torch.device("cpu")


@dataclass
class SubgridCoefs:
    """Material coefficients ``ca``/``cb`` of the main grid and every subgrid."""

    main_ca: torch.Tensor
    main_cb: torch.Tensor
    sub_ca: List[torch.Tensor]
    sub_cb: List[torch.Tensor]

    def tensors(self) -> List[torch.Tensor]:
        return [self.main_ca, self.main_cb] + list(self.sub_ca) + list(self.sub_cb)

    def replaced(self, tensors: Sequence[torch.Tensor]) -> SubgridCoefs:
        t = list(tensors)
        n = len(self.sub_ca)
        return SubgridCoefs(t[0], t[1], t[2:2 + n], t[2 + n:2 + 2 * n])


# --------------------------------------------------------------------------- static
def yee_grid(shape, spacing, pml, dt, dtype, device, nshot, cuda: bool) -> fn3.YeeStatic3D:
    """Pure Yee static (DeepGPR CPML on the faces with ``pml`` cells, mu_r = 1)."""
    coeffs = build_pml_coeffs(None, None, dt, list(spacing), *(n - 1 for n in shape),
                              torch.tensor(list(pml), dtype=torch.int32), device, dtype)
    chr = torch.full(tuple(shape), dt / VACUUM_PERMEABILITY, dtype=dtype, device=device)
    y = fn3.yee_static_3d(tuple(shape), tuple(spacing), list(pml), coeffs[6:], chr, None,
                          dtype, device)
    if cuda:
        cuda3d.attach_yee(y, nshot, device)
    return y


def build_subgrid(main: fn3.YeeStatic3D, lo, hi, ratio: int, is_os_sep: int, pml_cells: int,
                  pml_sep: Optional[int], dt_main: float, filt: bool, dtype, device, nshot,
                  cuda: bool, bounds=None, time_ratio: Optional[int] = None) -> SubgridStatic:
    """Fine grid, surfaces and precursors of a subgrid with IS box ``[lo, hi]``
    (main E-node indices). ``bounds``: per axis the first and last main E-node plane
    the subgrid (with its CPML) may cover (default: outside the main CPML).
    ``time_ratio``: ``ratio`` (sub-cycling, ``dt / r``; default) or 1 (the main ``dt``,
    allowed when it satisfies the subgrid's CFL limit, e.g. under a DG band)."""
    r = int(ratio)
    rt = r if time_ratio is None else int(time_ratio)
    if rt not in (1, r):
        raise ValueError("time_ratio must be 1 or the spatial ratio.")
    if r < 3 or r % 2 == 0:
        raise ValueError("The subgrid ratio must be an odd integer >= 3.")
    s = int(is_os_sep)
    if s < (2 if filt else 1):
        raise ValueError("is_os_sep must be >= 2 with the precursor filter (>= 1 without).")
    sep = r // 2 + 2 if pml_sep is None else int(pml_sep)
    if sep < r // 2 + 1:
        raise ValueError("pml_sep must leave the OS stencil outside the subgrid CPML.")
    lo, hi = tuple(int(v) for v in lo), tuple(int(v) for v in hi)
    if any(b - a < 1 for a, b in zip(lo, hi)):
        raise ValueError("The subgrid box must span at least one main cell per axis.")
    NX, NY, NZ = main.shape
    os_lo = tuple(v - s for v in lo)
    os_hi = tuple(v + s for v in hi)
    nb = s * r + sep + int(pml_cells)
    pad = -(-nb // r) + 1                           # main cells covered beyond the IS
    for a in range(3):
        p_lo, p_hi = _main_interior(main, a) if bounds is None else bounds[a]
        if lo[a] - pad < p_lo or hi[a] + pad > p_hi:
            raise ValueError(f"Subgrid (with its CPML) must lie inside the main interior "
                             f"along axis {a}: needs [{lo[a] - pad}, {hi[a] + pad}] within "
                             f"[{p_lo}, {p_hi}].")
    shape_f = tuple((b - a) * r + 2 * nb + 1 for a, b in zip(lo, hi))
    spacing_f = tuple(v / r for v in main.spacing)
    dt_f = dt_main / rt
    limit = 1.0 / (SPEED_OF_LIGHT * math.sqrt(sum(1.0 / s_ ** 2 for s_ in spacing_f)))
    if dt_f > limit * (1 + 1e-9):
        raise ValueError(f"Subgrid time step {dt_f:.3e} s exceeds its CFL limit {limit:.3e} s "
                         "(use time_ratio = ratio).")
    yf = yee_grid(shape_f, spacing_f, [int(pml_cells)] * 6, dt_f, dtype, device, nshot, cuda)

    lo_f = (nb, nb, nb)
    hi_f = tuple(nb + (b - a) * r for a, b in zip(lo, hi))
    chr_f = dt_f / VACUUM_PERMEABILITY
    chr_m = dt_main / VACUUM_PERMEABILITY
    T = lambda a, dt=torch.long: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    Tf = lambda a: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731

    def main_coord(ijk_f, off):
        return np.asarray(lo)[None] + (ijk_f + np.asarray(off)[None] - nb) / r

    def fine_index(ijk_m, off):
        o = np.asarray(off)[None]
        return nb + (ijk_m - np.asarray(lo)[None]) * r + (o * (r - 1)).astype(np.int64)

    is_h = Correction([], [], [], [], False)
    is_e = Correction([], [], [], [], True)
    os_h = Correction([], [], [], [], False)
    os_e = Correction([], [], [], [], True)
    pe, ph, pe_comp, ph_comp = [], [], [], []
    for corr, kind, box, grid in ((is_h, "h", (lo_f, hi_f), "fine"), (is_e, "e", (lo_f, hi_f), "fine"),
                                  (os_h, "h", (os_lo, os_hi), "main"),
                                  (os_e, "e", (os_lo, os_hi), "main")):
        start = 0
        for t in TERM_ORDER:
            c, a, f, sign = TERMS[t]
            us, qs, coefs = [], [], []
            for side in (-1, 1):
                u, q = face_entries(box[0], box[1], kind, t, side)
                us.append(u)
                qs.append(q)
                if grid == "fine":     # IS: d += s inc
                    k = (-chr_f if kind == "h" else 1.0) * sign * side / spacing_f[a]
                else:                  # OS: d -= s scat
                    k = (chr_m if kind == "h" else -1.0) * sign * side / main.spacing[a]
                coefs.append(np.full(len(u), k))
            u, q, k = np.concatenate(us), np.concatenate(qs), np.concatenate(coefs)
            shape_u = shape_f if grid == "fine" else main.shape
            corr.comp.append(c)
            corr.idx.append(T(flat(u, shape_u)))
            corr.coef.append(Tf(k))
            corr.slices.append((start, start + len(u)))
            start += len(u)
            okind = "e" if kind == "h" else "h"
            off_q = offsets(okind, f)
            if grid == "fine":
                # precursor: interpolate the main lattice of (okind, f) at q's position
                X = main_coord(q, off_q) - np.asarray(off_q)[None]
                taps = [_lin_taps(X[:, b], filt and b == a) for b in range(3)]
                idx, w = _tensor_taps(taps, main.shape)
                op = TapOp(idx, w, int(np.prod(main.shape)), dtype, device)
                (pe if kind == "h" else ph).append(op)
                (pe_comp if kind == "h" else ph_comp).append(f)
            else:
                corr.src_comp.append(f)
                corr.src_idx.append(T(flat(fine_index(q, off_q), shape_f)))
    _check_targets(yf, is_h, is_e)
    _check_targets(main, os_h, os_e)
    sg = SubgridStatic(r, yf, dt_f, lo, hi, os_lo, os_hi, nb, is_h, is_e, os_h, os_e, pe, ph,
                       pe_comp, ph_comp, filt, time_ratio=rt)
    if cuda:
        from . import cuda as sg_cuda
        sg_cuda.attach(sg, main.shape, dtype, device)
    return sg


def _main_interior(main: fn3.YeeStatic3D, axis: int) -> Tuple[int, int]:
    """First and last main E-node plane outside the CPML along ``axis``."""
    n = main.shape[axis]
    lo, hi = 1, n - 2
    if main.pe[axis] is not None:
        idx = main.pe[axis][4].cpu().numpy()
        low = idx[idx < n // 2]
        high = idx[idx >= n // 2]
        if low.size:
            lo = int(low.max()) + 1
        if high.size:
            hi = int(high.min()) - 1
    return lo, hi


def _tensor_taps(taps, shape):
    (ix, wx), (iy, wy), (iz, wz) = taps
    idx = ((ix[:, :, None, None] * shape[1] + iy[:, None, :, None]) * shape[2]
           + iz[:, None, None, :])
    w = wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
    n = idx.shape[0]
    return idx.reshape(n, -1), w.reshape(n, -1)


def _check_targets(y: fn3.YeeStatic3D, corr_h: Correction, corr_e: Correction):
    for corr, base in ((corr_h, 3), (corr_e, 0)):
        for c, idx in zip(corr.comp, corr.idx):
            if idx.numel() and not bool(y.act[base + c].reshape(-1)[idx].all()):
                raise ValueError("A Huygens-surface correction touches an inactive node.")


# --------------------------------------------------------------------------- state
def zero_state(st: SubgridStepStatic, recs: Optional[PointSet] = None) -> Tuple[torch.Tensor, ...]:
    """Zero state: main Yee (18 tensors), then per subgrid :data:`SUB_STATE` tensors."""
    S = st.nshot
    kw = dict(dtype=st.dtype, device=st.device)
    out = list(fn3.zero_state(fn3.StepStatic3D(S, st.main, None, [], st.dtype, st.device))[:18])
    for g, sg in enumerate(st.subs):
        out += list(fn3.zero_state(fn3.StepStatic3D(S, sg.yee, None, [], st.dtype, st.device))[:18])
        out.append(torch.zeros((S, sg.is_h.size), **kw))
        n = 0 if recs is None else recs.sub[g].shot.numel()
        out.append(torch.zeros((1, n), **kw))       # receiver buffer (flat point order)
    return tuple(out)


# --------------------------------------------------------------------------- Yee halves
_EMPTY = {}


def _empty(ref):
    key = (ref.dtype, ref.device)
    if key not in _EMPTY:
        _EMPTY[key] = torch.zeros(0, dtype=ref.dtype, device=ref.device)
    return _EMPTY[key]


def yee_h(y: fn3.YeeStatic3D, fe, fh, psh):
    """``H <- H - (dt/mu) curl E`` (CPML), fused kernel or PyTorch."""
    if y.cuda is not None and fe[0].is_cuda:
        fh, psh, _ = y.cuda.h_update(list(fe), list(fh), list(psh), _empty(fe[0]))
        return fh, psh
    rhs, psh = fn3._yee_curl(y, fe, y.ph, psh, fn3._fwd)
    return [torch.where(y.act[3 + k], fh[k] - y.chr * rhs[k], fh[k]) for k in range(3)], psh


def yee_e(y: fn3.YeeStatic3D, fh, fe, pse, ca, cb):
    """``E <- ca E + cb curl H`` (CPML), fused kernel or PyTorch."""
    if y.cuda is not None and fe[0].is_cuda:
        c = SimpleNamespace(fd_ca=ca, fd_cb=cb, fu_hist=None, fu_new=None, fu_old=None)
        fe, pse, _ = y.cuda.e_update(list(fh), list(fe), list(pse), _empty(fe[0]), c, None)
        return fe, pse
    rhs, pse = fn3._yee_curl(y, fh, y.pe, pse, fn3._bwd)
    return [torch.where(y.act[k], ca * fe[k] + cb * rhs[k], fe[k]) for k in range(3)], pse


def apply_correction(fields, corr: Correction, values: torch.Tensor, cb=None):
    """In place: ``fields[c_t].flat[idx_t] += coef_t (cb) values[:, slice_t]``."""
    S = values.shape[0]
    for c, idx, coef, (a, b) in zip(corr.comp, corr.idx, corr.coef, corr.slices):
        if b == a:
            continue
        v = values[:, a:b] * coef
        if cb is not None:
            v = v * cb.reshape(-1).index_select(0, idx)
        fields[c].view(S, -1).index_add_(1, idx, v)


def gather_sources(fields, corr: Correction) -> torch.Tensor:
    """OS values: subgrid fields at the coinciding fine nodes, in correction order."""
    S = fields[0].shape[0]
    parts = [fields[f].reshape(S, -1).index_select(1, idx)
             for f, idx in zip(corr.src_comp, corr.src_idx)]
    return torch.cat(parts, 1)


def precursors(ops: List[TapOp], comps: List[int], fields) -> torch.Tensor:
    return torch.cat([op(fields[c]) for op, c in zip(ops, comps)], 1)


def _fused(sg: SubgridStatic, ref: torch.Tensor) -> bool:
    return sg.cuda is not None and ref.is_cuda


def pre(sg: SubgridStatic, kind: str, fields) -> torch.Tensor:
    """Precursor vector of ``kind`` (``"e"``: incident E for the IS H corrections,
    ``"h"``: incident H for the IS E corrections) from the main fields."""
    if _fused(sg, fields[0]):
        return sg.cuda.precursors(kind, fields)
    return precursors(sg.pe, sg.pe_comp, fields) if kind == "e" else precursors(sg.ph, sg.ph_comp,
                                                                              fields)


def correct_is(sg: SubgridStatic, kind: str, fields, V0, V1=None, w: float = 1.0, cb=None):
    """IS correction with the incident vector ``(1 - w) V0 + w V1`` (``V1=None``: ``V0``)."""
    corr = sg.is_h if kind == "h" else sg.is_e
    if V1 is None:
        a, b = 1.0, 0.0
    else:
        a, b = 1.0 - w, w
    if _fused(sg, fields[0]):
        return sg.cuda.correct_is(kind, fields, V0, V1, a, b, cb)
    apply_correction(fields, corr, V0 if V1 is None else a * V0 + b * V1, cb)
    return fields


def correct_os(sg: SubgridStatic, kind: str, fields, src, cb=None):
    """OS correction of the main fields from the subgrid fields ``src``."""
    corr = sg.os_h if kind == "h" else sg.os_e
    if _fused(sg, fields[0]):
        return sg.cuda.correct_os(kind, fields, src, cb)
    apply_correction(fields, corr, gather_sources(src, corr), cb)
    return fields


def inject(fe, pts: Points, comp: int, cb, amp):
    """``E_comp -= cb w amp`` at the points (DeepGPR convention ``J = w dl delta``)."""
    if not pts.shot.numel():
        return fe
    fe = list(fe)
    f = fe[comp]
    vals = cb.reshape(-1).index_select(0, pts.node) * pts.w * amp.index_select(0, pts.item)
    # a fresh base tensor (not a view): later in-place corrections may act on it
    out = f.clone()
    out.view(-1).index_add_(0, pts.shot * f[0].numel() + pts.node, -vals)
    fe[comp] = out
    return fe


def _record(fe, pts: Points, comp: int, S: int) -> torch.Tensor:
    f = fe[comp]
    if not pts.shot.numel():
        return f.new_zeros((1, 0))
    v = f.reshape(-1).index_select(0, pts.shot * f[0].numel() + pts.node) * pts.w
    return v.view(1, -1)


# --------------------------------------------------------------------------- step
def unpack_subgrids(state, n_sub: int, base: int = MAIN_STATE):
    """Per subgrid ``[fe, fh, psi_e, psi_h, pe_prev, rec_buffer]`` from a flat state."""
    subs = []
    for g in range(n_sub):
        b = base + SUB_STATE * g
        subs.append([list(state[b:b + 3]), list(state[b + 3:b + 6]), list(state[b + 6:b + 12]),
                     list(state[b + 12:b + 18]), state[b + 18], state[b + 19]])
    return subs


def subgrids_first_half(sgs: List[SubgridStatic], c: SubgridCoefs, subs, fh, pe_cur, ph_prev,
                        ph_cur, src, amp_fine, rec, S: int):
    """``hsg_2``: every subgrid from E ``n - h/r``, H ``n - 1/2`` to E ``n``, H ``n + 1/(2r)``
    (main H already at ``n + 1/2``), then the OS H correction of the main ``fh``.
    Updates ``subs`` in place; returns ``(fh, new receiver buffers)``."""
    new_rec = []
    for g, sg in enumerate(sgs):
        r = sg.time_ratio
        h = (r - 1) // 2
        sfe, sfh, spe, sph, pe_prev, _ = subs[g]
        ca, cb = c.sub_ca[g], c.sub_cb[g]
        for k in range(1, h + 1):
            sfh, sph = yee_h(sg.yee, sfe, sfh, sph)
            sfh = correct_is(sg, "h", sfh, pe_prev, pe_cur[g], (k + h) / r)
            sfe, spe = yee_e(sg.yee, sfh, sfe, spe, ca, cb)
            sfe = correct_is(sg, "e", sfe, ph_prev[g], ph_cur[g], k / r, cb)
            sfe = _fine_source(sfe, src, g, cb, amp_fine, k - 1)
        sfh, sph = yee_h(sg.yee, sfe, sfh, sph)
        sfh = correct_is(sg, "h", sfh, pe_cur[g])
        new_rec.append(_record(sfe, rec.sub[g], rec.comp, S) if rec is not None
                       else subs[g][5])
        fh = correct_os(sg, "h", fh, sfe)
        subs[g][0:4] = [sfe, sfh, spe, sph]
    return fh, new_rec


def subgrids_second_half(sgs: List[SubgridStatic], c: SubgridCoefs, subs, fe, main_cb, pe_cur,
                         pe_next, ph_prev, ph_cur, src, amp_fine):
    """``hsg_1``: every subgrid to E ``n + 1 - h/r``, H ``n + 1/2`` (main E already at
    ``n + 1``), then the OS E correction of the main ``fe``. Returns ``fe``."""
    for g, sg in enumerate(sgs):
        r = sg.time_ratio
        h = (r - 1) // 2
        sfe, sfh, spe, sph = subs[g][0:4]
        ca, cb = c.sub_ca[g], c.sub_cb[g]
        for k in range(1, h + 1):
            sfe, spe = yee_e(sg.yee, sfh, sfe, spe, ca, cb)
            sfe = correct_is(sg, "e", sfe, ph_prev[g], ph_cur[g], (k + h) / r, cb)
            sfe = _fine_source(sfe, src, g, cb, amp_fine, h + k - 1)
            sfh, sph = yee_h(sg.yee, sfe, sfh, sph)
            sfh = correct_is(sg, "h", sfh, pe_cur[g], pe_next[g], k / r)
        sfe, spe = yee_e(sg.yee, sfh, sfe, spe, ca, cb)
        sfe = correct_is(sg, "e", sfe, ph_cur[g], None, 1.0, cb)
        sfe = _fine_source(sfe, src, g, cb, amp_fine, r - 1)
        fe = correct_os(sg, "e", fe, sfh, main_cb)
        subs[g][0:4] = [sfe, sfh, spe, sph]
    return fe


def pack_subgrids(subs, pe_cur, new_rec) -> list:
    out = []
    for g, sub in enumerate(subs):
        sfe, sfh, spe, sph = sub[0:4]
        out += list(sfe) + list(sfh) + list(spe) + list(sph) + [pe_cur[g], new_rec[g]]
    return out


def step(st: SubgridStepStatic, c: SubgridCoefs, state, src: Optional[PointSet] = None,
         amp=None, amp_fine=None, rec: Optional[PointSet] = None, pert=None):
    """One main time step of the subgridded system.

    ``amp``: ``(n_items,)`` main-rate samples (main-grid sources), ``amp_fine``:
    ``(n_items, r)`` per subgrid list of the subgrid-rate samples of this step
    (see :func:`fine_source_samples`). Receivers inside subgrids (``rec``) are
    recorded at main time ``n`` into the state (read by :func:`sample`). ``pert``:
    ``(dH, dE)``, three tensors each added to the main H and E updates (zero in
    value; their cotangents weight the error indicators of :mod:`.indicators`)."""
    S = st.nshot
    m = st.main
    fe, fh = list(state[0:3]), list(state[3:6])
    pse, psh = list(state[6:12]), list(state[12:18])
    subs = unpack_subgrids(state, len(st.subs))
    pe_cur = [pre(sg, "e", fe) for sg in st.subs]
    ph_prev = [pre(sg, "h", fh) for sg in st.subs]

    fh, psh = yee_h(m, fe, fh, psh)
    if pert is not None:
        fh = [f + d for f, d in zip(fh, pert[0])]
    ph_cur = [pre(sg, "h", fh) for sg in st.subs]
    fh, new_rec = subgrids_first_half(st.subs, c, subs, fh, pe_cur, ph_prev, ph_cur, src,
                                      amp_fine, rec, S)

    fe, pse = yee_e(m, fh, fe, pse, c.main_ca, c.main_cb)
    if pert is not None:
        fe = [f + d for f, d in zip(fe, pert[1])]
    if src is not None and amp is not None:
        fe = inject(fe, src.main, src.comp, c.main_cb, amp)
    pe_next = [pre(sg, "e", fe) for sg in st.subs]
    fe = subgrids_second_half(st.subs, c, subs, fe, c.main_cb, pe_cur, pe_next, ph_prev, ph_cur,
                              src, amp_fine)
    return tuple(list(fe) + list(fh) + list(pse) + list(psh) + pack_subgrids(subs, pe_cur, new_rec))


def _fine_source(sfe, src: Optional[PointSet], g: int, cb, amp_fine, j: int):
    if src is None or amp_fine is None or not src.sub[g].shot.numel():
        return sfe
    return inject(sfe, src.sub[g], src.comp, cb, amp_fine[g][:, j])


def sample(st: SubgridStepStatic, rec: PointSet, state):
    """``(main, fine)`` receiver values ``(S, count)``: main-grid receivers at
    ``n + 1`` (after the step), subgrid receivers at ``n`` (recorded in the step)."""
    S = st.nshot
    fe = state[0:3]
    ref = fe[rec.comp]
    out_m = ref.new_zeros((S * rec.count,))
    p = rec.main
    if p.shot.numel():
        f = fe[rec.comp]
        vals = f.reshape(-1).index_select(0, p.shot * f[0].numel() + p.node) * p.w
        out_m = out_m.index_add(0, p.shot * rec.count + p.item, vals)
    out_f = ref.new_zeros((S * rec.count,))
    for g in range(len(st.subs)):
        p = rec.sub[g]
        if p.shot.numel():
            buf = state[MAIN_STATE + SUB_STATE * g + 19].reshape(-1)
            out_f = out_f.index_add(0, p.shot * rec.count + p.item, buf)
    return out_m.view(S, rec.count), out_f.view(S, rec.count)


def fine_source_samples(w: torch.Tensor, r: int) -> torch.Tensor:
    """Band-limited (FFT) resampling of main-rate samples ``w (n, nt)`` at ``dt``
    to ``dt / r``: ``(n, (nt - 1) r + 1)`` with ``out[:, n r] = w[:, n]``."""
    n, nt = w.shape
    if r == 1:
        return w
    dev, dt = w.device, w.dtype
    w = w.to("cpu", torch.float64)       # tiny; avoids cuFFT plan failures on some GPUs
    L = 2 * nt
    spec = torch.fft.rfft(torch.nn.functional.pad(w, (0, L - nt)), dim=-1)
    k = spec.shape[-1]
    # split the Nyquist bin of the coarse spectrum (L even)
    spec = torch.cat([spec[:, :k - 1], 0.5 * spec[:, k - 1:]], 1)
    full = torch.nn.functional.pad(spec, (0, L * r // 2 + 1 - k))
    up = torch.fft.irfft(full, n=L * r, dim=-1) * r
    return up[:, :(nt - 1) * r + 1].to(dev, dt)


def fine_step_samples(w_fine: torch.Tensor, n: int, r: int) -> torch.Tensor:
    """Subgrid samples of main step ``n``: ``w_fine[:, (n - 1) r + 1 + j]``, ``j < r``
    (zero before the first sample), so that the subgrid source current has the
    main grid's half-step delay convention."""
    start = (n - 1) * r + 1
    idx = torch.arange(start, start + r, device=w_fine.device)
    ok = (idx >= 0) & (idx < w_fine.shape[1])
    vals = w_fine.index_select(1, idx.clamp(0, w_fine.shape[1] - 1))
    return vals * ok.to(vals.dtype)


__all__ = [
    "Correction", "FILTER", "MAIN_STATE", "PointSet", "Points", "SUB_STATE", "SubgridCoefs",
    "SubgridStatic", "SubgridStepStatic", "TapOp", "apply_correction", "build_subgrid",
    "face_entries", "fine_source_samples", "fine_step_samples", "gather_sources", "inject",
    "offsets", "pre", "precursors", "sample", "step", "yee_e", "yee_grid", "yee_h", "zero_state",
]
