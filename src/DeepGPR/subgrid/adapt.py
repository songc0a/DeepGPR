r"""Adaptive grid sizes: where to refine and by how much (a priori refinement plan).

From a model on the model grid (spacing ``h``) and the highest frequency of
interest ``f_max``, :func:`plan_refinement` chooses

1. the main spacing ``dx = coarsen * h`` (``coarsen`` odd): the largest odd
   multiple of ``h`` that still resolves the local wavelength
   ``lambda(x) = c0 / (f_max sqrt(eps_r(x)))`` with ``ppw`` points per wavelength
   on all but a fraction ``refine_fraction`` of the volume (or a given value);
2. per main cell the required ratio ``r`` (odd, a divisor of ``coarsen``) from two
   criteria:

   * dispersion: ``r >= dx / (lambda_min / ppw)`` in the ``coarsen^3`` block of
     the node;
   * representation: the block-averaged main-grid material, interpolated back to
     the model grid (trilinear), differs from the model by more than
     ``contrast_tol`` (relative, ``eps_r``; ``sigma`` relative to its largest
     value): a structure smaller than the main cells (``r = coarsen``, the model
     resolution);

3. boxes (subgrids) covering the flagged cells: connected groups of flags that
   reach the domain sides cannot be refined by a Huygens subgrid (its own CPML
   must fit inside the main grid) and are reported, not refined (a rugged
   surface, for example, is the job of the DGTD layer of :mod:`DeepGPR.hybrid`);
   the others are clipped to the region where boxes are admissible (reported as
   ``clipped``) and clustered with the Berger-Rigoutsos
   algorithm (signature holes and inflections, IEEE Trans. Syst. Man Cybern.
   21(5), 1278-1286, 1991, doi:10.1109/21.120081) to boxes with at least
   ``efficiency`` flagged cells, padded by ``margin`` cells, and merged while
   their outer surfaces come closer than two main cells. Points that must not
   lie between an inner and an outer surface (antennas) enlarge the box that
   would trap them or, when the enlarged box would not fit, push the nearest
   face away from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..config.constants import SPEED_OF_LIGHT
from .solver import Subgrid, block_average


def odd_divisors(n: int) -> List[int]:
    return [d for d in range(1, n + 1, 2) if n % d == 0]


def local_spacing(eps_r, f_max: float, ppw: float, mu_r=None) -> np.ndarray:
    """Required spacing ``lambda / ppw`` [m] at every model node."""
    e = np.asarray(eps_r, dtype=np.float64)
    m = 1.0 if mu_r is None else np.asarray(mu_r, dtype=np.float64)
    return SPEED_OF_LIGHT / (f_max * np.sqrt(e * m)) / ppw


@dataclass
class RefinementPlan:
    """Result of :func:`plan_refinement`."""

    coarsen: int
    subgrids: List[Subgrid]
    ratio_required: np.ndarray            # per main node (1 = main grid suffices)
    flags: np.ndarray                     # main nodes that need refinement
    refinable: np.ndarray                 # flags that boxes may cover
    boxes_cells: List[Tuple[Tuple[int, int], ...]] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def summary(self) -> str:
        i = self.info
        lines = [f"coarsen {self.coarsen} (dx = {i['dx'] * 100:.2f} cm), "
                 f"{int(self.flags.sum())} flagged main nodes, {len(self.subgrids)} subgrid(s)"]
        for sg, cells in zip(self.subgrids, self.boxes_cells):
            lines.append(f"  ratio {sg.ratio}: box {[tuple(round(v, 4) for v in b) for b in sg.box]} "
                         f"({'x'.join(str(c[1] - c[0]) for c in cells)} main cells)")
        if i.get("unrefinable"):
            lines.append(f"  not refinable (touch the boundary): {i['unrefinable']} flagged nodes")
        return "\n".join(lines)


def choose_coarsen(h_req: np.ndarray, h: float, refine_fraction: float, max_coarsen: int) -> int:
    target = float(np.quantile(h_req, refine_fraction))
    best = 1
    for R in range(1, max_coarsen + 1, 2):
        if R * h <= target * (1 + 1e-9):
            best = R
    return best


def required_ratio(eps_r, sigma, h: float, coarsen: int, f_max: float, ppw: float,
                   contrast_tol: float, sigma_tol: float) -> Tuple[np.ndarray, dict]:
    """Per main node: the smallest admissible odd ratio (divisor of ``coarsen``)."""
    R = coarsen
    eps = torch.as_tensor(np.asarray(eps_r, dtype=np.float64))
    sig = torch.as_tensor(np.asarray(sigma, dtype=np.float64))
    h_req = torch.as_tensor(local_spacing(eps.numpy(), f_max, ppw))
    # dispersion: smallest required spacing in the node's block
    p = (R - 1) // 2
    hmin = -F.max_pool3d(F.pad(-h_req[None, None], (p,) * 6, mode="replicate"), R, stride=R)[0, 0]
    r_disp = np.ceil((R * h / hmin.numpy()) - 1e-9).astype(np.int64)
    # representation: block averages interpolated back to the model grid
    shape = eps.shape
    main_shape = tuple((n - 1) // R + 1 for n in shape)

    def back(t):
        m = block_average(t, R)
        return F.interpolate(m[None, None], size=shape, mode="trilinear", align_corners=True)[0, 0]

    e_err = (back(eps) - eps).abs() / eps
    s_scale = float(sig.abs().max())
    s_err = (back(sig) - sig).abs() / s_scale if s_scale > 0 else torch.zeros_like(sig)
    bad = ((e_err > contrast_tol) | (s_err > sigma_tol)).double()
    # a main node needs refinement when its block contains a misrepresented model node
    bad_main = F.max_pool3d(F.pad(bad[None, None], (p,) * 6, mode="replicate"), R, stride=R)[0, 0]
    r_repr = np.where(bad_main.numpy() > 0, R, 1)
    r_need = np.maximum(r_disp, r_repr)
    if r_need.shape != main_shape:
        raise RuntimeError("Internal error: main-node shape mismatch.")
    allowed = np.array(odd_divisors(R))
    # round up to an admissible ratio (> coarsen is not representable: capped, reported)
    r_adm = np.full(r_need.shape, R, dtype=np.int64)
    for d in allowed[::-1]:
        r_adm = np.where(r_need <= d, d, r_adm)
    info = {"under_resolved_nodes": int((r_need > R).sum()),
            "dispersion_nodes": int((r_disp > 1).sum()), "representation_nodes": int((r_repr > 1).sum())}
    return r_adm, info


def subgrid_padding(r: int, is_os_sep: int, pml_cells: int, pml_sep: Optional[int] = None) -> int:
    """Main cells a subgrid of ratio ``r`` needs beyond its inner surface
    (outer surface, CPML separation and CPML), as checked by the solver."""
    sep = r // 2 + 2 if pml_sep is None else pml_sep
    nb = is_os_sep * r + sep + pml_cells
    return -(-nb // r) + 1


# --------------------------------------------------------------------------- clustering
def _label(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """6-connected components (BFS; masks are small: main-grid nodes)."""
    lab = np.zeros(mask.shape, dtype=np.int64)
    n = 0
    idx = np.argwhere(mask)
    for start in map(tuple, idx):
        if lab[start]:
            continue
        n += 1
        stack = [start]
        lab[start] = n
        while stack:
            c = stack.pop()
            for a in range(3):
                for d in (-1, 1):
                    q = list(c)
                    q[a] += d
                    q = tuple(q)
                    if 0 <= q[a] < mask.shape[a] and mask[q] and not lab[q]:
                        lab[q] = n
                        stack.append(q)
    return lab, n


def berger_rigoutsos(flags: np.ndarray, efficiency: float = 0.7, min_size: int = 2,
                     offset=(0, 0, 0)) -> List[Tuple[Tuple[int, int], ...]]:
    """Boxes ``((i0, i1), (j0, j1), (k0, k1))`` (inclusive node ranges) covering ``flags``."""
    idx = np.argwhere(flags)
    if idx.size == 0:
        return []
    lo, hi = idx.min(0), idx.max(0)
    sub = flags[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    off = tuple(int(o + v) for o, v in zip(offset, lo))
    eff = sub.mean()
    if eff >= efficiency or all(s <= 2 * min_size for s in sub.shape):
        return [tuple((off[a], off[a] + sub.shape[a] - 1) for a in range(3))]
    cut = None
    # 1. holes in the signatures
    for a in np.argsort(sub.shape)[::-1]:
        sig = sub.sum(axis=tuple(b for b in range(3) if b != a))
        holes = [i for i in range(min_size, len(sig) - min_size) if sig[i] == 0]
        if holes:
            mid = len(sig) / 2
            cut = (a, min(holes, key=lambda i: abs(i - mid)))
            break
    # 2. strongest inflection of the signature Laplacian
    if cut is None:
        best = (0, None)
        for a in range(3):
            sig = sub.sum(axis=tuple(b for b in range(3) if b != a)).astype(np.float64)
            if len(sig) < 2 * min_size + 2:
                continue
            lap = sig[2:] - 2 * sig[1:-1] + sig[:-2]
            for i in range(min_size - 1, len(lap) - min_size):
                if lap[i] * lap[i + 1] < 0:
                    jump = abs(lap[i + 1] - lap[i])
                    if jump > best[0]:
                        best = (jump, (a, i + 2))
        cut = best[1]
    # 3. bisect the longest axis
    if cut is None:
        a = int(np.argmax(sub.shape))
        if sub.shape[a] < 2 * min_size:
            return [tuple((off[b], off[b] + sub.shape[b] - 1) for b in range(3))]
        cut = (a, sub.shape[a] // 2)
    a, i = cut
    left = [slice(None)] * 3
    right = [slice(None)] * 3
    left[a] = slice(0, i)
    right[a] = slice(i, None)
    off_r = list(off)
    off_r[a] += i
    return (berger_rigoutsos(sub[tuple(left)], efficiency, min_size, off)
            + berger_rigoutsos(sub[tuple(right)], efficiency, min_size, tuple(off_r)))


def _os_overlap(a, b, gap: int) -> bool:
    """Inclusive node ranges closer than ``gap`` cells along every axis."""
    return all(b[k][0] < a[k][1] + gap and a[k][0] < b[k][1] + gap for k in range(3))


def merge_boxes(boxes, gap: int):
    """Merge boxes (inclusive cell ranges) whose ranges come closer than ``gap``."""
    boxes = [tuple(tuple(r) for r in b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            b = boxes.pop()
            for i, o in enumerate(out):
                if _os_overlap(b, o, gap):
                    out[i] = tuple((min(b[k][0], o[k][0]), max(b[k][1], o[k][1])) for k in range(3))
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


# --------------------------------------------------------------------------- plan
def plan_refinement(eps_r, sigma, h: float, f_max: float, *, ppw: float = 10.0,
                    coarsen: Optional[int] = None, refine_fraction: float = 0.02,
                    max_coarsen: int = 9, contrast_tol: float = 0.15, sigma_tol: float = 0.15,
                    efficiency: float = 0.6, margin: int = 1, is_os_sep: int = 3,
                    pml_cells: int = 6, pmlthick: int = 10, keep_points=None,
                    min_ratio: int = 3, extra_flags: Optional[np.ndarray] = None,
                    filter: bool = True) -> RefinementPlan:
    """Refinement plan for :class:`~DeepGPR.subgrid.SubgridMaxwell3D` (see the module
    docstring). ``keep_points``: physical points (antennas) that must end up either
    inside a box or outside every outer surface; ``extra_flags``: additional main-node
    flags (for example from an error indicator, :mod:`.indicators`)."""
    eps = np.asarray(eps_r, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    h_req = local_spacing(eps, f_max, ppw)
    if coarsen is None:
        coarsen = choose_coarsen(h_req, h, refine_fraction, max_coarsen)
        while coarsen > 1 and any((n - 1) % coarsen for n in eps.shape):
            coarsen -= 2
    R = int(coarsen)
    if R < 1 or R % 2 == 0 or any((n - 1) % R for n in eps.shape):
        raise ValueError("coarsen must be odd and divide every model dimension minus one.")
    dx = R * h
    info = {"dx": dx, "h": h, "f_max": f_max, "ppw": ppw,
            "model_resolves": bool(h <= h_req.min() * (1 + 1e-9))}
    if R == 1:
        z = np.zeros(tuple(eps.shape), dtype=bool)
        return RefinementPlan(1, [], np.ones(eps.shape, dtype=np.int64), z, z, [], info)
    r_adm, rinfo = required_ratio(eps, sig, h, R, f_max, ppw, contrast_tol, sigma_tol)
    info.update(rinfo)
    flags = r_adm > 1
    extra = None
    if extra_flags is not None:
        extra = np.asarray(extra_flags, dtype=bool)
        r_adm = np.where(extra & (r_adm < min_ratio), min(max(min_ratio, 3), R), r_adm)
    # admissible region: the subgrid with its CPML must fit inside the main interior
    pad = max(subgrid_padding(r, is_os_sep, pml_cells) for r in odd_divisors(R) if r >= 3)
    main_shape = flags.shape
    ok = np.zeros(main_shape, dtype=bool)
    # box [lo - margin, hi + margin] must satisfy lo - pad >= 1 and hi + pad <= n - 1
    # (the solver's check, ``hsg.build_subgrid``)
    sl = tuple(slice(pad + 1 + margin, n - pad - margin) for n in main_shape)
    ok[sl] = True
    # a priori flags: connected groups that reach the domain boundary (e.g. a surface
    # spanning the domain) are not refinable; the others are clipped to the admissible
    # region (a target near its edge is refined where a box can be placed)
    edge = np.zeros(main_shape, dtype=bool)
    for a in range(3):
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[a] = slice(0, 1)
        sl_hi[a] = slice(main_shape[a] - 1, main_shape[a])
        edge[tuple(sl_lo)] = True
        edge[tuple(sl_hi)] = True
    lab, n = _label(flags)
    refinable = np.zeros_like(flags)
    unref = clipped = 0
    for comp in range(1, n + 1):
        m = lab == comp
        if (m & edge).any():
            unref += int(m.sum())
        else:
            refinable |= m & ok
            clipped += int((m & ~ok).sum())
    info["unrefinable"] = unref
    info["clipped"] = clipped
    if extra is not None:
        # indicator flags (a posteriori): clipped to the admissible region
        info["extra_outside"] = int((extra & ~ok).sum())
        refinable |= extra & ok
        flags = flags | extra
    raw = berger_rigoutsos(refinable, efficiency)
    gap = 2 * is_os_sep + 2
    boxes = []
    for b in raw:
        boxes.append(tuple((max(b[k][0] - margin, 0), min(b[k][1] + margin, main_shape[k] - 1))
                           for k in range(3)))
    boxes = merge_boxes(boxes, gap)
    # antennas: inside a box (with one fine cell) or outside every OS (with one main cell)
    def admissible(lo, hi):
        return all(lo[k] - pad >= 1 and hi[k] + pad <= main_shape[k] - 1 for k in range(3))

    if keep_points is not None:
        pts = np.asarray(keep_points, dtype=np.float64).reshape(-1, 3) / dx
        for _ in range(10):
            moved = False
            new = []
            for b in boxes:
                lo = np.array([b[k][0] for k in range(3)], dtype=np.float64)
                hi = np.array([b[k][1] for k in range(3)], dtype=np.float64)
                inner = np.all((pts > lo + 0.34) & (pts < hi - 0.34), axis=1)
                ring = np.all((pts > lo - is_os_sep - 1) & (pts < hi + is_os_sep + 1), axis=1) & ~inner
                if ring.any():
                    q = pts[ring]
                    lo_e = np.minimum(lo, np.floor(q.min(0) - 1))
                    hi_e = np.maximum(hi, np.ceil(q.max(0) + 1))
                    if admissible(lo_e, hi_e):         # take the points inside the box
                        lo, hi = lo_e, hi_e
                    else:                              # or move the nearest face away from them
                        for p_ in q:
                            out_lo = lo - p_          # > 0: point below the box on that axis
                            out_hi = p_ - hi          # > 0: point above
                            k = int(np.argmax(np.maximum(out_lo, out_hi)))
                            if out_hi[k] >= out_lo[k]:
                                hi[k] = min(hi[k], np.floor(p_[k] - is_os_sep - 1.01))
                            else:
                                lo[k] = max(lo[k], np.ceil(p_[k] + is_os_sep + 1.01))
                        info["shrunk_for_points"] = info.get("shrunk_for_points", 0) + 1
                    moved = True
                if np.all(hi > lo):
                    new.append(tuple((int(lo[k]), int(hi[k])) for k in range(3)))
            boxes = merge_boxes(new, gap)
            if not moved:
                break
    subgrids, cells = [], []
    divs = odd_divisors(R)
    for b in boxes:
        if any(b[k][0] - pad < 1 or b[k][1] + pad > main_shape[k] - 1 for k in range(3)):
            info.setdefault("dropped_boxes", []).append(b)
            continue
        region = r_adm[b[0][0]:b[0][1] + 1, b[1][0]:b[1][1] + 1, b[2][0]:b[2][1] + 1]
        r = int(region.max())
        r = min([d for d in divs if d >= max(r, min_ratio)] or [R])
        if b[0][1] == b[0][0] or b[1][1] == b[1][0] or b[2][1] == b[2][0]:
            b = tuple((lo_, max(hi_, lo_ + 1)) for lo_, hi_ in b)
        box = tuple((b[k][0] * dx, b[k][1] * dx) for k in range(3))
        subgrids.append(Subgrid(box, ratio=r, is_os_sep=is_os_sep, pml_cells=pml_cells,
                                filter=filter))
        cells.append(b)
    info["pad_cells"] = pad
    return RefinementPlan(R, subgrids, r_adm, flags, refinable, cells, info)


__all__ = ["RefinementPlan", "berger_rigoutsos", "choose_coarsen", "local_spacing",
           "merge_boxes", "odd_divisors", "plan_refinement", "required_ratio", "subgrid_padding"]
