r"""2D hybrid with a staircase DGTD-FDTD interface that follows the surface (prototype).

The planar hybrid (:class:`~DeepGPR.hybrid.HybridTM2D`) puts the DGTD region in a
band between two horizontal Ez rows. Here the DGTD region ``D`` is a layer
around the surface: per model column ``c`` (2.5 cm cells of the model grid,
``r x r`` Yee cells each) the model rows ``[t_c, b_c)``, with ``t_c`` at least
``above`` above and ``b_c`` at least ``below`` below the surface in that column.
The interface ``Gamma = dD`` is therefore a staircase of horizontal and vertical
Yee grid lines. It is an artificial interface (the same medium on both sides),
so staircasing it costs no geometric accuracy; the ground surface itself stays
inside ``D`` as a chain of triangle edges.

**Yee side (summation by parts on an arbitrary union of cells).** With ``R`` the
Yee cells outside ``D``, the Ez update is the circulation of ``H`` around the
part of the node's dual cell inside ``R``:

* node weight ``w`` = fraction of the dual cell in ``R`` (1 inside, 1/2 on a
  straight piece of ``Gamma``, 1/4 at a convex and 3/4 at a concave corner of
  ``R``); H-edge weight = (number of adjacent cells in ``R``) / 2;
* per node and side the fraction ``f`` of the dual side in ``R`` (0, 1/2, 1);
* the missing part of the circulation is the tangential DG field along
  ``Gamma`` (Hy on horizontal, Hx on vertical pieces), entering through the
  hat-function mortar of :class:`SegmentCoupling` as ``T = C^T H_t / h``;

``eps w dE/dt = (f+ Hy[x+] - f- Hy[x-] + Th)/dx - (g+ Hx[y+] - g- Hx[y-] + Tv)/dy``
(scaled by ``w``), with ``Th``/``Tv`` the signed traces of the horizontal and
vertical pieces at the node. For nodes on straight pieces and at convex corners
the side that faces ``D`` is an inactive edge used by this node only, so the
regular Yee update applies with a ghost value written into that edge (as the
planar ghost rows); concave corners (all four sides active, fractional weights)
are updated explicitly after the Yee update. With central DG fluxes and no
absorbers the coupled scheme conserves the discrete energy (checked in the
tests). The DG side is unchanged: ``E* = I(E_b)`` (continuous piecewise-linear
along ``Gamma``) on interface faces, ``H* = H-``.

Side CPML columns: the layer is flat there (vertical pieces and corners are
kept out of the CPML), so the CPML terms act as in the planar band.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from . import cuda2d
from . import functional as fn
from .coupling import SparseOp
from .dgtd import FACE_INTERFACE, FACE_PEC, FACE_ABC, FACE_INTERIOR, DGGeometry
from .mesh import AIR, GROUND, TriMesh, _distance_to_polyline, _hex_lattice, _resample_polyline
from .reference import gauss_legendre, lagrange_1d
from .solver import HybridTM2D
from .yee import YeeTM2D

_GAUSS_POINTS = 4


# --------------------------------------------------------------------------- layer
def layer_columns(surface_ext, ncol: int, mdx: float, mdy: float, above: float, below: float,
                  flat_left: int, flat_right: int, row_min: int, row_max: int):
    """Model rows ``[t_c, b_c)`` of the DGTD layer per model column ``c``.

    ``flat_left``/``flat_right`` columns at each side get one common extent (no
    vertical pieces or corners there); adjacent columns overlap by at least one
    model cell (no saddle points)."""
    t = np.empty(ncol, dtype=np.int64)
    b = np.empty(ncol, dtype=np.int64)
    for c in range(ncol):
        ys = np.linspace(c * mdy, (c + 1) * mdy, 41)
        s = np.asarray(surface_ext(ys), dtype=np.float64)
        t[c] = int(np.floor((s.min() - above) / mdx + 1e-9))
        b[c] = int(np.ceil((s.max() + below) / mdx - 1e-9))
    for sl in (slice(0, flat_left + 1), slice(ncol - flat_right - 1, ncol)):
        t[sl], b[sl] = t[sl].min(), b[sl].max()
    for _ in range(ncol):                       # overlap of neighbours >= 1 cell
        changed = False
        for c in range(ncol - 1):
            lo, hi = max(t[c], t[c + 1]), min(b[c], b[c + 1])
            if hi - lo < 2:
                b[c] = b[c + 1] = max(b[c], b[c + 1], lo + 2)
                changed = True
        if not changed:
            break
    if t.min() < row_min or b.max() > row_max:
        raise ValueError("The DGTD layer reaches the top or bottom CPML; reduce the layer.")
    return t, b


def layer_polygon(t, b, mdx, mdy):
    """Closed boundary polygon ``(n, 2)`` ``(x, y)`` of the layer (corners only)."""
    ncol = len(t)
    top = []
    for c in range(ncol):
        top += [(t[c] * mdx, c * mdy), (t[c] * mdx, (c + 1) * mdy)]
    bot = []
    for c in reversed(range(ncol)):
        bot += [(b[c] * mdx, (c + 1) * mdy), (b[c] * mdx, c * mdy)]
    pts = np.array(top + bot, dtype=np.float64)
    keep = [0]
    for k in range(1, len(pts)):
        if np.hypot(*(pts[k] - pts[keep[-1]])) > 1e-12:
            keep.append(k)
    pts = pts[keep]
    # drop collinear points
    out = []
    n = len(pts)
    for k in range(n):
        a, p, q = pts[k - 1], pts[k], pts[(k + 1) % n]
        if abs((p[0] - a[0]) * (q[1] - p[1]) - (p[1] - a[1]) * (q[0] - p[0])) > 1e-14:
            out.append(p)
    return np.array(out)


def _inside_polygon(poly, pts, tol=0.0):
    """Strictly inside (by more than ``tol`` from every edge) a simple polygon."""
    from matplotlib.path import Path
    inside = Path(poly).contains_points(pts)
    if tol > 0:
        closed = np.concatenate([poly, poly[:1]])
        inside &= _distance_to_polyline(pts, closed) > tol
    return inside


def staircase_layer_mesh(t, b, mdx, mdy, surface_ext, h, h_surface=None, seed=0) -> TriMesh:
    """Body-fitted triangulation of the staircase layer (Delaunay of boundary,
    surface and graded interior points; checked for conformity)."""
    from scipy.spatial import Delaunay, cKDTree
    h_surface = h if h_surface is None else float(h_surface)
    rng = np.random.default_rng(seed)
    poly = layer_polygon(t, b, mdx, mdy)
    width = len(t) * mdy
    # boundary points: every polygon edge resampled (corners kept)
    bnd = []
    n = len(poly)
    for k in range(n):
        p, q = poly[k], poly[(k + 1) % n]
        m = max(1, int(np.ceil(np.hypot(*(q - p)) / h - 1e-9)))
        s = np.linspace(0.0, 1.0, m + 1)[:-1]
        bnd.append(p[None] + s[:, None] * (q - p)[None])
    boundary = np.concatenate(bnd, axis=0)
    dense_y = np.linspace(0.0, width, max(2, int(np.ceil(width / (0.05 * h_surface))) + 1))
    dense_x = np.asarray(surface_ext(dense_y), dtype=np.float64)
    surface = _resample_polyline(dense_y, dense_x, h_surface)
    surface[0, 1], surface[-1, 1] = 0.0, width
    surface[0, 0], surface[-1, 0] = dense_x[0], dense_x[-1]
    if not np.all(_inside_polygon(poly, surface[1:-1], tol=0.4 * h)):
        raise ValueError("The surface must stay inside the layer by at least 0.4 h.")
    lat = _hex_lattice(poly[:, 0].min(), poly[:, 0].max(), 0.0, width, h, phase=rng.random())
    closed = np.concatenate([poly, poly[:1]])
    ok = _inside_polygon(poly, lat) & (_distance_to_polyline(lat, closed) > 0.45 * h) \
        & (_distance_to_polyline(lat, surface) > 0.6 * h)
    accepted = np.concatenate([boundary, surface], axis=0)
    cand = lat[ok]
    if len(cand):
        dist, _ = cKDTree(accepted).query(cand)
        accepted = np.concatenate([accepted, cand[dist >= 0.7 * h]], axis=0)
    points = np.unique(np.round(accepted, 12), axis=0)
    tri = Delaunay(points)
    tris = tri.simplices.astype(np.int64)
    p = points[tris]
    area = np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                  - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])) / 2
    centroid = p.mean(axis=1)
    keep = (area > 1e-10 * h * h) & _inside_polygon(poly, centroid)
    tris, p, area = tris[keep], p[keep], area[keep]
    # conformity: the kept triangles tile the polygon exactly
    xs, ys = poly[:, 0], poly[:, 1]
    poly_area = 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
    if abs(area.sum() - poly_area) > 1e-9 * poly_area:
        raise RuntimeError("Staircase layer mesh does not tile the layer (boundary not recovered).")
    surf_at = lambda yv: np.interp(yv, surface[:, 1], surface[:, 0])  # noqa: E731
    side = p[:, :, 0] - surf_at(p[:, :, 1])
    tol = 1e-9 * max(1.0, h)
    if np.any(np.any(side > tol, axis=1) & np.any(side < -tol, axis=1)):
        raise RuntimeError("Triangles cross the surface; reduce h_surface.")
    region = np.where(centroid[keep][:, 0] < surf_at(centroid[keep][:, 1]), AIR, GROUND)
    return TriMesh(points, tris, region, surface,
                   (float(poly[:, 0].min()), float(poly[:, 0].max()), float(width)),
                   info={"generator": "staircase-layer", "polygon": poly, "t": t, "b": b})


# --------------------------------------------------------------------------- Yee side
@dataclass
class Segment:
    axis: str                 # "h": x = line dx, along y;  "v": y = line dy, along x
    sign: float               # +1: Yee region on the -axis side (above / left), -1 otherwise
    line: int                 # Yee node row (h) or column (v)
    n0: int                   # first node index along the segment
    n1: int                   # last node index (inclusive)

    def nodes_flat(self, NY):
        k = np.arange(self.n0, self.n1 + 1)
        return self.line * NY + k if self.axis == "h" else k * NY + self.line


def yee_layer_structure(dmask_cells: np.ndarray):
    """Weights, fractions, ghosts, concave nodes and interface segments of the
    Yee region ``R`` = cells outside ``dmask_cells`` (``(NX-1, NY-1)`` bool)."""
    NXc, NYc = dmask_cells.shape
    NX, NY = NXc + 1, NYc + 1
    Rp = np.zeros((NXc + 2, NYc + 2))
    Rp[1:-1, 1:-1] = ~dmask_cells
    q_ul, q_ur, q_ll, q_lr = Rp[:-1, :-1], Rp[:-1, 1:], Rp[1:, :-1], Rp[1:, 1:]   # cells around nodes
    w = (q_ul + q_ur + q_ll + q_lr) / 4
    fxp, fxm = (q_ll + q_lr) / 2, (q_ul + q_ur) / 2          # dual sides x+ (Hy[i]) / x- (Hy[i-1])
    gyp, gym = (q_ur + q_lr) / 2, (q_ul + q_ll) / 2          # y+ (Hx[j]) / y- (Hx[j-1])
    interior = np.zeros((NX, NY), dtype=bool)
    interior[1:-1, 1:-1] = True
    w = np.where(interior, w, 0.0)
    w_hx, w_hy = gyp.copy(), fxp.copy()
    w_hx[:, -1] = 0.0
    w_hy[-1, :] = 0.0
    bnode = interior & (w > 0) & (w < 1)
    if np.any(bnode & (q_ul == q_lr) & (q_ur == q_ll) & (q_ul != q_ur)):
        raise ValueError("Saddle node in the staircase interface (thicken the layer).")
    ghosts = {"hy": [], "hx": []}
    concave = []
    for i, j in zip(*np.nonzero(bnode)):
        wn = w[i, j]
        mx = (fxp[i, j] == 0, fxm[i, j] == 0)
        my = (gyp[i, j] == 0, gym[i, j] == 0)
        x_std = fxp[i, j] == wn and fxm[i, j] == wn
        y_std = gyp[i, j] == wn and gym[i, j] == wn
        if (x_std or any(mx)) and (y_std or any(my)):
            if mx[0]:
                ghosts["hy"].append((i * NY + j, (i - 1) * NY + j, 1 - fxm[i, j] / wn, 1 / wn, i * NY + j))
            if mx[1]:
                ghosts["hy"].append(((i - 1) * NY + j, i * NY + j, 1 - fxp[i, j] / wn, -1 / wn, i * NY + j))
            if my[0]:
                ghosts["hx"].append((i * NY + j, i * NY + j - 1, 1 - gym[i, j] / wn, 1 / wn, i * NY + j))
            if my[1]:
                ghosts["hx"].append((i * NY + j - 1, i * NY + j, 1 - gyp[i, j] / wn, -1 / wn, i * NY + j))
        else:
            concave.append((i, j))
    for key, wts in (("hy", w_hy), ("hx", w_hx)):
        tg = [g[0] for g in ghosts[key]]
        if len(set(tg)) != len(tg):
            raise ValueError(f"A ghost {key} edge is claimed by two nodes (thicken the layer).")
        if any(wts.reshape(-1)[g] != 0 for g in tg):
            raise ValueError(f"A ghost {key} edge is an active edge.")
    # interface segments: horizontal pieces are Hx edges with one adjacent cell in R
    segs: List[Segment] = []
    for i in range(1, NX - 1):
        on = (w_hx[i, : NY - 1] == 0.5)
        sign = np.where(q_ur[i, : NY - 1] > 0, 1.0, -1.0)    # cell (i-1, j) in R: R above
        j = 0
        while j < NY - 1:
            if on[j]:
                k = j
                while k + 1 < NY - 1 and on[k + 1] and sign[k + 1] == sign[j]:
                    k += 1
                segs.append(Segment("h", float(sign[j]), i, j, k + 1))
                j = k + 1
            else:
                j += 1
    for j in range(1, NY - 1):
        on = (w_hy[: NX - 1, j] == 0.5)
        sign = np.where(q_ll[: NX - 1, j] > 0, 1.0, -1.0)    # cell (i, j-1) in R: R left
        i = 0
        while i < NX - 1:
            if on[i]:
                k = i
                while k + 1 < NX - 1 and on[k + 1] and sign[k + 1] == sign[i]:
                    k += 1
                segs.append(Segment("v", float(sign[i]), j, i, k + 1))
                i = k + 1
            else:
                i += 1
    return dict(NX=NX, NY=NY, w=w, w_hx=w_hx, w_hy=w_hy, fxp=fxp, fxm=fxm, gyp=gyp, gym=gym,
                ghosts=ghosts, concave=concave, segments=segs, bnode=bnode)


# --------------------------------------------------------------------------- mortar
class SegmentCoupling:
    """Mortar between DG faces on one straight interface segment and the Yee Ez
    nodes on it (hat functions along the segment; end nodes get half hats).

    Attributes as :class:`~DeepGPR.hybrid.coupling.InterfaceCoupling`: ``slots``,
    ``nodes`` (DG face nodes), ``to_dg`` (Yee row values -> projected ``E*``),
    ``to_fd`` (DG tangential H at the face nodes -> ``T = C^T H / h``)."""

    def __init__(self, geometry: DGGeometry, faces, along: str, spacing: float, n0: int, nn: int,
                 dtype=torch.float64, device="cpu"):
        ref = geometry.ref
        nfp = ref.nfp
        xg, wg = gauss_legendre(_GAUSS_POINTS)
        coord = (geometry.y if along == "y" else geometry.x).reshape(-1)
        rows, cols, vals, prow, pcol, pval, slots, nodes = [], [], [], [], [], [], [], []
        for n, (k, f) in enumerate(faces):
            fnodes = geometry.vmapM[k, f]
            c = coord[fnodes]
            ca, cb = c[0], c[-1]
            lo, hi = min(ca, cb), max(ca, cb)
            j0, j1 = int(np.floor(lo / spacing + 1e-12)), int(np.ceil(hi / spacing - 1e-12))
            br = np.unique(np.concatenate(([lo, hi], np.arange(j0, j1 + 1) * spacing)))
            br = br[(br >= lo - 1e-12) & (br <= hi + 1e-12)]
            block = np.zeros((nfp, nn))
            for a, bb in zip(br[:-1], br[1:]):
                if bb - a <= 1e-14 * max(1.0, hi):
                    continue
                yq = 0.5 * (a + bb) + 0.5 * (bb - a) * xg
                wq = 0.5 * (bb - a) * wg
                xi = -1.0 + 2.0 * (yq - ca) / (cb - ca)
                lag = lagrange_1d(ref.face_coordinate[f], xi)
                jl = np.floor(yq / spacing + 1e-12).astype(np.int64)
                tt = yq / spacing - jl
                for qq in range(yq.size):
                    for jj, hat in ((jl[qq], 1.0 - tt[qq]), (jl[qq] + 1, tt[qq])):
                        loc = jj - n0
                        if 0 <= loc < nn and hat != 0.0:
                            block[:, loc] += wq[qq] * lag[qq] * hat
            sJ = 0.5 * (hi - lo)
            if not np.isclose(sJ, geometry.sJ[k, f], rtol=1e-9, atol=0.0):
                raise RuntimeError("Interface face is not on the segment line.")
            proj = np.linalg.solve(sJ * ref.face_mass, block)
            for i in range(nfp):
                row = n * nfp + i
                nz = np.nonzero(block[i])[0]
                rows += [row] * nz.size
                cols += nz.tolist()
                vals += block[i, nz].tolist()
                nzp = np.nonzero(proj[i])[0]
                prow += [row] * nzp.size
                pcol += nzp.tolist()
                pval += proj[i, nzp].tolist()
                slots.append(k * 3 * nfp + f * nfp + i)
                nodes.append(int(fnodes[i]))
        nif = len(slots)
        self.slots = torch.as_tensor(slots, dtype=torch.long, device=device)
        self.nodes = torch.as_tensor(nodes, dtype=torch.long, device=device)
        self.to_dg = SparseOp(prow, pcol, pval, (nif, nn), dtype, device)
        self.to_fd = SparseOp(cols, rows, np.asarray(vals) / spacing, (nn, nif), dtype, device)
        self.C = SparseOp(rows, cols, vals, (nif, nn), dtype, device)


class StaircaseStatic:
    """Coupling operations of the staircase interface used by :func:`functional.step`."""

    def __init__(self, struct, couplings, segs, dx, dy, dtype, device):
        NY = struct["NY"]
        self.dx, self.dy = float(dx), float(dy)
        L = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=device)  # noqa: E731
        T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype, device=device)  # noqa: E731
        bflat = np.nonzero(struct["bnode"].reshape(-1))[0]
        pos = -np.ones(struct["NX"] * NY, dtype=np.int64)
        pos[bflat] = np.arange(bflat.size)
        self.nB = int(bflat.size)
        self.segs = []
        NX = struct["NX"]
        for sg, cp in zip(segs, couplings):
            yf = sg.nodes_flat(NY)
            bp = pos[yf].copy()
            bad = bp < 0
            ii, jj = yf // NY, yf % NY
            outer = (ii == 0) | (ii == NX - 1) | (jj == 0) | (jj == NY - 1)
            if np.any(bad & ~outer):
                raise RuntimeError("Segment node is not a boundary node.")
            bp[bad] = self.nB          # outer PEC nodes (Ez = 0): dummy slot, never read
            self.segs.append(dict(axis=sg.axis, sign=float(sg.sign), yee=L(yf), bpos=L(bp),
                                  cp=cp))
        self.ghost = {}
        for key in ("hy", "hx"):
            g = struct["ghosts"][key]
            arr = np.asarray(g, dtype=np.float64).reshape(-1, 5)
            self.ghost[key] = dict(tgt=L(arr[:, 0]), src=L(arr[:, 1]), c1=T(arr[:, 2]),
                                   c2=T(arr[:, 3]), own=L(pos[arr[:, 4].astype(np.int64)]))
        cc = struct["concave"]
        self.n_concave = len(cc)
        if cc:
            i = np.array([c[0] for c in cc])
            j = np.array([c[1] for c in cc])
            f = lambda a: T(struct[a][i, j])  # noqa: E731
            self.cv = dict(node=L(i * NY + j), hyp=L(i * NY + j), hym=L((i - 1) * NY + j),
                           hxp=L(i * NY + j), hxm=L(i * NY + j - 1), fxp=f("fxp"),
                           fxm=f("fxm"), gyp=f("gyp"), gym=f("gym"), w=f("w"),
                           own=L(pos[i * NY + j]))
        self.slots = torch.cat([s["cp"].slots for s in self.segs]) if self.segs else None
        # All segments in one sparse operator each (a few kernels per step):
        # E* = P_all Ez[yee_all];  Th = TH Hy_DG[face nodes of h segments], Tv likewise.
        rows, cols, vals, yee_all = [], [], [], []
        th = {"h": ([], [], [], []), "v": ([], [], [], [])}   # rows, cols, vals, dg nodes
        r0 = c0 = 0
        for sg in self.segs:
            cp = sg["cp"]
            pr, pc, pv = (t.cpu().numpy() for t in (cp.to_dg.rows, cp.to_dg.cols, cp.to_dg.vals))
            rows.append(pr + r0)
            cols.append(pc + c0)
            vals.append(pv)
            yee_all.append(sg["yee"].cpu().numpy())
            fr, fc, fv = (t.cpu().numpy() for t in (cp.to_fd.rows, cp.to_fd.cols, cp.to_fd.vals))
            R_, C_, V_, N_ = th[sg["axis"]]
            off = sum(len(n) for n in N_)
            R_.append(sg["bpos"].cpu().numpy()[fr])
            C_.append(fc + off)
            V_.append(sg["sign"] * fv)
            N_.append(cp.nodes.cpu().numpy())
            r0 += cp.slots.numel()
            c0 += sg["yee"].numel()
        cat = lambda a: np.concatenate(a) if a else np.zeros(0)  # noqa: E731
        self.yee_all = L(cat(yee_all))
        self.P = SparseOp(cat(rows), cat(cols), cat(vals), (r0, c0), dtype, device)
        self.T = {}
        for ax in ("h", "v"):
            R_, C_, V_, N_ = th[ax]
            nodes = cat(N_)
            self.T[ax] = (L(nodes), SparseOp(cat(R_), cat(C_), cat(V_),
                                             (self.nB + 1, len(nodes)), dtype, device))

    def trace(self, Ez, compact, K=None, NF=None):
        """``E* = P Ez|Gamma`` on the interface slots: compact ``(S, nX)`` or dense."""
        S = Ez.shape[0]
        X = self.P(Ez.reshape(S, -1).index_select(1, self.yee_all))
        if compact:
            return X
        out = torch.zeros((S, K * NF), dtype=Ez.dtype, device=Ez.device)
        return out.index_copy(1, self.slots, X)

    def traces(self, Hxd, Hyd):
        """Signed tangential DG traces ``(Th, Tv)`` at the boundary nodes ``(S, nB + 1)``
        (the last column collects the outer PEC nodes and is never read)."""
        S = Hxd.shape[0]
        nh, Th_op = self.T["h"]
        nv, Tv_op = self.T["v"]
        Th = Th_op(Hyd.reshape(S, -1).index_select(1, nh))
        Tv = Tv_op(Hxd.reshape(S, -1).index_select(1, nv))
        return Th, Tv

    # ------------------------------------------------------------- transposes (vjp2d)
    def trace_transpose(self, gX, lEz):
        """``lEz += (P Ez|Gamma)^T gX`` (in place)."""
        S = gX.shape[0]
        lEz.view(S, -1).index_add_(1, self.yee_all, self.P.transpose_apply(gX))

    def traces_transpose(self, gTh, gTv, lHxd, lHyd):
        """``lHyd += TH^T gTh``, ``lHxd += TV^T gTv`` (in place)."""
        S = gTh.shape[0]
        nh, Th_op = self.T["h"]
        nv, Tv_op = self.T["v"]
        lHyd.view(S, -1).index_add_(1, nh, Th_op.transpose_apply(gTh))
        lHxd.view(S, -1).index_add_(1, nv, Tv_op.transpose_apply(gTv))

    def ghosts_transpose(self, lHx, lHy, gTh, gTv):
        """Transpose of :meth:`ghosts`: cotangents of the H fields before the ghost
        writes (new tensors) and ``gTh``/``gTv`` updated in place."""
        S = lHx.shape[0]
        out = []
        for key, L2, gT in (("hx", lHx, gTv), ("hy", lHy, gTh)):
            g = self.ghost[key]
            flat = L2.reshape(S, -1)
            if g["tgt"].numel():
                gt = flat.index_select(1, g["tgt"])
                flat = flat.index_fill(1, g["tgt"], 0.0).index_add(1, g["src"], g["c1"] * gt)
                gT.index_add_(1, g["own"], g["c2"] * gt)
            out.append(flat.view(L2.shape))
        return out[0], out[1]

    def concave_transpose(self, lEz, Ez_old, Hx, Hy, ca, cb, Th, Tv, gTh, gTv):
        """Transpose of :meth:`concave_update`. Returns ``(lEz without the overridden
        nodes, (node, gEz_old), [(edge, gHx)], [(edge, gHy)], gca, gcb)`` (``gca``/``gcb``
        summed over shots, per concave node); ``gTh``/``gTv`` updated in place."""
        S = lEz.shape[0]
        c = self.cv
        lf = lEz.reshape(S, -1)
        g = lf.index_select(1, c["node"])
        lEz_m = lf.index_fill(1, c["node"], 0.0).view(lEz.shape)
        hx, hy = Hx.reshape(S, -1), Hy.reshape(S, -1)
        dyx = (c["fxp"] * hy.index_select(1, c["hyp"]) - c["fxm"] * hy.index_select(1, c["hym"])
               + Th.index_select(1, c["own"])) / c["w"]
        dxy = (c["gyp"] * hx.index_select(1, c["hxp"]) - c["gym"] * hx.index_select(1, c["hxm"])
               + Tv.index_select(1, c["own"])) / c["w"]
        rhs = dyx / self.dx - dxy / self.dy
        can = ca.reshape(-1).index_select(0, c["node"])
        cbn = cb.reshape(-1).index_select(0, c["node"])
        e_old = Ez_old.reshape(S, -1).index_select(1, c["node"])
        a = cbn * g / (c["w"] * self.dx)
        b = cbn * g / (c["w"] * self.dy)
        gTh.index_add_(1, c["own"], a)
        gTv.index_add_(1, c["own"], -b)
        gHy = [(c["hyp"], c["fxp"] * a), (c["hym"], -c["fxm"] * a)]
        gHx = [(c["hxp"], -c["gyp"] * b), (c["hxm"], c["gym"] * b)]
        return lEz_m, (c["node"], can * g), gHx, gHy, (g * e_old).sum(0), (g * rhs).sum(0)

    def ghosts(self, Hx, Hy, Th, Tv):
        """Write the ghost values of the inactive edges that face ``D``."""
        S = Hx.shape[0]
        out = []
        for key, F, Tr in (("hx", Hx, Tv), ("hy", Hy, Th)):
            g = self.ghost[key]
            flat = F.reshape(S, -1)
            if g["tgt"].numel():
                val = g["c1"] * flat.index_select(1, g["src"]) + g["c2"] * Tr.index_select(1, g["own"])
                flat = flat.index_copy(1, g["tgt"], val)
            out.append(flat.view(F.shape))
        return out[0], out[1]

    def concave_update(self, Ez_new, Ez_old, Hx, Hy, ca, cb, Th, Tv):
        """Explicit SBP update of the concave-corner nodes (overrides the Yee update)."""
        if not self.n_concave:
            return Ez_new
        S = Ez_new.shape[0]
        c = self.cv
        hx, hy = Hx.reshape(S, -1), Hy.reshape(S, -1)
        dyx = (c["fxp"] * hy.index_select(1, c["hyp"]) - c["fxm"] * hy.index_select(1, c["hym"])
               + Th.index_select(1, c["own"])) / c["w"]
        dxy = (c["gyp"] * hx.index_select(1, c["hxp"]) - c["gym"] * hx.index_select(1, c["hxm"])
               + Tv.index_select(1, c["own"])) / c["w"]
        rhs = dyx / self.dx - dxy / self.dy
        val = (ca.reshape(-1).index_select(0, c["node"]) * Ez_old.reshape(S, -1).index_select(1, c["node"])
               + cb.reshape(-1).index_select(0, c["node"]) * rhs)
        return Ez_new.reshape(S, -1).index_copy(1, c["node"], val).view(Ez_new.shape)


class StaircaseYee(YeeTM2D):
    """Yee part of the staircase hybrid: the full grid (no band) with 2D energy weights."""

    def __init__(self, *args, struct=None, **kw):
        super().__init__(*args, band=None, **kw)
        dev, dt = self.device, self.dtype
        self.w_e = torch.as_tensor(struct["w"], dtype=dt, device=dev)
        self.w_hx = torch.as_tensor(struct["w_hx"], dtype=dt, device=dev)
        self.w_hy = torch.as_tensor(struct["w_hy"], dtype=dt, device=dev)

    def node_weights(self):
        return self.w_e

    def energy(self, Ez_prev, Hx_prev, Hy_prev):
        cell = self.dx * self.dy
        e = (self.w_e * self.eps * Ez_prev ** 2).sum(dim=(1, 2))
        hx = (self.w_hx * self.mu * Hx_prev * self.Hx).sum(dim=(1, 2))
        hy = (self.w_hy * self.mu * Hy_prev * self.Hy).sum(dim=(1, 2))
        return 0.5 * cell * (e + hx + hy)


def staircase_yee_static(fd: StaircaseYee) -> fn.YeeStatic:
    """:func:`functional.yee_static` of the full grid, restricted to the Yee region."""
    ys = fn.yee_static(fd)
    act_e = ys.act_e & (fd.w_e > 0)
    mhx = (fd.w_hx > 0).to(ys.cHx.dtype)
    mhy = (fd.w_hy > 0).to(ys.cHx.dtype)
    A, B, E, F = ys.h_y
    h_y = (A * mhx, B * mhx, E, F)
    A, B, E, F = ys.h_x
    h_x = (A * mhy, B * mhy, E, F)
    return fn.YeeStatic(ys.dx, ys.dy, ys.shape, act_e, ys.cHx * mhx, ys.cHy * mhy, h_y, h_x,
                        ys.e_x, ys.e_y)


# --------------------------------------------------------------------------- solver
class StaircaseHybridTM2D(HybridTM2D):
    """:class:`~DeepGPR.hybrid.HybridTM2D` with the DGTD region a staircase layer
    around the surface (see the module docstring).

    Args:
        layer: ``(above, below)`` minimum DGTD thickness [m] above and below the
            surface in every model column.
        flat_columns: Extra model columns next to the side CPML in which the
            layer is kept flat (vertical pieces and corners stay out of the CPML).
        Other arguments as :class:`HybridTM2D` (``dg_band`` and ``mesh`` are set
        internally; ``mesh_options`` may give ``h_surface``).
    """

    def __init__(self, eps_r, sigma, dx, *, surface, layer=(0.10, 0.075), flat_columns: int = 2,
                 **kw):
        if surface is None:
            raise ValueError("The staircase hybrid needs a surface.")
        self.layer = (float(layer[0]), float(layer[1]))
        self.flat_columns = int(flat_columns)
        kw.pop("dg_band", None)
        mesh_opts = dict(kw.pop("mesh_options", None) or {})
        ys = np.linspace(0, (np.asarray(eps_r).shape[1] - 1) * (dx if np.isscalar(dx) else dx[1]),
                         2001)
        s = surface(ys)
        band = (float(s.min() - self.layer[0] - 0.05), float(s.max() + self.layer[1] + 0.05))
        self._mesh_opts = mesh_opts
        super().__init__(eps_r, sigma, dx, surface=surface, dg_band=band, mesh="staircase",
                         mesh_options=mesh_opts, **kw)

    # ------------------------------------------------------------ geometry
    def _build_mesh(self, mesh, options) -> TriMesh:
        mdx, mdy = self.model_dx, self.model_dy
        ncol = int(round(self.width / mdy))
        nrow = int(round(self.nx_ext * self.dx / mdx))
        p0m, p1m, p2m, p3m = self.model_pml[:4]
        self._t, self._b = layer_columns(self.surface_x_ext, ncol, mdx, mdy, *self.layer,
                                         p2m + self.flat_columns, p3m + self.flat_columns,
                                         p0m + 2, nrow - p1m - 2)
        # Yee cells of the layer
        a = (np.arange(self.NX - 1) + 0.5) * self.dx
        c = (np.arange(self.NY - 1) + 0.5) * self.dy
        I = np.floor(a / mdx).astype(np.int64)[:, None]
        C = np.clip(np.floor(c / mdy).astype(np.int64), 0, ncol - 1)[None, :]
        self._dcells = (I >= self._t[C]) & (I < self._b[C])
        self._struct = yee_layer_structure(self._dcells)
        opts = dict(options)
        return staircase_layer_mesh(self._t, self._b, mdx, mdy, self.surface_x_ext, self.dg_h,
                                    h_surface=opts.get("h_surface"))

    def _face_tags(self, side_boundary) -> np.ndarray:
        g = self.geometry
        tags = np.full((g.K, 3), FACE_INTERIOR, dtype=np.int64)
        fm = self.ref.fmask
        tol = 1e-6 * min(self.dx, self.dy)
        side = {"pec": FACE_PEC, "abc": FACE_ABC}[side_boundary]
        for k, f in zip(*np.nonzero(g.boundary)):
            yf = g.y[k, fm[f]]
            if np.all(np.abs(yf) < tol) or np.all(np.abs(yf - self.width) < tol):
                tags[k, f] = side
            else:
                tags[k, f] = FACE_INTERFACE
        return tags

    def _segment_faces(self, sg: Segment):
        """Boundary faces of the DG mesh lying on an interface segment."""
        g, fm = self.geometry, self.ref.fmask
        if getattr(self, "_bfaces", None) is None or self._bfaces[0] is not g:
            kf = np.argwhere(g.boundary)
            xf = np.stack([g.x[k, fm[f]] for k, f in kf])
            yf = np.stack([g.y[k, fm[f]] for k, f in kf])
            self._bfaces = (g, kf, xf, yf)
        _, kf, xf, yf = self._bfaces
        tol = 1e-6 * min(self.dx, self.dy)
        if sg.axis == "h":
            on, cc, h = np.all(np.abs(xf - sg.line * self.dx) < tol, axis=1), yf, self.dy
        else:
            on, cc, h = np.all(np.abs(yf - sg.line * self.dy) < tol, axis=1), xf, self.dx
        on &= np.all(cc >= sg.n0 * h - tol, axis=1) & np.all(cc <= sg.n1 * h + tol, axis=1)
        return kf[on].astype(np.int64).reshape(-1, 2)

    # ------------------------------------------------------------ sub-solvers
    def _build_subsolvers(self, dt, conservative: bool, nshot: Optional[int] = None,
                          dtype: Optional[torch.dtype] = None):
        from ..solver.pml import build_pml_coeffs
        from .dgtd import DGTM2D
        nshot = self.nshot if nshot is None else nshot
        dtype = self.dtype if dtype is None else dtype
        cast = lambda t: t.to(dtype)  # noqa: E731
        pml = [0] * 6 if conservative else self.pml
        sig = cast(torch.zeros_like(self.sig_fd_pad) if conservative else self.sig_fd_pad)
        velocity = self.pml_max_velocity
        if self.pml_profile == "matched":
            velocity *= self.dg_pml_strength * self.dx / self.model_dx
        coeffs = build_pml_coeffs(None, None, dt, [self.dx, self.dy, self.dx], self.nx_ext,
                                  self.ny_ext, 1, torch.tensor(pml, dtype=torch.int32),
                                  self.device, dtype, max_velocity=velocity)
        fd = StaircaseYee(cast(self.eps_fd_pad), sig, cast(self.mu_fd_pad), self.dx, self.dy, dt,
                          pml, coeffs[6:], nshot, struct=self._struct)
        eps, sigd, mu = (cast(m) for m in self.dg_materials)
        tags = self.face_tags.copy()
        if conservative:
            sigd = torch.zeros_like(sigd)
            tags[tags == FACE_ABC] = FACE_PEC
        sx, sy = (None, None) if conservative else self.pml_sigma
        dg = DGTM2D(self.geometry, eps, sigd, mu, dt, tags,
                    flux_alpha=0.0 if conservative else self.flux_alpha,
                    pml_sigma_x=sx, pml_sigma_y=sy, nshot=nshot, dtype=dtype, device=self.device)
        cps = []
        for sg in self._struct["segments"]:
            faces = self._segment_faces(sg)
            if faces.size == 0:
                raise RuntimeError("No DG faces on an interface segment.")
            along, h = ("y", self.dy) if sg.axis == "h" else ("x", self.dx)
            cps.append(SegmentCoupling(self.geometry, faces, along, h, sg.n0, sg.n1 - sg.n0 + 1,
                                       dtype=dtype, device=self.device))
        if not conservative:
            self.fd, self.dg, self.couplings = fd, dg, cps
        return fd, dg, cps

    def _make_static(self, fd, dg, couplings, nshot, dtype=None) -> fn.StepStatic:
        dtype = self.dtype if dtype is None else dtype
        stair = StaircaseStatic(self._struct, couplings, self._struct["segments"], self.dx,
                                self.dy, dtype, self.device)
        ys = staircase_yee_static(fd)
        if stair.n_concave:
            nodes = stair.cv["node"]
            for arr in (*ys.e_x[:2], *ys.e_y[:2]):
                if float(arr.reshape(-1).index_select(0, nodes).abs().max()) != 0.0:
                    raise RuntimeError("A concave interface corner lies in the CPML.")
        st = fn.StepStatic(nshot=nshot, yee=ys, dg=fn.dg_static(dg), couplings=[], dtype=dtype,
                           device=self.device, staircase=stair)
        if self.cuda_kernels and self.device.type == "cuda":
            cuda2d.attach_yee(st.yee, self.device)
            cuda2d.attach(st.dg, self.device, slots=[s["cp"].slots for s in stair.segs])
        return st

    def _node_weight_grid(self, fd) -> torch.Tensor:
        return fd.node_weights()

    # ------------------------------------------------------------ points
    def _in_layer(self, x_ext, y_ext):
        x_ext, y_ext = np.asarray(x_ext, dtype=float), np.asarray(y_ext, dtype=float)
        poly = self.mesh.info["polygon"]
        return _inside_polygon(poly, np.stack([x_ext.reshape(-1), y_ext.reshape(-1)], 1),
                               tol=1e-9 * self.dx).reshape(x_ext.shape)

    def classify(self, x_ext, y_ext=None) -> str:
        if y_ext is None:
            raise ValueError("The staircase hybrid classifies points by (x, y).")
        return "dg" if bool(self._in_layer(x_ext, y_ext)) else "fd"

    def in_band(self, x_ext, y_ext=None):
        if y_ext is None:
            raise ValueError("The staircase hybrid classifies points by (x, y).")
        return self._in_layer(x_ext, y_ext)

    def summary(self) -> dict:
        info = super().summary()
        info.update(layer=self.layer, segments=len(self._struct["segments"]),
                    concave_corners=len(self._struct["concave"]))
        return info


__all__ = ["SegmentCoupling", "StaircaseHybridTM2D", "StaircaseStatic", "layer_columns",
           "staircase_layer_mesh", "yee_layer_structure"]
