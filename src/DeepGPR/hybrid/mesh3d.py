"""Tetrahedral meshes of the 3D DGTD band that follow the ground surface.

Coordinates ``(x, y, z)`` [m] in the extended solver frame, ``x`` = depth
(DeepGPR axis 0), ``y``/``z`` horizontal (axes 1/2). The surface is a depth
``x = d(y, z)``.

:func:`terrain_following_tet_mesh` builds columns of vertices on a horizontal
``(y, z)`` grid, stretches the layers between the band boundaries and the
surface (the surface is a vertex sheet), and splits every hexahedral cell into
six tetrahedra along its main diagonal (Kuhn/Freudenthal subdivision, which is
conforming across neighbouring cells). The surface is therefore a chain of
tetrahedron faces (a body-fitted, piecewise-linear material interface).
Optional grading towards the surface and jitter of interior vertices. Every
column has the same number of layers, so relief makes layers thin (the
thinnest element sets the DG time step).

:func:`adaptive_column_tet_mesh` keeps the column structure but gives every
column its own number of layers, ``n = round(gap / h)`` uniform layers between
each band plane and the surface. The layer sheets of the index lattice that a
column does not need collapse onto the surface; tetrahedra of the Kuhn split
with a collapsed vertical edge have zero volume and are dropped. For a Kuhn
tetrahedron the volume is ``h_y h_z / 6`` times the height difference along its
single vertical step, so column-wise monotone sheets never invert an element,
and the mesh stays conforming (collapse identifies consecutive vertices of a
column). Layers are 2/3 to 2 ``h`` thick (thinner only where the gap itself
is), instead of ``gap_min / n`` for the terrain-following mesh.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np

AIR, GROUND = 0, 1


def signed_volumes(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p = vertices[tets]
    a, b, c = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
    return np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0


@dataclass
class TetMesh:
    vertices: np.ndarray        # (nv, 3)
    tets: np.ndarray            # (k, 4), positive orientation
    region: np.ndarray          # (k,) AIR / GROUND
    bounds: Tuple[float, float, float, float]   # (x_top, x_bottom, width_y, width_z)
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        self.tets = np.ascontiguousarray(self.tets, dtype=np.int64)
        self.region = np.ascontiguousarray(self.region, dtype=np.int8)
        vol = signed_volumes(self.vertices, self.tets)
        flip = vol < 0
        if np.any(flip):
            self.tets[flip] = self.tets[flip][:, [0, 2, 1, 3]]
            vol = np.abs(vol)
        if np.any(vol <= 1e-14 * vol.max()):
            raise ValueError("Mesh contains degenerate tetrahedra.")

    @property
    def num_elements(self) -> int:
        return int(self.tets.shape[0])

    def quality(self) -> dict:
        p = self.vertices[self.tets]
        edges = [p[:, b] - p[:, a] for a, b in itertools.combinations(range(4), 2)]
        lengths = np.stack([np.linalg.norm(e, axis=1) for e in edges], 1)
        vol = signed_volumes(self.vertices, self.tets)
        area = 0.0
        for f in ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)):
            area = area + 0.5 * np.linalg.norm(np.cross(p[:, f[1]] - p[:, f[0]],
                                                        p[:, f[2]] - p[:, f[0]]), axis=1)
        inradius = 3 * vol / area
        return {"elements": self.num_elements, "min_edge": float(lengths.min()),
                "max_edge": float(lengths.max()), "min_inradius": float(inradius.min()),
                "radius_ratio_min": float((inradius / lengths.max(1)).min())}


def _fractions(n: int, refinement: float) -> np.ndarray:
    if n == 1 or refinement == 1.0:
        return np.linspace(0.0, 1.0, n + 1)
    q = refinement ** (-1.0 / (n - 1))
    w = q ** np.arange(n)
    return np.concatenate(([0.0], np.cumsum(w) / w.sum()))


def terrain_following_tet_mesh(
    y_nodes: np.ndarray, z_nodes: np.ndarray, x_top: float, x_bottom: float,
    surface_x: Callable[[np.ndarray, np.ndarray], np.ndarray], h: float, *,
    surface_refinement: float = 1.0, jitter: float = 0.0, seed: int = 0,
) -> TetMesh:
    """Body-fitted tetrahedral band ``x_top <= x <= x_bottom``.

    Args:
        y_nodes, z_nodes: Increasing horizontal column positions [m].
        x_top, x_bottom: Band boundaries [m].
        surface_x: Vectorised depth ``x = d(y, z)``.
        h: Target layer thickness [m].
        surface_refinement: Ratio (<= 1) of the layer next to the surface to the
            outermost layer.
        jitter: Random interior-vertex perturbation as a fraction of ``h``.
    """
    Y, Z = np.meshgrid(y_nodes, z_nodes, indexing="ij")
    s = np.asarray(surface_x(Y, Z), dtype=np.float64)
    if np.any(s <= x_top) or np.any(s >= x_bottom):
        raise ValueError("The surface must lie strictly inside the DGTD band.")
    n_air = max(1, int(np.ceil(np.max(s - x_top) / h - 1e-9)))
    n_gnd = max(1, int(np.ceil(np.max(x_bottom - s) / h - 1e-9)))
    fa = _fractions(n_air, surface_refinement)[::-1]
    fg = _fractions(n_gnd, surface_refinement)
    nl = n_air + n_gnd + 1
    X = np.zeros(Y.shape + (nl,))
    X[..., :n_air + 1] = s[..., None] - (s - x_top)[..., None] * fa
    X[..., n_air:] = s[..., None] + (x_bottom - s)[..., None] * fg
    YY = np.broadcast_to(Y[..., None], X.shape).copy()
    ZZ = np.broadcast_to(Z[..., None], X.shape).copy()
    if jitter > 0:
        rng = np.random.default_rng(seed)
        inner = np.zeros(X.shape, dtype=bool)
        inner[1:-1, 1:-1, 1:-1] = True
        inner[:, :, n_air] = False
        amp = jitter * h * 0.5
        X = X + inner * amp * (rng.random(X.shape) - 0.5)
        YY = YY + inner * amp * (rng.random(X.shape) - 0.5)
        ZZ = ZZ + inner * amp * (rng.random(X.shape) - 0.5)
    ny, nz = Y.shape
    vid = np.arange(ny * nz * nl).reshape(ny, nz, nl)
    vertices = np.stack([X.reshape(-1), YY.reshape(-1), ZZ.reshape(-1)], 1)
    # Kuhn subdivision of the unit cube along (0,0,0)-(1,1,1): one tetrahedron per
    # permutation of the three axes.
    corners = np.array(list(itertools.product((0, 1), repeat=3)))   # (8, 3) in (a, b, m)
    index = {tuple(c): i for i, c in enumerate(corners)}
    kuhn = []
    for perm in itertools.permutations(range(3)):
        v = np.zeros(3, int)
        chain = [index[tuple(v)]]
        for ax in perm:
            v[ax] = 1
            chain.append(index[tuple(v)])
        kuhn.append(chain)
    kuhn = np.array(kuhn)                                              # (6, 4)
    a, b, m = np.meshgrid(np.arange(ny - 1), np.arange(nz - 1), np.arange(nl - 1), indexing="ij")
    a, b, m = a.reshape(-1), b.reshape(-1), m.reshape(-1)
    cell_vertices = np.stack([vid[a + c[0], b + c[1], m + c[2]] for c in corners], 1)  # (nc, 8)
    tets = cell_vertices[:, kuhn].reshape(-1, 4)
    region = np.repeat(np.where(m < n_air, AIR, GROUND), 6)
    return TetMesh(vertices, tets, region, (float(x_top), float(x_bottom),
                                             float(y_nodes[-1]), float(z_nodes[-1])),
                   info={"n_air_layers": n_air, "n_ground_layers": n_gnd,
                         "surface_sheet": n_air, "columns": (ny, nz),
                         "y_nodes": np.asarray(y_nodes, dtype=np.float64),
                         "z_nodes": np.asarray(z_nodes, dtype=np.float64),
                         "tet_column": np.repeat(a * (nz - 1) + b, 6)})


def _kuhn_simplices() -> np.ndarray:
    """The six Kuhn tetrahedra of the unit cube as corner indices of
    ``itertools.product((0, 1), repeat=3)`` (one per axis permutation)."""
    corners = list(itertools.product((0, 1), repeat=3))
    index = {c: i for i, c in enumerate(corners)}
    out = []
    for perm in itertools.permutations(range(3)):
        v = [0, 0, 0]
        chain = [index[tuple(v)]]
        for ax in perm:
            v[ax] = 1
            chain.append(index[tuple(v)])
        out.append(chain)
    return np.array(out)


def adaptive_column_tet_mesh(
    y_nodes: np.ndarray, z_nodes: np.ndarray, x_top: float, x_bottom: float,
    surface_x: Callable[[np.ndarray, np.ndarray], np.ndarray], h: float,
) -> TetMesh:
    """Body-fitted tetrahedral band with a per-column number of layers.

    Each column gets ``n = clip(round(gap / h), 1, n_max)`` uniform layers between
    the band plane and the surface (air and ground separately); unused sheets of
    the index lattice collapse onto the surface and the degenerate tetrahedra
    are removed (see the module docstring).
    """
    Y, Z = np.meshgrid(y_nodes, z_nodes, indexing="ij")
    s = np.asarray(surface_x(Y, Z), dtype=np.float64)
    if np.any(s <= x_top) or np.any(s >= x_bottom):
        raise ValueError("The surface must lie strictly inside the DGTD band.")
    A = max(1, int(np.ceil(np.max(s - x_top) / h - 1e-9)))
    G = max(1, int(np.ceil(np.max(x_bottom - s) / h - 1e-9)))
    M = A + G
    ny, nz = Y.shape
    X = np.empty((ny, nz, M + 1))
    coll = np.zeros((ny, nz, M + 1), dtype=bool)
    for side, nmax in (("air", A), ("ground", G)):
        gap = (s - x_top) if side == "air" else (x_bottom - s)
        n = np.clip(np.rint(gap / h), 1, nmax)
        for k in range(nmax + 1):                  # k sheets away from the band plane
            pos = np.where(k < n, k * gap / n, gap)
            m = k if side == "air" else M - k
            X[..., m] = x_top + pos if side == "air" else x_bottom - pos
            coll[..., m] = k >= n
    vid = np.arange(ny * nz * (M + 1)).reshape(ny, nz, M + 1)
    canon = np.where(coll, vid[..., A:A + 1], vid)          # collapsed -> surface vertex
    corners = np.array(list(itertools.product((0, 1), repeat=3)))
    a, b, m = (g.reshape(-1) for g in np.meshgrid(np.arange(ny - 1), np.arange(nz - 1),
                                                  np.arange(M), indexing="ij"))
    cell_vertices = np.stack([canon[a + c[0], b + c[1], m + c[2]] for c in corners], 1)
    tets = cell_vertices[:, _kuhn_simplices()].reshape(-1, 4)
    region = np.repeat(np.where(m < A, AIR, GROUND), 6)
    column = np.repeat(a * (nz - 1) + b, 6)
    st = np.sort(tets, axis=1)
    ok = np.all(st[:, 1:] != st[:, :-1], axis=1)
    tets, region, column = tets[ok], region[ok], column[ok]
    used, inverse = np.unique(tets, return_inverse=True)
    vertices = np.stack([X.reshape(-1), np.broadcast_to(Y[..., None], X.shape).reshape(-1),
                         np.broadcast_to(Z[..., None], X.shape).reshape(-1)], axis=1)[used]
    return TetMesh(vertices, inverse.reshape(-1, 4), region,
                   (float(x_top), float(x_bottom), float(y_nodes[-1]), float(z_nodes[-1])),
                   info={"n_air_layers": A, "n_ground_layers": G, "surface_sheet": A,
                         "columns": (ny, nz), "y_nodes": np.asarray(y_nodes, dtype=np.float64),
                         "z_nodes": np.asarray(z_nodes, dtype=np.float64),
                         "tet_column": column})


def _column_index(mesh: TetMesh):
    """CSR lists of the tetrahedra of each (y, z) column (cached in ``mesh.info``)."""
    if "column_csr" not in mesh.info:
        col = mesh.info["tet_column"]
        order = np.argsort(col, kind="stable")
        ny, nz = mesh.info["columns"]
        start = np.searchsorted(col[order], np.arange((ny - 1) * (nz - 1) + 1))
        mesh.info["column_csr"] = (order, start)
    return mesh.info["column_csr"]


def locate_points_columns(mesh: TetMesh, points: np.ndarray, tol: float = 1e-9):
    """Fast location in a column mesh (:func:`terrain_following_tet_mesh`,
    :func:`adaptive_column_tet_mesh`): one containing tetrahedron per point
    (``-1`` if outside) and its barycentric coordinates."""
    yn, zn = mesh.info["y_nodes"], mesh.info["z_nodes"]
    ny, nz = mesh.info["columns"]
    order, start = _column_index(mesh)
    width = int(np.max(np.diff(start)))
    pts = np.atleast_2d(points)
    a = np.searchsorted(yn, pts[:, 1], side="right") - 1
    b = np.searchsorted(zn, pts[:, 2], side="right") - 1
    # The column of the point and its neighbours (interior vertices may be jittered).
    cols = []
    for da in (0, -1, 1):
        for db in (0, -1, 1):
            aa = np.clip(a + da, 0, ny - 2)
            bb = np.clip(b + db, 0, nz - 2)
            cols.append(aa * (nz - 1) + bb)
    col = np.stack(cols, 1)                                  # (n, 9)
    k = np.arange(width)
    slot = start[col][..., None] + k                         # (n, 9, width)
    valid = slot < start[col + 1][..., None]
    cand = np.where(valid, order[np.minimum(slot, order.size - 1)], -1).reshape(len(pts), -1)
    cand_safe = np.where(cand >= 0, cand, 0)
    p = mesh.vertices[mesh.tets[cand_safe]]                  # (n, c, 4, 3)
    T = np.stack([p[..., 1, :] - p[..., 0, :], p[..., 2, :] - p[..., 0, :],
                  p[..., 3, :] - p[..., 0, :]], axis=-1)
    lam123 = np.linalg.solve(T, (pts[:, None, :] - p[..., 0, :])[..., None])[..., 0]
    lam = np.concatenate([1 - lam123.sum(-1, keepdims=True), lam123], axis=-1)
    ok = np.all(lam >= -tol, axis=-1) & (cand >= 0)
    hit = ok.any(axis=1)
    j = np.argmax(ok, axis=1)
    elem = np.where(hit, cand[np.arange(len(pts)), j], -1)
    return elem, lam[np.arange(len(pts)), j]


def locate_points_3d(mesh: TetMesh, points: np.ndarray, tol: float = 1e-9):
    """``[(element_ids, barycentric (m, 4)), ...]`` per point (all containing tets)."""
    p = mesh.vertices[mesh.tets]
    T = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=2)  # (k,3,3)
    Tinv = np.linalg.inv(T)
    out = []
    for q in np.atleast_2d(points):
        lam123 = np.einsum("kij,kj->ki", Tinv, q[None] - p[:, 0])
        lam = np.concatenate([1 - lam123.sum(1, keepdims=True), lam123], 1)
        ids = np.nonzero(np.all(lam >= -tol, axis=1))[0]
        out.append((ids, lam[ids]))
    return out


__all__ = ["AIR", "GROUND", "TetMesh", "adaptive_column_tet_mesh", "locate_points_3d",
           "locate_points_columns", "signed_volumes", "terrain_following_tet_mesh"]
