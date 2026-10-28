"""Triangle meshes of the DGTD band that follow the ground surface exactly.

Coordinates are ``(x, y)`` in metres with ``x`` the depth (DeepGPR axis 0)
and ``y`` the horizontal position (axis 1), in the *extended* solver frame
(the frame that includes the CPML; extended Yee node ``(i, j)`` sits at
``(i dx, j dy)``).

Two generators are provided. Both put vertices on the surface curve so that the
air-ground interface is a chain of mesh edges (a conforming, body-fitted
material interface, never a staircase):

* :func:`terrain_following_band_mesh` - columns of vertices whose layers are
  stretched between the band boundaries and the surface (structured topology,
  robust, optional grading towards the surface and vertex jitter);
* :func:`delaunay_band_mesh` - a genuinely unstructured mesh: hexagonal
  lattice points, surface points and boundary points triangulated by Delaunay
  (needs SciPy), with an optional finer strip around the surface. Conformity of
  the surface edges is verified, not assumed.

Any other triangulation (for example from Gmsh) can be wrapped in
:class:`TriMesh` directly, as long as surface vertices lie on the surface and
the band boundaries are straight lines ``x = x_top``, ``x = x_bottom``,
``y = 0`` and ``y = width``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

AIR = 0
GROUND = 1


@dataclass
class TriMesh:
    """A conforming triangle mesh with a region tag per element.

    Attributes:
        vertices: ``(nv, 2)`` float64 ``(x, y)``.
        triangles: ``(k, 3)`` int64 vertex indices, counter-clockwise in
            ``(x, y)`` (fixed on construction).
        region: ``(k,)`` int8, :data:`AIR` or :data:`GROUND`.
        surface: ``(ns, 2)`` vertices of the surface polyline sorted by ``y``
            (the discrete air-ground interface represented by the mesh).
        bounds: ``(x_top, x_bottom, width)`` of the band.
    """

    vertices: np.ndarray
    triangles: np.ndarray
    region: np.ndarray
    surface: np.ndarray
    bounds: Tuple[float, float, float]
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        self.triangles = np.ascontiguousarray(self.triangles, dtype=np.int64)
        self.region = np.ascontiguousarray(self.region, dtype=np.int8)
        area = signed_areas(self.vertices, self.triangles)
        flip = area < 0
        if np.any(flip):
            self.triangles[flip] = self.triangles[flip][:, [0, 2, 1]]
            area = np.abs(area)
        if np.any(area <= 1e-14 * max(1.0, float(np.max(area)))):
            raise ValueError("Mesh contains degenerate (zero-area) triangles.")

    @property
    def num_elements(self) -> int:
        return int(self.triangles.shape[0])

    def quality(self) -> dict:
        """Minimum angle [deg], edge-length range and smallest inscribed radius."""
        p = self.vertices[self.triangles]
        edges = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]], axis=1)
        lengths = np.linalg.norm(edges, axis=2)
        area = signed_areas(self.vertices, self.triangles)
        angles = []
        for k in range(3):
            a = -edges[:, (k + 2) % 3]
            b = edges[:, k]
            cosang = np.sum(a * b, axis=1) / (lengths[:, (k + 2) % 3] * lengths[:, k])
            angles.append(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
        inradius = 2 * area / lengths.sum(axis=1)
        return {
            "elements": self.num_elements,
            "min_angle_deg": float(np.min(angles)),
            "min_edge": float(lengths.min()),
            "max_edge": float(lengths.max()),
            "min_inradius": float(inradius.min()),
        }

    def surface_depth(self, y: np.ndarray) -> np.ndarray:
        """Depth of the mesh's surface polyline at ``y`` (extended frame)."""
        return np.interp(y, self.surface[:, 1], self.surface[:, 0])


def signed_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Signed areas in the ``(x, y)`` orientation used by the DG geometric factors."""
    p = vertices[triangles]
    return 0.5 * (
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
        - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    )


def _layer_fractions(n: int, refinement: float) -> np.ndarray:
    """Cumulative fractions ``0..1`` of ``n`` layers growing geometrically away from 0.

    ``refinement`` is the ratio between the layer next to the surface and the
    outermost layer (``1`` = uniform).
    """
    if n == 1 or refinement == 1.0:
        return np.linspace(0.0, 1.0, n + 1)
    q = refinement ** (-1.0 / (n - 1))
    widths = q ** np.arange(n)
    return np.concatenate(([0.0], np.cumsum(widths) / widths.sum()))


def terrain_following_band_mesh(
    y_nodes: np.ndarray,
    x_top: float,
    x_bottom: float,
    surface_x: Callable[[np.ndarray], np.ndarray],
    h: float,
    *,
    surface_refinement: float = 1.0,
    diagonal: str = "alternate",
    jitter: float = 0.0,
    seed: int = 0,
) -> TriMesh:
    """Terrain-following triangulation of ``x_top <= x <= x_bottom``.

    Args:
        y_nodes: Increasing column positions from ``0`` to the band width [m].
        x_top, x_bottom: Band boundaries (depth, extended frame) [m].
        surface_x: Vectorised surface depth ``x = d(y)`` (extended frame).
        h: Target vertical layer thickness [m].
        surface_refinement: Ratio (<= 1) of the layer thickness next to the
            surface to the outermost layer thickness.
        diagonal: ``"alternate"`` or ``"shortest"`` quad split.
        jitter: Random interior-vertex perturbation as a fraction of ``h``.
        seed: Random seed for the jitter.
    """
    y_nodes = np.asarray(y_nodes, dtype=np.float64)
    s = np.asarray(surface_x(y_nodes), dtype=np.float64)
    if np.any(s <= x_top) or np.any(s >= x_bottom):
        raise ValueError("The surface must lie strictly inside the DGTD band.")
    n_air = max(1, int(np.ceil(np.max(s - x_top) / h - 1e-9)))
    n_ground = max(1, int(np.ceil(np.max(x_bottom - s) / h - 1e-9)))
    air_t = _layer_fractions(n_air, surface_refinement)[::-1]  # 1 at the top ... 0 at surface
    ground_t = _layer_fractions(n_ground, surface_refinement)  # 0 at surface ... 1 at bottom
    ncol = y_nodes.size
    nlay = n_air + n_ground + 1
    x = np.zeros((ncol, nlay))
    x[:, : n_air + 1] = s[:, None] - (s - x_top)[:, None] * air_t[None, :]
    x[:, n_air:] = s[:, None] + (x_bottom - s)[:, None] * ground_t[None, :]
    y = np.broadcast_to(y_nodes[:, None], x.shape).copy()

    if jitter > 0:
        rng = np.random.default_rng(seed)
        interior = np.zeros_like(x, dtype=bool)
        interior[1:-1, 1:-1] = True
        interior[:, n_air] = False  # keep the surface vertices on the surface
        dy_local = np.minimum(np.gradient(y_nodes)[:, None], h)
        x = x + interior * jitter * h * (rng.random(x.shape) - 0.5) * 0.5
        y = y + interior * jitter * dy_local * (rng.random(x.shape) - 0.5) * 0.5

    vid = np.arange(ncol * nlay).reshape(ncol, nlay)
    vertices = np.stack([x.reshape(-1), y.reshape(-1)], axis=1)
    triangles, region = [], []
    for k in range(ncol - 1):
        for m in range(nlay - 1):
            a, b, c, d = vid[k, m], vid[k + 1, m], vid[k + 1, m + 1], vid[k, m + 1]
            if diagonal == "shortest":
                use_ac = (np.linalg.norm(vertices[a] - vertices[c])
                          <= np.linalg.norm(vertices[b] - vertices[d]))
            elif diagonal == "alternate":
                use_ac = (k + m) % 2 == 0
            else:
                raise ValueError("diagonal must be 'alternate' or 'shortest'.")
            tag = AIR if m < n_air else GROUND
            if use_ac:
                triangles += [(a, b, c), (a, c, d)]
            else:
                triangles += [(a, b, d), (b, c, d)]
            region += [tag, tag]
    surface = np.stack([s, y_nodes], axis=1)
    return TriMesh(
        vertices, np.array(triangles), np.array(region), surface,
        (float(x_top), float(x_bottom), float(y_nodes[-1])),
        info={"generator": "terrain_following", "n_air_layers": n_air,
              "n_ground_layers": n_ground},
    )


def _resample_polyline(y: np.ndarray, x: np.ndarray, spacing: float) -> np.ndarray:
    """Points along the polyline ``(x(y), y)`` at (almost) uniform arc length."""
    seg = np.hypot(np.diff(x), np.diff(y))
    arc = np.concatenate(([0.0], np.cumsum(seg)))
    n = max(1, int(np.round(arc[-1] / spacing)))
    target = np.linspace(0.0, arc[-1], n + 1)
    return np.stack([np.interp(target, arc, x), np.interp(target, arc, y)], axis=1)


def _distance_to_polyline(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    a = poly[:-1][None]
    b = poly[1:][None]
    p = points[:, None]
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, axis=2) / np.maximum(np.sum(ab * ab, axis=2), 1e-300), 0, 1)
    closest = a + t[..., None] * ab
    return np.min(np.linalg.norm(p - closest, axis=2), axis=1)


def _hex_lattice(x0, x1, y0, y1, h, phase=0.0):
    dxr = h * np.sqrt(3.0) / 2
    rows = np.arange(x0, x1 + dxr, dxr)
    pts = []
    for m, xr in enumerate(rows):
        offset = (0.5 * (m % 2) + phase) * h
        ys = np.arange(y0 + offset, y1 + h, h)
        pts.append(np.stack([np.full_like(ys, xr), ys], axis=1))
    return np.concatenate(pts, axis=0)


def delaunay_band_mesh(
    width: float,
    x_top: float,
    x_bottom: float,
    surface_x: Callable[[np.ndarray], np.ndarray],
    h: float,
    *,
    h_surface: Optional[float] = None,
    refine_width: float = 0.0,
    h_interface: Optional[float] = None,
    seed: int = 0,
) -> TriMesh:
    """Unstructured body-fitted triangulation of the band (requires SciPy).

    Args:
        width: Band width [m] (``0 <= y <= width``).
        x_top, x_bottom: Band boundaries [m].
        surface_x: Vectorised surface depth ``x = d(y)``.
        h: Target element size away from the surface [m].
        h_surface: Element size in a strip of half-width ``refine_width``
            around the surface (local refinement); default ``h``.
        refine_width: Half-width of the refined strip [m].
        h_interface: Vertex spacing on the horizontal band boundaries
            (the DGTD-FDTD interfaces); default ``h``.
        seed: Seed of the small random lattice phase.
    """
    try:
        from scipy.spatial import Delaunay, cKDTree
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("delaunay_band_mesh requires SciPy; use "
                          "terrain_following_band_mesh instead.") from exc
    h_surface = h if h_surface is None else float(h_surface)
    h_interface = h if h_interface is None else float(h_interface)
    rng = np.random.default_rng(seed)

    dense_y = np.linspace(0.0, width, max(2, int(np.ceil(width / (0.05 * h_surface))) + 1))
    dense_x = np.asarray(surface_x(dense_y), dtype=np.float64)
    if np.any(dense_x <= x_top + 0.5 * h) or np.any(dense_x >= x_bottom - 0.5 * h):
        raise ValueError("The surface must stay at least h/2 inside the DGTD band.")
    surface = _resample_polyline(dense_y, dense_x, h_surface)
    surface[0, 1], surface[-1, 1] = 0.0, width
    surface[0, 0], surface[-1, 0] = dense_x[0], dense_x[-1]

    # Element size as a function of the distance d to the surface: h_surface
    # within refine_width, then growing linearly (rate 0.25) up to h.
    grading = 0.25

    def size(d):
        return np.minimum(h, h_surface + grading * np.maximum(np.asarray(d) - refine_width, 0.0))

    def line(nseg, start, end):
        t = np.linspace(0.0, 1.0, nseg + 1)
        return start[None] + t[:, None] * (end - start)[None]

    nh = max(1, int(np.round(width / h_interface)))
    top = line(nh, np.array([x_top, 0.0]), np.array([x_top, width]))
    bottom = line(nh, np.array([x_bottom, 0.0]), np.array([x_bottom, width]))
    sides = []
    for yb, xs in ((0.0, surface[0, 0]), (width, surface[-1, 0])):
        for x_end in (x_top, x_bottom):
            # March away from the surface with the local size, then rescale
            # so that the last step ends on the band boundary.
            steps, x = [], xs
            direction = np.sign(x_end - xs)
            while abs(x_end - x) > 1e-12:
                hh = float(size(abs(x - xs)))
                if abs(x_end - x) < 1.5 * hh:
                    steps.append(abs(x_end - x))
                    break
                steps.append(hh)
                x += direction * hh
            cum = np.cumsum(steps)
            cum *= abs(x_end - xs) / cum[-1]
            sides.append(np.stack([xs + direction * cum[:-1], np.full(len(cum) - 1, yb)], 1))
    boundary = np.concatenate([top, bottom, *sides], axis=0)

    # Candidate interior points: hexagonal lattices of geometrically graded
    # spacing; each candidate is kept on the level closest to its local size.
    levels = [h_surface]
    while levels[-1] < h * (1 - 1e-9):
        levels.append(min(h, levels[-1] * 1.5))
    candidates, candidate_h = [], []
    for spacing in levels:
        lat = _hex_lattice(x_top, x_bottom, 0.0, width, spacing, phase=rng.random())
        d = _distance_to_polyline(lat, surface)
        hloc = size(d)
        mine = np.abs(np.log(hloc / spacing)) <= 0.5 * np.log(1.5) + 1e-9
        if len(levels) == 1:
            mine[:] = True
        clear = (
            (lat[:, 0] > x_top + 0.45 * hloc) & (lat[:, 0] < x_bottom - 0.45 * hloc)
            & (lat[:, 1] > 0.45 * hloc) & (lat[:, 1] < width - 0.45 * hloc)
            & (d > 0.6 * hloc)
        )
        candidates.append(lat[mine & clear])
        candidate_h.append(hloc[mine & clear])
    # Greedy acceptance, finest level first: a candidate must keep 0.7 of its
    # local size from every point accepted before it (removes transition slivers).
    accepted = np.concatenate([boundary, surface], axis=0)
    for pts, hloc in zip(candidates, candidate_h):
        if len(pts) == 0:
            continue
        dist, _ = cKDTree(accepted).query(pts)
        accepted = np.concatenate([accepted, pts[dist >= 0.7 * hloc]], axis=0)
    points = accepted
    points = np.unique(np.round(points, 12), axis=0)
    tri = Delaunay(points)
    triangles = tri.simplices.astype(np.int64)
    area = signed_areas(points, triangles)
    triangles = triangles[np.abs(area) > 1e-10 * h * h]

    # Region by the centroid; a triangle with vertices strictly on both sides
    # of the surface would cut the interface.
    surf_at = lambda yv: np.interp(yv, surface[:, 1], surface[:, 0])  # noqa: E731
    p = points[triangles]
    side = p[:, :, 0] - surf_at(p[:, :, 1])
    tol = 1e-9 * max(1.0, h)
    crossing = np.any(side > tol, axis=1) & np.any(side < -tol, axis=1)
    if np.any(crossing):
        raise RuntimeError(
            f"{int(crossing.sum())} Delaunay triangles cross the surface; reduce h_surface "
            "or smooth the topography.")
    centroid = p.mean(axis=1)
    region = np.where(centroid[:, 0] < surf_at(centroid[:, 1]), AIR, GROUND)
    return TriMesh(points, triangles, region, surface,
                   (float(x_top), float(x_bottom), float(width)),
                   info={"generator": "delaunay"})


def locate_points(mesh: TriMesh, points: np.ndarray, tol: float = 1e-9):
    """Elements containing each point.

    Returns:
        A list (one entry per point) of ``(element_ids, barycentric)`` with
        ``barycentric`` of shape ``(m, 3)``; points on shared edges or vertices
        return every containing element.
    """
    p = mesh.vertices[mesh.triangles]
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    out = []
    for q in np.atleast_2d(points):
        l1 = ((q[0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (q[1] - a[:, 1])) / det
        l2 = ((b[:, 0] - a[:, 0]) * (q[1] - a[:, 1]) - (q[0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / det
        l0 = 1 - l1 - l2
        lam = np.stack([l0, l1, l2], axis=1)
        inside = np.all(lam >= -tol, axis=1)
        ids = np.nonzero(inside)[0]
        out.append((ids, lam[ids]))
    return out


__all__ = [
    "AIR",
    "GROUND",
    "TriMesh",
    "delaunay_band_mesh",
    "locate_points",
    "signed_areas",
    "terrain_following_band_mesh",
]
