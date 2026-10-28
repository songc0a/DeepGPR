"""Nodal operators on the reference tetrahedron for the 3D DGTD solver.

Reference tetrahedron: vertices ``(-1,-1,-1)``, ``(1,-1,-1)``, ``(-1,1,-1)``,
``(-1,-1,1)``. Orthonormal (Proriol-Koornwinder-Dubiner) modes, Vandermonde,
``Dr = Vr V^-1`` (likewise ``Ds``, ``Dt``), ``M = (V V^T)^-1`` and
``LIFT = M^-1 E_f M_f``, as in Hesthaven & Warburton, *Nodal Discontinuous
Galerkin Methods* (Springer 2008, doi:10.1007/978-0-387-72067-8), chapter 10.
Nodes are equidistant, which for ``N <= 2`` coincide with the warp-and-blend
nodes (vertices and edge midpoints).

Faces: 0 ``t = -1`` (v0 v1 v2), 1 ``s = -1`` (v0 v1 v3),
2 ``r + s + t = -1`` (v1 v2 v3), 3 ``r = -1`` (v0 v2 v3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .reference import jacobi_p, grad_jacobi_p

FACE_VERTICES = ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3))
_VERTS = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])


def equispaced_nodes_3d(n: int):
    """Equidistant nodes ``(r, s, t)`` of order ``n`` (lexicographic in barycentric)."""
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                l1, l2, l3 = i / n, j / n, k / n
                l0 = 1.0 - l1 - l2 - l3
                pts.append(l0 * _VERTS[0] + l1 * _VERTS[1] + l2 * _VERTS[2] + l3 * _VERTS[3])
    p = np.array(pts)
    return p[:, 0], p[:, 1], p[:, 2]


def rst_to_abc(r, s, t):
    r, s, t = (np.asarray(v, dtype=np.float64) for v in (r, s, t))
    a = np.full_like(r, -1.0)
    b = np.full_like(r, -1.0)
    ok = np.abs(s + t) > 1e-14
    a[ok] = 2 * (1 + r[ok]) / (-s[ok] - t[ok]) - 1
    ok = np.abs(t - 1) > 1e-14
    b[ok] = 2 * (1 + s[ok]) / (1 - t[ok]) - 1
    return a, b, t.copy()


def simplex_3d_mode(a, b, c, i, j, k):
    h1 = jacobi_p(a, 0, 0, i)
    h2 = jacobi_p(b, 2 * i + 1, 0, j)
    h3 = jacobi_p(c, 2 * (i + j) + 2, 0, k)
    return 2 * math.sqrt(2.0) * h1 * h2 * ((1 - b) ** i) * h3 * ((1 - c) ** (i + j))


def grad_simplex_3d_mode(a, b, c, i, j, k):
    fa, dfa = jacobi_p(a, 0, 0, i), grad_jacobi_p(a, 0, 0, i)
    gb, dgb = jacobi_p(b, 2 * i + 1, 0, j), grad_jacobi_p(b, 2 * i + 1, 0, j)
    hc, dhc = jacobi_p(c, 2 * (i + j) + 2, 0, k), grad_jacobi_p(c, 2 * (i + j) + 2, 0, k)
    V3Dr = dfa * (gb * hc)
    if i > 0:
        V3Dr = V3Dr * ((0.5 * (1 - b)) ** (i - 1))
    if i + j > 0:
        V3Dr = V3Dr * ((0.5 * (1 - c)) ** (i + j - 1))
    V3Ds = 0.5 * (1 + a) * V3Dr
    tmp = dgb * ((0.5 * (1 - b)) ** i)
    if i > 0:
        tmp = tmp + (-0.5 * i) * (gb * (0.5 * (1 - b)) ** (i - 1))
    if i + j > 0:
        tmp = tmp * ((0.5 * (1 - c)) ** (i + j - 1))
    tmp = fa * (tmp * hc)
    V3Ds = V3Ds + tmp
    V3Dt = 0.5 * (1 + a) * V3Dr + 0.5 * (1 + b) * tmp
    tmp = dhc * ((0.5 * (1 - c)) ** (i + j))
    if i + j > 0:
        tmp = tmp - 0.5 * (i + j) * (hc * ((0.5 * (1 - c)) ** (i + j - 1)))
    tmp = fa * (gb * tmp)
    tmp = tmp * ((0.5 * (1 - b)) ** i)
    V3Dt = V3Dt + tmp
    scale = 2 ** (2 * i + j + 1.5)
    return V3Dr * scale, V3Ds * scale, V3Dt * scale


def _modes(n):
    return [(i, j, k) for i in range(n + 1) for j in range(n + 1 - i) for k in range(n + 1 - i - j)]


def vandermonde_3d(n, r, s, t):
    a, b, c = rst_to_abc(r, s, t)
    return np.stack([simplex_3d_mode(a, b, c, i, j, k) for i, j, k in _modes(n)], axis=1)


def grad_vandermonde_3d(n, r, s, t):
    a, b, c = rst_to_abc(r, s, t)
    out = [grad_simplex_3d_mode(a, b, c, i, j, k) for i, j, k in _modes(n)]
    return tuple(np.stack([o[d] for o in out], axis=1) for d in range(3))


def _vandermonde_2d_tri(n, x, y):
    """Orthonormal triangle basis on the reference triangle (-1,-1),(1,-1),(-1,1)."""
    from .reference import vandermonde_2d
    return vandermonde_2d(n, x, y)


@dataclass(frozen=True)
class ReferenceTetrahedron:
    order: int
    np_: int
    nfp: int
    r: np.ndarray
    s: np.ndarray
    t: np.ndarray
    V: np.ndarray
    Dr: np.ndarray
    Ds: np.ndarray
    Dt: np.ndarray
    M: np.ndarray
    Minv: np.ndarray
    fmask: np.ndarray        # (4, nfp) node indices of each face
    face_mass: np.ndarray    # (4, nfp, nfp) for reference face area 2
    LIFT: np.ndarray         # (np_, 4 nfp)

    def basis_at(self, r, s, t) -> np.ndarray:
        modes = vandermonde_3d(self.order, np.atleast_1d(r), np.atleast_1d(s), np.atleast_1d(t))
        return np.linalg.solve(self.V.T, modes.T).T


def reference_tetrahedron(order: int) -> ReferenceTetrahedron:
    if order not in (1, 2):
        raise ValueError("The 3D DGTD supports orders 1 and 2 (equidistant nodes).")
    r, s, t = equispaced_nodes_3d(order)
    V = vandermonde_3d(order, r, s, t)
    Vr, Vs, Vt = grad_vandermonde_3d(order, r, s, t)
    Vinv = np.linalg.inv(V)
    Dr, Ds, Dt = Vr @ Vinv, Vs @ Vinv, Vt @ Vinv
    Minv = V @ V.T
    M = np.linalg.inv(Minv)
    tol = 1e-10
    masks = (np.abs(1 + t) < tol, np.abs(1 + s) < tol, np.abs(1 + r + s + t) < tol,
             np.abs(1 + r) < tol)
    nfp = (order + 1) * (order + 2) // 2
    fmask = np.array([np.nonzero(m)[0] for m in masks])
    if fmask.shape != (4, nfp):
        raise RuntimeError("Reference tetrahedron face nodes not found.")
    # Face mass matrices in 2D coordinates of each face (reference face area 2).
    face_coords = ((r, s), (r, t), (s, t), (s, t))
    face_mass = np.zeros((4, nfp, nfp))
    E = np.zeros((r.size, 4 * nfp))
    for f in range(4):
        x, y = face_coords[f]
        Vf = _vandermonde_2d_tri(order, x[fmask[f]], y[fmask[f]])
        mf = np.linalg.inv(Vf @ Vf.T)
        face_mass[f] = mf
        E[fmask[f], f * nfp:(f + 1) * nfp] = mf
    LIFT = Minv @ E
    return ReferenceTetrahedron(order, r.size, nfp, r, s, t, V, Dr, Ds, Dt, M, Minv, fmask,
                                face_mass, LIFT)


__all__ = ["FACE_VERTICES", "ReferenceTetrahedron", "reference_tetrahedron", "vandermonde_3d"]
