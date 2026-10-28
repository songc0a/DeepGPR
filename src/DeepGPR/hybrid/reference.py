"""Nodal operators on the reference triangle for the 2D DGTD solver.

The reference triangle has vertices ``(-1, -1)``, ``(1, -1)`` and ``(-1, 1)``.
Nodes, the orthonormal (Dubiner) modal basis and the operators follow the
nodal discontinuous Galerkin construction of Hesthaven & Warburton,
*Nodal Discontinuous Galerkin Methods*, Springer 2008,
doi:10.1007/978-0-387-72067-8 (chapters 3, 6 and 10): warp-and-blend nodes,
``V`` (Vandermonde), ``Dr = Vr V^-1``, ``Ds = Vs V^-1``, ``M = (V V^T)^-1``
and ``LIFT = M^-1 E_f M_f``.

Faces are numbered as in that book: face 0 is ``s = -1`` (vertices 0-1),
face 1 is ``r + s = 0`` (vertices 1-2) and face 2 is ``r = -1`` (vertices 2-0).
Everything here is NumPy float64; the solver converts once to torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

_NODE_TOLERANCE = 1e-10

# Optimised warp-and-blend blending parameters (Hesthaven & Warburton, Table 6.1).
_ALPHA_OPT = (
    0.0, 0.0, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999, 1.2832,
    1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258,
)


def jacobi_p(x: np.ndarray, alpha: float, beta: float, n: int) -> np.ndarray:
    """Orthonormal Jacobi polynomial ``P_n^(alpha, beta)`` evaluated at ``x``."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    values = np.zeros((n + 1, x.size))
    gamma0 = (
        2.0 ** (alpha + beta + 1) / (alpha + beta + 1)
        * math.gamma(alpha + 1) * math.gamma(beta + 1) / math.gamma(alpha + beta + 1)
    )
    values[0] = 1.0 / math.sqrt(gamma0)
    if n == 0:
        return values[0]
    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    values[1] = ((alpha + beta + 2) * x / 2 + (alpha - beta) / 2) / math.sqrt(gamma1)
    a_old = 2.0 / (2 + alpha + beta) * math.sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))
    for i in range(1, n):
        h1 = 2 * i + alpha + beta
        a_new = 2.0 / (h1 + 2) * math.sqrt(
            (i + 1) * (i + 1 + alpha + beta) * (i + 1 + alpha) * (i + 1 + beta)
            / (h1 + 1) / (h1 + 3)
        )
        b_new = -(alpha**2 - beta**2) / h1 / (h1 + 2)
        values[i + 1] = (-a_old * values[i - 1] + (x - b_new) * values[i]) / a_new
        a_old = a_new
    return values[n]


def grad_jacobi_p(x: np.ndarray, alpha: float, beta: float, n: int) -> np.ndarray:
    """Derivative of :func:`jacobi_p`."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if n == 0:
        return np.zeros_like(x)
    return math.sqrt(n * (n + alpha + beta + 1)) * jacobi_p(x, alpha + 1, beta + 1, n - 1)


def jacobi_gauss_quadrature(alpha: float, beta: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Jacobi points and weights of order ``n`` (``n + 1`` points)."""
    if n == 0:
        return np.array([-(alpha - beta) / (alpha + beta + 2)]), np.array([2.0])
    h1 = 2 * np.arange(n + 1) + alpha + beta
    diagonal = -0.5 * (alpha**2 - beta**2) / (h1 + 2) / h1
    k = np.arange(1, n + 1)
    off = 2.0 / (h1[:-1] + 2) * np.sqrt(
        k * (k + alpha + beta) * (k + alpha) * (k + beta) / (h1[:-1] + 1) / (h1[:-1] + 3)
    )
    if alpha + beta < 10 * np.finfo(float).eps:
        diagonal[0] = 0.0
    matrix = np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)
    points, vectors = np.linalg.eigh(matrix)
    weights = vectors[0] ** 2 * 2 ** (alpha + beta + 1) / (alpha + beta + 1) * (
        math.gamma(alpha + 1) * math.gamma(beta + 1) / math.gamma(alpha + beta + 1)
    )
    return points, weights


def jacobi_gauss_lobatto(alpha: float, beta: float, n: int) -> np.ndarray:
    """Gauss-Lobatto-Jacobi points of order ``n``."""
    if n == 1:
        return np.array([-1.0, 1.0])
    interior, _ = jacobi_gauss_quadrature(alpha + 1, beta + 1, n - 2)
    return np.concatenate(([-1.0], interior, [1.0]))


def vandermonde_1d(n: int, r: np.ndarray) -> np.ndarray:
    """1D Vandermonde matrix of the orthonormal Legendre basis."""
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    return np.stack([jacobi_p(r, 0, 0, j) for j in range(n + 1)], axis=1)


def _warp_factor(n: int, rout: np.ndarray) -> np.ndarray:
    lgl = jacobi_gauss_lobatto(0, 0, n)
    req = np.linspace(-1.0, 1.0, n + 1)
    veq = vandermonde_1d(n, req)
    pmat = np.stack([jacobi_p(rout, 0, 0, i) for i in range(n + 1)], axis=0)
    lmat = np.linalg.solve(veq.T, pmat)
    warp = lmat.T @ (lgl - req)
    zerof = (np.abs(rout) < 1.0 - 1e-10).astype(float)
    sf = 1.0 - (zerof * rout) ** 2
    return warp / sf + warp * (zerof - 1.0)


def nodes_2d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Warp-and-blend nodes on the reference triangle in ``(r, s)``."""
    alpha = _ALPHA_OPT[n - 1] if n < 16 else 5.0 / 3.0
    l1, l3 = [], []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            l1.append(i / n)
            l3.append(j / n)
    l1 = np.array(l1)
    l3 = np.array(l3)
    l2 = 1.0 - l1 - l3
    x = -l2 + l3
    y = (-l2 - l3 + 2 * l1) / math.sqrt(3.0)
    blend1, blend2, blend3 = 4 * l2 * l3, 4 * l1 * l3, 4 * l1 * l2
    warp1 = blend1 * _warp_factor(n, l3 - l2) * (1 + (alpha * l1) ** 2)
    warp2 = blend2 * _warp_factor(n, l1 - l3) * (1 + (alpha * l2) ** 2)
    warp3 = blend3 * _warp_factor(n, l2 - l1) * (1 + (alpha * l3) ** 2)
    x = x + warp1 + math.cos(2 * math.pi / 3) * warp2 + math.cos(4 * math.pi / 3) * warp3
    y = y + math.sin(2 * math.pi / 3) * warp2 + math.sin(4 * math.pi / 3) * warp3
    # Equilateral (x, y) to reference (r, s).
    b1 = (math.sqrt(3.0) * y + 1) / 3
    b2 = (-3 * x - math.sqrt(3.0) * y + 2) / 6
    b3 = (3 * x - math.sqrt(3.0) * y + 2) / 6
    return -b2 + b3 - b1, -b2 - b3 + b1


def rs_to_ab(r: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Collapsed coordinates of the reference triangle."""
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    a = np.full_like(r, -1.0)
    regular = np.abs(s - 1.0) > 1e-14
    a[regular] = 2 * (1 + r[regular]) / (1 - s[regular]) - 1
    return a, s


def simplex_2d_mode(a: np.ndarray, b: np.ndarray, i: int, j: int) -> np.ndarray:
    """Orthonormal mode ``(i, j)`` of the triangle in collapsed coordinates."""
    h1 = jacobi_p(a, 0, 0, i)
    h2 = jacobi_p(b, 2 * i + 1, 0, j)
    return math.sqrt(2.0) * h1 * h2 * (1 - b) ** i


def grad_simplex_2d_mode(a: np.ndarray, b: np.ndarray, i: int, j: int):
    """``(d/dr, d/ds)`` of :func:`simplex_2d_mode`."""
    fa = jacobi_p(a, 0, 0, i)
    dfa = grad_jacobi_p(a, 0, 0, i)
    gb = jacobi_p(b, 2 * i + 1, 0, j)
    dgb = grad_jacobi_p(b, 2 * i + 1, 0, j)
    dr = dfa * gb
    if i > 0:
        dr = dr * (0.5 * (1 - b)) ** (i - 1)
    ds = dfa * (gb * (0.5 * (1 + a)))
    if i > 0:
        ds = ds * (0.5 * (1 - b)) ** (i - 1)
    tmp = dgb * (0.5 * (1 - b)) ** i
    if i > 0:
        tmp = tmp - 0.5 * i * gb * (0.5 * (1 - b)) ** (i - 1)
    ds = ds + fa * tmp
    return 2 ** (i + 0.5) * dr, 2 ** (i + 0.5) * ds


def vandermonde_2d(n: int, r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """2D Vandermonde matrix ``V[k, m] = psi_m(r_k, s_k)``."""
    a, b = rs_to_ab(np.asarray(r).reshape(-1), np.asarray(s).reshape(-1))
    columns = [simplex_2d_mode(a, b, i, j) for i in range(n + 1) for j in range(n + 1 - i)]
    return np.stack(columns, axis=1)


def grad_vandermonde_2d(n: int, r: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gradient Vandermonde matrices ``(Vr, Vs)``."""
    a, b = rs_to_ab(np.asarray(r).reshape(-1), np.asarray(s).reshape(-1))
    vr, vs = [], []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            dr, ds = grad_simplex_2d_mode(a, b, i, j)
            vr.append(dr)
            vs.append(ds)
    return np.stack(vr, axis=1), np.stack(vs, axis=1)


@dataclass(frozen=True)
class ReferenceTriangle:
    """Nodal operators of polynomial order ``order`` on the reference triangle.

    Attributes:
        order: Polynomial order ``N``.
        np_: Nodes per element ``(N + 1)(N + 2) / 2``.
        nfp: Nodes per face ``N + 1``.
        r, s: Node coordinates, shape ``(np_,)``.
        V: Vandermonde matrix ``(np_, np_)``.
        Dr, Ds: Nodal differentiation matrices.
        M: Reference mass matrix ``(V V^T)^-1``.
        Minv: Its inverse ``V V^T``.
        fmask: ``(3, nfp)`` node indices of each face.
        face_mass: ``(nfp, nfp)`` mass matrix of a face of reference length 2.
        LIFT: ``(np_, 3 nfp)`` surface lift ``M^-1 E_f M_f``.
        face_coordinate: ``(3, nfp)`` 1D coordinate of the face nodes in [-1, 1].
    """

    order: int
    np_: int
    nfp: int
    r: np.ndarray
    s: np.ndarray
    V: np.ndarray
    Dr: np.ndarray
    Ds: np.ndarray
    M: np.ndarray
    Minv: np.ndarray
    fmask: np.ndarray
    face_mass: np.ndarray
    LIFT: np.ndarray
    face_coordinate: np.ndarray

    def basis_at(self, r: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Nodal (Lagrange) basis values at points ``(r, s)``: shape ``(npts, np_)``."""
        modes = vandermonde_2d(self.order, r, s)
        return np.linalg.solve(self.V.T, modes.T).T


def reference_triangle(order: int) -> ReferenceTriangle:
    """Build the nodal operators of order ``order`` (1 <= order <= 6)."""
    if not isinstance(order, int) or isinstance(order, bool) or not 1 <= order <= 6:
        raise ValueError("DGTD polynomial order must be an integer between 1 and 6.")
    r, s = nodes_2d(order)
    vandermonde = vandermonde_2d(order, r, s)
    vr, vs = grad_vandermonde_2d(order, r, s)
    vinv = np.linalg.inv(vandermonde)
    dr = vr @ vinv
    ds = vs @ vinv
    minv = vandermonde @ vandermonde.T
    mass = np.linalg.inv(minv)

    faces = (
        np.nonzero(np.abs(s + 1) < _NODE_TOLERANCE)[0],
        np.nonzero(np.abs(r + s) < _NODE_TOLERANCE)[0],
        np.nonzero(np.abs(r + 1) < _NODE_TOLERANCE)[0],
    )
    nfp = order + 1
    if any(face.size != nfp for face in faces):
        raise RuntimeError("Reference triangle face nodes were not found.")
    # Order nodes along each face from its first to its second vertex
    # (face 0: v0->v1, face 1: v1->v2, face 2: v2->v0).
    coordinates = (r[faces[0]], s[faces[1]], -s[faces[2]])
    fmask = np.zeros((3, nfp), dtype=np.int64)
    face_coordinate = np.zeros((3, nfp))
    for f in range(3):
        order_along = np.argsort(coordinates[f])
        fmask[f] = faces[f][order_along]
        face_coordinate[f] = coordinates[f][order_along]
    v1d = vandermonde_1d(order, face_coordinate[0])
    face_mass = np.linalg.inv(v1d @ v1d.T)
    emat = np.zeros((r.size, 3 * nfp))
    for f in range(3):
        # The 1D coordinate along every face is a rigid copy of face 0's, so
        # the same face mass matrix applies.
        emat[fmask[f], f * nfp:(f + 1) * nfp] = face_mass
    lift = minv @ emat
    return ReferenceTriangle(
        order=order, np_=r.size, nfp=nfp, r=r, s=s, V=vandermonde, Dr=dr, Ds=ds,
        M=mass, Minv=minv, fmask=fmask, face_mass=face_mass, LIFT=lift,
        face_coordinate=face_coordinate,
    )


def lagrange_1d(nodes: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Lagrange basis on ``nodes`` evaluated at ``x``: shape ``(x.size, nodes.size)``."""
    nodes = np.asarray(nodes, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    values = np.ones((x.size, nodes.size))
    for k in range(nodes.size):
        for m in range(nodes.size):
            if m != k:
                values[:, k] *= (x - nodes[m]) / (nodes[k] - nodes[m])
    return values


def gauss_legendre(npoints: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre points and weights on [-1, 1]."""
    return np.polynomial.legendre.leggauss(npoints)


__all__ = [
    "ReferenceTriangle",
    "gauss_legendre",
    "jacobi_p",
    "lagrange_1d",
    "nodes_2d",
    "reference_triangle",
    "vandermonde_1d",
    "vandermonde_2d",
]
