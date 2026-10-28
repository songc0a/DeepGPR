r"""Geometry and connectivity of the 3D nodal DGTD discretisation on tetrahedra.

Affine geometric factors, outward unit normals, face Jacobians and the trace
maps ``vmapM``/``vmapP`` follow Hesthaven & Warburton, *Nodal Discontinuous
Galerkin Methods* (Springer 2008, doi:10.1007/978-0-387-72067-8), chapter 10
(``GeometricFactors3D``, ``Normals3D``, ``BuildMaps3D``). Exterior trace nodes
are matched by coordinates.

Equations (DeepGPR convention, ``x`` = axis 0 = depth)::

    eps dE/dt + sigma E =  curl H - J
    mu  dH/dt           = -curl E

Strong form with the DG derivative ``D_a f = d_a f + LIFT(Fscale n_a (f* - f-))``
applied to every Cartesian component, so ``curl_D H = curl H + LIFT(n x (H* - H-))``.
Fluxes (``[f] = f+ - f-``, ``Y = sqrt(eps/mu)``, ``Z = 1/Y``, ``alpha`` in
``[0, 1]`` blends central (0) and upwind (1)):

    H* - H- = 1/2 [H] + alpha (Z+/(Z- + Z+) - 1/2) [H] - alpha/(Z- + Z+) n x [E]
    E* - E- = 1/2 [E] + alpha (Y+/(Y- + Y+) - 1/2) [E] + alpha/(Y- + Y+) n x [H]

PEC faces use ``E+ = -E-``, ``H+ = H-``; first-order Silver-Mueller faces the
full upwind flux against a zero exterior state; interface faces with the Yee
grid ``E* = E_FD`` (tangential, projected) and ``H* = H-``. In 2D these
formulas reduce to those of :mod:`DeepGPR.hybrid.dgtd`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh3d import TetMesh, locate_points_3d
from .reference3d import FACE_VERTICES, ReferenceTetrahedron

FACE_INTERIOR = 0
FACE_PEC = 1
FACE_ABC = 2
FACE_INTERFACE = 3


@dataclass
class DGGeometry3D:
    ref: ReferenceTetrahedron
    x: np.ndarray            # (K, Np)
    y: np.ndarray
    z: np.ndarray
    rst_x: np.ndarray        # (K, 3, 3): [d(r,s,t)/d(x,y,z)], row = r/s/t, col = x/y/z
    J: np.ndarray            # (K,)
    normals: np.ndarray      # (K, 4, 3) outward unit normals
    sJ: np.ndarray           # (K, 4) face Jacobian (physical area = 2 sJ)
    EToE: np.ndarray         # (K, 4)
    EToF: np.ndarray         # (K, 4)
    vmapM: np.ndarray        # (K, 4, Nfp) flat node indices
    vmapP: np.ndarray        # (K, 4, Nfp)
    boundary: np.ndarray     # (K, 4) bool

    @property
    def K(self) -> int:
        return int(self.J.shape[0])


def build_geometry_3d(mesh: TetMesh, ref: ReferenceTetrahedron) -> DGGeometry3D:
    v = mesh.vertices[mesh.tets]                     # (K, 4, 3)
    r, s, t = ref.r[None, :, None], ref.s[None, :, None], ref.t[None, :, None]
    nodes = 0.5 * (-(1 + r + s + t) * v[:, 0:1] + (1 + r) * v[:, 1:2] + (1 + s) * v[:, 2:3]
                   + (1 + t) * v[:, 3:4])             # (K, Np, 3)
    # Columns of d(x,y,z)/d(r,s,t).
    A = 0.5 * np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]], axis=2)
    J = np.linalg.det(A)
    if np.any(J <= 0):
        raise ValueError("Tetrahedra must be positively oriented.")
    rst_x = np.linalg.inv(A)                         # rows: grad r, grad s, grad t
    gr, gs, gt = rst_x[:, 0], rst_x[:, 1], rst_x[:, 2]
    n = np.stack([-gt, -gs, gr + gs + gt, -gr], axis=1)   # (K, 4, 3)
    sJ = np.linalg.norm(n, axis=2)
    normals = n / sJ[:, :, None]
    sJ = sJ * J[:, None]

    K = mesh.num_elements
    nv = mesh.vertices.shape[0]
    tets = mesh.tets
    fv = np.stack([np.sort(tets[:, list(f)], axis=1) for f in FACE_VERTICES], axis=1)  # (K, 4, 3)
    key = (fv[..., 0] * nv + fv[..., 1]) * nv + fv[..., 2]
    flat = key.reshape(-1)
    order = np.argsort(flat, kind="stable")
    sk = flat[order]
    if np.unique(flat, return_counts=True)[1].max() > 2:
        raise ValueError("Non-manifold mesh: a face is shared by more than two tetrahedra.")
    same = np.nonzero(sk[1:] == sk[:-1])[0]
    a, b = order[same], order[same + 1]
    EToE = np.repeat(np.arange(K)[:, None], 4, axis=1)
    EToF = np.repeat(np.arange(4)[None, :], K, axis=0)
    EToE[a // 4, a % 4], EToF[a // 4, a % 4] = b // 4, b % 4
    EToE[b // 4, b % 4], EToF[b // 4, b % 4] = a // 4, a % 4
    boundary = EToE == np.arange(K)[:, None]

    npn = ref.np_
    vmapM = np.arange(K)[:, None, None] * npn + ref.fmask[None, :, :]       # (K, 4, Nfp)
    cand = EToE[:, :, None] * npn + ref.fmask[EToF]                         # (K, 4, Nfp)
    pts = nodes.reshape(-1, 3)
    pm, pc = pts[vmapM], pts[cand]                                          # (K, 4, Nfp, 3)
    dist = np.linalg.norm(pm[:, :, :, None, :] - pc[:, :, None, :, :], axis=-1)
    match = np.argmin(dist, axis=3)
    vmapP = np.take_along_axis(cand, match, axis=2)
    vmapP = np.where(boundary[:, :, None], vmapM, vmapP)
    scale = np.cbrt(np.mean(J))
    if np.max(np.linalg.norm(pts[vmapM] - pts[vmapP], axis=-1)) > 1e-8 * scale:
        raise RuntimeError("DG trace maps do not match across faces.")
    return DGGeometry3D(ref, nodes[..., 0], nodes[..., 1], nodes[..., 2], rst_x, J, normals,
                        sJ, EToE, EToF, vmapM, vmapP, boundary)


def point_operators_3d(geometry: DGGeometry3D, mesh: TetMesh, points: np.ndarray,
                       tol: float = 1e-9):
    """Per point ``(elements, basis values (m, Np))`` of every containing element."""
    out = []
    for ids, lam in locate_points_3d(mesh, points, tol):
        if ids.size == 0:
            out.append((ids, np.zeros((0, geometry.ref.np_))))
            continue
        r, s, t = (-1 + 2 * lam[:, 1], -1 + 2 * lam[:, 2], -1 + 2 * lam[:, 3])
        out.append((ids, geometry.ref.basis_at(r, s, t)))
    return out


__all__ = [
    "DGGeometry3D", "FACE_ABC", "FACE_INTERFACE", "FACE_INTERIOR", "FACE_PEC",
    "build_geometry_3d", "point_operators_3d",
]
