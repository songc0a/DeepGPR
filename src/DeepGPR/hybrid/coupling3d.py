r"""Energy-neutral 3D DGTD-FDTD coupling on a horizontal plane ``x = x_I``.

The plane is an E plane ``i_I`` of the Yee grid; it carries the tangential
``Ey`` nodes ``((j + 1/2) dy, k dz)`` and ``Ez`` nodes ``(j dy, (k + 1/2) dz)``.
Let ``phi^y_m``/``phi^z_m`` be the bilinear hat functions of the two lattices
and ``L_i`` the Lagrange basis of a DG face (triangle) on the plane. The mortar
matrices

.. math::  C^{(y)}_{im} = \int_\Gamma L_i\, \phi^y_m\, dA,\qquad
           C^{(z)}_{im} = \int_\Gamma L_i\, \phi^z_m\, dA

are integrated exactly: every triangle is clipped against every lattice cell
(Sutherland-Hodgman), the convex pieces are fan-triangulated and integrated
with a collapsed Gauss rule that is exact for the polynomial degree of
``L_i phi_m``.

Coupling, for each tangential component ``c`` in ``{y, z}`` (``H_t`` partner
``Hz`` for ``Ey`` and ``Hy`` for ``Ez``):

* DG side: ``E*_c = (sJ M_f)^-1 C^(c) E_c`` (L2 projection of the hat
  interpolant) and ``H* = H-``;
* Yee side: ``H_b = C^(c)T H_t,DG / (dy dz)`` on the lattice nodes, used by the
  half-cell (SBP) closure of the interface plane.

As in 2D the interface terms of the discrete energy balance are
``-+ sum dA E_c H_b`` (Yee) and ``+- int E*_c H_t,DG`` (DG), which are equal, so
the coupled leapfrog scheme conserves the discrete energy.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from .coupling import SparseOp
from .reference import gauss_legendre
from .reference3d import FACE_VERTICES

_GAUSS = 3          # collapsed Gauss points per direction: exact to degree 4 (P2 x bilinear)
_MAXV = 8


def horizontal_faces(geometry, x_plane: float, tol: float) -> np.ndarray:
    """``(n, 2)`` boundary ``(element, face)`` pairs on ``x = x_plane``."""
    fm = geometry.ref.fmask
    ks, fs = np.nonzero(geometry.boundary)
    xf = geometry.x[ks[:, None], fm[fs]]
    on = np.all(np.abs(xf - x_plane) < tol, axis=1)
    return np.stack([ks[on], fs[on]], axis=1).astype(np.int64)


def _collapsed_rule():
    xg, wg = gauss_legendre(_GAUSS)
    u, wu = 0.5 * (xg + 1), 0.5 * wg
    U, Vv = np.meshgrid(u, u, indexing="ij")
    W = np.outer(wu, wu) * U
    xi, eta = U * (1 - Vv), U * Vv
    return xi.reshape(-1), eta.reshape(-1), W.reshape(-1)


class InterfaceCoupling3D:
    """Mortar operators between DG faces on ``x = x_plane`` and the Yee E plane.

    Args:
        mesh, geometry: The DG tetrahedral mesh and its geometry.
        faces: ``(n, 2)`` interface ``(element, face)`` pairs.
        x_plane: Plane position [m].
        dy, dz: Yee spacings [m].
        NY, NZ: Yee field-grid sizes (plane arrays are ``(NY, NZ)``, flat ``j NZ + k``).

    Attributes:
        slots, nodes: Face-node slots in the ``(K * 4 * Nfp)`` layout and the
            flat DG node indices.
        to_dg: ``(Ey plane -> E*_y, Ez plane -> E*_z)`` at the slots.
        to_fd: ``(DG Hz -> Hz_b at Ey nodes, DG Hy -> Hy_b at Ez nodes)``.
        C: Raw mortar matrices ``(C^(y), C^(z))`` as ``scipy.sparse`` (tests).
    """

    def __init__(self, mesh, geometry, faces, x_plane, dy, dz, NY, NZ,
                 dtype=torch.float64, device="cpu"):
        ref = geometry.ref
        nfp, npn = ref.nfp, ref.np_
        nface = faces.shape[0]
        k_of, f_of = faces[:, 0], faces[:, 1]
        fv = np.array(FACE_VERTICES)
        tri = mesh.vertices[mesh.tets[k_of[:, None], fv[f_of]]]          # (n, 3, 3)
        if np.max(np.abs(tri[..., 0] - x_plane)) > 1e-9 * max(dy, dz):
            raise ValueError("Interface faces are not on the plane.")
        tri2 = tri[..., 1:]                                               # (n, 3, 2) (y, z)
        tets = mesh.vertices[mesh.tets[k_of]]                             # (n, 4, 3)
        Tinv = np.linalg.inv(np.stack([tets[:, 1] - tets[:, 0], tets[:, 2] - tets[:, 0],
                                       tets[:, 3] - tets[:, 0]], axis=2))
        xi, eta, wq = _collapsed_rule()
        self.C = []
        self.to_dg, self.to_fd = [], []
        face_minv = np.stack([np.linalg.inv(ref.face_mass[f]) for f in range(4)])
        sJ = geometry.sJ[k_of, f_of]
        area = 0.5 * np.abs(np.cross(tri2[:, 1] - tri2[:, 0], tri2[:, 2] - tri2[:, 0]))
        if not np.allclose(area, 2 * sJ, rtol=1e-9):
            raise RuntimeError("Interface face area does not match sJ.")
        # Lattices: (offset_y, offset_z, nodes along y, nodes along z)
        for oy, oz, nyl, nzl in ((0.5 * dy, 0.0, NY - 1, NZ), (0.0, 0.5 * dz, NY, NZ - 1)):
            lo = tri2.min(axis=1)
            hi = tri2.max(axis=1)
            j0 = np.floor((lo[:, 0] - oy) / dy + 1e-12).astype(np.int64)
            j1 = np.ceil((hi[:, 0] - oy) / dy - 1e-12).astype(np.int64)
            k0 = np.floor((lo[:, 1] - oz) / dz + 1e-12).astype(np.int64)
            k1 = np.ceil((hi[:, 1] - oz) / dz - 1e-12).astype(np.int64)
            pf, pj, pk = [], [], []
            for n in range(nface):
                jj, kk = np.meshgrid(np.arange(j0[n], j1[n]), np.arange(k0[n], k1[n]),
                                     indexing="ij")
                pf.append(np.full(jj.size, n))
                pj.append(jj.reshape(-1))
                pk.append(kk.reshape(-1))
            pf, pj, pk = np.concatenate(pf), np.concatenate(pj), np.concatenate(pk)
            y0, z0 = oy + pj * dy, oz + pk * dz
            poly = np.zeros((pf.size, _MAXV, 2))
            poly[:, :3] = tri2[pf]
            cnt = np.full(pf.size, 3)
            for axis, value, keep in ((0, y0, True), (0, y0 + dy, False),
                                      (1, z0, True), (1, z0 + dz, False)):
                poly, cnt = _clip(poly, cnt, axis, value, keep)
            # Fan triangulation and quadrature.
            Cy_rows, Cy_cols, Cy_vals = [], [], []
            for m in range(1, _MAXV - 1):
                ok = cnt >= m + 2
                if not np.any(ok):
                    continue
                p0, p1, p2 = poly[ok, 0], poly[ok, m], poly[ok, m + 1]
                ar = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                                  - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
                qp = (p0[:, None] + xi[None, :, None] * (p1 - p0)[:, None]
                      + eta[None, :, None] * (p2 - p0)[:, None])                # (P, Q, 2)
                w = 2 * ar[:, None] * wq[None]
                f_ok = pf[ok]
                # Basis of the containing element at (x_plane, y, z).
                X = np.concatenate([np.full(qp.shape[:2] + (1,), x_plane), qp], axis=2)
                lam = np.einsum("pij,pqj->pqi", Tinv[f_ok], X - tets[f_ok, 0][:, None])
                r, s, t = -1 + 2 * lam[..., 0], -1 + 2 * lam[..., 1], -1 + 2 * lam[..., 2]
                phi = ref.basis_at(r.reshape(-1), s.reshape(-1), t.reshape(-1)).reshape(
                    r.shape + (npn,))
                L = np.take_along_axis(phi, ref.fmask[f_of[f_ok]][:, None, :].repeat(
                    phi.shape[1], axis=1), axis=2)                               # (P, Q, nfp)
                a = (qp[..., 0] - y0[ok][:, None]) / dy
                b = (qp[..., 1] - z0[ok][:, None]) / dz
                for dj, dk, hat in ((0, 0, (1 - a) * (1 - b)), (1, 0, a * (1 - b)),
                                    (0, 1, (1 - a) * b), (1, 1, a * b)):
                    jn, kn = pj[ok] + dj, pk[ok] + dk
                    inside = (jn >= 0) & (jn < nyl) & (kn >= 0) & (kn < nzl)
                    val = np.einsum("pq,pqi->pi", w * hat, L)                    # (P, nfp)
                    rows = f_ok[:, None] * nfp + np.arange(nfp)[None]
                    cols = np.broadcast_to((jn * NZ + kn)[:, None], rows.shape)
                    sel = np.broadcast_to(inside[:, None], rows.shape)
                    Cy_rows.append(rows[sel])
                    Cy_cols.append(cols[sel])
                    Cy_vals.append(val[sel])
            C = sp.coo_matrix((np.concatenate(Cy_vals), (np.concatenate(Cy_rows),
                                                         np.concatenate(Cy_cols))),
                              shape=(nface * nfp, NY * NZ)).tocsr()
            C.sum_duplicates()
            C.eliminate_zeros()
            blocks = sp.block_diag([face_minv[f] / s for f, s in zip(f_of, sJ)], format="csr")
            P = (blocks @ C).tocoo()
            Ct = C.tocoo()
            self.C.append(C)
            self.to_dg.append(SparseOp(P.row, P.col, P.data, P.shape, dtype, device))
            self.to_fd.append(SparseOp(Ct.col, Ct.row, Ct.data / (dy * dz), (NY * NZ, nface * nfp),
                                       dtype, device))
        vm = geometry.vmapM[k_of, f_of]                                   # (n, nfp)
        self.slots = torch.as_tensor(((k_of * 4 + f_of)[:, None] * nfp + np.arange(nfp)).reshape(-1),
                                     dtype=torch.long, device=device)
        self.nodes = torch.as_tensor(vm.reshape(-1), dtype=torch.long, device=device)
        self.nfaces = nface


def _clip(poly, cnt, axis, value, keep_greater):
    """Clip convex polygons ``(P, V, 2)`` with ``cnt`` vertices by the half-plane
    ``coord[axis] >= value`` (``<=`` unless ``keep_greater``); per-polygon ``value``."""
    P, V, _ = poly.shape
    value = np.broadcast_to(np.asarray(value, dtype=np.float64), (P,))[:, None]
    idx = np.arange(V)
    nxt = np.where(idx[None, :] + 1 < cnt[:, None], idx[None, :] + 1, 0)
    a = poly
    b = np.take_along_axis(poly, nxt[:, :, None].repeat(2, axis=2), axis=1)
    sa = a[..., axis] - value
    sb = b[..., axis] - value
    if not keep_greater:
        sa, sb = -sa, -sb
    ina, inb = sa >= 0, sb >= 0
    edge = idx[None, :] < cnt[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        tq = np.where(ina != inb, sa / np.where(ina != inb, sa - sb, 1.0), 0.0)
    inter = a + tq[..., None] * (b - a)
    pts = np.stack([inter, b], axis=2).reshape(P, 2 * V, 2)
    valid = np.stack([edge & (ina != inb), edge & inb], axis=2).reshape(P, 2 * V)
    order = np.argsort(~valid, axis=1, kind="stable")
    pts = np.take_along_axis(pts, order[:, :, None].repeat(2, axis=2), axis=1)[:, :V]
    count = valid.sum(axis=1)
    if count.max(initial=0) > V:
        raise RuntimeError("Polygon clipping overflow.")
    return pts, count


__all__ = ["InterfaceCoupling3D", "horizontal_faces"]
