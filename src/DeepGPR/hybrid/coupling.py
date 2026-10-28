r"""Energy-neutral DGTD-FDTD coupling on a horizontal interface ``x = x_I``.

The interface coincides with a row ``i_I`` of Ez nodes of the Yee grid. Let
``l_j(y)`` be the piecewise-linear hat functions of the Ez nodes ``y_j = j dy``
of that row and ``L_k`` the Lagrange basis of a DGTD face lying on ``x = x_I``.
The mortar matrix

.. math::  C_{kj} = \int_\Gamma L_k(y)\, l_j(y)\, dy

is evaluated exactly (Gauss quadrature on every sub-segment between DGTD
vertices and Yee nodes), so DGTD and FDTD meshes need not match.

Coupling (``Hy`` is the tangential magnetic field on ``x = const``):

* DGTD side: on interface faces the flux uses ``E* = I(E_b)``, the hat
  interpolant of the FDTD row values, and ``H* = H-``. In the nodal (LIFT)
  form ``E*`` enters as its face-wise L2 projection
  ``(sJ M_f)^{-1} C E_b``, which reproduces the exact surface integral
  ``\int L_k I(E_b) dy = (C E_b)_k``.
* FDTD side: the interface row is an SBP boundary with weight ``1/2``; its
  Hy boundary value is ``H_b = C^T Hy_DG / dy`` (see :mod:`.yee`).

With these two choices the interface contributions to the rate of change of
the discrete energy are ``+-(Hy_DG^T C E_b)`` and ``-+(dy E_b^T H_b)``, which
cancel exactly. The global semi-discrete operator therefore stays of the form
``M_e e' = K h, M_h h' = -K^T e`` and leapfrog conserves the discrete energy
of the coupled system (checked in the tests to round-off).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from .dgtd import DGGeometry
from .reference import gauss_legendre, lagrange_1d

_GAUSS_POINTS = 4


class SparseOp:
    """``out[:, r] += sum val * x[:, c]`` for a fixed COO pattern (batched)."""

    def __init__(self, rows, cols, vals, shape, dtype, device):
        self.rows = torch.as_tensor(rows, dtype=torch.long, device=device)
        self.cols = torch.as_tensor(cols, dtype=torch.long, device=device)
        self.vals = torch.as_tensor(vals, dtype=dtype, device=device)
        self.shape = tuple(int(s) for s in shape)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((x.shape[0], self.shape[0]), dtype=x.dtype, device=x.device)
        # index_select (not x[:, cols]): its backward is an index_add, not a sorted index_put
        return out.index_add(1, self.rows, x.index_select(1, self.cols) * self.vals)

    def transpose_apply(self, y: torch.Tensor) -> torch.Tensor:
        """``out[:, c] += sum val * y[:, r]`` (the transpose, used by hand-written adjoints)."""
        out = torch.zeros((y.shape[0], self.shape[1]), dtype=y.dtype, device=y.device)
        return out.index_add_(1, self.cols, y.index_select(1, self.rows) * self.vals)

    def dense(self) -> np.ndarray:
        a = np.zeros(self.shape)
        np.add.at(a, (self.rows.cpu().numpy(), self.cols.cpu().numpy()), self.vals.cpu().numpy())
        return a


def interface_faces(geometry: DGGeometry, x_line: float, tol: float) -> np.ndarray:
    """``(n, 2)`` array of boundary ``(element, face)`` pairs lying on ``x = x_line``."""
    fm = geometry.ref.fmask
    xf = geometry.x
    out = []
    for k, f in zip(*np.nonzero(geometry.boundary)):
        if np.all(np.abs(xf[k, fm[f]] - x_line) < tol):
            out.append((k, f))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


class InterfaceCoupling:
    """Mortar operators between DGTD faces on ``x = x_line`` and a Yee Ez row.

    Args:
        geometry: DGTD geometry.
        faces: ``(n, 2)`` interface ``(element, face)`` pairs.
        dy: Yee spacing along the interface [m].
        ny_nodes: Number of Ez nodes on the row (``NY``; nodes ``0`` and
            ``NY - 1`` are PEC).

    Attributes:
        slots: Positions of the interface face nodes in the DGTD face-node
            layout ``(K * 3 * Nfp)`` (where :attr:`DGTM2D.Eext` is written).
        nodes: Flat DGTD node indices of the same face nodes.
        to_dg: ``SparseOp`` ``E_b (NY) -> projected E*`` at the face nodes.
        to_fd: ``SparseOp`` ``Hy (face nodes) -> H_b (NY)``.
        C: ``SparseOp`` of the raw mortar matrix (diagnostics/tests).
    """

    def __init__(self, geometry: DGGeometry, faces: np.ndarray, dy: float, ny_nodes: int,
                 dtype=torch.float64, device="cpu"):
        ref = geometry.ref
        nfp = ref.nfp
        xg, wg = gauss_legendre(_GAUSS_POINTS)
        yflat = geometry.y.reshape(-1)
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        slots, nodes = [], []
        proj_rows, proj_cols, proj_vals = [], [], []
        for n, (k, f) in enumerate(faces):
            fnodes = geometry.vmapM[k, f]
            y_nodes = yflat[fnodes]
            ya, yb = y_nodes[0], y_nodes[-1]
            lo, hi = min(ya, yb), max(ya, yb)
            j0, j1 = int(np.floor(lo / dy + 1e-12)), int(np.ceil(hi / dy - 1e-12))
            breaks = np.unique(np.concatenate(([lo, hi], np.arange(j0, j1 + 1) * dy)))
            breaks = breaks[(breaks >= lo - 1e-12) & (breaks <= hi + 1e-12)]
            block = np.zeros((nfp, ny_nodes))
            for a, b in zip(breaks[:-1], breaks[1:]):
                if b - a <= 1e-14 * max(1.0, hi):
                    continue
                yq = 0.5 * (a + b) + 0.5 * (b - a) * xg
                wq = 0.5 * (b - a) * wg
                xi = -1.0 + 2.0 * (yq - ya) / (yb - ya)
                lag = lagrange_1d(ref.face_coordinate[f], xi)       # (q, nfp)
                jl = np.floor(yq / dy).astype(np.int64)
                t = yq / dy - jl
                for q in range(yq.size):
                    for jj, hat in ((jl[q], 1.0 - t[q]), (jl[q] + 1, t[q])):
                        if 0 <= jj < ny_nodes and hat != 0.0:
                            block[:, jj] += wq[q] * lag[q] * hat
            sJ = 0.5 * (hi - lo)
            if not np.isclose(sJ, geometry.sJ[k, f], rtol=1e-9, atol=0.0):
                raise RuntimeError("Interface face is not horizontal or sJ mismatch.")
            proj = np.linalg.solve(sJ * ref.face_mass, block)      # (nfp, NY)
            for i in range(nfp):
                row = n * nfp + i
                nz = np.nonzero(block[i])[0]
                rows += [row] * nz.size
                cols += nz.tolist()
                vals += block[i, nz].tolist()
                nzp = np.nonzero(proj[i])[0]
                proj_rows += [row] * nzp.size
                proj_cols += nzp.tolist()
                proj_vals += proj[i, nzp].tolist()
                slots.append(k * 3 * nfp + f * nfp + i)
                nodes.append(int(fnodes[i]))
        nif = len(slots)
        self.nfaces = int(faces.shape[0])
        self.slots = torch.as_tensor(slots, dtype=torch.long, device=device)
        self.nodes = torch.as_tensor(nodes, dtype=torch.long, device=device)
        self.C = SparseOp(rows, cols, vals, (nif, ny_nodes), dtype, device)
        self.to_dg = SparseOp(proj_rows, proj_cols, proj_vals, (nif, ny_nodes), dtype, device)
        self.to_fd = SparseOp(cols, rows, np.asarray(vals) / dy, (ny_nodes, nif), dtype, device)
        self.dy = float(dy)

    def project_to_dg(self, e_row: torch.Tensor) -> torch.Tensor:
        """``E*`` at the interface face nodes from the FDTD row ``(S, NY)``."""
        return self.to_dg(e_row)

    def trace_to_fd(self, hy_dg_flat: torch.Tensor) -> torch.Tensor:
        """``H_b (S, NY)`` from the flat DGTD ``Hy`` ``(S, K * Np)``."""
        return self.to_fd(hy_dg_flat[:, self.nodes])


__all__ = ["InterfaceCoupling", "SparseOp", "interface_faces"]
