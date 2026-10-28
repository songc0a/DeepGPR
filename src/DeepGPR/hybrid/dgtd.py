r"""Nodal DGTD discretisation of the 2D Ez-TM Maxwell equations on triangles.

Equations (DeepGPR convention, ``x`` = axis 0, ``y`` = axis 1)::

    eps dEz/dt + sigma Ez = dHy/dx - dHx/dy - Jz
    mu  dHx/dt            = -dEz/dy
    mu  dHy/dt            =  dEz/dx

Strong-form nodal DG (Hesthaven & Warburton 2008, doi:10.1007/978-0-387-72067-8),
with the DG derivatives including their surface terms::

    D_x E  = dE/dx  + LIFT(Fscale n_x (E* - E-))
    D_y E  = dE/dy  + LIFT(Fscale n_y (E* - E-))
    D_x Hy - D_y Hx = dHy/dx - dHx/dy + LIFT(Fscale (H_t* - H_t-)),

where ``H_t = Hy n_x - Hx n_y`` is the tangential magnetic field. Numerical
fluxes (``-`` interior, ``+`` exterior trace, ``Y = sqrt(eps/mu)``,
``Z = 1/Y``)::

    central: E* = {E},  H* = {H}
    upwind:  E*   = (Y- E- + Y+ E+ + H_t+ - H_t-) / (Y- + Y+)
             H_t* = (Z- H_t- + Z+ H_t+ + E+ - E-) / (Z- + Z+)

``flux_alpha`` blends them (0 central, 1 upwind). Boundary faces are PEC
(``E* = 0``), first-order Silver-Mueller absorbing (upwind flux against a
zero exterior state) or interfaces with the Yee grid, where ``E*`` is the
projected FDTD trace and ``H* = H-`` (see :mod:`DeepGPR.hybrid.coupling`).

Time stepping is the same staggered leapfrog as DeepGPR's Yee scheme, with
the same Crank-Nicolson conductivity term::

    H^(n+1/2) = H^(n-1/2) + dt/mu  (-D_y E^n, D_x E^n)
    E^(n+1)   = ca E^n + cb (D_x Hy - D_y Hx)^(n+1/2),
    ca = (eps/dt - sigma/2) / (eps/dt + sigma/2),  cb = 1 / (eps/dt + sigma/2).

With central fluxes the semi-discrete operator is skew-adjoint in the mass
inner products, so leapfrog conserves a discrete energy (tested). Upwind
penalty terms are evaluated explicitly at the old time level (dissipative,
smaller stable time step).

The CPML stretches the DG derivatives exactly as DeepGPR's CPML stretches the
Yee differences (same profile and recursion, coefficients evaluated at the DG
nodes): ``d -> RA d + RB phi``, ``phi <- RE phi - RF d``.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from ..config.constants import VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY
from .mesh import TriMesh, locate_points
from .reference import ReferenceTriangle

FACE_INTERIOR = 0
FACE_PEC = 1
FACE_ABC = 2
FACE_INTERFACE = 3


@dataclass
class DGGeometry:
    """Geometric factors and connectivity of a mesh for one reference element."""

    ref: ReferenceTriangle
    x: np.ndarray            # (K, Np) node coordinates
    y: np.ndarray
    rx: np.ndarray           # (K,)
    ry: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    J: np.ndarray            # (K,)
    nx: np.ndarray           # (K, 3) outward unit normals
    ny: np.ndarray
    sJ: np.ndarray           # (K, 3) half physical edge length
    EToE: np.ndarray         # (K, 3)
    EToF: np.ndarray         # (K, 3)
    vmapM: np.ndarray        # (K, 3, Nfp) flat node index of the interior trace
    vmapP: np.ndarray        # (K, 3, Nfp) flat node index of the exterior trace
    boundary: np.ndarray     # (K, 3) bool

    @property
    def K(self) -> int:
        return int(self.J.shape[0])


def build_geometry(mesh: TriMesh, ref: ReferenceTriangle) -> DGGeometry:
    """Affine geometric factors, normals, connectivity and trace maps."""
    vx, vy = mesh.vertices[:, 0], mesh.vertices[:, 1]
    tri = mesh.triangles
    va, vb, vc = tri[:, 0], tri[:, 1], tri[:, 2]
    r, s = ref.r[None, :], ref.s[None, :]
    x = 0.5 * (-(r + s) * vx[va, None] + (1 + r) * vx[vb, None] + (1 + s) * vx[vc, None])
    y = 0.5 * (-(r + s) * vy[va, None] + (1 + r) * vy[vb, None] + (1 + s) * vy[vc, None])
    xr, xs = 0.5 * (vx[vb] - vx[va]), 0.5 * (vx[vc] - vx[va])
    yr, ys = 0.5 * (vy[vb] - vy[va]), 0.5 * (vy[vc] - vy[va])
    J = xr * ys - xs * yr
    if np.any(J <= 0):
        raise ValueError("Mesh elements must be counter-clockwise with positive area.")
    rx, ry, sx, sy = ys / J, -xs / J, -yr / J, xr / J
    nx = np.stack([yr, ys - yr, -ys], axis=1)
    ny = np.stack([-xr, -xs + xr, xs], axis=1)
    sJ = np.hypot(nx, ny)
    nx, ny = nx / sJ, ny / sJ

    K = tri.shape[0]
    nv = mesh.vertices.shape[0]
    faces = np.stack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=1)  # (K, 3, 2)
    key = np.sort(faces, axis=2)
    key = key[:, :, 0].astype(np.int64) * nv + key[:, :, 1]
    flat = key.reshape(-1)
    order = np.argsort(flat, kind="stable")
    sorted_keys = flat[order]
    EToE = np.repeat(np.arange(K)[:, None], 3, axis=1)
    EToF = np.repeat(np.arange(3)[None, :], K, axis=0)
    same = np.nonzero(sorted_keys[1:] == sorted_keys[:-1])[0]
    if np.unique(flat, return_counts=True)[1].max() > 2:
        raise ValueError("Non-manifold mesh: an edge is shared by more than two triangles.")
    a, b = order[same], order[same + 1]
    ka, fa = a // 3, a % 3
    kb, fb = b // 3, b % 3
    EToE[ka, fa], EToF[ka, fa] = kb, fb
    EToE[kb, fb], EToF[kb, fb] = ka, fa
    boundary = EToE == np.arange(K)[:, None]

    npn = ref.np_
    vmapM = np.arange(K)[:, None, None] * npn + ref.fmask[None, :, :]
    # Both elements traverse a shared edge counter-clockwise, i.e. in opposite
    # directions, so the exterior nodes are the neighbour's face nodes reversed.
    vmapP = EToE[:, :, None] * npn + ref.fmask[EToF][:, :, ::-1]
    vmapP = np.where(boundary[:, :, None], vmapM, vmapP)
    xf, yf = x.reshape(-1), y.reshape(-1)
    mismatch = np.hypot(xf[vmapM] - xf[vmapP], yf[vmapM] - yf[vmapP])
    scale = np.sqrt(np.mean(J)) + 1e-300
    if mismatch.max() > 1e-8 * scale:
        raise RuntimeError("DG trace maps do not match across faces.")
    return DGGeometry(ref, x, y, rx, ry, sx, sy, J, nx, ny, sJ, EToE, EToF, vmapM, vmapP,
                      boundary)


def cpml_profile_coefficients(sigma: np.ndarray, dt: float):
    """DeepGPR CPML recursion coefficients for ``kappa = 1``, ``alpha = 0``.

    Same formulas as :func:`DeepGPR.solver.pml.calculate_pml_update_coeffs`.
    Returns ``(RA, RB, RE, RF)`` arrays shaped like ``sigma``.
    """
    e0 = VACUUM_PERMITTIVITY
    tmp = 2 * e0 + dt * sigma
    return 2 * e0 / tmp, 2 * e0 / tmp, (2 * e0 - dt * sigma) / tmp, 2 * sigma * dt / tmp


class DGTM2D:
    """Fields and update operators of the DGTD region.

    Args:
        geometry: :class:`DGGeometry`.
        eps_r, sigma, mu_r: Element-wise materials ``(K,)``.
        dt: Time step [s].
        face_tags: ``(K, 3)`` ints (``FACE_*``); interior faces must be
            :data:`FACE_INTERIOR`.
        flux_alpha: 0 central, 1 upwind (interior faces).
        pml_sigma_x, pml_sigma_y: ``(K, Np)`` CPML conductivity profiles at
            the nodes (``None`` or zeros outside the CPML).
        nshot: Number of simultaneous shots.
    """

    def __init__(self, geometry: DGGeometry, eps_r, sigma, mu_r, dt: float, face_tags,
                 *, flux_alpha: float = 0.0, pml_sigma_x=None, pml_sigma_y=None,
                 nshot: int = 1, dtype=torch.float64, device="cpu"):
        g = geometry
        ref = g.ref
        self.geometry = g
        self.K, self.Np, self.Nfp = g.K, ref.np_, ref.nfp
        self.nshot = int(nshot)
        self.dtype, self.device = dtype, torch.device(device)
        self.dt = float(dt)
        self.flux_alpha = float(flux_alpha)
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=self.device)  # noqa: E731
        ti = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.long, device=self.device)  # noqa: E731
        K, Np, Nfp = self.K, self.Np, self.Nfp
        self.DrT, self.DsT = t(ref.Dr.T), t(ref.Ds.T)
        self.LIFTT = t(ref.LIFT.T)
        self.rx, self.ry = t(g.rx[:, None]), t(g.ry[:, None])
        self.sx, self.sy = t(g.sx[:, None]), t(g.sy[:, None])
        rep = lambda a: np.repeat(a, Nfp, axis=1)  # (K,3)->(K,3Nfp)  # noqa: E731
        self.nxf, self.nyf = t(rep(g.nx)), t(rep(g.ny))
        self.fscale = t(rep(g.sJ / g.J[:, None]))
        self.vmapM = ti(g.vmapM.reshape(-1))
        self.vmapP = ti(g.vmapP.reshape(-1))
        tags = np.asarray(face_tags)
        if np.any(tags[~g.boundary] != FACE_INTERIOR):
            raise ValueError("Interior faces must be tagged FACE_INTERIOR.")
        if np.any(tags[g.boundary] == FACE_INTERIOR):
            raise ValueError("Every boundary face needs a boundary tag.")
        self.face_tags = tags
        tagf = rep(tags)
        self.m_int = t(tagf == FACE_INTERIOR)
        self.m_pec = t(tagf == FACE_PEC)
        self.m_abc = t(tagf == FACE_ABC)
        self.m_if = t(tagf == FACE_INTERFACE)
        self.has_abc = bool(np.any(tags == FACE_ABC))

        self.set_materials(eps_r, sigma, mu_r)
        self._setup_pml(pml_sigma_x, pml_sigma_y)
        shape = (self.nshot, K, Np)
        self.E = torch.zeros(shape, dtype=dtype, device=self.device)
        self.Hx = torch.zeros_like(self.E)
        self.Hy = torch.zeros_like(self.E)
        self.Eext = torch.zeros((self.nshot, K * 3 * Nfp), dtype=dtype, device=self.device)
        self.mass = t(ref.M)
        self.J = t(g.J[:, None])

    # ----------------------------------------------------------------- setup
    def set_materials(self, eps_r, sigma, mu_r) -> None:
        """Element materials ``(K,)`` (tensors keep their autograd graph)."""
        as_t = lambda a: torch.as_tensor(a, dtype=self.dtype, device=self.device)  # noqa: E731
        eps = VACUUM_PERMITTIVITY * as_t(eps_r)
        mu = VACUUM_PERMEABILITY * as_t(mu_r)
        sig = as_t(sigma)
        self.eps_e, self.mu_e, self.sigma_e = eps, mu, sig
        dt = self.dt
        ea = eps / dt + 0.5 * sig
        self.ca = ((eps / dt - 0.5 * sig) / ea)[:, None]
        self.cb = (1.0 / ea)[:, None]
        self.ch = (dt / mu)[:, None]
        Y = torch.sqrt(eps / mu)
        Nfp = self.Nfp
        g = self.geometry
        neighbour = torch.as_tensor(g.EToE, device=self.device)
        rep = lambda a: a.repeat_interleave(Nfp, dim=1)  # noqa: E731
        Ym = rep(Y[:, None].expand(-1, 3))
        Yp = rep(Y[neighbour])
        self.Ym, self.Yp = Ym, Yp
        self.Zm, self.Zp = 1.0 / Ym, 1.0 / Yp

    def _setup_pml(self, sig_x, sig_y) -> None:
        self.pml_idx = None
        if sig_x is None and sig_y is None:
            return
        sx = np.zeros((self.K, self.Np)) if sig_x is None else np.asarray(sig_x)
        sy = np.zeros((self.K, self.Np)) if sig_y is None else np.asarray(sig_y)
        active = np.nonzero(np.any(sx > 0, axis=1) | np.any(sy > 0, axis=1))[0]
        if active.size == 0:
            return
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=self.dtype, device=self.device)  # noqa: E731
        self.pml_idx = torch.as_tensor(active, dtype=torch.long, device=self.device)
        cx = cpml_profile_coefficients(sx[active], self.dt)
        cy = cpml_profile_coefficients(sy[active], self.dt)
        self.pml_x = [t(c) for c in cx]
        self.pml_y = [t(c) for c in cy]
        shape = (self.nshot, active.size, self.Np)
        z = lambda: torch.zeros(shape, dtype=self.dtype, device=self.device)  # noqa: E731
        self.psi = {"ex": z(), "ey": z(), "hx": z(), "hy": z()}

    # -------------------------------------------------------------- operators
    def _traces(self, f: torch.Tensor):
        flat = f.reshape(f.shape[0], -1)
        shape = (f.shape[0], self.K, 3 * self.Nfp)
        return flat[:, self.vmapM].view(shape), flat[:, self.vmapP].view(shape)

    def _lift(self, flux: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.fscale * flux, self.LIFTT)

    def grad_e(self, E, Hx, Hy, Eext):
        """``(D_x E, D_y E)`` with the configured fluxes; ``Eext`` holds the
        projected FDTD trace at interface face nodes (zeros elsewhere)."""
        Em, Ep = self._traces(E)
        a = self.flux_alpha
        Eext = Eext.view(Em.shape)
        dE = self.m_int * (0.5 * (Ep - Em)) + self.m_pec * (-Em) + self.m_if * (Eext - Em)
        if a > 0 or self.has_abc:
            Hxm, Hxp = self._traces(Hx)
            Hym, Hyp = self._traces(Hy)
            Htm = Hym * self.nxf - Hxm * self.nyf
            if a > 0:
                Htp = Hyp * self.nxf - Hxp * self.nyf
                up = (self.Yp * (Ep - Em) + (Htp - Htm)) / (self.Ym + self.Yp)
                dE = dE + self.m_int * a * (up - 0.5 * (Ep - Em))
            if self.has_abc:
                dE = dE + self.m_abc * (-0.5 * Em - 0.5 * Htm / self.Ym)
        dr = torch.matmul(E, self.DrT)
        ds = torch.matmul(E, self.DsT)
        dxE = self.rx * dr + self.sx * ds + self._lift(self.nxf * dE)
        dyE = self.ry * dr + self.sy * ds + self._lift(self.nyf * dE)
        return dxE, dyE

    def curl_h(self, E, Hx, Hy):
        """``(D_x Hy, D_y Hx)`` with the configured fluxes."""
        Hxm, Hxp = self._traces(Hx)
        Hym, Hyp = self._traces(Hy)
        a = self.flux_alpha
        dHx = self.m_int * (0.5 * (Hxp - Hxm))
        dHy = self.m_int * (0.5 * (Hyp - Hym))
        P = None
        if a > 0:
            w = self.Zp / (self.Zm + self.Zp)
            dHx = dHx + self.m_int * a * (w - 0.5) * (Hxp - Hxm)
            dHy = dHy + self.m_int * a * (w - 0.5) * (Hyp - Hym)
            Em, Ep = self._traces(E)
            P = self.m_int * a * (Ep - Em) / (self.Zm + self.Zp) - self.m_pec * a * Em / self.Zm
        if self.has_abc:
            dHx = dHx - self.m_abc * 0.5 * Hxm
            dHy = dHy - self.m_abc * 0.5 * Hym
            if P is None:
                Em, _ = self._traces(E)
                P = torch.zeros_like(Em)
            P = P - self.m_abc * 0.5 * Em / self.Zm
        fx = self.nxf * dHy
        fy = self.nyf * dHx
        if P is not None:
            fx = fx + self.nxf * self.nxf * P
            fy = fy - self.nyf * self.nyf * P
        dxHy = self.rx * torch.matmul(Hy, self.DrT) + self.sx * torch.matmul(Hy, self.DsT) + self._lift(fx)
        dyHx = self.ry * torch.matmul(Hx, self.DrT) + self.sy * torch.matmul(Hx, self.DsT) + self._lift(fy)
        return dxHy, dyHx

    # ------------------------------------------------------------- time step
    def update_h(self) -> None:
        """``H^(n+1/2)`` from ``E^n`` and the FDTD trace in :attr:`Eext`."""
        dxE, dyE = self.grad_e(self.E, self.Hx, self.Hy, self.Eext)
        rhs_hx, rhs_hy = -dyE, dxE
        if self.pml_idx is not None:
            idx = self.pml_idx
            ra_x, rb_x, re_x, rf_x = self.pml_x
            ra_y, rb_y, re_y, rf_y = self.pml_y
            dx_p, dy_p = dxE[:, idx], dyE[:, idx]
            psi_hx, psi_hy = self.psi["hx"], self.psi["hy"]
            rhs_hx = rhs_hx.index_copy(1, idx, -(ra_y * dy_p + rb_y * psi_hx))
            rhs_hy = rhs_hy.index_copy(1, idx, ra_x * dx_p + rb_x * psi_hy)
            self.psi["hx"] = re_y * psi_hx - rf_y * dy_p
            self.psi["hy"] = re_x * psi_hy - rf_x * dx_p
        self.Hx = self.Hx + self.ch * rhs_hx
        self.Hy = self.Hy + self.ch * rhs_hy

    def update_e(self) -> None:
        """``E^(n+1)`` from ``H^(n+1/2)`` (upwind penalties use ``E^n``)."""
        dxHy, dyHx = self.curl_h(self.E, self.Hx, self.Hy)
        rhs = dxHy - dyHx
        if self.pml_idx is not None:
            idx = self.pml_idx
            ra_x, rb_x, re_x, rf_x = self.pml_x
            ra_y, rb_y, re_y, rf_y = self.pml_y
            a_p, b_p = dxHy[:, idx], dyHx[:, idx]
            psi_ex, psi_ey = self.psi["ex"], self.psi["ey"]
            rhs = rhs.index_copy(
                1, idx, (ra_x * a_p + rb_x * psi_ex) - (ra_y * b_p + rb_y * psi_ey))
            self.psi["ex"] = re_x * psi_ex - rf_x * a_p
            self.psi["ey"] = re_y * psi_ey - rf_y * b_p
        self.E = self.ca * self.E + self.cb * rhs

    # ------------------------------------------------------- energy, helpers
    def energy(self, E_prev, Hx_prev, Hy_prev) -> torch.Tensor:
        """``1/2 sum_K (eps E^nT M_K E^n + mu H^(n-1/2)T M_K H^(n+1/2))`` per shot
        (``E^n``, ``H^(n-1/2)`` saved before the step)."""
        def quad(a, b):
            return (torch.matmul(a, self.mass) * b).sum(-1)
        jm = self.J[:, 0]
        e = (self.eps_e * jm * quad(E_prev, E_prev)).sum(-1)
        h = (self.mu_e * jm * (quad(Hx_prev, self.Hx) + quad(Hy_prev, self.Hy))).sum(-1)
        return 0.5 * (e + h)

    def point_operators(self, mesh: TriMesh, points: np.ndarray, tol: float = 1e-9):
        """Per point: ``(elements, basis_values (m, Np))`` of every containing
        element (``m > 1`` on shared edges/vertices; callers average)."""
        ref = self.geometry.ref
        out = []
        for ids, lam in locate_points(mesh, points, tol):
            if ids.size == 0:
                out.append((ids, np.zeros((0, self.Np))))
                continue
            # Barycentric (l0, l1, l2) of vertices (v0, v1, v2) -> (r, s).
            r = -lam[:, 0] + lam[:, 1] - lam[:, 2]
            s = -lam[:, 0] - lam[:, 1] + lam[:, 2]
            out.append((ids, ref.basis_at(r, s)))
        return out

    def zero_(self) -> None:
        self.E = torch.zeros_like(self.E)
        self.Hx = torch.zeros_like(self.E)
        self.Hy = torch.zeros_like(self.E)
        if self.pml_idx is not None:
            for key in self.psi:
                self.psi[key] = torch.zeros_like(self.psi[key])


__all__ = [
    "DGGeometry",
    "DGTM2D",
    "FACE_ABC",
    "FACE_INTERFACE",
    "FACE_INTERIOR",
    "FACE_PEC",
    "build_geometry",
    "cpml_profile_coefficients",
]
