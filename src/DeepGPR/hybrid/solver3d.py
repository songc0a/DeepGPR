"""3D hybrid DGTD-FDTD Maxwell solver (pure Yee or Yee + DGTD band).

DeepGPR's 3D layout: model ``eps_r[i, j, k]`` with axis 0 (``x``) the depth,
axes 1/2 (``y``, ``z``) horizontal; physical node ``(i, j, k)`` at
``(i dx, j dy, k dz)``. A band ``x_T <= x <= x_B`` of body-fitted tetrahedra
(terrain-following, see :mod:`.mesh3d`) carries the ground surface
``x = d(y, z)``; the Yee grid above and below it is DeepGPR's second-order
3D scheme. Both are advanced by :func:`DeepGPR.hybrid.functional3d.step`.

Sources and receivers act on one electric component ``component`` (DeepGPR
``source_direction``/``receiver_component``, default ``2`` = ``Ez``). The
source amplitude follows DeepGPR's convention ``J = w dl delta`` with ``dl``
the *model* spacing along the component (equal to DeepGPR's for
``fdtd_refinement = 1``).
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..config.constants import (
    CPML_SCALING_PROFILES,
    CPML_SIGMA_MAX_FACTOR,
    SPEED_OF_LIGHT,
    VACUUM_PERMITTIVITY,
)
from ..preprocessing.grid import pmlthick_revert
from ..preprocessing.model_setup import _ExtendModel
from ..solver.pml import build_pml_coeffs
from . import cuda3d
from . import functional3d as fn3
from . import graphs
from .coupling3d import InterfaceCoupling3D, horizontal_faces
from .dgtd3d import (FACE_ABC, FACE_INTERFACE, FACE_INTERIOR, FACE_PEC, build_geometry_3d,
                     point_operators_3d)
from .mesh3d import AIR, TetMesh, adaptive_column_tet_mesh, terrain_following_tet_mesh
from .reference3d import reference_tetrahedron
from .solver import DEFAULT_DG_PML_STRENGTH, DEFAULT_FLUX_ALPHA, FLUX_ALPHA, default_cfl_safety
from .topography import Surface3D


def _as_3d(t, name, dtype, device):
    t = torch.as_tensor(t)
    if t.ndim != 3:
        raise ValueError(f"{name} must be a 3D model (nx, ny, nz).")
    return t.to(device=device, dtype=dtype)


class HybridMaxwell3D:
    """Hybrid DGTD-FDTD (or pure Yee) 3D forward solver.

    Args:
        eps_r, sigma: Physical model ``(nx, ny, nz)``; with a ``surface`` every
            point above it becomes air.
        dx: Scalar or ``(dx, dy, dz)`` model spacing [m].
        surface: :class:`~DeepGPR.hybrid.topography.Surface3D` (physical frame).
        dg_band: ``None`` (pure Yee) or ``(top, bottom)`` physical depths [m] of
            the DGTD band, rounded to Yee E planes (strictly inside the grid,
            below/above the x CPML).
        mu_r: Relative permeability (``None`` = 1).
        pmlthick: DeepGPR CPML thickness (int or six values).
        dg_order: 1 or 2.
        dg_h: DG layer thickness and column spacing [m] (default model ``dx``).
        mesh: ``"adaptive"`` (default: per-column number of layers, layers 2/3-2
            ``dg_h`` thick, :func:`~DeepGPR.hybrid.mesh3d.adaptive_column_tet_mesh`),
            ``"terrain"`` (same number of layers in every column; relief makes
            them thin) or a :class:`~DeepGPR.hybrid.mesh3d.TetMesh` in the
            extended frame.
        mesh_options: ``h_y``, ``h_z``; for ``"terrain"`` also
            ``surface_refinement`` and ``jitter``.
        flux: ``"central"``, ``"upwind"`` or the penalty weight (default 0.1).
        dt, cfl_safety: Time step, or the fraction of the coupled leapfrog
            limit (Lanczos estimate) used when ``dt`` is None.
        side_boundary: ``"pec"`` or ``"abc"`` on the DG side faces.
        dg_pml_strength, fdtd_refinement: As in :class:`~DeepGPR.hybrid.HybridTM2D`.
            In the hybrid the four side faces (y, z) are absorbed by a UPML in
            auxiliary-equation form in both the Yee grid and the DG band, with
            DeepGPR's quartic profile and ``dg_pml_strength`` times its
            ``sigma_max``; the x faces keep DeepGPR's CPML. Pure Yee mode is
            DeepGPR's scheme (CPML on all faces).
        pml_profile: Kept for API symmetry with 2D (no effect in 3D).
        component: Source/receiver component (0, 1, 2).
        interface_penalty: Weight ``alpha_if`` of the dissipative penalty that
            relaxes the DG trace and the Yee interface-plane values towards
            each other (``gamma = alpha_if Y / 2``); it removes a spurious
            grid-scale mode trapped on the interface planes (0.1 suffices).
            The penalty is explicit: measured stability limits are
            ``alpha_if`` 1-3 for ``fdtd_refinement=1`` and 0.3-1 for 3. 0
            disables it (energy-conserving coupling).
        cuda_kernels: Use the fused CUDA kernels on GPUs (Yee and DG updates, see
            :mod:`.cuda3d`; :meth:`run` and the adjoint replay the step as a CUDA
            Graph, :mod:`.graphs`); otherwise the PyTorch implementation of the
            same operator.
    """

    def __init__(self, eps_r, sigma, dx, *, surface: Optional[Surface3D] = None,
                 dg_band: Optional[Tuple[float, float]] = None, mu_r=None, pmlthick=10,
                 dg_order: int = 1, dg_h: Optional[float] = None, mesh="adaptive",
                 mesh_options: Optional[dict] = None, flux=DEFAULT_FLUX_ALPHA,
                 dt: Optional[float] = None, cfl_safety: Optional[float] = None,
                 side_boundary: str = "pec", air_eps_r: float = 1.0, air_sigma: float = 0.0,
                 pml_max_velocity: float = SPEED_OF_LIGHT,
                 dg_pml_strength: float = DEFAULT_DG_PML_STRENGTH, fdtd_refinement: int = 1,
                 pml_profile: str = "matched", component: int = 2, nshot: int = 1,
                 interface_penalty: float = 0.1, cuda_kernels: bool = True, device="cpu",
                 dtype=torch.float64):
        t0 = time.perf_counter()
        self.device, self.dtype = torch.device(device), dtype
        self.nshot = int(nshot)
        self.component = int(component)
        eps = _as_3d(eps_r, "eps_r", dtype, self.device)
        sig = _as_3d(sigma, "sigma", dtype, self.device)
        mu = torch.ones_like(eps) if mu_r is None else _as_3d(mu_r, "mu_r", dtype, self.device)
        r = int(fdtd_refinement)
        if r < 1 or r != fdtd_refinement:
            raise ValueError("fdtd_refinement must be a positive integer.")
        md = tuple(float(v) for v in (dx if isinstance(dx, (tuple, list)) else (dx,) * 3))
        self.model_spacing = md
        self.fdtd_refinement = r
        self.spacing = tuple(v / r for v in md)
        self.dx, self.dy, self.dz = self.spacing
        self.model_shape = tuple(eps.shape)
        self.model_pml = [int(v) for v in pmlthick_revert(pmlthick, eps).tolist()]
        self.pml = [v * r for v in self.model_pml]
        self.origin = (self.pml[0] * self.dx, self.pml[2] * self.dy, self.pml[4] * self.dz)
        self.surface = surface
        self.flux_alpha = FLUX_ALPHA[flux] if isinstance(flux, str) else float(flux)
        self.pml_max_velocity = float(pml_max_velocity)
        self.dg_pml_strength = float(dg_pml_strength)
        if pml_profile not in ("matched", "deepgpr"):
            raise ValueError("pml_profile must be 'matched' or 'deepgpr'.")
        self.pml_profile = pml_profile
        self.side_boundary = side_boundary
        self.cuda_kernels = bool(cuda_kernels)
        self.interface_penalty = float(interface_penalty)
        self.air_values = (float(air_eps_r), float(air_sigma))
        self.model = (eps.detach(), sig.detach(), mu.detach())

        self._build_material_maps()
        self.eps_fd_pad, self.sig_fd_pad, self.mu_fd_pad = self._yee_fields(eps, sig, mu)
        self.NX, self.NY, self.NZ = self.eps_fd_pad.shape
        self.ext = (self.NX - 1, self.NY - 1, self.NZ - 1)
        self.widths = (self.ext[1] * self.dy, self.ext[2] * self.dz)

        self.mode = "fdtd" if dg_band is None else "hybrid"
        self.band = None
        self.mesh = self.geometry = None
        self._couplings = {}
        if dg_band is not None:
            p0 = self.pml[0]
            i_t = p0 + int(round(dg_band[0] / self.dx))
            i_b = p0 + int(round(dg_band[1] / self.dx))
            if not (p0 + 2 <= i_t < i_b <= self.NX - 3 - self.pml[1]):
                raise ValueError("dg_band must lie strictly between the x CPML layers.")
            self.band = (i_t, i_b)
            self.x_top, self.x_bottom = i_t * self.dx, i_b * self.dx
            self.dg_order = int(dg_order)
            self.dg_h = float(dg_h) if dg_h is not None else md[0]
            self.mesh = mesh if isinstance(mesh, TetMesh) else self._build_mesh(
                mesh, mesh_options or {})
            self.ref = reference_tetrahedron(self.dg_order)
            self.geometry = build_geometry_3d(self.mesh, self.ref)
            self.face_tags = self._face_tags()
            self._build_dg_index()
            self.dg_materials = self._dg_fields(*self.model)
            self.pml_sigma = self._dg_pml_profiles()
        self.setup_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        self.lambda_max = None
        if dt is None:
            self.lambda_max = self.estimate_lambda_max()
            if cfl_safety is None:
                cfl_safety = default_cfl_safety(self.flux_alpha)
            dt = cfl_safety * 2.0 / math.sqrt(self.lambda_max)
        self.dt = float(dt)
        self.dt_seconds = time.perf_counter() - t1
        self.static = self.make_static(self.dt)
        self.coefs = fn3.build_coefficients(self.static, self._fields_of_model(), self.dt)
        self.state = fn3.zero_state(self.static)

    # ------------------------------------------------------------ geometry
    def surface_x_ext(self, y_ext, z_ext):
        """Surface depth in the extended frame (flat continuation into the side CPML)."""
        oy, oz = self.origin[1], self.origin[2]
        _, mny, mnz = self.model_shape
        y = np.clip(np.asarray(y_ext) - oy, 0.0, (mny - 1) * self.model_spacing[1])
        z = np.clip(np.asarray(z_ext) - oz, 0.0, (mnz - 1) * self.model_spacing[2])
        return self.surface(y, z) + self.origin[0]

    def _build_mesh(self, kind, options) -> TetMesh:
        opts = dict(options)
        h_y = float(opts.pop("h_y", self.dg_h))
        h_z = float(opts.pop("h_z", self.dg_h))
        wy, wz = self.widths
        y_nodes = np.linspace(0.0, wy, max(1, int(round(wy / h_y))) + 1)
        z_nodes = np.linspace(0.0, wz, max(1, int(round(wz / h_z))) + 1)
        if self.surface is not None:
            surf = self.surface_x_ext
        else:
            mid = 0.5 * (self.x_top + self.x_bottom)
            surf = lambda Y, Z: np.full_like(np.asarray(Y, dtype=float), mid)  # noqa: E731
        if kind == "adaptive":
            m = adaptive_column_tet_mesh(y_nodes, z_nodes, self.x_top, self.x_bottom, surf,
                                         self.dg_h, **opts)
        elif kind == "terrain":
            m = terrain_following_tet_mesh(y_nodes, z_nodes, self.x_top, self.x_bottom, surf,
                                           self.dg_h, **opts)
        else:
            raise ValueError("mesh must be 'adaptive', 'terrain' or a TetMesh.")
        if self.surface is None:
            m.region[:] = 1
        return m

    def _face_tags(self):
        g, fm = self.geometry, self.ref.fmask
        tags = np.full((g.K, 4), FACE_INTERIOR, dtype=np.int64)
        tol = 1e-6 * min(self.spacing)
        side = {"pec": FACE_PEC, "abc": FACE_ABC}[self.side_boundary]
        wy, wz = self.widths
        for k, f in zip(*np.nonzero(g.boundary)):
            xf, yf, zf = g.x[k, fm[f]], g.y[k, fm[f]], g.z[k, fm[f]]
            if np.all(np.abs(xf - self.x_top) < tol) or np.all(np.abs(xf - self.x_bottom) < tol):
                tags[k, f] = FACE_INTERFACE
            elif (np.all(np.abs(yf) < tol) or np.all(np.abs(yf - wy) < tol)
                  or np.all(np.abs(zf) < tol) or np.all(np.abs(zf - wz) < tol)):
                tags[k, f] = side
            else:
                raise ValueError("A DG boundary face is not on the band boundary.")
        return tags

    # ------------------------------------------------------------ materials
    def _build_material_maps(self):
        dev = self.device
        mnx, mny, mnz = self.model_shape
        mdx, mdy, mdz = self.model_spacing
        r = self.fdtd_refinement
        self._fill = None
        if self.surface is not None:
            Y, Z = np.meshgrid(np.arange(mny) * mdy, np.arange(mnz) * mdz, indexing="ij")
            depth = torch.as_tensor(self.surface(Y, Z))
            first = torch.clamp(torch.ceil(depth / mdx - 1e-9).long(), 0, mnx - 1)
            rows = torch.maximum(torch.arange(mnx)[:, None, None], first[None])
            jj = torch.arange(mny)[None, :, None].expand(mnx, mny, mnz)
            kk = torch.arange(mnz)[None, None, :].expand(mnx, mny, mnz)
            self._fill = (rows.to(dev), jj.to(dev), kk.to(dev))
        self._up = None
        if r > 1:
            up = lambda n: torch.clamp(torch.floor(torch.arange(n * r) / r + 0.5).long(), 0, n - 1)  # noqa: E731
            self._up = tuple(up(n).to(dev) for n in self.model_shape)
        self.shape = (mnx * r, mny * r, mnz * r)
        self._air_fd = None
        if self.surface is not None:
            nx, ny, nz = self.shape
            Y, Z = np.meshgrid(np.arange(ny) * self.dy, np.arange(nz) * self.dz, indexing="ij")
            depth = torch.as_tensor(self.surface(Y, Z))
            xs = torch.arange(nx, dtype=torch.float64) * self.dx
            self._air_fd = (xs[:, None, None] < depth[None]).to(dev)

    def _yee_fields(self, eps, sig, mu):
        if self._up is not None:
            i, j, k = self._up
            eps, sig, mu = (t[i][:, j][:, :, k] for t in (eps, sig, mu))
        if self._air_fd is not None:
            air = self._air_fd
            eps = torch.where(air, torch.full_like(eps, self.air_values[0]), eps)
            sig = torch.where(air, torch.full_like(sig, self.air_values[1]), sig)
            mu = torch.where(air, torch.ones_like(mu), mu)
        # The one-cell halo is never updated; it is replicated (not zero) so that no
        # coefficient divides by zero there (0 * inf would poison the adjoint).
        ext = lambda t: F.pad(_ExtendModel.apply(t, self.pml)[None, None], (0, 1, 0, 1, 0, 1),  # noqa: E731
                              mode="replicate")[0, 0]
        return ext(eps), ext(sig), ext(mu)

    def _build_dg_index(self):
        p = self.mesh.vertices[self.mesh.tets].mean(axis=1)
        dims = [n + self.model_pml[2 * a] + self.model_pml[2 * a + 1]
                for a, n in enumerate(self.model_shape)]
        idx = [np.clip(np.rint(p[:, a] / self.model_spacing[a]).astype(np.int64), 0, dims[a] - 1)
               for a in range(3)]
        self._dg_index = tuple(torch.as_tensor(i, device=self.device) for i in idx)
        self._dg_air = (torch.as_tensor(self.mesh.region == AIR, device=self.device)
                        if self.surface is not None else None)

    def _dg_fields(self, eps, sig, mu):
        if self._fill is not None:
            rows, jj, kk = self._fill
            eps, sig, mu = (t[rows, jj, kk] for t in (eps, sig, mu))
        i, j, k = self._dg_index
        eps, sig, mu = (_ExtendModel.apply(t, self.model_pml)[i, j, k] for t in (eps, sig, mu))
        if self._dg_air is not None:
            air = self._dg_air
            eps = torch.where(air, torch.full_like(eps, self.air_values[0]), eps)
            sig = torch.where(air, torch.full_like(sig, self.air_values[1]), sig)
            mu = torch.where(air, torch.ones_like(mu), mu)
        return eps, sig, mu

    def material_fields(self, eps_r, sigma, mu_r=None):
        """Differentiable ``(eps_fd_pad, sig_fd_pad, dg_eps, dg_sig, dg_mu)``."""
        mu = self.model[2] if mu_r is None else mu_r
        eps_fd, sig_fd, _ = self._yee_fields(eps_r, sigma, mu)
        dg = (None, None, None)
        if self.mode == "hybrid":
            dg = self._dg_fields(eps_r, sigma, mu)
        return (eps_fd, sig_fd) + tuple(dg)

    def _fields_of_model(self, sigma_zero=False, dtype=None):
        cast = (lambda t: t) if dtype is None else (lambda t: t.to(dtype))  # noqa: E731
        sig = torch.zeros_like(self.sig_fd_pad) if sigma_zero else self.sig_fd_pad
        dg = (None, None, None)
        if self.mode == "hybrid":
            e, s, m = self.dg_materials
            dg = (cast(e), cast(torch.zeros_like(s) if sigma_zero else s), cast(m))
        return (cast(self.eps_fd_pad), cast(sig)) + dg

    def upml_sigma(self, axis: int, coord) -> np.ndarray:
        """Side-UPML conductivity [S/m] along ``axis`` (1 = y, 2 = z) at extended
        coordinates: DeepGPR's quartic CPML profile with ``sigma_max`` of the model
        spacing times ``dg_pml_strength`` (fixed reference speed)."""
        coord = np.asarray(coord, dtype=np.float64)
        order = CPML_SCALING_PROFILES["quartic"]
        sp, msp = self.spacing[axis], self.model_spacing[axis]
        sig_max = (CPML_SIGMA_MAX_FACTOR * (order + 1) * VACUUM_PERMITTIVITY
                   * self.pml_max_velocity / msp * self.dg_pml_strength)
        out = np.zeros_like(coord)
        for thick, inner, outward in ((self.pml[2 * axis], self.pml[2 * axis] * sp, -1.0),
                                      (self.pml[2 * axis + 1],
                                       (self.ext[axis] - self.pml[2 * axis + 1]) * sp, 1.0)):
            if thick > 0:
                d = (coord - inner) * outward / (thick * sp)
                out = out + sig_max * np.clip(d, 0.0, None) ** order
        return out

    def _dg_pml_profiles(self):
        g = self.geometry
        out = [None]
        for a, coord in ((1, g.y), (2, g.z)):
            s = self.upml_sigma(a, coord)
            out.append(s if np.any(s > 0) else None)
        return tuple(out)

    # ------------------------------------------------------------ static data
    def _coupling(self, dtype):
        key = str(dtype)
        if key not in self._couplings:
            tol = 1e-6 * min(self.spacing)
            out = []
            i_t, i_b = self.band
            for name, row, xp in (("top", i_t, self.x_top), ("bottom", i_b, self.x_bottom)):
                faces = horizontal_faces(self.geometry, xp, tol)
                cp = InterfaceCoupling3D(self.mesh, self.geometry, faces, xp, self.dy, self.dz,
                                         self.NY, self.NZ, dtype=dtype, device=self.device)
                ghost, inner = (row, row - 1) if name == "top" else (row - 1, row)
                out.append(fn3.Coupling3D(row, ghost, inner, cp.slots, cp.nodes,
                                          tuple(cp.to_dg), tuple(cp.to_fd)))
            self._couplings[key] = out
        return self._couplings[key]

    def make_static(self, dt, conservative=False, nshot=None, dtype=None) -> fn3.StepStatic3D:
        """Static operators for ``dt`` (``conservative``: no CPML, ABC -> PEC, central flux)."""
        nshot = self.nshot if nshot is None else nshot
        dtype = self.dtype if dtype is None else dtype
        pml = [0] * 6 if conservative else list(self.pml)
        upml = self.mode == "hybrid" and not conservative
        if upml:
            pml[2:6] = [0, 0, 0, 0]          # y/z faces: UPML (below), x faces: CPML
        velocity = self.pml_max_velocity
        coeffs = build_pml_coeffs(None, None, dt, list(self.spacing), *self.ext,
                                  torch.tensor(pml, dtype=torch.int32), self.device, dtype,
                                  max_velocity=velocity)
        chr = dt / (fn3.VACUUM_PERMEABILITY * self.mu_fd_pad.to(dtype))
        yee = fn3.yee_static_3d((self.NX, self.NY, self.NZ), self.spacing, pml, coeffs[6:], chr,
                                self.band, dtype, self.device)
        if upml and any(self.pml[2:6]):
            yee.upml = fn3.yee_upml_3d(yee, self.upml_sigma, self.mu_fd_pad, dt, dtype, self.device)
        dg, cps = None, []
        if self.mode == "hybrid":
            tags = self.face_tags.copy()
            if conservative:
                tags[tags == FACE_ABC] = FACE_PEC
            dg = fn3.dg_static_3d(self.geometry, tags, self.dg_materials[2].detach().cpu(), dt,
                                  0.0 if conservative else self.flux_alpha,
                                  None if conservative else self.pml_sigma, dtype, self.device,
                                  alpha_if=0.0 if conservative else self.interface_penalty)
            cps = self._coupling(dtype)
            if self.cuda_kernels:
                cuda3d.attach(dg, self.device, slots=[cp.slots for cp in cps])
        if self.cuda_kernels:
            cuda3d.attach_yee(yee, nshot, self.device, rows=[cp.row for cp in cps])
        return fn3.StepStatic3D(nshot, yee, dg, cps, dtype, self.device)

    # ------------------------------------------------------------ CFL
    def _e_inner(self, st, fields, a, b):
        eps_fd, dg_eps = fields
        y, d = st.yee, st.dg
        cell = self.dx * self.dy * self.dz
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        for k in range(3):
            total = total + cell * (y.weights[k] * eps_fd * a[k] * b[k]).sum()
        if d is not None:
            m = (dg_eps * d.J[:, 0])[None, None, :, None]
            total = total + (m * torch.matmul(a[3], d.mass) * b[3]).sum()
        return total

    def estimate_lambda_max(self, iterations: int = 60, seed: int = 0) -> float:
        """Largest eigenvalue of the conservative coupled operator (Lanczos, float64)."""
        st = self.make_static(1.0, conservative=True, nshot=1, dtype=torch.float64)
        fields = self._fields_of_model(sigma_zero=True, dtype=torch.float64)
        coefs = fn3.build_coefficients(st, fields, 1.0)
        eps_rel = (fields[0], fields[2])
        gen = torch.Generator(device="cpu").manual_seed(seed)
        y, d = st.yee, st.dg

        def rand():
            out = [torch.randn((1,) + y.shape, generator=gen, dtype=torch.float64).to(self.device)
                   * y.act[k] for k in range(3)]
            out.append(None if d is None else torch.randn((1, 3, d.K, d.Np), generator=gen,
                                                          dtype=torch.float64).to(self.device))
            return out

        def apply(v):
            s = list(fn3.zero_state(st))
            s[0:3] = v[0:3]
            if d is not None:
                s[18] = v[3]
            s = fn3.step(st, coefs, tuple(s))
            out = [v[k] - s[k] for k in range(3)]
            out.append(None if d is None else v[3] - s[18])
            return out

        ip = lambda a, b: self._e_inner(st, eps_rel, a, b)  # noqa: E731
        comb = lambda a, s, b: [None if x is None else x + s * z for x, z in zip(a, b)]  # noqa: E731
        scale = lambda a, s: [None if x is None else x * s for x in a]  # noqa: E731
        v = rand()
        v = scale(v, 1.0 / float(torch.sqrt(ip(v, v))))
        # Full re-orthogonalisation while the Krylov basis fits in ~2 GB; otherwise
        # the plain three-term recurrence (its largest Ritz value still converges
        # to lambda_max from below; lost orthogonality only duplicates Ritz values).
        ndof = sum(x.numel() for x in v if x is not None)
        reorth = ndof * iterations * 8 < 2e9
        basis, alphas, betas, w_prev, beta = [v], [], [], None, 0.0
        for _ in range(iterations):
            w = apply(v)
            alpha = float(ip(w, v))
            w = comb(w, -alpha, v)
            if w_prev is not None:
                w = comb(w, -beta, w_prev)
            for q in (basis if reorth else ()):
                w = comb(w, -float(ip(w, q)), q)
            alphas.append(alpha)
            beta = float(torch.sqrt(torch.clamp(ip(w, w), min=0.0)))
            if beta < 1e-12 * abs(alpha):
                break
            betas.append(beta)
            w_prev, v = v, scale(w, 1.0 / beta)
            if reorth:
                basis.append(v)
        n = len(alphas)
        T = np.diag(alphas) + np.diag(betas[: n - 1], 1) + np.diag(betas[: n - 1], -1)
        return float(np.linalg.eigvalsh(T).max())

    # ------------------------------------------------------------ points
    def in_band(self, x_ext) -> np.ndarray:
        x_ext = np.asarray(x_ext, dtype=np.float64)
        if self.mode == "fdtd":
            return np.zeros(x_ext.shape, dtype=bool)
        tol = 1e-9 * self.dx
        return (x_ext > self.x_top + tol) & (x_ext < self.x_bottom - tol)

    def point_ops(self, positions, *, source: bool) -> fn3.PointOps3D:
        """Operators for physical ``(x, y, z)`` positions ``(nshot, n, 3)`` [m].

        Yee: trilinear weights on the lattice of :attr:`component` (sources
        divided by the cell volume and the interface half weight); DG: basis
        values (sources ``M^-1 phi / J``), averaged over containing elements.
        Sources carry the dipole length ``dl`` (model spacing along the component)."""
        pos = np.asarray(torch.as_tensor(positions, dtype=torch.float64).cpu())
        if pos.ndim == 2:
            pos = np.broadcast_to(pos[None], (self.nshot,) + pos.shape)
        if pos.shape[0] != self.nshot or pos.shape[2] != 3:
            raise ValueError("positions must have shape (nshot, n, 3) in metres.")
        c = self.component
        dl = self.model_spacing[c] if source else 1.0
        off = [0.0, 0.0, 0.0]
        off[c] = 0.5
        act = self.static.yee.act[c].cpu().numpy()
        wrow = self.static.yee.weights[c].cpu().numpy()
        cell = self.dx * self.dy * self.dz
        fd = {"shot": [], "node": [], "item": [], "w": []}
        dg = {"shot": [], "elem": [], "item": [], "w": []}
        for s in range(pos.shape[0]):
            for item in range(pos.shape[1]):
                pe = pos[s, item] + np.asarray(self.origin)
                if not np.all((pe > 0) & (pe < np.asarray(self.ext) * np.asarray(self.spacing))):
                    raise ValueError(f"Point {pos[s, item]} lies outside the model.")
                if not self.in_band(pe[0]):
                    f = pe / np.asarray(self.spacing) - np.asarray(off)
                    i0 = np.floor(f + 1e-9).astype(int)
                    t = f - i0
                    for corner in np.ndindex(2, 2, 2):
                        w = np.prod([t[a] if corner[a] else 1 - t[a] for a in range(3)])
                        if abs(w) < 1e-12:
                            continue
                        ijk = tuple(i0[a] + corner[a] for a in range(3))
                        if source:
                            if not act[ijk]:
                                raise ValueError("FDTD source stencil touches an inactive node.")
                            w = w * dl / (wrow[ijk] * cell)
                        fd["shot"].append(s)
                        fd["node"].append((ijk[0] * self.NY + ijk[1]) * self.NZ + ijk[2])
                        fd["item"].append(item)
                        fd["w"].append(w)
                else:
                    (ids, phi), = point_operators_3d(self.geometry, self.mesh, pe[None])
                    if ids.size == 0:
                        raise ValueError(f"Point {pos[s, item]} is not inside a DG element.")
                    for k, ph in zip(ids, phi):
                        vec = dl * self.ref.Minv @ ph / self.geometry.J[k] if source else ph
                        dg["shot"].append(s)
                        dg["elem"].append(int(k))
                        dg["item"].append(item)
                        dg["w"].append(vec / ids.size)
        L = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=self.device)  # noqa: E731
        T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=self.dtype,  # noqa: E731
                                      device=self.device)
        npn = self.ref.np_ if self.mode == "hybrid" else 1
        return fn3.PointOps3D(L(fd["shot"]), L(fd["node"]), L(fd["item"]), T(fd["w"]),
                              L(dg["shot"]), L(dg["elem"]), L(dg["item"]),
                              T(np.asarray(dg["w"]).reshape(-1, npn)), int(pos.shape[1]), c)

    # ------------------------------------------------------------ run
    def reset(self):
        self.state = fn3.zero_state(self.static)

    def energy(self, prev, now):
        dg = (None, None) if self.mode == "fdtd" else (self.dg_materials[0], self.dg_materials[2])
        return fn3.energy(self.static, self.eps_fd_pad, self.mu_fd_pad, dg[0], dg[1], prev, now)

    def run(self, source_amplitudes, source_positions, receiver_positions, *,
            callback=None, callback_every: int = 0, energy_every: int = 0) -> dict:
        """Run ``nt`` steps; positions are physical ``(x, y, z)`` [m]."""
        w = torch.as_tensor(source_amplitudes, dtype=self.dtype, device=self.device)
        if w.ndim == 1:
            w = w[None]
        nt = w.shape[1]
        src = self.point_ops(source_positions, source=True)
        rec = self.point_ops(receiver_positions, source=False)
        nsrc = int(np.asarray(source_positions).shape[-2])
        if w.shape[0] == 1 and nsrc > 1:
            w = w.expand(nsrc, -1)
        self.reset()
        data = torch.zeros((self.nshot, nt, rec.count), dtype=self.dtype, device=self.device)
        energies, steps = [], []
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        graph = None
        if not energy_every and self.cuda_kernels:
            graph = graphs.make(self.static, self.coefs, src, rec, w.shape[0])
        with torch.no_grad():
            if graph is not None:
                # CUDA Graph replay of the same step (removes the per-step host overhead)
                graph.set_state(self.state)
                for n in range(nt):
                    data[:, n] = graph(w[:, n])
                    if callback is not None and callback_every and n % callback_every == 0:
                        self.state = tuple(graph.state)
                        callback(n, self)
                self.state = graph.snapshot()
            else:
                for n in range(nt):
                    prev = self.state
                    self.state = fn3.step(self.static, self.coefs, self.state, src, w[:, n])
                    data[:, n] = fn3.sample(self.static, rec, self.state)
                    if energy_every and n % energy_every == 0:
                        energies.append(self.energy(prev, self.state))
                        steps.append(n)
                    if callback is not None and callback_every and n % callback_every == 0:
                        callback(n, self)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - t0
        return {"receivers": data, "energy": torch.stack(energies) if energies else None,
                "energy_steps": steps, "seconds": seconds, "dt": self.dt}

    def summary(self) -> dict:
        info = {"mode": self.mode, "dt": self.dt, "spacing": self.spacing,
                "model_spacing": self.model_spacing, "fdtd_refinement": self.fdtd_refinement,
                "pml": self.pml, "lambda_max": self.lambda_max, "grid": (self.NX, self.NY, self.NZ),
                "setup_seconds": self.setup_seconds, "dt_seconds": self.dt_seconds}
        y = self.static.yee
        info["fdtd_dofs"] = int(sum(int(a.sum()) for a in y.act))
        if self.mode == "hybrid":
            info.update(self.mesh.quality())
            info["dg_order"] = self.dg_order
            info["dg_dofs"] = 6 * self.geometry.K * self.ref.np_
            info["band"] = self.band
        return info


class FieldSampler3D:
    """Sample one electric component of a :class:`HybridMaxwell3D` at physical
    points (snapshots, GIFs): trilinear on the Yee lattice of the component
    outside the band, DG basis values inside it.

    Args:
        solver: The solver.
        points: ``(n, 3)`` physical ``(x, y, z)`` [m].
        component: 0, 1 or 2 (default: the solver's).
    """

    def __init__(self, solver: HybridMaxwell3D, points, component: Optional[int] = None):
        from .mesh3d import locate_points_columns

        self.solver = solver
        self.c = solver.component if component is None else int(component)
        pe = np.atleast_2d(np.asarray(points, dtype=np.float64)) + np.asarray(solver.origin)
        self.n = pe.shape[0]
        inb = solver.in_band(pe[:, 0])
        dev = solver.device
        off = np.zeros(3)
        off[self.c] = 0.5
        f = pe[~inb] / np.asarray(solver.spacing) - off
        i0 = np.floor(f).astype(np.int64)
        t = f - i0
        nodes, weights = [], []
        for corner in np.ndindex(2, 2, 2):
            ijk = i0 + np.array(corner)
            w = np.prod(np.where(np.array(corner)[None] == 1, t, 1 - t), axis=1)
            ijk = np.clip(ijk, 0, np.array([solver.NX, solver.NY, solver.NZ]) - 1)
            nodes.append((ijk[:, 0] * solver.NY + ijk[:, 1]) * solver.NZ + ijk[:, 2])
            weights.append(w)
        self.fd_idx = torch.as_tensor(np.nonzero(~inb)[0], device=dev)
        self.fd_nodes = torch.as_tensor(np.stack(nodes, 1), device=dev)
        self.fd_w = torch.as_tensor(np.stack(weights, 1), dtype=solver.dtype, device=dev)
        self.dg_idx = torch.as_tensor(np.nonzero(inb)[0], device=dev)
        if inb.any():
            elem, lam = locate_points_columns(solver.mesh, pe[inb])
            if np.any(elem < 0):
                raise ValueError("A sample point in the band is not inside a DG element.")
            r, s_, t_ = -1 + 2 * lam[:, 1], -1 + 2 * lam[:, 2], -1 + 2 * lam[:, 3]
            self.dg_elem = torch.as_tensor(elem, device=dev)
            self.dg_phi = torch.as_tensor(solver.ref.basis_at(r, s_, t_), dtype=solver.dtype,
                                          device=dev)

    def __call__(self, shot: int = 0) -> torch.Tensor:
        st = self.solver.state
        out = torch.zeros(self.n, dtype=self.solver.dtype, device=self.solver.device)
        if self.fd_idx.numel():
            f = st[self.c][shot].reshape(-1)
            out[self.fd_idx] = (f[self.fd_nodes] * self.fd_w).sum(1)
        if self.dg_idx.numel():
            e = st[18][shot, self.c]
            out[self.dg_idx] = (e[self.dg_elem] * self.dg_phi).sum(1)
        return out


__all__ = ["FieldSampler3D", "HybridMaxwell3D"]
