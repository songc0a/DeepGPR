"""2D Ez-TM DGTD-Yee-FDTD hybrid solver with body-fitted topography.

Layout (DeepGPR 2D convention: ``x`` = axis 0 = depth, ``y`` = axis 1)::

      x = 0     +--------------------------------------+  PEC
                |  CPML (Yee)                          |
                |  Yee-FDTD (air)                      |
      x_top     |======== interface (Ez row i_top) ====|
                |  DGTD band: triangles fitted to the  |
                |  ground surface x = d(y) (air above, |
                |  ground below), CPML in the side     |
                |  columns                             |
      x_bottom  |====== interface (Ez row i_bottom) ===|
                |  Yee-FDTD (ground)                   |
                |  CPML (Yee)                          |
                +--------------------------------------+

The Yee part is DeepGPR's second-order scheme (:mod:`.yee`), the DGTD part a
nodal DG scheme (:mod:`.dgtd`), and the two are coupled by the energy-neutral
mortar of :mod:`.coupling`. All three share one leapfrog step::

    Yee H^(n+1/2)  <- Ez^n
    DG  H^(n+1/2)  <- E_DG^n, FDTD interface rows Ez^n (E* = I E_b)
    Yee ghost Hy   <- 2 H_b - Hy_inner,  H_b = C^T Hy_DG^(n+1/2) / dy
    Yee Ez^(n+1)   <- H^(n+1/2)
    DG  E^(n+1)    <- H_DG^(n+1/2)
    sources (w[n]) and receivers (E^(n+1)), as in DeepGPR.

``dt`` defaults to ``cfl_safety`` times the leapfrog limit ``2 / sqrt(lambda_max)``
of the *coupled* operator, with ``lambda_max`` from a Lanczos iteration.

Special cases: ``dg_band=None`` runs the pure Yee scheme (identical to
:func:`DeepGPR.compute` with ``fdtd_order=2``), ``dg_band=(None, None)``
runs DGTD on the whole extended domain (CPML in DG on all faces).
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional, Tuple, Union

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
from ..utils.logger import get_logger
from . import cuda2d, graphs
from . import functional as fn
from .coupling import InterfaceCoupling, interface_faces
from .dgtd import (
    FACE_ABC,
    FACE_INTERFACE,
    FACE_INTERIOR,
    FACE_PEC,
    DGTM2D,
    build_geometry,
)
from .mesh import AIR, TriMesh, delaunay_band_mesh, terrain_following_band_mesh
from .reference import reference_triangle
from .topography import Surface
from .yee import YeeTM2D

_LOGGER = get_logger(__name__)

FLUX_ALPHA = {"central": 0.0, "upwind": 1.0}

#: Default interior flux: central plus 10 % of the upwind penalty. Pure central
#: fluxes leave the grid-scale spurious modes of DG on unstructured meshes
#: undamped (a point source excites them strongly); alpha = 0.1 removes them
#: while keeping ~92 % of the central-flux leapfrog limit (P1 and P2).
DEFAULT_FLUX_ALPHA = 0.1

#: Measured stable fraction of the central-flux limit 2/sqrt(lambda_max) for
#: leapfrog with explicit penalty terms (random initial data, 15000 steps,
#: bisection; see docs/HYBRID_DGTD.md). Default dt = 0.9 * fraction * limit.
_PENALTY_STABLE_FRACTION = ((0.0, 1.0), (0.05, 0.96), (0.1, 0.92), (0.25, 0.82),
                            (0.5, 0.67), (1.0, 0.30))


#: Factor on DeepGPR's optimal Yee ``sigma_max`` used for the CPML inside the
#: DGTD band. The Yee-optimal profile is too steep for P1/P2 DG: measured
#: reflection (vacuum, 1 GHz, 20 layers of 1 cm) 1.2e-3 at 1.0, 5.3e-4 at 0.4
#: (P1) and 1.8e-4 -> 7.1e-5 (P2); the continuous-PML term exp(-1.6 f N)
#: stays below 3e-6 for N = 20 at f = 0.4.
DEFAULT_DG_PML_STRENGTH = 0.4


def default_cfl_safety(alpha: float) -> float:
    """0.9 times the (interpolated) stable fraction for penalty weight ``alpha``."""
    a, f = zip(*_PENALTY_STABLE_FRACTION)
    return 0.9 * float(np.interp(alpha, a, f))


def _as_2d(t, name, dtype, device):
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    if t.ndim == 3 and t.shape[2] == 1:
        t = t[:, :, 0]
    if t.ndim != 2:
        raise ValueError(f"{name} must be a 2D model (nx, ny).")
    return t.to(device=device, dtype=dtype)


def _pad_halo(t3):
    return F.pad(t3, (0, 1, 0, 1, 0, 1))[:, :, 0]


class HybridTM2D:
    """Hybrid DGTD-FDTD (or pure Yee / pure DGTD) 2D Ez-TM forward solver.

    Args:
        eps_r, sigma: Physical model ``(nx, ny)`` (DeepGPR layout: axis 0
            depth). With a ``surface`` they describe the ground; every point
            above the surface becomes air (``air_eps_r``, ``air_sigma``).
        dx: Grid spacing, scalar or ``(dx, dy)`` [m].
        surface: :class:`~DeepGPR.hybrid.topography.Surface` in the physical
            frame, or ``None`` (materials from the grid only).
        dg_band: ``None`` (pure Yee), or ``(top, bottom)`` physical depths
            [m] of the DGTD band, rounded to Ez rows; ``None`` entries extend
            the band to that outer boundary (DGTD then owns that CPML face).
        mu_r: Relative permeability (``None`` = 1, not differentiable).
        pmlthick: DeepGPR CPML thickness (int or four values).
        dg_order: DG polynomial order (1 or 2 recommended).
        dg_h: Target DG element size [m] (default ``dx``; ``2 dx`` is a good
            choice for ``dg_order=2``).
        mesh: ``"terrain"``, ``"delaunay"`` or a :class:`TriMesh` in the
            extended frame.
        mesh_options: Extra keyword arguments of the mesh generator
            (e.g. ``h_y``, ``surface_refinement``, ``jitter``, ``h_surface``,
            ``refine_width``, ``h_interface``).
        flux: ``"central"`` (energy conserving), ``"upwind"`` or the upwind
            penalty weight ``alpha`` in ``[0, 1]`` (default 0.1).
        dt: Time step [s]; default from the coupled CFL limit.
        cfl_safety: Fraction of the central-flux leapfrog limit used when
            ``dt`` is None; default :func:`default_cfl_safety` of ``alpha``.
        side_boundary: ``"pec"`` or ``"abc"`` (Silver-Mueller) on the DG
            side faces (behind the side CPML).
        dg_pml_strength: Factor on DeepGPR's ``sigma_max`` for the CPML inside
            the DGTD band (the Yee CPML keeps DeepGPR's profile).
        fdtd_refinement: Yee spacing ``dx / r`` (the model stays on ``dx``).
            The global ``dt`` is set by the DGTD band, so a Yee grid refined
            until its own CFL limit reaches that ``dt`` costs almost no extra
            time steps and reduces the interface reflection and the Yee
            dispersion by ``r^2``. For P1 with ``h = dx`` the DG limit is close
            to the Yee limit at ``dx/3``: ``r = 3`` costs 2-20 % more steps
            (measured), ``r = 2`` none. Odd ``r`` avoids Yee nodes on
            model-pixel boundaries.
        pml_profile: ``"matched"`` (default): in the hybrid the Yee CPML gets
            the same physical sigma profile as the DG CPML; ``"deepgpr"``: the
            Yee CPML keeps DeepGPR's profile. Pure Yee always uses DeepGPR's.
        nshot: Number of simultaneous shots.
        cuda_kernels: Use the fused CUDA kernels on GPUs (Yee and DG updates
            with exact transposes, :mod:`.cuda2d`; the adjoint then uses the
            hand-written step transpose of :mod:`.vjp2d`, and :meth:`run` replays
            small steps as CUDA Graphs); otherwise the PyTorch implementation of
            the same step.
    """

    def __init__(
        self,
        eps_r,
        sigma,
        dx,
        *,
        surface: Optional[Surface] = None,
        dg_band: Optional[Tuple[Optional[float], Optional[float]]] = None,
        mu_r=None,
        pmlthick=10,
        dg_order: int = 1,
        dg_h: Optional[float] = None,
        mesh: Union[str, TriMesh] = "terrain",
        mesh_options: Optional[dict] = None,
        flux: Union[str, float] = DEFAULT_FLUX_ALPHA,
        dt: Optional[float] = None,
        cfl_safety: Optional[float] = None,
        side_boundary: str = "pec",
        air_eps_r: float = 1.0,
        air_sigma: float = 0.0,
        pml_max_velocity: float = SPEED_OF_LIGHT,
        dg_pml_strength: float = DEFAULT_DG_PML_STRENGTH,
        fdtd_refinement: int = 1,
        pml_profile: str = "matched",
        nshot: int = 1,
        cuda_kernels: bool = True,
        device="cpu",
        dtype=torch.float64,
    ):
        t_setup = time.perf_counter()
        self.cuda_kernels = bool(cuda_kernels)
        self.device = torch.device(device)
        self.dtype = dtype
        self.nshot = int(nshot)
        eps = _as_2d(eps_r, "eps_r", dtype, self.device)
        sig = _as_2d(sigma, "sigma", dtype, self.device)
        mu = torch.ones_like(eps) if mu_r is None else _as_2d(mu_r, "mu_r", dtype, self.device)
        r = int(fdtd_refinement)
        if r < 1 or r != fdtd_refinement:
            raise ValueError("fdtd_refinement must be a positive integer.")
        if isinstance(dx, (tuple, list)):
            mdx, mdy = float(dx[0]), float(dx[1])
        else:
            mdx = mdy = float(dx)
        # Model grid (what the user supplies) and Yee grid (refined by r).
        self.model_dx, self.model_dy = mdx, mdy
        self.fdtd_refinement = r
        self.dx, self.dy = mdx / r, mdy / r
        mnx, mny = eps.shape
        self.model_shape = (mnx, mny)
        self.model_pml = [int(v) for v in pmlthick_revert(pmlthick, eps[:, :, None]).tolist()]
        self.pml = [v * r for v in self.model_pml]
        p0, p1, p2, p3 = self.pml[:4]
        self.ox, self.oy = p0 * self.dx, p2 * self.dy
        self.surface = surface
        self.flux_alpha = FLUX_ALPHA[flux] if isinstance(flux, str) else float(flux)
        self.pml_max_velocity = float(pml_max_velocity)
        self.dg_pml_strength = float(dg_pml_strength)
        if pml_profile not in ("matched", "deepgpr"):
            raise ValueError("pml_profile must be 'matched' or 'deepgpr'.")
        self.pml_profile = pml_profile
        self.air_values = (float(air_eps_r), float(air_sigma), 1.0)

        # ---- material maps (index maps kept so that material_fields() can be
        # re-evaluated differentiably during FWI) and Yee materials.
        self._build_material_maps(mnx, mny, r)
        self.model = tuple(t.detach() for t in (eps, sig, mu))
        self.eps_fd_pad, self.sig_fd_pad, self.mu_fd_pad = self._yee_fields(eps, sig, mu)
        self.NX, self.NY = self.eps_fd_pad.shape
        self.nx_ext, self.ny_ext = self.NX - 1, self.NY - 1
        self.width = self.ny_ext * self.dy

        # ---- band and mesh
        self.mode = "fdtd" if dg_band is None else (
            "dgtd" if dg_band[0] is None and dg_band[1] is None else "hybrid")
        self.i_top = self.i_bottom = None
        self.mesh = None
        self.dg = None
        self.fd = None
        self.couplings = {}
        if dg_band is not None:
            top, bottom = dg_band
            self.i_top = None if top is None else p0 + int(round(top / self.dx))
            self.i_bottom = None if bottom is None else p0 + int(round(bottom / self.dx))
            self.x_top = 0.0 if self.i_top is None else self.i_top * self.dx
            self.x_bottom = self.nx_ext * self.dx if self.i_bottom is None else self.i_bottom * self.dx
            if self.x_bottom <= self.x_top:
                raise ValueError("dg_band must satisfy top < bottom.")
            self.dg_order = int(dg_order)
            self.dg_h = float(dg_h) if dg_h is not None else self.model_dx
            self.mesh = self._build_mesh(mesh, mesh_options or {})
            self.ref = reference_triangle(self.dg_order)
            self.geometry = build_geometry(self.mesh, self.ref)
            self.face_tags = self._face_tags(side_boundary)
            self._build_dg_index()
            self.dg_materials = self._dg_fields(*self.model)
            self.pml_sigma = self._dg_pml_profiles()
        self.setup_seconds = time.perf_counter() - t_setup

        t_dt = time.perf_counter()
        self.lambda_max = None
        if dt is None:
            self.lambda_max = self.estimate_lambda_max()
            if cfl_safety is None:
                cfl_safety = default_cfl_safety(self.flux_alpha)
            dt = cfl_safety * 2.0 / math.sqrt(self.lambda_max)
        self.dt = float(dt)
        self.dt_seconds = time.perf_counter() - t_dt
        self._build_subsolvers(self.dt, conservative=False)
        self.static = self._make_static(self.fd, self.dg, self.couplings, self.nshot)
        self.coefs = fn.build_coefficients(self.static, self._fields_of_model(), self.dt)
        self.state = fn.zero_state(self.static)

    # ------------------------------------------------------------ geometry
    def surface_x_ext(self, y_ext: np.ndarray) -> np.ndarray:
        """Surface depth in the extended frame (flat continuation into the side CPML)."""
        y_phys = np.clip(np.asarray(y_ext) - self.oy, 0.0,
                         (self.model_shape[1] - 1) * self.model_dy)
        return self.surface(y_phys) + self.ox

    def _build_mesh(self, mesh, options) -> TriMesh:
        if isinstance(mesh, TriMesh):
            return mesh
        if self.surface is not None:
            surf = self.surface_x_ext
        else:
            mid = 0.5 * (self.x_top + self.x_bottom)
            surf = lambda y: np.full_like(np.asarray(y, dtype=float), mid)  # noqa: E731
        opts = dict(options)
        if mesh == "terrain":
            h_y = float(opts.pop("h_y", self.dg_h))
            ncol = max(1, int(round(self.width / h_y)))
            y_nodes = np.linspace(0.0, self.width, ncol + 1)
            m = terrain_following_band_mesh(y_nodes, self.x_top, self.x_bottom, surf,
                                            self.dg_h, **opts)
        elif mesh == "delaunay":
            m = delaunay_band_mesh(self.width, self.x_top, self.x_bottom, surf, self.dg_h, **opts)
        else:
            raise ValueError("mesh must be 'terrain', 'delaunay' or a TriMesh.")
        if self.surface is None:
            m.region[:] = 1  # no topography: every element samples the grid
        return m

    def _face_tags(self, side_boundary) -> np.ndarray:
        g = self.geometry
        tags = np.full((g.K, 3), FACE_INTERIOR, dtype=np.int64)
        fm = self.ref.fmask
        tol = 1e-6 * min(self.dx, self.dy)
        side = {"pec": FACE_PEC, "abc": FACE_ABC}[side_boundary]
        for k, f in zip(*np.nonzero(g.boundary)):
            xf, yf = g.x[k, fm[f]], g.y[k, fm[f]]
            if np.all(np.abs(xf - self.x_top) < tol):
                tags[k, f] = FACE_PEC if self.i_top is None else FACE_INTERFACE
            elif np.all(np.abs(xf - self.x_bottom) < tol):
                tags[k, f] = FACE_PEC if self.i_bottom is None else FACE_INTERFACE
            elif np.all(np.abs(yf) < tol) or np.all(np.abs(yf - self.width) < tol):
                tags[k, f] = side
            else:
                raise ValueError("A DG boundary face is not on the band boundary.")
        return tags

    # ------------------------------------------------------------ materials
    def _build_material_maps(self, mnx: int, mny: int, r: int) -> None:
        """Index maps from the model grid to the Yee grid and the DG ground model."""
        dev = self.device
        mdx, mdy = self.model_dx, self.model_dy
        self._fill = None
        if self.surface is not None:
            # Ground values continued upwards over the surface (for DG sampling),
            # so near-surface ground elements never pick up staircase air.
            depth_m = torch.as_tensor(self.surface(np.arange(mny) * mdy), dtype=torch.float64)
            first = torch.clamp(torch.ceil(depth_m / mdx - 1e-9).long(), 0, mnx - 1)
            rows = torch.maximum(torch.arange(mnx)[:, None], first[None, :]).to(dev)
            cols = torch.arange(mny, device=dev)[None, :].expand(mnx, mny)
            self._fill = (rows, cols)
        self._up = None
        if r > 1:
            # Model pixels (squares centred on the model nodes) resampled to Yee nodes.
            ii = torch.clamp(torch.floor(torch.arange(mnx * r) / r + 0.5).long(), 0, mnx - 1)
            jj = torch.clamp(torch.floor(torch.arange(mny * r) / r + 0.5).long(), 0, mny - 1)
            self._up = (ii.to(dev), jj.to(dev))
        self.nx, self.ny = mnx * r, mny * r
        self._air_fd = None
        if self.surface is not None:
            xs = torch.arange(self.nx, dtype=torch.float64) * self.dx
            depth = torch.as_tensor(self.surface(np.arange(self.ny) * self.dy), dtype=torch.float64)
            self._air_fd = (xs[:, None] < depth[None, :]).to(dev)

    def _yee_fields(self, eps, sig, mu):
        """Yee materials (extended + halo) from model-grid materials (differentiable)."""
        if self._up is not None:
            ii, jj = self._up
            eps, sig, mu = (t[ii][:, jj] for t in (eps, sig, mu))
        if self._air_fd is not None:
            air = self._air_fd
            eps = torch.where(air, torch.full_like(eps, self.air_values[0]), eps)
            sig = torch.where(air, torch.full_like(sig, self.air_values[1]), sig)
            mu = torch.where(air, torch.ones_like(mu), mu)
        ext = lambda t: _pad_halo(_ExtendModel.apply(t[:, :, None], self.pml))  # noqa: E731
        return ext(eps), ext(sig), ext(mu)

    def _build_dg_index(self) -> None:
        """Model pixel (extended model grid) containing each element centroid."""
        p = self.mesh.vertices[self.mesh.triangles].mean(axis=1)
        mnx, mny = self.model_shape
        gx = mnx + self.model_pml[0] + self.model_pml[1]
        gy = mny + self.model_pml[2] + self.model_pml[3]
        i = np.clip(np.rint(p[:, 0] / self.model_dx).astype(np.int64), 0, gx - 1)
        j = np.clip(np.rint(p[:, 1] / self.model_dy).astype(np.int64), 0, gy - 1)
        self._dg_index = (torch.as_tensor(i, device=self.device),
                          torch.as_tensor(j, device=self.device))
        self._dg_air = (torch.as_tensor(self.mesh.region == AIR, device=self.device)
                        if self.surface is not None else None)

    def _dg_fields(self, eps, sig, mu):
        """Element materials (differentiable): air above the surface, else the
        model pixel containing the centroid (ground continued over the surface)."""
        if self._fill is not None:
            rows, cols = self._fill
            eps, sig, mu = (t[rows, cols] for t in (eps, sig, mu))
        ext = lambda t: _ExtendModel.apply(t[:, :, None], self.model_pml)[:, :, 0]  # noqa: E731
        it, jt = self._dg_index
        eps, sig, mu = (ext(t)[it, jt] for t in (eps, sig, mu))
        if self._dg_air is not None:
            air = self._dg_air
            eps = torch.where(air, torch.full_like(eps, self.air_values[0]), eps)
            sig = torch.where(air, torch.full_like(sig, self.air_values[1]), sig)
            mu = torch.where(air, torch.ones_like(mu), mu)
        return eps, sig, mu

    def material_fields(self, eps_r, sigma, mu_r=None):
        """Differentiable map from model-grid ``(eps_r, sigma)`` to the solver
        materials ``(eps_fd_pad, sig_fd_pad, dg_eps, dg_sig, dg_mu)`` (``None``
        for absent parts). ``mu_r`` defaults to the solver's (not inverted)."""
        mu = self.model[2] if mu_r is None else mu_r
        eps_fd = sig_fd = dg_eps = dg_sig = dg_mu = None
        if self.mode != "dgtd":
            eps_fd, sig_fd, _ = self._yee_fields(eps_r, sigma, mu)
        if self.mode != "fdtd":
            dg_eps, dg_sig, dg_mu = self._dg_fields(eps_r, sigma, mu)
        return eps_fd, sig_fd, dg_eps, dg_sig, dg_mu

    def _dg_pml_profiles(self):
        """CPML conductivity at the DG nodes (DeepGPR profile, fixed speed)."""
        g = self.geometry
        order = CPML_SCALING_PROFILES["quartic"]
        p0, p1, p2, p3 = self.pml[:4]

        def profile(coord, inner, thickness, spacing, model_spacing, outward):
            if thickness <= 0:
                return np.zeros_like(coord)
            # sigma_max follows the model (DG) cell size, not a refined Yee grid.
            sig_max = CPML_SIGMA_MAX_FACTOR * (order + 1) * VACUUM_PERMITTIVITY \
                * self.pml_max_velocity / model_spacing * self.dg_pml_strength
            d = (coord - inner) * outward / (thickness * spacing)
            return sig_max * np.clip(d, 0.0, None) ** order

        dx, dy, mdx, mdy = self.dx, self.dy, self.model_dx, self.model_dy
        sy = profile(g.y, p2 * dy, p2, dy, mdy, -1.0) + profile(
            g.y, (self.ny_ext - p3) * dy, p3, dy, mdy, 1.0)
        sx = np.zeros_like(g.x)
        if self.i_top is None:
            sx += profile(g.x, p0 * dx, p0, dx, mdx, -1.0)
        if self.i_bottom is None:
            sx += profile(g.x, (self.nx_ext - p1) * dx, p1, dx, mdx, 1.0)
        return sx, sy

    # ------------------------------------------------------------ subsolvers
    def _build_subsolvers(self, dt, conservative: bool, nshot: Optional[int] = None,
                          dtype: Optional[torch.dtype] = None):
        """Create the Yee/DG parts (``conservative``: no sigma/CPML/penalties)."""
        nshot = self.nshot if nshot is None else nshot
        dtype = self.dtype if dtype is None else dtype
        cast = lambda t: t.to(dtype)  # noqa: E731
        pml = [0] * 6 if conservative else self.pml
        fd = dg = None
        couplings = {}
        if self.mode != "dgtd":
            sig = cast(torch.zeros_like(self.sig_fd_pad) if conservative else self.sig_fd_pad)
            pml_t = torch.tensor(pml, dtype=torch.int32)
            # In the hybrid the Yee CPML uses the same physical sigma profile as
            # the DG CPML (sigma_max = strength * optimal value of the model
            # spacing), so the DG-FDTD interfaces inside the side CPML columns
            # see no jump of the absorber. Pure Yee keeps DeepGPR's profile.
            velocity = self.pml_max_velocity
            if self.mode == "hybrid" and self.pml_profile == "matched":
                velocity *= self.dg_pml_strength * self.dx / self.model_dx
            coeffs = build_pml_coeffs(
                None, None, dt, [self.dx, self.dy, self.dx], self.nx_ext, self.ny_ext, 1, pml_t,
                self.device, dtype, max_velocity=velocity)
            band = None if self.mode == "fdtd" else (self.i_top, self.i_bottom)
            fd = YeeTM2D(cast(self.eps_fd_pad), sig, cast(self.mu_fd_pad), self.dx, self.dy, dt, pml,
                         coeffs[6:], nshot, band=band)
        if self.mode != "fdtd":
            eps, sig, mu = (cast(m) for m in self.dg_materials)
            tags = self.face_tags.copy()
            if conservative:
                sig = torch.zeros_like(sig)
                tags[tags == FACE_ABC] = FACE_PEC
            sx, sy = (None, None) if conservative else self.pml_sigma
            dg = DGTM2D(self.geometry, eps, sig, mu, dt, tags,
                        flux_alpha=0.0 if conservative else self.flux_alpha,
                        pml_sigma_x=sx, pml_sigma_y=sy, nshot=nshot,
                        dtype=dtype, device=self.device)
            tol = 1e-6 * min(self.dx, self.dy)
            for name, row, xline in (("top", self.i_top, getattr(self, "x_top", None)),
                                     ("bottom", self.i_bottom, getattr(self, "x_bottom", None))):
                if row is None or self.mode == "fdtd":
                    continue
                faces = interface_faces(self.geometry, xline, tol)
                couplings[name] = (row, InterfaceCoupling(self.geometry, faces, self.dy, self.NY,
                                                          dtype=dtype, device=self.device))
        if conservative:
            return fd, dg, couplings
        self.fd, self.dg, self.couplings = fd, dg, couplings
        return fd, dg, couplings

    # ------------------------------------------------------------ stepping
    def _make_static(self, fd, dg, couplings, nshot, dtype=None) -> fn.StepStatic:
        st = fn.StepStatic(
            nshot=nshot,
            yee=None if fd is None else fn.yee_static(fd),
            dg=None if dg is None else fn.dg_static(dg),
            couplings=fn.coupling_static(couplings, fd) if couplings else [],
            dtype=self.dtype if dtype is None else dtype, device=self.device)
        if self.cuda_kernels and self.device.type == "cuda":
            if st.yee is not None:
                cuda2d.attach_yee(st.yee, self.device)
            if st.dg is not None:
                cuda2d.attach(st.dg, self.device, slots=[cp.slots for cp in st.couplings])
        return st

    def _fields_of_model(self, sigma_zero: bool = False, dtype=None):
        cast = (lambda t: t) if dtype is None else (lambda t: t.to(dtype))  # noqa: E731
        sig_fd = None if self.mode == "dgtd" else cast(self.sig_fd_pad)
        eps_fd = None if self.mode == "dgtd" else cast(self.eps_fd_pad)
        dg = (None, None, None) if self.mode == "fdtd" else tuple(cast(t) for t in self.dg_materials)
        if sigma_zero:
            sig_fd = None if sig_fd is None else torch.zeros_like(sig_fd)
            dg = (dg[0], None if dg[1] is None else torch.zeros_like(dg[1]), dg[2])
        return eps_fd, sig_fd, dg[0], dg[1], dg[2]

    def _sync(self) -> None:
        """Expose the functional state through the Yee/DG objects (snapshots, energy)."""
        st = self.state
        if self.fd is not None:
            self.fd.Ez, self.fd.Hx, self.fd.Hy = st[0], st[1], st[2]
        if self.dg is not None:
            self.dg.E, self.dg.Hx, self.dg.Hy = st[7], st[8], st[9]

    def _pull(self) -> None:
        """Take fields that were set directly on the Yee/DG objects into the state."""
        st = list(self.state)
        if self.fd is not None:
            st[0], st[1], st[2] = self.fd.Ez, self.fd.Hx, self.fd.Hy
        if self.dg is not None:
            st[7], st[8], st[9] = self.dg.E, self.dg.Hx, self.dg.Hy
        self.state = tuple(st)

    def step(self) -> None:
        """Advance all fields by one leapfrog step (no sources)."""
        self._pull()
        self.state = fn.step(self.static, self.coefs, self.state)
        self._sync()

    def energy(self, saved) -> torch.Tensor:
        """Total leapfrog-invariant energy per shot after a step; ``saved`` is
        :meth:`save_state` taken before that step."""
        total = 0.0
        if self.fd is not None:
            total = total + self.fd.energy(*saved[0])
        if self.dg is not None:
            total = total + self.dg.energy(*saved[1])
        return total

    def save_state(self):
        """``(E^n, Hx^(n-1/2), Hy^(n-1/2))`` of the Yee and DG parts."""
        self._sync()
        fd, dg = self.fd, self.dg
        return (None if fd is None else (fd.Ez.clone(), fd.Hx.clone(), fd.Hy.clone()),
                None if dg is None else (dg.E.clone(), dg.Hx.clone(), dg.Hy.clone()))

    def reset(self) -> None:
        self.state = fn.zero_state(self.static)
        self._sync()

    # ------------------------------------------------------------ CFL
    def _node_weight_grid(self, fd) -> torch.Tensor:
        """Energy weights of the Ez nodes ``(NX, NY)`` (1/2 on the interface rows)."""
        return fd.node_weights()[:, None].expand(self.NX, self.NY)

    def _e_inner(self, fd, dg, a, b):
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        if fd is not None:
            w = self._node_weight_grid(fd) * fd.eps * (self.dx * self.dy)
            w = w.clone()
            w[:, 0] = 0
            w[:, -1] = 0
            total = total + (w * a[0][0] * b[0][0]).sum()
        if dg is not None:
            m = dg.eps_e[:, None] * dg.J
            total = total + (m * (torch.matmul(a[1][0], dg.mass) * b[1][0])).sum()
        return total

    def estimate_lambda_max(self, iterations: int = 80, seed: int = 0) -> float:
        """Largest eigenvalue of ``M_e^-1 K M_h^-1 K^T`` of the coupled scheme.

        Lanczos in the ``M_e`` inner product with full re-orthogonalisation on
        the conservative operator (central fluxes, no sigma/CPML/ABC). The
        leapfrog limit is ``dt < 2 / sqrt(lambda_max)``.
        """
        fd, dg, cps = self._build_subsolvers(1.0, conservative=True, nshot=1,
                                             dtype=torch.float64)
        static = self._make_static(fd, dg, cps, 1, dtype=torch.float64)
        coefs = fn.build_coefficients(
            static, self._fields_of_model(sigma_zero=True, dtype=torch.float64), 1.0)
        gen = torch.Generator(device="cpu").manual_seed(seed)

        def random_state():
            parts = [None, None]
            if fd is not None:
                v = torch.randn((1, self.NX, self.NY), generator=gen, dtype=torch.float64)
                v = v.to(self.device)
                mask = self._node_weight_grid(fd) > 0
                v = v * mask
                v[:, :, 0] = 0
                v[:, :, -1] = 0
                parts[0] = v
            if dg is not None:
                parts[1] = torch.randn((1, dg.K, dg.Np), generator=gen,
                                       dtype=torch.float64).to(self.device)
            return parts

        def apply(parts):
            st = list(fn.zero_state(static))
            if fd is not None:
                st[0] = parts[0]
            if dg is not None:
                st[7] = parts[1]
            st = fn.step(static, coefs, tuple(st))
            out = [None, None]
            if fd is not None:
                out[0] = parts[0] - st[0]
            if dg is not None:
                out[1] = parts[1] - st[7]
            return out

        ip = lambda a, b: self._e_inner(fd, dg, [(a[0],), (a[1],)], [(b[0],), (b[1],)])  # noqa: E731
        comb = lambda a, s, b: [None if x is None else x + s * y for x, y in zip(a, b)]  # noqa: E731
        scale = lambda a, s: [None if x is None else x * s for x in a]  # noqa: E731
        v = random_state()
        v = scale(v, 1.0 / float(torch.sqrt(ip(v, v))))
        basis, alphas, betas = [v], [], []
        w_prev, beta = None, 0.0
        for _ in range(iterations):
            w = apply(v)
            alpha = float(ip(w, v))
            w = comb(w, -alpha, v)
            if w_prev is not None:
                w = comb(w, -beta, w_prev)
            for q in basis:  # full re-orthogonalisation
                w = comb(w, -float(ip(w, q)), q)
            alphas.append(alpha)
            beta = float(torch.sqrt(torch.clamp(ip(w, w), min=0.0)))
            if beta < 1e-12 * abs(alpha):
                break
            betas.append(beta)
            w_prev, v = v, scale(w, 1.0 / beta)
            basis.append(v)
        n = len(alphas)
        T = np.diag(alphas) + np.diag(betas[: n - 1], 1) + np.diag(betas[: n - 1], -1)
        return float(np.linalg.eigvalsh(T).max())

    # ------------------------------------------------------------ points
    def classify(self, x_ext: float, y_ext: Optional[float] = None) -> str:
        """``"dg"`` for points strictly inside the band, else ``"fd"``."""
        if self.mode == "fdtd":
            return "fd"
        if self.mode == "dgtd":
            return "dg"
        tol = 1e-9 * self.dx
        lo = -np.inf if self.i_top is None else self.x_top + tol
        hi = np.inf if self.i_bottom is None else self.x_bottom - tol
        return "dg" if lo < x_ext < hi else "fd"

    def in_band(self, x_ext: np.ndarray, y_ext: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorised :meth:`classify` (``True`` where the DGTD band owns the point)."""
        x_ext = np.asarray(x_ext, dtype=np.float64)
        if self.mode == "fdtd":
            return np.zeros(x_ext.shape, dtype=bool)
        if self.mode == "dgtd":
            return np.ones(x_ext.shape, dtype=bool)
        tol = 1e-9 * self.dx
        lo = -np.inf if self.i_top is None else self.x_top + tol
        hi = np.inf if self.i_bottom is None else self.x_bottom - tol
        return (x_ext > lo) & (x_ext < hi)

    def point_ops(self, positions, *, source: bool) -> fn.PointOps:
        """Operators for physical ``(x, y)`` positions ``(nshot, n, 2)``.

        Source weights are geometric (``1/(w_row dx dy)`` bilinear or
        ``M_K^-1 phi / m``); the material factor ``cb`` is applied per step."""
        pos = np.asarray(torch.as_tensor(positions, dtype=torch.float64).cpu())
        if pos.ndim == 2:
            pos = np.broadcast_to(pos[None], (self.nshot,) + pos.shape)
        if pos.shape[0] != self.nshot or pos.shape[2] != 2:
            raise ValueError("positions must have shape (nshot, n, 2) with (x, y) in metres.")
        fd = {"shot": [], "node": [], "item": [], "w": []}
        dg = {"shot": [], "elem": [], "item": [], "w": []}
        node_w = self._node_weight_grid(self.fd).cpu().numpy() if self.fd is not None else None
        for s in range(pos.shape[0]):
            for item in range(pos.shape[1]):
                xe, ye = pos[s, item, 0] + self.ox, pos[s, item, 1] + self.oy
                if not (0 < xe < self.nx_ext * self.dx and 0 < ye < self.width):
                    raise ValueError(f"Point {pos[s, item]} lies outside the model.")
                if self.classify(xe, ye) == "fd":
                    fi, fj = xe / self.dx, ye / self.dy
                    i0, j0 = int(np.floor(fi + 1e-9)), int(np.floor(fj + 1e-9))
                    ti, tj = fi - i0, fj - j0
                    for di, wi in ((0, 1 - ti), (1, ti)):
                        for dj, wj in ((0, 1 - tj), (1, tj)):
                            w = wi * wj
                            if abs(w) < 1e-12:
                                continue
                            i, j = i0 + di, j0 + dj
                            if source:
                                if node_w[i, j] == 0:
                                    raise ValueError("FDTD source stencil touches an inactive node.")
                                w = w / (node_w[i, j] * self.dx * self.dy)
                            fd["shot"].append(s)
                            fd["node"].append(i * self.NY + j)
                            fd["item"].append(item)
                            fd["w"].append(w)
                else:
                    (ids, phi), = self.dg.point_operators(self.mesh, np.array([[xe, ye]]))
                    if ids.size == 0:
                        raise ValueError(f"Point {pos[s, item]} is not inside a DG element.")
                    for k, ph in zip(ids, phi):
                        if source:
                            vec = self.ref.Minv @ ph / self.geometry.J[k]
                        else:
                            vec = ph
                        dg["shot"].append(s)
                        dg["elem"].append(int(k))
                        dg["item"].append(item)
                        dg["w"].append(vec / ids.size)
        L = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=self.device)  # noqa: E731
        T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=self.dtype,  # noqa: E731
                                      device=self.device)
        npn = self.ref.np_ if self.dg is not None else 1
        return fn.PointOps(
            L(fd["shot"]), L(fd["node"]), L(fd["item"]), T(fd["w"]),
            L(dg["shot"]), L(dg["elem"]), L(dg["item"]),
            T(np.asarray(dg["w"]).reshape(-1, npn)), int(pos.shape[1]),
        )

    _point_ops = point_ops

    def _inject(self, ops: fn.PointOps, amplitudes: torch.Tensor) -> None:
        """Add sources with samples ``amplitudes (nsrc,)`` to the current state."""
        self._pull()
        st = list(self.state)
        st[0], st[7] = fn.inject(self.static, self.coefs, ops, amplitudes, st[0], st[7])
        self.state = tuple(st)
        self._sync()

    def _sample(self, ops: fn.PointOps, nrx: int = None) -> torch.Tensor:
        self._pull()
        return fn.sample(self.static, ops, self.state)

    # ------------------------------------------------------------ run
    def run(
        self,
        source_amplitudes: torch.Tensor,
        source_positions,
        receiver_positions,
        *,
        callback: Optional[Callable[[int, "HybridTM2D"], None]] = None,
        callback_every: int = 0,
        energy_every: int = 0,
        reset: bool = True,
    ) -> dict:
        """Run ``nt`` steps.

        Args:
            source_amplitudes: ``(nsrc, nt)`` or ``(nt,)`` waveforms (DeepGPR
                amplitude convention: ``J = w delta``).
            source_positions: ``(nshot, nsrc, 2)`` physical ``(x, y)`` [m].
            receiver_positions: ``(nshot, nrx, 2)`` physical ``(x, y)`` [m].
            callback: ``callback(n, solver)`` after step ``n`` every
                ``callback_every`` steps.
            energy_every: Record the total discrete energy every ``k`` steps.

        Returns:
            ``{"receivers": (nshot, nt, nrx), "energy": (m, nshot) or None,
            "energy_steps": list, "seconds": float, "dt": float}``.
        """
        w = torch.as_tensor(source_amplitudes, dtype=self.dtype, device=self.device)
        if w.ndim == 1:
            w = w[None]
        if w.ndim == 3:
            w = w[:, :, 0]
        nt = w.shape[1]
        src_pos = torch.as_tensor(np.asarray(source_positions, dtype=np.float64))
        rec_pos = torch.as_tensor(np.asarray(receiver_positions, dtype=np.float64))
        if src_pos.ndim == 2:
            src_pos = src_pos[None].expand(self.nshot, -1, -1)
        if rec_pos.ndim == 2:
            rec_pos = rec_pos[None].expand(self.nshot, -1, -1)
        nsrc, nrx = src_pos.shape[1], rec_pos.shape[1]
        if w.shape[0] == 1 and nsrc > 1:
            w = w.expand(nsrc, -1)
        if w.shape[0] != nsrc:
            raise ValueError("source_amplitudes must have one waveform or one per source.")
        if reset:
            self.reset()
        else:
            self._pull()
        src_ops = self.point_ops(src_pos, source=True)
        rec_ops = self.point_ops(rec_pos, source=False)
        data = torch.zeros((self.nshot, nt, nrx), dtype=self.dtype, device=self.device)
        energies, energy_steps = [], []
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        st, coefs = self.static, self.coefs
        graph = graphs.make(st, coefs, src_ops, rec_ops, w.shape[0]) \
            if not energy_every and self.cuda_kernels else None
        if graph is not None:
            # CUDA Graph replay of the same step (small states, host-bound otherwise)
            graph.set_state(self.state)
            with torch.no_grad():
                for n in range(nt):
                    data[:, n, :] = graph(w[:, n])
                    if callback is not None and callback_every and n % callback_every == 0:
                        self.state = tuple(graph.state)
                        self._sync()
                        callback(n, self)
            self.state = graph.snapshot()
        else:
            for n in range(nt):
                h_prev = self.save_state() if energy_every and n % energy_every == 0 else None
                self.state = fn.step(st, coefs, self.state, src_ops, w[:, n])
                data[:, n, :] = fn.sample(st, rec_ops, self.state)
                if h_prev is not None or (callback is not None and callback_every):
                    self._sync()
                if h_prev is not None:
                    energies.append(self.energy(h_prev))
                    energy_steps.append(n)
                if callback is not None and callback_every and n % callback_every == 0:
                    callback(n, self)
        self._sync()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - t0
        energy = torch.stack([torch.as_tensor(e) for e in energies]) if energies else None
        return {"receivers": data, "energy": energy, "energy_steps": energy_steps,
                "seconds": seconds, "dt": self.dt}

    # ------------------------------------------------------------ reporting
    def summary(self) -> dict:
        """Discretisation sizes for benchmarks and reports."""
        info = {
            "mode": self.mode, "dt": self.dt, "dx": self.dx, "dy": self.dy,
            "model_dx": self.model_dx, "fdtd_refinement": self.fdtd_refinement,
            "pml": self.pml[:4], "lambda_max": self.lambda_max,
            "setup_seconds": self.setup_seconds, "dt_seconds": self.dt_seconds,
        }
        if self.fd is not None:
            rows = sum(b - a for a, b in self.fd.e_ranges)
            info["fdtd_active_ez_nodes"] = rows * (self.NY - 2)
            info["fdtd_dofs"] = 3 * rows * (self.NY - 2)
        if self.dg is not None:
            info.update(self.mesh.quality())
            info["dg_order"] = self.dg_order
            info["dg_elements"] = self.dg.K
            info["dg_dofs"] = 3 * self.dg.K * self.dg.Np
            info["band"] = (self.i_top, self.i_bottom)
        return info


class FieldSampler:
    """Sample ``Ez`` of a :class:`HybridTM2D` on a regular display grid.

    Args:
        solver: The solver.
        x, y: 1D physical coordinates [m] of the display grid.
    """

    def __init__(self, solver: HybridTM2D, x: np.ndarray, y: np.ndarray):
        from matplotlib.tri import Triangulation

        self.solver = solver
        X, Y = np.meshgrid(np.asarray(x) + solver.ox, np.asarray(y) + solver.oy, indexing="ij")
        self.shape = X.shape
        xe, ye = X.reshape(-1), Y.reshape(-1)
        in_dg = solver.in_band(xe, ye)
        dev = solver.device
        self.fd_idx = self.dg_idx = None
        fd_pts = np.nonzero(~in_dg)[0]
        if fd_pts.size:
            fi, fj = xe[fd_pts] / solver.dx, ye[fd_pts] / solver.dy
            i0 = np.clip(np.floor(fi).astype(np.int64), 0, solver.NX - 2)
            j0 = np.clip(np.floor(fj).astype(np.int64), 0, solver.NY - 2)
            ti, tj = fi - i0, fj - j0
            nodes, weights = [], []
            for di, wi in ((0, 1 - ti), (1, ti)):
                for dj, wj in ((0, 1 - tj), (1, tj)):
                    nodes.append((i0 + di) * solver.NY + (j0 + dj))
                    weights.append(wi * wj)
            self.fd_idx = torch.as_tensor(fd_pts, device=dev)
            self.fd_nodes = torch.as_tensor(np.stack(nodes, 1), device=dev)
            self.fd_w = torch.as_tensor(np.stack(weights, 1), dtype=solver.dtype, device=dev)
        dg_pts = np.nonzero(in_dg)[0]
        if dg_pts.size:
            mesh = solver.mesh
            tri = Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.triangles)
            finder = tri.get_trifinder()
            elem = finder(xe[dg_pts], ye[dg_pts])
            ok = elem >= 0
            dg_pts, elem = dg_pts[ok], elem[ok]
            p = mesh.vertices[mesh.triangles[elem]]
            a, b, c = p[:, 0], p[:, 1], p[:, 2]
            q = np.stack([xe[dg_pts], ye[dg_pts]], 1)
            det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
            l1 = ((q[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (q[:, 1] - a[:, 1])) / det
            l2 = ((b[:, 0] - a[:, 0]) * (q[:, 1] - a[:, 1]) - (q[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / det
            l0 = 1 - l1 - l2
            r, s = -l0 + l1 - l2, -l0 - l1 + l2
            phi = solver.ref.basis_at(r, s)
            self.dg_idx = torch.as_tensor(dg_pts, device=dev)
            self.dg_elem = torch.as_tensor(elem.astype(np.int64), device=dev)
            self.dg_phi = torch.as_tensor(phi, dtype=solver.dtype, device=dev)

    def __call__(self, shot: int = 0) -> np.ndarray:
        s = self.solver
        out = torch.full((self.shape[0] * self.shape[1],), float("nan"), dtype=s.dtype,
                         device=s.device)
        if self.fd_idx is not None:
            ez = s.fd.Ez[shot].reshape(-1)
            out[self.fd_idx] = (ez[self.fd_nodes] * self.fd_w).sum(-1)
        if self.dg_idx is not None:
            e = s.dg.E[shot][self.dg_elem]
            out[self.dg_idx] = (e * self.dg_phi).sum(-1)
        return out.view(self.shape).cpu().numpy()


__all__ = ["FieldSampler", "HybridTM2D"]
