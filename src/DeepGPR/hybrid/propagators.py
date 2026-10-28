"""Uniform forward-model interface for FWI: pick the solver with one option.

``make_propagator(solver="fdtd" | "hybrid_dgtd", ...)`` returns an object with

* ``data = prop(eps_r, sigma, source_amplitudes)`` - receiver data
  ``(nshot, nt, nrx)``, differentiable with respect to ``eps_r`` and ``sigma``
  (and the source waveform);
* ``prop.dt`` and ``prop.time_axis(nt)`` - the sample times ``(n + 1) dt``.

Both take the *ground* model on the regular model grid (DeepGPR layout, axis 0
= depth) plus the surface; points above the surface are air in both solvers,
so the inversion variables and the FWI loop are identical:

* ``"fdtd"``: native DeepGPR Yee-FDTD (``DeepGPR.compute``, native discrete
  adjoint). The surface is a node staircase (DeepGPR's representation);
  antennas are rounded to the nearest node.
* ``"hybrid_dgtd"``: :class:`HybridTM2D` with the body-fitted DGTD band and
  the discrete adjoint of :mod:`DeepGPR.hybrid.adjoint`.

3D models ``(nx, ny, nz)`` with a :class:`~DeepGPR.hybrid.topography.Surface3D`
use :class:`~DeepGPR.hybrid.HybridMaxwell3D` and DeepGPR's 3D solver
(``Ez`` sources and receivers, nodes at ``(i dx, j dy, (k + 1/2) dz)``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..solver.modeling import compute
from .adjoint import HybridPropagator
from .solver import HybridTM2D
from .solver3d import HybridMaxwell3D
from .topography import Surface


class FDTDPropagator:
    """Native DeepGPR FDTD with the surface as a node staircase.

    Args:
        model_shape: ``(nx, ny)`` or ``(nx, ny, nz)`` of the model grid.
        dx: Grid spacing [m].
        dt: Time step [s] (CFL-checked by DeepGPR).
        source_positions: ``(nshot, 1, 2|3)`` physical positions [m] (3D: of the
            ``Ez`` dipole, nodes at ``(i, j, k + 1/2) dx``).
        receiver_positions: ``(nshot, nrx, 2|3)``.
        surface: Optional surface (:class:`Surface` / ``Surface3D``); nodes above
            it are air.
        pmlthick: DeepGPR CPML thickness (model cells).
        fdtd_refinement: Run DeepGPR on ``dx / r`` with the model pixels resampled
            (nearest pixel, gradients summed back to the model grid); in 3D the
            source amplitude is scaled by ``r`` so the dipole moment stays
            ``w * dx``. ``dt`` must satisfy the CFL limit of the fine grid.
    """

    solver_name = "fdtd"

    def __init__(self, model_shape, dx, dt, source_positions, receiver_positions, *,
                 surface: Optional[Surface] = None, pmlthick=10, device="cuda",
                 air_eps_r: float = 1.0, air_sigma: float = 0.0, fdtd_refinement: int = 1,
                 **compute_kwargs):
        rf = int(fdtd_refinement)
        if rf < 1 or rf != fdtd_refinement:
            raise ValueError("fdtd_refinement must be a positive integer.")
        self.refinement = rf
        self.model_dx = float(dx)
        model_shape = tuple(int(n) for n in model_shape)
        self.model_shape = model_shape
        self.up = None
        if rf > 1:
            # Model pixels (cells centred on the model nodes) resampled to the fine grid.
            self.up = [torch.clamp(torch.floor(torch.arange(n * rf) / rf + 0.5).long(), 0, n - 1)
                       .to(device) for n in model_shape]
            model_shape = tuple(n * rf for n in model_shape)
            pmlthick = int(pmlthick) * rf if not isinstance(pmlthick, (list, tuple)) else \
                [int(v) * rf for v in pmlthick]
        dx = float(dx) / rf
        self.dx, self.dt = float(dx), float(dt)
        self.device = torch.device(device)
        self.pmlthick = pmlthick
        self.compute_kwargs = compute_kwargs
        src = np.asarray(source_positions, dtype=float)
        rec = np.asarray(receiver_positions, dtype=float)
        if src.shape[1] != 1:
            raise ValueError("FDTDPropagator supports one source per shot.")
        three_d = len(model_shape) == 3
        off = np.array([0.0, 0.0, 0.5]) if three_d else np.zeros(2)
        node = lambda p: np.rint(p / self.dx - off).astype(int)  # noqa: E731
        s, r = node(src), node(rec)
        self.rounding = float(max(np.abs(src - (s + off) * self.dx).max(),
                                  np.abs(rec - (r + off) * self.dx).max()))
        if not three_d:
            s = np.concatenate([s, np.zeros(s.shape[:2] + (1,), int)], axis=2)
            r = np.concatenate([r, np.zeros(r.shape[:2] + (1,), int)], axis=2)
        else:
            self.compute_kwargs.setdefault("mode", 3)
        # DeepGPR's 3D dipole moment is w * dz of its grid: keep w * (model dz).
        self.amplitude_scale = float(rf) if three_d else 1.0
        self.source_location = torch.tensor(s, dtype=torch.int, device=self.device)
        self.receiver_location = torch.tensor(r, dtype=torch.int, device=self.device)
        self.air = None
        if surface is not None:
            if three_d:
                nx, ny, nz = model_shape
                Y, Z = np.meshgrid(np.arange(ny) * self.dx, np.arange(nz) * self.dx, indexing="ij")
                air = (np.arange(nx) * self.dx)[:, None, None] < surface(Y, Z)[None]
            else:
                nx, ny = model_shape
                air = (np.arange(nx) * self.dx)[:, None] < surface(np.arange(ny) * self.dx)[None]
            self.air = torch.as_tensor(air, device=self.device)
        self.air_values = (air_eps_r, air_sigma)

    def __call__(self, eps_r, sigma, source_amplitudes):
        eps = eps_r.to(self.device, torch.float32)
        sig = sigma.to(self.device, torch.float32)
        if self.up is not None:
            if eps.ndim == 3:
                i, j, k = self.up
                eps, sig = (t[i][:, j][:, :, k] for t in (eps, sig))
            else:
                i, j = self.up
                eps, sig = (t[i][:, j] for t in (eps, sig))
        if self.air is not None:
            eps = torch.where(self.air, torch.full_like(eps, self.air_values[0]), eps)
            sig = torch.where(self.air, torch.full_like(sig, self.air_values[1]), sig)
        w = torch.as_tensor(source_amplitudes, device=self.device).to(torch.float32)
        w = w * self.amplitude_scale
        if w.ndim == 1:
            w = w[None]
        if w.ndim == 2:
            w = w[:, :, None]
        out = compute(device=self.device, dx=self.dx, dt=self.dt, source_amplitudes=w,
                      source_location=self.source_location,
                      receiver_location=self.receiver_location, eps_r=eps, sigma=sig,
                      pmlthick=self.pmlthick, **self.compute_kwargs)
        return out[-1]

    def time_axis(self, nt: int) -> torch.Tensor:
        return (torch.arange(nt, dtype=torch.float64) + 1) * self.dt


def make_propagator(solver: str, eps_r, sigma, dx, source_positions, receiver_positions, *,
                    surface: Optional[Surface] = None, pmlthick=10, dt: Optional[float] = None,
                    device="cuda", dtype=torch.float32, checkpoint_every: Optional[int] = None,
                    **solver_kwargs):
    """Build a propagator for ``solver`` (``"fdtd"`` or ``"hybrid_dgtd"``).

    ``eps_r``/``sigma`` are a reference model used for the set-up (the hybrid
    mesh, material maps and its coupled CFL limit); the propagator itself is
    evaluated at any model. ``solver_kwargs`` go to :class:`HybridTM2D` /
    :class:`HybridMaxwell3D` (``dg_band``, ``fdtd_refinement``, ``dg_order``,
    ...) or to :func:`DeepGPR.compute`. A 3D model selects the 3D solvers.
    """
    src = np.asarray(source_positions, dtype=float)
    rec = np.asarray(receiver_positions, dtype=float)
    three_d = len(tuple(eps_r.shape)) == 3
    if solver == "fdtd":
        if dt is None:
            from ..preprocessing.stability import max_stable_time_step
            dt = 0.95 * max_stable_time_step(dx, 10, 10, 10 if three_d else 1)
        return FDTDPropagator(tuple(eps_r.shape), dx, dt, src, rec, surface=surface,
                              pmlthick=pmlthick, device=device, **solver_kwargs)
    if solver == "hybrid_dgtd":
        cls = HybridMaxwell3D if three_d else HybridTM2D
        sol = cls(eps_r, sigma, dx, surface=surface, pmlthick=pmlthick, dt=dt,
                  nshot=src.shape[0], device=device, dtype=dtype, **solver_kwargs)
        return HybridPropagator(sol, src, rec, checkpoint_every=checkpoint_every)
    raise ValueError("solver must be 'fdtd' or 'hybrid_dgtd'.")


def resample(data: torch.Tensor, t_from: torch.Tensor, t_to: torch.Tensor) -> torch.Tensor:
    """Linear resampling of ``(nshot, nt, nrx)`` data between time axes."""
    t_from = t_from.to(data.device, data.dtype)
    t_to = t_to.to(data.device, data.dtype)
    idx = torch.clamp(torch.searchsorted(t_from, t_to) - 1, 0, t_from.numel() - 2)
    t0, t1 = t_from[idx], t_from[idx + 1]
    a = ((t_to - t0) / (t1 - t0)).clamp(0, 1)[None, :, None]
    return data[:, idx] * (1 - a) + data[:, idx + 1] * a


__all__ = ["FDTDPropagator", "HybridPropagator", "make_propagator", "resample"]
