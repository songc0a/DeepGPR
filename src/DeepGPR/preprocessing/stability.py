"""CFL stability check for the staggered FDTD scheme."""

from __future__ import annotations

import math
from typing import Optional

import torch

from ..config.constants import (
    FDTD_STENCIL_COEFFICIENT_SUMS,
    SPEED_OF_LIGHT,
    SUPPORTED_FDTD_ORDERS,
)
from ..utils.exceptions import CFLConditionError
from ..utils.validators import require_positive_finite
from .grid import GridSpacingLike, normalize_grid_spacing


def max_stable_time_step(
    dx: GridSpacingLike,
    nx: int,
    ny: int,
    nz: int,
    *,
    fdtd_order: int = 2,
    min_er_mr: Optional[float] = None,
) -> float:
    """Return the largest CFL-stable time step for a grid.

    ``dt_max = sqrt(min(eps_r mu_r)) / (c * S * sqrt(sum_active 1/d^2))`` where
    ``S`` is the stencil coefficient sum of the chosen FDTD order and only axes
    with more than one cell contribute.

    Args:
        dx: Grid spacing (scalar or ``(dx, dy, dz)``).
        nx, ny, nz: Grid dimensions (including PML).
        fdtd_order: 2, 4 or 8.
        min_er_mr: Minimum of ``eps_r * mu_r`` in the model, or ``None`` for
            vacuum (factor 1).

    Raises:
        ValueError: For unsupported orders, a degenerate grid, or an invalid
            material factor.
    """
    spacing = normalize_grid_spacing(dx)
    if fdtd_order not in SUPPORTED_FDTD_ORDERS:
        raise ValueError("fdtd_order must be one of 2, 4, or 8.")

    active_inverse_spacing_squared = sum(
        1.0 / (axis_spacing * axis_spacing)
        for size, axis_spacing in zip((nx, ny, nz), spacing)
        if size > 1
    )
    if active_inverse_spacing_squared == 0.0:
        raise ValueError("At least one model dimension must contain more than one cell.")

    material_factor = 1.0
    if min_er_mr is not None:
        value = float(min_er_mr)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("epsilon_r * mu_r must be finite and positive for the CFL check.")
        material_factor = math.sqrt(value)

    spectral_factor = FDTD_STENCIL_COEFFICIENT_SUMS[fdtd_order]
    return material_factor / (
        SPEED_OF_LIGHT * spectral_factor * math.sqrt(active_inverse_spacing_squared)
    )


def check_cfl(
    dx: GridSpacingLike,
    dt: float,
    nx: int,
    ny: int,
    nz: int,
    er: Optional[torch.Tensor] = None,
    mr: Optional[torch.Tensor] = None,
    fdtd_order: int = 2,
    _material_min_er_mr: Optional[float] = None,
) -> None:
    """Raise if ``dt`` violates the CFL stability condition.

    Args:
        dx: Scalar grid spacing or a three-value ``(dx, dy, dz)`` sequence.
        dt: Time step [s].
        nx, ny, nz: Grid dimensions (including PML).
        er: Optional relative permittivity, used with ``mr`` to find the
            fastest material.
        mr: Optional relative permeability.
        fdtd_order: Spatial finite-difference order, 2, 4 or 8.
        _material_min_er_mr: Internal fast path: precomputed
            ``min(er * mr)``; takes precedence over ``er``/``mr``.

    Raises:
        CFLConditionError: If ``dt`` exceeds the stable limit (a
            :class:`ValueError` subclass).
    """
    normalize_grid_spacing(dx)
    time_step = require_positive_finite("dt", dt)
    if fdtd_order not in SUPPORTED_FDTD_ORDERS:
        raise ValueError("fdtd_order must be one of 2, 4, or 8.")

    min_er_mr: Optional[float] = None
    if _material_min_er_mr is not None:
        min_er_mr = float(_material_min_er_mr)
    elif er is not None and mr is not None:
        min_er_mr = float((er.detach() * mr.detach()).amin().item())

    dt_max = max_stable_time_step(dx, nx, ny, nz, fdtd_order=fdtd_order, min_er_mr=min_er_mr)
    if time_step > dt_max:
        raise CFLConditionError(
            f"Does not meet CFL conditions: dt={time_step:.3e} > dt_max={dt_max:.3e}"
        )


__all__ = ["check_cfl", "max_stable_time_step"]
