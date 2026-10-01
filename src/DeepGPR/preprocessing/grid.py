"""Grid-spacing and PML-thickness normalisation."""

from __future__ import annotations

import math
import numbers
from typing import Any, Sequence, Tuple, Union

import torch

from ..config.constants import INT32_MAX

GridSpacing = Tuple[float, float, float]
GridSpacingLike = Union[float, Sequence[float], torch.Tensor]


def normalize_grid_spacing(value: GridSpacingLike) -> GridSpacing:
    """Return grid spacing as a validated ``(dx, dy, dz)`` tuple in metres.

    Args:
        value: A positive scalar (isotropic grid), a three-element list/tuple,
            or a one- or three-element tensor.

    Raises:
        TypeError: If the value has an unsupported type.
        ValueError: If the wrong number of values is given or any value is
            not finite and positive.
    """
    values: Sequence[Any]
    if isinstance(value, torch.Tensor):  # same test as torch.is_tensor
        if value.numel() == 1:
            values = [value.detach().item()] * 3
        elif value.numel() == 3:
            values = value.detach().cpu().reshape(-1).tolist()
        else:
            raise ValueError("dx tensor must contain either one or three values.")
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        values = [value] * 3
    elif isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError("dx list or tuple must contain exactly three values.")
        values = value
    else:
        raise TypeError("dx must be a positive scalar or a three-value list, tuple, or tensor.")

    try:
        spacing = tuple(float(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise TypeError("dx, dy, and dz must be finite positive scalars.") from exc
    if any(not math.isfinite(item) or item <= 0.0 for item in spacing):
        raise ValueError("dx, dy, and dz must be finite positive scalars.")
    return spacing  # type: ignore[return-value]


def pmlthick_revert(p: Any, er: torch.Tensor) -> torch.Tensor:
    """Convert a user PML-thickness specification to six face thicknesses.

    Args:
        p: One of

            * an integer: every active face gets ``p`` cells (z faces are
              zero for 2D models);
            * four values ``[x0, xm, y0, ym]`` (z faces zero);
            * six values ``[x0, xm, y0, ym, z0, zm]``;
            * a one-dimensional tensor with four or six values.

            NumPy integer scalars are accepted wherever ``int`` is.
        er: Relative permittivity, shape ``(nx, ny, nz)``; only ``nz`` is used
            to detect 2D models.

    Returns:
        ``torch.int32`` tensor ``[x0, xm, y0, ym, z0, zm]`` on the CPU.

    Raises:
        TypeError: For unsupported types or non-integer values.
        ValueError: For wrong lengths or negative / oversized thicknesses.
    """
    if isinstance(p, bool):
        raise TypeError("PML thickness must contain integer values, not bool.")
    if isinstance(p, numbers.Integral):
        p = int(p)
        values = [p, p, p, p, 0, 0] if er.shape[2] == 1 else [p] * 6
    elif isinstance(p, (list, tuple)):
        if len(p) == 4:
            values = [*p, 0, 0]
        elif len(p) == 6:
            values = list(p)
        else:
            raise ValueError(f"Unsupported PML length: {len(p)}. Must be 4 or 6.")
    elif isinstance(p, torch.Tensor):
        if p.ndim != 1:
            raise ValueError("PML thickness tensor must be one-dimensional.")
        if p.numel() == 4:
            values = [*p.detach().cpu().tolist(), 0, 0]
        elif p.numel() == 6:
            values = p.detach().cpu().tolist()
        else:
            raise ValueError(f"Unsupported PML length: {p.numel()}. Must be 4 or 6.")
    else:
        raise TypeError(f"Unsupported PML thickness type: {type(p)}")

    normalized = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("PML thickness must contain integer values, not bool.")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("PML thickness must contain finite integer values.") from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("PML thickness must contain finite integer values.")
        integer = int(numeric)
        if integer < 0 or integer > INT32_MAX:
            raise ValueError("PML thickness values must fit in non-negative int32.")
        normalized.append(integer)
    return torch.tensor(normalized, dtype=torch.int32)


__all__ = ["GridSpacing", "GridSpacingLike", "normalize_grid_spacing", "pmlthick_revert"]
