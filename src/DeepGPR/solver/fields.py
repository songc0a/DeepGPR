"""Electromagnetic field state: allocation, validation and checkpoints."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch

from ..preprocessing.model_setup import initialization
from .pml import build_pml_phi, pml_face_descriptors

FieldTriple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def create_or_separate(
    fields: Optional[Sequence[torch.Tensor]],
    nx: int,
    ny: int,
    nz: int,
    nstep: int,
    device: Any,
    dtype: torch.dtype,
) -> FieldTriple:
    """Allocate zero field components or validate supplied ones.

    Fields live on the extended grid plus a one-cell Yee halo, i.e. shape
    ``(nstep, nx + 1, ny + 1, nz + 1)``.

    Args:
        fields: ``None`` to allocate, or three tensors ``(x, y, z)``.
        nx, ny, nz: Extended grid dimensions.
        nstep: Number of shots.
        device: Device of the tensors.
        dtype: Dtype of the tensors.

    Returns:
        Three contiguous tensors. Supplied tensors already matching
        ``device``/``dtype`` are returned as is (the solver updates them in place).

    Raises:
        ValueError: For a wrong count or shape.
    """
    expected_shape = (nstep, nx + 1, ny + 1, nz + 1)
    if fields is None:
        return tuple(  # type: ignore[return-value]
            torch.zeros(expected_shape, device=device, dtype=dtype).contiguous() for _ in range(3)
        )
    if not isinstance(fields, (list, tuple)) or len(fields) != 3:
        raise ValueError("E and H must each contain exactly three field tensors.")

    for component in fields:
        if not isinstance(component, torch.Tensor) or component.shape != expected_shape:
            actual_shape = getattr(component, "shape", None)
            raise ValueError(
                f"Field shape mismatch: got {actual_shape}, expected {expected_shape}."
            )

    return tuple(  # type: ignore[return-value]
        component.to(device=device, dtype=dtype).contiguous() for component in fields
    )


def checkpoint_initial_field(
    device=None,
    per_nstep: Optional[int] = None,
    dx=None,
    dt=None,
    source_amplitudes=None,
    source_location=None,
    receiver_location=None,
    er=None,
    se=None,
    mr=None,
    pmlthick=10,
    fdtd_order: int = 2,
):
    """Create zero electric, magnetic and CPML states on the full solver grid.

    Supply only the physical model (including air), exactly as for
    :func:`DeepGPR.compute`. The returned states include the external CPML
    and the one-cell Yee halo; pass them back to ``compute`` unchanged with
    the same model shape, PML thickness and shot batch.

    Args:
        device: Device of the allocated tensors.
        per_nstep: Optional number of leading shots to keep.
        dx: Scalar grid spacing or ``(dx, dy, dz)`` [m].
        dt: Time step [s].
        source_amplitudes: Source waveforms (validated like ``compute``).
        source_location: ``(nstep, nsr, 3)``.
        receiver_location: ``(nstep, nrx, 3)``.
        er: Relative permittivity.
        se: Electrical conductivity.
        mr: Relative permeability or ``None``.
        pmlthick: PML thickness.
        fdtd_order: FDTD order used for the CFL check.

    Returns:
        ``((Ex, Ey, Ez), (Hx, Hy, Hz), PML)`` where ``PML`` holds 24 tensors.
        Disabled CPML faces use one-dimensional empty tensors.

    Raises:
        TypeError, ValueError: For invalid inputs or ``per_nstep``.
    """
    prepared = initialization(
        device,
        er,
        se,
        mr,
        source_amplitudes,
        source_location,
        receiver_location,
        dx,
        dt,
        pmlthick,
        fdtd_order,
    )
    nx, ny, nz = prepared[2], prepared[3], prepared[4]
    nstep = prepared[6]
    dtype = prepared[13]
    pml = prepared[14]

    if per_nstep is not None:
        if isinstance(per_nstep, bool) or not isinstance(per_nstep, int):
            raise TypeError("per_nstep must be an integer shot count or None.")
        if not 1 <= per_nstep <= nstep:
            raise ValueError("per_nstep must be between 1 and the input shot count.")

    electric = create_or_separate(None, nx, ny, nz, nstep, device, dtype)
    magnetic = create_or_separate(None, nx, ny, nz, nstep, device, dtype)
    # Only the face descriptors are needed; the coefficients (which the
    # original implementation computed and then discarded) are skipped.
    x0, xm, y0, ym, z0, zm = pml_face_descriptors(nx, ny, nz, pml)
    phi = build_pml_phi(x0, xm, y0, ym, z0, zm, nstep, None, device)

    fields = (electric, magnetic, phi)
    if per_nstep is None:
        return fields
    # Disabled faces use one-dimensional empty tensors, with no shot axis.
    return tuple(
        tuple(tensor[:per_nstep] if tensor.numel() else tensor for tensor in group)
        for group in fields
    )


def zero_field(*tensors: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
    """Return zero tensors shaped like the inputs (``None`` is preserved)."""
    return tuple(torch.zeros_like(t) if t is not None else None for t in tensors)


__all__ = ["checkpoint_initial_field", "create_or_separate", "zero_field"]
