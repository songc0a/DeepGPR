"""Validation of simulation inputs and extension of the model into the CPML.

Users supply only the physical model (air plus target). Every call of
:func:`initialization` validates the inputs in that frame, checks the CFL
limit on the extended grid, replicates the boundary materials outward into
the CPML (:class:`_ExtendModel`), and pads one extra Yee halo cell.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..config.constants import (
    CUDA_MAX_SHOTS_PER_CALL,
    INT32_MAX,
    MIN_RELATIVE_PERMITTIVITY,
    PEC_CONDUCTIVITY_THRESHOLD,
)
from ..utils.logger import get_logger
from ..utils.validators import require_positive_finite
from .grid import GridSpacingLike, normalize_grid_spacing, pmlthick_revert
from .stability import check_cfl

_LOGGER = get_logger(__name__)

#: Dtype used by the native solvers for every floating-point array.
NATIVE_FLOAT_DTYPE = torch.float32

#: Zero padding that adds the one-cell Yee halo on the high side of each axis.
_YEE_HALO_PADDING = (0, 1, 0, 1, 0, 1)


class PreparedModel(NamedTuple):
    """Result of :func:`initialization` (a tuple, for backward compatibility).

    Attributes:
        er, se: Materials extended into the CPML, shape ``(nx, ny, nz)``.
        nx, ny, nz: Extended grid dimensions (physical model + CPML).
        nt: Number of time samples.
        nstep: Number of shots.
        nsr: Sources per shot.
        nrx: Receivers per shot.
        ere, see: ``er``/``se`` with the Yee halo, ``(nx+1, ny+1, nz+1)``.
        mr: Relative permeability with CPML and Yee halo.
        mode: Spatial dimensionality, 2 (``nz == 1``) or 3.
        dtype: Native floating-point dtype (``torch.float32``).
        pmlthick: ``int32`` tensor ``[x0, xm, y0, ym, z0, zm]``.
        source_amplitudes: ``(nsr, nt, 1)`` float32 waveforms on the device.
    """

    er: torch.Tensor
    se: torch.Tensor
    nx: int
    ny: int
    nz: int
    nt: int
    nstep: int
    nsr: int
    nrx: int
    ere: torch.Tensor
    see: torch.Tensor
    mr: torch.Tensor
    mode: int
    dtype: torch.dtype
    pmlthick: torch.Tensor
    source_amplitudes: torch.Tensor


class _ExtendModel(torch.autograd.Function):
    """Replicate materials into fixed PML cells; backward returns only the model.

    PML is rebuilt from current boundary values on every forward call, but is
    not an inversion parameter. In particular, halo sensitivities are *not*
    summed into edge cells as an ordinary replicate-pad backward would do.
    """

    @staticmethod
    def forward(ctx, model: torch.Tensor, pml: Sequence[int]) -> torch.Tensor:  # type: ignore[override]
        ctx.model_slices = tuple(
            slice(pml[2 * axis], pml[2 * axis] + size) for axis, size in enumerate(model.shape)
        )
        padding = (pml[4], pml[5], pml[2], pml[3], pml[0], pml[1])
        return F.pad(model[None, None], padding, mode="replicate")[0, 0]

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):  # type: ignore[override]
        return gradient[ctx.model_slices].contiguous(), None


def _shift_locations(
    locations: torch.Tensor, pmlthick: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Map validated physical-model coordinates onto the extended solver grid."""
    offset = pmlthick[::2].to(device=device, dtype=torch.int32)
    return (locations.to(device=device, dtype=torch.int32) + offset).contiguous()


def _as_3d(tensor: torch.Tensor, message: str) -> torch.Tensor:
    """Append a unit z axis to 2D models; reject other ranks."""
    if len(tensor.shape) == 2:
        return tensor.reshape(*tensor.shape, 1)
    if len(tensor.shape) != 3:
        raise ValueError(message)
    return tensor


def _validate_locations(
    source_location: torch.Tensor, receiver_location: torch.Tensor, device: torch.device
) -> Tuple[int, int, int]:
    """Validate acquisition geometry shapes and return ``(nstep, nsr, nrx)``."""
    if not torch.is_tensor(source_location) or not torch.is_tensor(receiver_location):
        raise TypeError("source_location and receiver_location must be PyTorch tensors.")
    if source_location.ndim != 3 or receiver_location.ndim != 3:
        raise ValueError("source_location and receiver_location must have shape (nstep, count, 3).")
    if source_location.shape[2] != 3 or receiver_location.shape[2] != 3:
        raise ValueError("The last dimension of source_location and receiver_location must be 3.")
    if source_location.shape[0] != receiver_location.shape[0]:
        raise ValueError(
            "The first dimension (nstep) of source_location and receiver_location "
            "should be the same."
        )
    nstep, nsr, nrx = (
        source_location.shape[0],
        source_location.shape[1],
        receiver_location.shape[1],
    )
    if nstep < 1:
        raise ValueError("At least one shot is required.")
    if nsr < 1:
        raise ValueError("At least one source per shot is required.")
    if nrx < 1:
        raise ValueError("At least one receiver per shot is required.")
    if device.type == "cuda" and nstep > CUDA_MAX_SHOTS_PER_CALL:
        raise ValueError(
            f"CUDA supports at most {CUDA_MAX_SHOTS_PER_CALL} shots in one DeepGPR "
            "compute call. Split a larger acquisition batch into smaller calls."
        )
    for name, locations in (
        ("source_location", source_location),
        ("receiver_location", receiver_location),
    ):
        if locations.dtype == torch.bool or locations.is_complex():
            raise TypeError(f"{name} must contain real integer-valued coordinates.")
    return nstep, nsr, nrx


def _prepare_source_amplitudes(
    source_amplitudes: torch.Tensor, nsr: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Validate waveforms and broadcast a single waveform to every source."""
    if not torch.is_tensor(source_amplitudes):
        raise TypeError("source_amplitudes must be a PyTorch tensor.")
    if source_amplitudes.ndim not in (2, 3):
        raise ValueError(
            "source_amplitudes must have shape (num_waveforms, nt) or (num_waveforms, nt, 1)."
        )
    if source_amplitudes.ndim == 2:
        source_amplitudes = source_amplitudes.unsqueeze(-1)
    if source_amplitudes.shape[2] != 1:
        raise ValueError("The last dimension of source_amplitudes must be 1.")
    if source_amplitudes.shape[0] < 1 or source_amplitudes.shape[1] < 1:
        raise ValueError(
            "source_amplitudes must contain at least one waveform and one time sample."
        )
    source_amplitudes = source_amplitudes.to(device=device, dtype=dtype).contiguous()

    waveforms = source_amplitudes.shape[0]
    if (1 < waveforms < nsr) or waveforms > nsr:
        raise ValueError("The number of source waveforms is incorrect.")
    if waveforms == 1 and nsr != 1:
        source_amplitudes = source_amplitudes.repeat(nsr, 1, 1).contiguous()
        _LOGGER.info(
            "One source waveform supplied for %d sources per shot; the waveform "
            "is repeated for all sources.",
            nsr,
        )
    return source_amplitudes


def initialization(
    device,
    er,
    se,
    mr,
    source_amplitudes,
    source_location,
    receiver_location,
    dx: GridSpacingLike,
    dt,
    pmlthick,
    fdtd_order: int = 2,
) -> PreparedModel:
    """Validate physical-model inputs and extend the materials into the CPML.

    Coordinates are validated in the input-model frame; callers map them to
    the extended grid with :func:`_shift_locations`. Only the original model
    slice participates in the material backward pass.

    Args:
        device: Target device (``None`` means CPU).
        er: Relative permittivity, ``(nx, ny)`` or ``(nx, ny, nz)``, >= 1.
        se: Electrical conductivity [S/m], same shape as ``er``, >= 0.
        mr: Relative permeability (same shape) or ``None`` for ones.
        source_amplitudes: ``(nwaveforms, nt)`` or ``(nwaveforms, nt, 1)``.
        source_location: ``(nstep, nsr, 3)`` integer-valued coordinates.
        receiver_location: ``(nstep, nrx, 3)`` integer-valued coordinates.
        dx: Scalar grid spacing or ``(dx, dy, dz)`` [m].
        dt: Time step [s].
        pmlthick: PML thickness (see :func:`pmlthick_revert`).
        fdtd_order: FDTD order used for the CFL check.

    Returns:
        A :class:`PreparedModel` (a 16-element tuple).

    Raises:
        TypeError, ValueError: For invalid inputs; CFL violations raise
            :class:`DeepGPR.utils.exceptions.CFLConditionError`.
    """
    device = torch.device("cpu" if device is None else device)
    dtype = NATIVE_FLOAT_DTYPE
    spacing = normalize_grid_spacing(dx)
    require_positive_finite("dt", dt)
    if not torch.is_tensor(er) or not torch.is_tensor(se):
        raise TypeError("er and se must be PyTorch tensors.")
    er = _as_3d(er, "The shape of epsilon should be 2-d or 3-d.")
    se = _as_3d(se, "The shape of sigma should be 2-d or 3-d.")
    if mr is not None and not torch.is_tensor(mr):
        raise TypeError("mr must be a PyTorch tensor or None.")
    if mr is not None:
        mr = _as_3d(mr, "The shape of mr should be 2-d or 3-d.")

    if er.shape != se.shape:
        raise ValueError("The shape of epsilon and sigma should be the same.")
    if any(size < 1 for size in er.shape):
        raise ValueError("The material model dimensions must all be non-empty.")
    nx, ny, nz = er.shape
    mode = 2 if nz == 1 else 3
    er = er.to(device=device, dtype=dtype)
    se = se.to(device=device, dtype=dtype)
    if mr is None:
        mr = torch.ones_like(er, device=device)
    elif mr.shape == er.shape:
        mr = mr.to(device=device, dtype=dtype)
    else:
        raise ValueError("The shape of mr should be the same as epsilon and sigma.")

    nstep, nsr, nrx = _validate_locations(source_location, receiver_location, device)
    source_location = source_location.to(device=device).contiguous()
    receiver_location = receiver_location.to(device=device).contiguous()
    source_amplitudes = _prepare_source_amplitudes(source_amplitudes, nsr, device, dtype)
    nt = source_amplitudes.shape[1]

    pmlthick = pmlthick_revert(pmlthick, er)
    if pmlthick.numel() != 6:
        raise ValueError("pmlthick must contain six boundary thicknesses.")
    pml_values = [int(value) for value in pmlthick.tolist()]
    if any(value < 0 for value in pml_values):
        raise ValueError("PML thicknesses must be non-negative.")
    if nz == 1 and any(pml_values[4:]):
        raise ValueError("2D models require zero PML thickness on the z boundaries.")
    extended_shape = tuple(
        size + pml_values[2 * axis] + pml_values[2 * axis + 1]
        for axis, size in enumerate((nx, ny, nz))
    )
    if any(size >= INT32_MAX for size in extended_shape):
        raise ValueError("Extended model dimensions must fit in positive int32 with a field halo.")

    # Gather every scalar needed for validation in one device->host transfer.
    shape_tensor = torch.tensor((nx, ny, nz), dtype=torch.int32, device=device)
    source_integral = (
        (torch.isfinite(source_location) & (source_location == source_location.trunc())).all()
        if source_location.is_floating_point()
        else torch.ones((), dtype=torch.bool, device=device)
    )
    receiver_integral = (
        (torch.isfinite(receiver_location) & (receiver_location == receiver_location.trunc())).all()
        if receiver_location.is_floating_point()
        else torch.ones((), dtype=torch.bool, device=device)
    )
    source_valid = ((source_location >= 0) & (source_location < shape_tensor)).all()
    receiver_valid = ((receiver_location >= 0) & (receiver_location < shape_tensor)).all()
    stats = (
        torch.stack(
            (
                torch.isfinite(er).all().to(dtype),
                torch.isfinite(se).all().to(dtype),
                torch.isfinite(mr).all().to(dtype),
                torch.isfinite(source_amplitudes).all().to(dtype),
                er.amin(),
                se.amin(),
                mr.amin(),
                (er.detach() * mr.detach()).amin(),
                source_valid.to(dtype),
                receiver_valid.to(dtype),
                source_integral.to(dtype),
                receiver_integral.to(dtype),
                se.amax(),
            )
        )
        .detach()
        .cpu()
        .tolist()
    )
    er_finite, se_finite, mr_finite, source_finite = (bool(value) for value in stats[:4])
    er_min, se_min, mr_min, min_er_mr = stats[4:8]
    sources_in_range, receivers_in_range = (bool(value) for value in stats[8:10])
    sources_integral, receivers_integral = (bool(value) for value in stats[10:12])
    se_max = stats[12]

    for name, finite in (
        ("er", er_finite),
        ("se", se_finite),
        ("mr", mr_finite),
        ("source_amplitudes", source_finite),
    ):
        if not finite:
            raise ValueError(f"`{name}` contains NaN or Inf values.")
    if er_min < MIN_RELATIVE_PERMITTIVITY:
        raise ValueError("The values of epsilon is incorrect.(should be greater than 1)")
    if se_min < 0:
        raise ValueError("The values of sigma is incorrect.(should be non-negative)")
    if mr_min <= 0:
        raise ValueError("The values of mr are incorrect (must be positive).")
    if not sources_integral:
        raise ValueError("source_location must contain finite integer-valued coordinates.")
    if not receivers_integral:
        raise ValueError("receiver_location must contain finite integer-valued coordinates.")
    if not sources_in_range:
        raise ValueError(
            "Error: Source coordinates out of range! "
            f"Valid ranges are x∈[0,{nx}), y∈[0,{ny}), z∈[0,{nz})"
        )
    if not receivers_in_range:
        raise ValueError(
            "Error: Receiver coordinates out of range! "
            f"Valid ranges are x∈[0,{nx}), y∈[0,{ny}), z∈[0,{nz})"
        )
    if se_max > PEC_CONDUCTIVITY_THRESHOLD:
        _LOGGER.warning(
            "Conductivity values above %g S/m are treated as perfect electric "
            "conductors by the native solvers (no field update, no gradient).",
            PEC_CONDUCTIVITY_THRESHOLD,
        )

    extended_nx, extended_ny, extended_nz = extended_shape
    check_cfl(
        spacing,
        dt,
        extended_nx,
        extended_ny,
        extended_nz,
        fdtd_order=fdtd_order,
        _material_min_er_mr=min_er_mr,
    )
    if any(pml_values):
        er = _ExtendModel.apply(er, pml_values)
        se = _ExtendModel.apply(se, pml_values)
        mr = _ExtendModel.apply(mr, pml_values)
    nx, ny, nz = extended_shape

    ere = F.pad(er, _YEE_HALO_PADDING)
    see = F.pad(se, _YEE_HALO_PADDING)
    mr = F.pad(mr, _YEE_HALO_PADDING)

    _LOGGER.debug(
        "Prepared model: extended grid=(%d, %d, %d) nt=%d shots=%d sources=%d "
        "receivers=%d pml=%s device=%s",
        nx,
        ny,
        nz,
        nt,
        nstep,
        nsr,
        nrx,
        pml_values,
        device,
    )
    return PreparedModel(
        er,
        se,
        nx,
        ny,
        nz,
        nt,
        nstep,
        nsr,
        nrx,
        ere,
        see,
        mr,
        mode,
        dtype,
        pmlthick,
        source_amplitudes,
    )


def extended_shape_of(shape: Sequence[int], pml: Sequence[int]) -> Tuple[int, int, int]:
    """Return the solver-grid shape of a physical model with the given PML.

    Args:
        shape: Physical model shape ``(nx, ny, nz)``.
        pml: Six face thicknesses ``[x0, xm, y0, ym, z0, zm]``.
    """
    return tuple(  # type: ignore[return-value]
        size + int(pml[2 * axis]) + int(pml[2 * axis + 1]) for axis, size in enumerate(shape)
    )


def physical_shape_of(shape: Sequence[int], pml: Sequence[int]) -> Tuple[int, int, int]:
    """Inverse of :func:`extended_shape_of`."""
    return tuple(  # type: ignore[return-value]
        size - int(pml[2 * axis]) - int(pml[2 * axis + 1]) for axis, size in enumerate(shape)
    )


__all__ = [
    "NATIVE_FLOAT_DTYPE",
    "PreparedModel",
    "extended_shape_of",
    "initialization",
    "physical_shape_of",
]
