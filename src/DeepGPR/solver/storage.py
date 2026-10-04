"""Saved-wavefield (``E_saved`` / ``R_saved``) storage options and INT8 layout.

The forward solver can store its history in float32, float16, bfloat16 or a
CUDA-only block-INT8 format. This module validates the user options, packs
them into the native ``storage_type`` integer, and provides the diagnostic
INT8 decoder.
"""

from __future__ import annotations

import math
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple

import torch

from ..config.constants import (
    FLOAT32_BYTES,
    INT8_BLOCK_DIM_MASK,
    INT8_BLOCK_X_SHIFT,
    INT8_BLOCK_Y_SHIFT,
    INT8_BLOCK_Z_SHIFT,
    INT8_DEFAULT_BLOCK_2D,
    INT8_DEFAULT_BLOCK_3D,
    INT8_MAX_BLOCK_VOLUME,
    INT8_REDUCTION_BACKENDS,
    INT8_REDUCTION_SHIFT,
    INT8_SCALE_ALIGNMENT_BYTES,
    WAVEFIELD_CONVERSION_BACKENDS,
    WAVEFIELD_CONVERSION_SHIFT,
    WAVEFIELD_HISTORY_DISABLED,
    WAVEFIELD_RHS_FROM_E,
    WAVEFIELD_RHS_HISTORY_MODES,
    WAVEFIELD_STORAGE_BFLOAT16,
    WAVEFIELD_STORAGE_FLOAT16,
    WAVEFIELD_STORAGE_FLOAT32,
    WAVEFIELD_STORAGE_INT8,
)

from ..utils.exceptions import NativeLibraryError

BlockSize = Tuple[int, int, int]

#: Native storage codes of the uncompressed history dtypes.
WAVEFIELD_STORAGE_TYPES: Dict[torch.dtype, int] = {
    torch.float32: WAVEFIELD_STORAGE_FLOAT32,
    torch.float16: WAVEFIELD_STORAGE_FLOAT16,
    torch.bfloat16: WAVEFIELD_STORAGE_BFLOAT16,
}

#: String aliases accepted for ``wavefield_storage_dtype``.
STORAGE_DTYPE_ALIASES: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

_COMPRESSION_MODES = ("none", "int8", "zfp")


class WavefieldStorageConfig(NamedTuple):
    """Resolved history-storage options passed to the autograd function.

    Attributes:
        dtype: Storage dtype of uncompressed histories.
        compression: ``"none"`` or ``"int8"``.
        block_size: INT8 block ``(bx, by, bz)`` or ``None``.
        conversion_backend: CUDA FP16/BF16 conversion backend name.
        int8_reduction_backend: CUDA INT8 reduction backend name.
    """

    dtype: torch.dtype
    compression: str
    block_size: Optional[BlockSize]
    conversion_backend: str
    int8_reduction_backend: str


def normalize_wavefield_storage_dtype(value: Any) -> torch.dtype:
    """Return the torch dtype for a ``wavefield_storage_dtype`` argument.

    Args:
        value: ``torch.float32``/``float16``/``bfloat16`` or an alias string
            such as ``"fp16"``.

    Raises:
        ValueError: For unsupported dtypes.
    """
    if isinstance(value, str):
        value = STORAGE_DTYPE_ALIASES.get(value.lower())
    if value not in WAVEFIELD_STORAGE_TYPES:
        raise ValueError("wavefield_storage_dtype must be float32, float16, or bfloat16.")
    return value


def normalize_wavefield_compression(value: Any) -> str:
    """Normalise the saved-wavefield compression mode (``None`` -> ``"none"``)."""
    if value is None:
        return "none"
    if not isinstance(value, str):
        raise TypeError("wavefield_compression must be 'none', 'int8', or 'zfp'.")
    value = value.lower()
    if value not in _COMPRESSION_MODES:
        raise ValueError("wavefield_compression must be 'none', 'int8', or 'zfp'.")
    return value


def normalize_wavefield_conversion_backend(value: Any) -> str:
    """Normalise the CUDA FP16/BF16 history-conversion backend name."""
    message = (
        "wavefield_conversion_backend must be 'auto', 'legacy', 'native_scalar', or 'native_vec2'."
    )
    if not isinstance(value, str):
        raise TypeError(message)
    value = value.lower()
    if value != "auto" and value not in WAVEFIELD_CONVERSION_BACKENDS:
        raise ValueError(message)
    return value


def normalize_int8_reduction_backend(value: Any) -> str:
    """Normalise the CUDA INT8 tile-reduction backend name."""
    message = "int8_reduction_backend must be 'auto', 'current', 'cub_block', or 'warp_shuffle'."
    if not isinstance(value, str):
        raise TypeError(message)
    value = value.lower()
    if value != "auto" and value not in INT8_REDUCTION_BACKENDS:
        raise ValueError(message)
    return value


def normalize_wavefield_rhs_history(value: Any) -> str:
    """Normalise ``wavefield_rhs_history`` (``"auto"``, ``"stored"``, ``"reconstructed"``)."""
    message = "wavefield_rhs_history must be 'auto', 'stored', or 'reconstructed'."
    if not isinstance(value, str):
        raise TypeError(message)
    value = value.lower()
    if value not in WAVEFIELD_RHS_HISTORY_MODES:
        raise ValueError(message)
    return value


def resolve_rhs_reconstruction(
    requested: str,
    *,
    sampling_interval: int,
    stores_model_history: bool,
    library_supported: bool,
    lossless_storage: bool = True,
) -> bool:
    """Decide whether R^n is rebuilt from consecutive E frames.

    ``"auto"`` selects the E-only history only for lossless (float32,
    uncompressed) storage: there R^n is rebuilt bitwise as the forward solver
    computed it. With float16/bfloat16/INT8 storage the difference of two
    independently rounded E frames can be markedly less accurate than a stored
    R^n (measured up to 3.5x larger permittivity-gradient errors), so those
    modes keep the E+R pair unless ``"reconstructed"`` is requested.

    Args:
        requested: Normalised ``wavefield_rhs_history``.
        sampling_interval: Resolved ``model_gradient_sampling_interval``.
        stores_model_history: Whether a material gradient will read the history.
        library_supported: Whether the native library reports
            ``deepgpr_supports_rhs_reconstruction``.
        lossless_storage: Whether the history is stored as uncompressed float32.

    Returns:
        ``True`` for the E-only history (only possible at sampling interval 1).

    Raises:
        ValueError: ``"reconstructed"`` with a sampling interval above 1.
        NativeLibraryError: ``"reconstructed"`` with a library that lacks the
            capability.
    """
    if requested == "stored":
        return False
    if requested == "reconstructed":
        if sampling_interval != 1:
            raise ValueError(
                "wavefield_rhs_history='reconstructed' requires "
                "model_gradient_sampling_interval=1 (R^n is rebuilt from E^n and E^(n+1))."
            )
        if not stores_model_history:
            return False
        if not library_supported:
            raise NativeLibraryError(
                "The loaded native library cannot rebuild R^n from the E history "
                "(deepgpr_supports_rhs_reconstruction). Rebuild the CPU/CUDA libraries "
                "or use wavefield_rhs_history='stored'."
            )
        return True
    return (
        sampling_interval == 1 and lossless_storage and library_supported and stores_model_history
    )


def default_compression_block_size(spatial_mode: int) -> BlockSize:
    """Default INT8 block for a 2D (``spatial_mode == 2``) or 3D model."""
    return INT8_DEFAULT_BLOCK_2D if spatial_mode == 2 else INT8_DEFAULT_BLOCK_3D


def normalize_compression_block_size(value: Any, spatial_mode: int) -> BlockSize:
    """Return a validated ``(x, y, z)`` INT8 block shape.

    Args:
        value: ``None`` for the default, or a 2-sequence (2D) / 3-sequence (3D).
        spatial_mode: 2 for 2D models, 3 for 3D models.

    Raises:
        TypeError: For scalars or non-integer entries.
        ValueError: For wrong length, out-of-range dimensions, or a block
            volume that is not a power of two <= 256.
    """
    if value is None:
        return default_compression_block_size(spatial_mode)
    if isinstance(value, int):
        raise TypeError(
            "wavefield_compression_block_size must be a 2D or 3D sequence, not a scalar."
        )
    try:
        block_size = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TypeError("wavefield_compression_block_size must be a sequence of integers.") from exc
    expected_dimensions = 2 if spatial_mode == 2 else 3
    if len(block_size) != expected_dimensions:
        raise ValueError(
            f"wavefield_compression_block_size must contain {expected_dimensions} "
            f"values for a {spatial_mode}D model."
        )
    if spatial_mode == 2:
        block_size = (*block_size, 1)
    if any(size < 1 or size > INT8_BLOCK_DIM_MASK for size in block_size):
        raise ValueError("Each INT8 compression block dimension must be in [1, 63].")
    block_volume = math.prod(block_size)
    if block_volume > INT8_MAX_BLOCK_VOLUME or block_volume & (block_volume - 1):
        raise ValueError(
            "The INT8 compression block volume must be a power of two no larger than 256."
        )
    return block_size  # type: ignore[return-value]


def encode_int8_storage_type(block_size: BlockSize, reduction_backend: str = "current") -> int:
    """Pack the INT8 block shape and reduction backend into ``storage_type``."""
    bx, by, bz = block_size
    return (
        WAVEFIELD_STORAGE_INT8
        | (bx << INT8_BLOCK_X_SHIFT)
        | (by << INT8_BLOCK_Y_SHIFT)
        | (bz << INT8_BLOCK_Z_SHIFT)
        | (INT8_REDUCTION_BACKENDS[reduction_backend] << INT8_REDUCTION_SHIFT)
    )


def encode_storage_type(
    config: WavefieldStorageConfig, save_wavefield_history: bool, rhs_from_e: bool = False
) -> int:
    """Return the complete native ``storage_type`` argument.

    Args:
        config: Resolved storage options.
        save_wavefield_history: ``False`` sets the history-disabled flag.
        rhs_from_e: Store only E and rebuild R^n in the adjoint
            (:data:`WAVEFIELD_RHS_FROM_E`).
    """
    if config.compression == "int8":
        assert config.block_size is not None
        storage_type = encode_int8_storage_type(config.block_size, config.int8_reduction_backend)
    else:
        storage_type = WAVEFIELD_STORAGE_TYPES[config.dtype] | (
            WAVEFIELD_CONVERSION_BACKENDS[config.conversion_backend] << WAVEFIELD_CONVERSION_SHIFT
        )
    if not save_wavefield_history:
        storage_type |= WAVEFIELD_HISTORY_DISABLED
    elif rhs_from_e:
        storage_type |= WAVEFIELD_RHS_FROM_E
    return storage_type


def saved_history_shape(
    mode: int, nt_saved: int, nstep: int, nx: int, ny: int, nz: int
) -> Tuple[int, ...]:
    """Shape of the uncompressed ``E_saved`` history for a gradient mode.

    ``mode == 3`` stores ``(3, nt_saved, nstep, nx, ny, nz)`` (Ex, Ey, Ez);
    otherwise Ez only, ``(nt_saved, nstep, nx, ny, nz)``.
    """
    if mode == 3:
        return (3, nt_saved, nstep, nx, ny, nz)
    return (nt_saved, nstep, nx, ny, nz)


def saved_time_steps(nt: int, sampling_interval: int) -> int:
    """Number of saved history steps, ``ceil(nt / sampling_interval)``."""
    return (nt + sampling_interval - 1) // sampling_interval


def int8_history_layout(shape: Sequence[int], block_size: BlockSize) -> Dict[str, Any]:
    """Return the packed block-INT8 layout of an uncompressed history shape.

    Args:
        shape: 5D ``(nt_saved, shots, nx, ny, nz)`` or 6D with a leading
            component axis.
        block_size: ``(bx, by, bz)``.

    Returns:
        Dictionary with component/step/shot counts, block grid, INT8 payload
        size (``q_count``/``q_bytes_aligned``), scale count and total bytes.
    """
    if len(shape) == 5:
        components = 1
        nt_saved, shots, nx, ny, nz = shape
    elif len(shape) == 6:
        components, nt_saved, shots, nx, ny, nz = shape
    else:
        raise ValueError("Wavefield history shape must have five or six dimensions.")
    bx, by, bz = block_size
    blocks_xyz = ((nx + bx - 1) // bx, (ny + by - 1) // by, (nz + bz - 1) // bz)
    blocks_per_shot = math.prod(blocks_xyz)
    q_count = components * nt_saved * shots * nx * ny * nz
    alignment_mask = INT8_SCALE_ALIGNMENT_BYTES - 1
    q_bytes_aligned = (q_count + alignment_mask) & ~alignment_mask
    scale_count = components * nt_saved * shots * blocks_per_shot
    return {
        "components": components,
        "nt_saved": nt_saved,
        "shots": shots,
        "spatial_shape": (nx, ny, nz),
        "blocks_xyz": blocks_xyz,
        "blocks_per_shot": blocks_per_shot,
        "q_count": q_count,
        "q_bytes_aligned": q_bytes_aligned,
        "scale_count": scale_count,
        "packed_bytes": q_bytes_aligned + scale_count * FLOAT32_BYTES,
    }


def decompress_wavefield_history(
    packed: torch.Tensor, shape: Sequence[int], block_size: Sequence[int]
) -> torch.Tensor:
    """Reconstruct a packed INT8 history for diagnostics (never used by backward).

    The fused CUDA material-gradient kernel decodes values inline; this
    helper materialises a float32 tensor only for validation, plotting or
    export.

    Args:
        packed: Contiguous 1D ``torch.int8`` tensor returned as ``E_saved``.
        shape: Uncompressed history shape (5D or 6D).
        block_size: INT8 block (2 or 3 values).

    Returns:
        float32 tensor of ``shape``.
    """
    block_size = tuple(block_size)
    if len(block_size) == 2:
        block_size = (*block_size, 1)
    if len(block_size) != 3:
        raise ValueError("block_size must contain two or three values.")
    layout = int8_history_layout(tuple(shape), block_size)  # type: ignore[arg-type]
    if packed.dtype != torch.int8 or packed.ndim != 1 or not packed.is_contiguous():
        raise TypeError("packed must be a contiguous one-dimensional torch.int8 tensor.")
    if packed.numel() != layout["packed_bytes"]:
        raise ValueError(
            f"packed contains {packed.numel()} bytes; expected {layout['packed_bytes']}."
        )

    components = layout["components"]
    nt_saved = layout["nt_saved"]
    shots = layout["shots"]
    nx, ny, nz = layout["spatial_shape"]
    nbx, nby, nbz = layout["blocks_xyz"]
    q = packed[: layout["q_count"]].reshape(components, nt_saved, shots, nx, ny, nz)
    scale_bytes = packed[layout["q_bytes_aligned"] : layout["packed_bytes"]]
    scales = scale_bytes.view(torch.float32).reshape(components, nt_saved, shots, nbx, nby, nbz)
    expanded_scales = scales.repeat_interleave(block_size[0], dim=-3)
    expanded_scales = expanded_scales.repeat_interleave(block_size[1], dim=-2)
    expanded_scales = expanded_scales.repeat_interleave(block_size[2], dim=-1)
    expanded_scales = expanded_scales[..., :nx, :ny, :nz]
    reconstructed = q.to(torch.float32) * expanded_scales
    return reconstructed.squeeze(0) if len(shape) == 5 else reconstructed


__all__ = [
    "BlockSize",
    "STORAGE_DTYPE_ALIASES",
    "WAVEFIELD_STORAGE_TYPES",
    "WavefieldStorageConfig",
    "decompress_wavefield_history",
    "default_compression_block_size",
    "encode_int8_storage_type",
    "encode_storage_type",
    "int8_history_layout",
    "normalize_compression_block_size",
    "normalize_int8_reduction_backend",
    "normalize_wavefield_compression",
    "normalize_wavefield_conversion_backend",
    "normalize_wavefield_rhs_history",
    "normalize_wavefield_storage_dtype",
    "resolve_rhs_reconstruction",
    "saved_history_shape",
    "saved_time_steps",
]
