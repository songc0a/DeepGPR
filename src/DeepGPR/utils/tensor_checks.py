"""Tensor sanity checks used by the solver's ``debug`` mode."""

from __future__ import annotations

from typing import Any, List

import torch

from .exceptions import NativeLibraryError, NonFiniteTensorError
from .logger import get_logger

_LOGGER = get_logger(__name__)


def check_tensors_for_nan_inf(d: str, **tensors: Any) -> None:
    """Raise if any named tensor contains NaN or Inf values.

    ``None`` entries and non-tensor values are reported as warnings and
    skipped. Each tensor is scanned once with :func:`torch.isfinite`; the more
    detailed NaN/Inf breakdown is only computed for tensors that fail.

    Args:
        d: Label of the calculation stage (for example ``"forward"``).
        **tensors: Tensors to validate, keyed by display name.

    Raises:
        NonFiniteTensorError: If at least one tensor is not finite. The
            exception derives from :class:`FloatingPointError`.
    """
    problems: List[str] = []
    for name, tensor in tensors.items():
        if tensor is None:
            _LOGGER.warning("%s: %s is None.", d, name)
            continue
        if not isinstance(tensor, torch.Tensor):
            _LOGGER.warning("%s: %s is not a tensor: %s", d, name, type(tensor))
            continue
        if bool(torch.isfinite(tensor).all().item()):
            continue

        kinds = []
        if bool(torch.isnan(tensor).any().item()):
            kinds.append("NaN")
        if bool(torch.isinf(tensor).any().item()):
            kinds.append("Inf")
        message = (
            f"Tensor `{name}` contains {' '.join(kinds)} | "
            f"shape={tuple(tensor.shape)} | dtype={tensor.dtype}"
        )
        _LOGGER.error("%s: %s", d, message)
        problems.append(message)

    if problems:
        raise NonFiniteTensorError(
            f"{d} produced tensors containing NaN or Inf values: " + "; ".join(problems)
        )


def check_nonzero_source_created_fields(
    c_lib: Any, source_amplitudes: torch.Tensor, *fields: torch.Tensor
) -> None:
    """Detect a stale native library that silently produced all-zero fields.

    Args:
        c_lib: Loaded native DeepGPR library (used for its path in messages).
        source_amplitudes: Source waveform tensor.
        *fields: Field tensors returned by the native backend.

    Raises:
        NativeLibraryError: If the source is non-zero but every field is zero.
    """
    if source_amplitudes.abs().max().item() == 0.0:
        return

    max_field = 0.0
    for field in fields:
        max_field = max(max_field, field.abs().max().item())

    if max_field == 0.0:
        lib_path = getattr(c_lib, "_deepgpr_path", "unknown")
        raise NativeLibraryError(
            "The native DeepGPR backend returned all-zero fields even though the "
            f"source waveform is nonzero. Loaded library: {lib_path}. "
            "This usually means the shared library is stale or incompatible with "
            "the current source code/device. Rebuild or pull the latest generated "
            "native libraries."
        )


__all__ = ["check_nonzero_source_created_fields", "check_tensors_for_nan_inf"]
