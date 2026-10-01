"""Persisting solver outputs to disk."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..config.defaults import FORWARD_WAVEFIELD_PREFIX, FORWARD_WAVEFIELD_TIME_FORMAT
from ..utils.logger import get_logger

_LOGGER = get_logger(__name__)


def normalize_forward_wavefield_directory(value: Any) -> Optional[Path]:
    """Validate an optional output directory for the forward wavefield.

    Args:
        value: ``None``, a string or a path-like object.

    Raises:
        TypeError: If ``value`` is not path-like.
        NotADirectoryError: If it names an existing non-directory.
    """
    if value is None:
        return None
    try:
        directory = Path(value).expanduser()
    except TypeError as exc:
        raise TypeError(
            "save_forward_wavefield_path must be a string, path-like object, or None."
        ) from exc
    if directory.exists() and not directory.is_dir():
        raise NotADirectoryError(f"save_forward_wavefield_path is not a directory: {directory}")
    return directory


def _reserve_output_path(directory: Path, time_label: str) -> Path:
    """Atomically create an unused ``forward_wavefield_<label>[_NN].pt`` file.

    Using exclusive creation avoids the check-then-write race of the previous
    implementation when several processes save into the same directory within
    the same minute. The naming scheme is unchanged.
    """
    candidate = directory / f"{FORWARD_WAVEFIELD_PREFIX}_{time_label}.pt"
    collision_index = 1
    while True:
        try:
            with open(candidate, "xb"):
                return candidate
        except FileExistsError:
            candidate = directory / (
                f"{FORWARD_WAVEFIELD_PREFIX}_{time_label}_{collision_index:02d}.pt"
            )
            collision_index += 1


def save_forward_wavefield(
    wavefield: torch.Tensor,
    directory: Path,
    run_time: datetime,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a detached CPU copy of ``E_saved`` and return the resolved path.

    Args:
        wavefield: Forward history tensor (``E_saved``).
        directory: Output directory (created if needed).
        run_time: Start time of the run; its ``HH-MM`` label names the file.
        metadata: Optional extra entries. When given, a dictionary
            ``{"wavefield": tensor, **metadata}`` is saved instead of the bare
            tensor (used for packed INT8 histories).
    """
    directory.mkdir(parents=True, exist_ok=True)
    output_path = _reserve_output_path(directory, run_time.strftime(FORWARD_WAVEFIELD_TIME_FORMAT))
    cpu_wavefield = wavefield.detach().to(device="cpu")
    payload: Any = (
        {"wavefield": cpu_wavefield, **metadata} if metadata is not None else cpu_wavefield
    )
    try:
        torch.save(payload, output_path)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    resolved = output_path.resolve()
    _LOGGER.info("Forward wavefield saved to %s", resolved)
    return resolved


__all__ = ["normalize_forward_wavefield_directory", "save_forward_wavefield"]
