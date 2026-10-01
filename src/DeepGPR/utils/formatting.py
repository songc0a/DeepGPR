"""Formatting helpers for human-readable reports."""

from __future__ import annotations

from ..config.constants import MEMORY_UNITS


def format_memory_size(num_bytes: float) -> str:
    """Format a byte count using binary (1024-based) units.

    Args:
        num_bytes: Number of bytes.

    Returns:
        A string such as ``"12.50 MiB"``.
    """
    value = float(num_bytes)
    for unit in MEMORY_UNITS:
        if abs(value) < 1024.0 or unit == MEMORY_UNITS[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")  # pragma: no cover


__all__ = ["format_memory_size"]
