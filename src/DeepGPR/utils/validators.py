"""Small, reusable argument validators.

Error messages are kept identical to the messages the package emitted before
the refactor because downstream code and the test-suite match on them.
"""

from __future__ import annotations

import math
import operator
from typing import Any


def require_positive_finite(name: str, value: Any) -> float:
    """Return ``value`` as a finite, strictly positive ``float``.

    Args:
        name: Argument name used in error messages.
        value: Value to convert.

    Raises:
        TypeError: If ``value`` cannot be converted to ``float``.
        ValueError: If ``value`` is not finite or not strictly positive.
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite positive scalar.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar.")
    return result


def require_finite(name: str, value: Any) -> float:
    """Return ``value`` as a finite ``float``.

    Args:
        name: Argument name used in error messages.
        value: Value to convert.

    Raises:
        TypeError: If ``value`` cannot be converted to ``float``.
        ValueError: If ``value`` is not finite.
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite scalar.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar.")
    return result


def require_positive_index(name: str, value: Any) -> int:
    """Return ``value`` as a strictly positive Python ``int``.

    ``bool`` is rejected even though it is an ``int`` subclass.

    Args:
        name: Argument name used in error messages.
        value: Integer-like value (``int``, NumPy integer, ...).

    Raises:
        TypeError: If ``value`` is not integer-like.
        ValueError: If ``value`` is not positive.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a positive integer.") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def require_bool(name: str, value: Any) -> bool:
    """Validate that ``value`` is a real ``bool``.

    Args:
        name: Argument name used in error messages.
        value: Value to check.

    Raises:
        TypeError: If ``value`` is not a ``bool``.
    """
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")
    return value


__all__ = [
    "require_bool",
    "require_finite",
    "require_positive_finite",
    "require_positive_index",
]
