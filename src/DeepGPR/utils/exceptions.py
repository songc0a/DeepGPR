"""Exception hierarchy for DeepGPR.

Each exception also derives from the built-in type that the package raised
before the refactor, so existing ``except ValueError`` / ``except
RuntimeError`` handlers (and tests) keep working unchanged.
"""

from __future__ import annotations


class DeepGPRError(Exception):
    """Base class for all DeepGPR-specific errors."""


class NativeLibraryError(DeepGPRError, RuntimeError):
    """A native backend is missing a capability, has the wrong ABI, or failed."""


class NativeLibraryNotFoundError(DeepGPRError, FileNotFoundError):
    """No loadable native backend library was found for the requested device."""


class CFLConditionError(DeepGPRError, ValueError):
    """The time step violates the CFL stability limit of the FDTD scheme."""


class NonFiniteTensorError(DeepGPRError, FloatingPointError):
    """A tensor produced or consumed by the solver contains NaN or Inf values."""


__all__ = [
    "CFLConditionError",
    "DeepGPRError",
    "NativeLibraryError",
    "NativeLibraryNotFoundError",
    "NonFiniteTensorError",
]
