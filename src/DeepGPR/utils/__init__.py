"""Shared utilities: logging, validation, errors, formatting and tensor checks.

Only torch-free helpers are imported eagerly; :mod:`.tensor_checks` is
imported on first use so that tooling can use these helpers without PyTorch.
"""

from .devices import device_type
from .exceptions import (
    CFLConditionError,
    DeepGPRError,
    NativeLibraryError,
    NativeLibraryNotFoundError,
    NonFiniteTensorError,
)
from .formatting import format_memory_size
from .logger import configure_logging, get_logger
from .validators import (
    require_bool,
    require_finite,
    require_positive_finite,
    require_positive_index,
)

__all__ = [
    "CFLConditionError",
    "DeepGPRError",
    "NativeLibraryError",
    "NativeLibraryNotFoundError",
    "NonFiniteTensorError",
    "configure_logging",
    "device_type",
    "format_memory_size",
    "get_logger",
    "require_bool",
    "require_finite",
    "require_positive_finite",
    "require_positive_index",
]
