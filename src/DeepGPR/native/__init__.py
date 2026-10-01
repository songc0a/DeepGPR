"""Native backend access: library loading, ABI description and call bridge."""

from .abi import BACKWARD_SIGNATURE, FORWARD_SIGNATURE, NATIVE_FUNCTIONS
from .loader import (
    LIB_DIR,
    available_library_files,
    candidate_library_paths,
    configure_library,
    get_deepgpr_lib,
    get_deepgpr_library_path,
    library_supports,
    set_library_fdtd_order,
)

__all__ = [
    "BACKWARD_SIGNATURE",
    "FORWARD_SIGNATURE",
    "LIB_DIR",
    "NATIVE_FUNCTIONS",
    "available_library_files",
    "candidate_library_paths",
    "configure_library",
    "get_deepgpr_lib",
    "get_deepgpr_library_path",
    "library_supports",
    "set_library_fdtd_order",
]
