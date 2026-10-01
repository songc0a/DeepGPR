"""Discovery, loading and configuration of the native DeepGPR libraries.

Two shared libraries implement the solver: a CUDA build (``deepgpr.so`` /
``deepgpr.dll``) and a C/OpenMP CPU build (``deepgpr_cpu.so`` / ``.dll`` /
``.dylib``). They are loaded lazily, once per process, and their ``ctypes``
signatures are generated from :mod:`DeepGPR.native.abi`.
"""

from __future__ import annotations

import ctypes
import os
import platform
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.constants import NATIVE_ABI_VERSION, SUPPORTED_FDTD_ORDERS
from ..utils.devices import device_type
from ..utils.exceptions import NativeLibraryError, NativeLibraryNotFoundError
from ..utils.logger import get_logger
from .abi import NATIVE_FUNCTIONS, REQUIRED_SYMBOLS, ctypes_signature, CTYPES_BY_KIND

_LOGGER = get_logger(__name__)

#: Directory that ships the native sources and prebuilt binaries.
LIB_DIR: Path = Path(__file__).resolve().parent.parent / "lib"

_SYSTEM_NAME = platform.system()
_LOADED_LIBS: Dict[str, ctypes.CDLL] = {}
_OPENMP_RUNTIME: Optional[ctypes.CDLL] = None
_LOAD_LOCK = threading.Lock()

#: Candidate filenames per backend kind and operating system.
_LIBRARY_CANDIDATES: Dict[str, Dict[str, List[str]]] = {
    "cuda": {
        "Windows": ["deepgpr.dll"],
        "Linux": ["deepgpr.so", "libdeepgpr.so"],
    },
    "cpu": {
        "Windows": ["deepgpr_cpu.dll"],
        "Darwin": ["libdeepgpr_cpu.dylib", "deepgpr_cpu.dylib"],
        "Linux": ["deepgpr_cpu.so", "libdeepgpr_cpu.so"],
    },
}


def candidate_library_paths(kind: str) -> List[Path]:
    """Return the platform-specific native library candidates.

    Args:
        kind: Backend kind, ``"cuda"`` or ``"cpu"``.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    if kind not in _LIBRARY_CANDIDATES:
        raise ValueError(f"Unknown DeepGPR library kind: {kind}")
    return [LIB_DIR / name for name in _LIBRARY_CANDIDATES[kind].get(_SYSTEM_NAME, [])]


def available_library_files() -> List[str]:
    """Return the filenames currently present in the package ``lib`` folder."""
    if not LIB_DIR.is_dir():
        return []
    return sorted(path.name for path in LIB_DIR.iterdir())


def _add_windows_dll_search_paths() -> None:
    """Register CUDA, conda and package directories for Windows DLL lookup."""
    if _SYSTEM_NAME != "Windows" or not hasattr(os, "add_dll_directory"):
        return

    search_dirs: List[Path] = [LIB_DIR]
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        search_dirs.append(Path(cuda_path) / "bin")
    for key, value in os.environ.items():
        if key.startswith("CUDA_PATH_V") and value:
            search_dirs.append(Path(value) / "bin")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        search_dirs.append(Path(conda_prefix) / "Library" / "bin")

    for directory in dict.fromkeys(Path(item) for item in search_dirs):
        try:
            if directory.is_dir():
                os.add_dll_directory(str(directory))
        except OSError as exc:
            _LOGGER.debug("Could not add DLL directory %s: %s", directory, exc)


def _preload_macos_openmp_runtime() -> None:
    """Preload the OpenMP runtime that PyTorch uses on macOS.

    Loading ``libomp`` with ``RTLD_GLOBAL`` before the CPU backend avoids
    two OpenMP runtimes being active in one process.
    """
    global _OPENMP_RUNTIME

    if _SYSTEM_NAME != "Darwin" or _OPENMP_RUNTIME is not None:
        return

    candidates: List[Path] = []
    try:
        import torch

        candidates.append(Path(torch.__file__).resolve().parent / "lib" / "libomp.dylib")
    except Exception as exc:  # pragma: no cover - depends on the installation
        _LOGGER.debug("PyTorch libomp lookup failed: %s", exc)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib" / "libomp.dylib")
    candidates.append(LIB_DIR / "libomp.dylib")

    for path in dict.fromkeys(Path(item) for item in candidates):
        if not path.is_file():
            continue
        try:
            _OPENMP_RUNTIME = ctypes.CDLL(str(path.resolve()), mode=ctypes.RTLD_GLOBAL)
            _LOGGER.debug("Preloaded OpenMP runtime %s", path)
            return
        except OSError:
            continue


def _require_exported_symbols(lib: ctypes.CDLL, symbols: tuple, path: Path) -> None:
    """Raise if ``lib`` does not export every name in ``symbols``."""
    missing = [name for name in symbols if not hasattr(lib, name)]
    if missing:
        raise NativeLibraryError(
            "The shared library was loaded, but the following exported C ABI symbols "
            f"were not found: {missing}.\nLibrary: {path}"
        )


def configure_library(lib: ctypes.CDLL) -> None:
    """Validate the ABI version and install ``ctypes`` signatures.

    Optional symbols (capability probes, ``set_fdtd_order``, the conversion
    self-test) are configured only when the library exports them.

    Args:
        lib: Freshly loaded library.

    Raises:
        NativeLibraryError: If the library reports an incompatible ABI.
    """
    if getattr(lib, "_deepgpr_argtypes_configured", False):
        return

    version_fn = lib.deepgpr_abi_version
    version_fn.argtypes = []
    version_fn.restype = ctypes.c_int
    abi_version = int(version_fn())
    if abi_version != NATIVE_ABI_VERSION:
        raise NativeLibraryError(
            f"Incompatible DeepGPR native ABI {abi_version}; expected ABI "
            f"{NATIVE_ABI_VERSION}. Rebuild the CPU/CUDA shared libraries from the "
            "current sources."
        )

    for name, (signature, return_kind) in NATIVE_FUNCTIONS.items():
        if not hasattr(lib, name):
            continue
        function = getattr(lib, name)
        function.argtypes = ctypes_signature(signature)
        function.restype = None if return_kind is None else CTYPES_BY_KIND[str(return_kind)]

    setattr(lib, "_deepgpr_argtypes_configured", True)  # noqa: B010


def _load_library(kind: str) -> ctypes.CDLL:
    """Load (once) and configure the native library for a backend kind."""
    cached = _LOADED_LIBS.get(kind)
    if cached is not None:
        return cached

    with _LOAD_LOCK:
        cached = _LOADED_LIBS.get(kind)
        if cached is not None:
            return cached

        _add_windows_dll_search_paths()
        if kind == "cpu":
            _preload_macos_openmp_runtime()

        candidates = candidate_library_paths(kind)
        load_errors: List[str] = []
        for path in candidates:
            if not path.is_file():
                continue
            try:
                lib = ctypes.CDLL(str(path.resolve()))
            except OSError as exc:
                load_errors.append(f"{path}: {exc}")
                continue

            _require_exported_symbols(lib, REQUIRED_SYMBOLS, path)
            configure_library(lib)
            from .bridge import FdtdOrderGate

            resolved = str(path.resolve())
            setattr(lib, "_deepgpr_path", resolved)  # noqa: B010
            setattr(lib, "_deepgpr_order_gate", FdtdOrderGate())  # noqa: B010
            _LOADED_LIBS[kind] = lib
            _LOGGER.info("Loaded DeepGPR %s backend from %s", kind.upper(), resolved)
            return lib

        expected = "\n".join(str(p) for p in candidates) or "No candidates for this platform."
        details = "\n".join(load_errors) if load_errors else "No load attempts succeeded."
        raise NativeLibraryNotFoundError(
            f"DeepGPR {kind.upper()} shared library was not found or could not be loaded.\n\n"
            f"Current platform: {_SYSTEM_NAME}\n"
            f"Expected one of:\n{expected}\n\n"
            f"Available files in {LIB_DIR}:\n{available_library_files()}\n\n"
            f"Load details:\n{details}"
        )


def get_deepgpr_lib(device: Any) -> ctypes.CDLL:
    """Return the native library that serves ``device``.

    Args:
        device: ``torch.device`` or device string (``"cpu"``, ``"cuda:0"``).

    Raises:
        ValueError: For device types other than CPU and CUDA.
        NativeLibraryNotFoundError: If no library can be loaded.
    """
    kind = device_type(device)
    if kind in ("cpu", "cuda"):
        return _load_library(kind)
    raise ValueError(f"Unsupported DeepGPR device: {device}")


def get_deepgpr_library_path(device: Any) -> str:
    """Return the filesystem path of the library serving ``device``."""
    return str(getattr(get_deepgpr_lib(device), "_deepgpr_path", ""))


def set_library_fdtd_order(lib: Any, fdtd_order: int) -> None:
    """Select the FDTD spatial order in a native library.

    The order is process-global state inside the library; solver calls go
    through :class:`DeepGPR.native.bridge.FdtdOrderGate` so concurrent calls
    with different orders cannot interfere.

    Args:
        lib: Loaded native library.
        fdtd_order: 2, 4 or 8.

    Raises:
        ValueError: If ``fdtd_order`` is unsupported.
        NativeLibraryError: If a non-2 order is requested from an old library.
    """
    if fdtd_order not in SUPPORTED_FDTD_ORDERS:
        raise ValueError("fdtd_order must be one of 2, 4, or 8.")

    if hasattr(lib, "set_fdtd_order"):
        lib.set_fdtd_order(int(fdtd_order))
        return

    if fdtd_order != 2:
        raise NativeLibraryError(
            "The loaded DeepGPR shared library does not support fdtd_order. "
            "Rebuild it from the updated C/CUDA sources to use 4th or 8th order FDTD."
        )


def library_supports(lib: Any, capability: str) -> bool:
    """Return ``True`` if ``lib`` exports ``capability`` and it reports 1.

    Args:
        lib: Loaded native library (or any object; missing symbols yield False).
        capability: Probe symbol name such as ``"deepgpr_supports_external_pml"``.
    """
    probe = getattr(lib, capability, None)
    return probe is not None and int(probe()) == 1


__all__ = [
    "LIB_DIR",
    "available_library_files",
    "candidate_library_paths",
    "configure_library",
    "get_deepgpr_lib",
    "get_deepgpr_library_path",
    "library_supports",
    "set_library_fdtd_order",
]
