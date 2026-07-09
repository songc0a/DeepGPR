from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path


_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_INT_P = ctypes.POINTER(ctypes.c_int)

_PACKAGE_DIR = Path(__file__).resolve().parent
_LIB_DIR = _PACKAGE_DIR / "lib"
_SYSTEM_NAME = platform.system()
_LOADED_LIBS: dict[str, ctypes.CDLL] = {}


def _candidate_library_paths(kind: str) -> list[Path]:
    if kind == "cuda":
        if _SYSTEM_NAME == "Windows":
            return [_LIB_DIR / "deepgpr.dll"]
        if _SYSTEM_NAME == "Linux":
            return [_LIB_DIR / "deepgpr.so", _LIB_DIR / "libdeepgpr.so"]
        return []

    if kind == "cpu":
        if _SYSTEM_NAME == "Windows":
            return [_LIB_DIR / "deepgpr_cpu.dll"]
        if _SYSTEM_NAME == "Darwin":
            return [_LIB_DIR / "libdeepgpr_cpu.dylib", _LIB_DIR / "deepgpr_cpu.dylib"]
        if _SYSTEM_NAME == "Linux":
            return [_LIB_DIR / "deepgpr_cpu.so", _LIB_DIR / "libdeepgpr_cpu.so"]
        return []

    raise ValueError(f"Unknown DeepGPR library kind: {kind}")


def _available_library_files() -> list[str]:
    if not _LIB_DIR.is_dir():
        return []
    return sorted(p.name for p in _LIB_DIR.iterdir())


def _add_windows_dll_search_paths() -> None:
    if _SYSTEM_NAME != "Windows" or not hasattr(os, "add_dll_directory"):
        return

    search_dirs: list[Path] = [_LIB_DIR]

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        search_dirs.append(Path(cuda_path) / "bin")

    for key, value in os.environ.items():
        if key.startswith("CUDA_PATH_V") and value:
            search_dirs.append(Path(value) / "bin")

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        search_dirs.append(Path(conda_prefix) / "Library" / "bin")

    seen = set()
    for directory in search_dirs:
        directory = Path(directory)
        if directory in seen:
            continue
        seen.add(directory)

        try:
            if directory.is_dir():
                os.add_dll_directory(str(directory))
        except OSError:
            pass


def _require_exported_symbols(lib: ctypes.CDLL, symbols: tuple[str, ...], path: Path) -> None:
    missing = [name for name in symbols if not hasattr(lib, name)]
    if missing:
        raise RuntimeError(
            "The shared library was loaded, but the following exported C ABI symbols "
            f"were not found: {missing}.\nLibrary: {path}"
        )


def _configure_deepgpr_library(lib: ctypes.CDLL) -> None:
    if getattr(lib, "_deepgpr_argtypes_configured", False):
        return

    lib.forward.argtypes = [
        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P,

        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,

        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,

        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,

        ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        _INT_P, _FLOAT_P,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        _INT_P, _FLOAT_P,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.forward.restype = None

    lib.backward.argtypes = [
        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P,

        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,

        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,

        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,

        ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        _INT_P, _FLOAT_P,
        ctypes.c_int,
        _FLOAT_P, _FLOAT_P,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.backward.restype = None

    if hasattr(lib, "set_fdtd_order"):
        lib.set_fdtd_order.argtypes = [ctypes.c_int]
        lib.set_fdtd_order.restype = None

    lib._deepgpr_argtypes_configured = True


def _load_deepgpr_library(kind: str) -> ctypes.CDLL:
    if kind in _LOADED_LIBS:
        return _LOADED_LIBS[kind]

    _add_windows_dll_search_paths()
    candidates = _candidate_library_paths(kind)
    load_errors: list[str] = []

    for path in candidates:
        if not path.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(path.resolve()))
        except OSError as exc:
            load_errors.append(f"{path}: {exc}")
            continue

        _require_exported_symbols(lib, ("forward", "backward"), path)
        _configure_deepgpr_library(lib)
        lib._deepgpr_path = str(path.resolve())
        _LOADED_LIBS[kind] = lib
        return lib

    expected = "\n".join(str(p) for p in candidates) or "No candidates for this platform."
    details = "\n".join(load_errors) if load_errors else "No load attempts succeeded."
    raise FileNotFoundError(
        f"DeepGPR {kind.upper()} shared library was not found or could not be loaded.\n\n"
        f"Current platform: {_SYSTEM_NAME}\n"
        f"Expected one of:\n{expected}\n\n"
        f"Available files in {_LIB_DIR}:\n{_available_library_files()}\n\n"
        f"Load details:\n{details}"
    )


def _device_type(device) -> str:
    device_type = getattr(device, "type", None)
    if device_type is not None:
        return str(device_type).lower()
    return str(device).split(":", 1)[0].lower()


def get_deepgpr_lib(device) -> ctypes.CDLL:
    device_type = _device_type(device)
    if device_type == "cpu":
        return _load_deepgpr_library("cpu")
    if device_type == "cuda":
        return _load_deepgpr_library("cuda")
    raise ValueError(f"Unsupported DeepGPR device: {device}")


def set_library_fdtd_order(lib: ctypes.CDLL, fdtd_order: int) -> None:
    if fdtd_order not in (2, 4, 8):
        raise ValueError("fdtd_order must be one of 2, 4, or 8.")

    if hasattr(lib, "set_fdtd_order"):
        lib.set_fdtd_order(int(fdtd_order))
        return

    if fdtd_order != 2:
        raise RuntimeError(
            "The loaded DeepGPR shared library does not support fdtd_order. "
            "Rebuild it from the updated C/CUDA sources to use 4th or 8th order FDTD."
        )


from .common import *
from .compute2 import *
from .multiscale import *
from .wavelet import *


_EXCLUDED_FROM_ALL = {
    "ctypes",
    "os",
    "platform",
    "Path",
}

__all__ = [
    name for name in globals()
    if not name.startswith("_") and name not in _EXCLUDED_FROM_ALL
]
