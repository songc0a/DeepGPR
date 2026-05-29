import ctypes
import os
import platform
from pathlib import Path


_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_INT_P = ctypes.POINTER(ctypes.c_int)

_PACKAGE_DIR = Path(__file__).resolve().parent
_LIB_DIR = _PACKAGE_DIR / "lib"
_SYSTEM_NAME = platform.system()


def _candidate_library_paths() -> list[Path]:
    """
    Return possible precompiled DeepGPR library paths for the current platform.

    Expected files:
        Windows:
            src/DeepGPR/lib/deepgpr.dll

        Linux:
            src/DeepGPR/lib/libdeepgpr.so
            or
            src/DeepGPR/lib/deepgpr.so

    This package no longer compiles deepgpr.cu during import.
    The corresponding precompiled library must already be included in the package.
    """
    if _SYSTEM_NAME == "Windows":
        return [
            _LIB_DIR / "deepgpr.dll",
        ]

    if _SYSTEM_NAME == "Linux":
        return [
            _LIB_DIR / "libdeepgpr.so",
            _LIB_DIR / "deepgpr.so",
        ]

    raise RuntimeError(f"Unsupported operating system: {_SYSTEM_NAME}")


def _select_library_path() -> Path:
    candidates = _candidate_library_paths()

    for path in candidates:
        if path.is_file():
            return path

    available_files = []
    if _LIB_DIR.is_dir():
        available_files = [p.name for p in _LIB_DIR.iterdir()]

    expected = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "DeepGPR precompiled shared library was not found.\n\n"
        f"Current platform: {_SYSTEM_NAME}\n"
        f"Expected one of:\n{expected}\n\n"
        f"Available files in {_LIB_DIR}:\n{available_files}\n\n"
        "Please put the compiled library file into src/DeepGPR/lib before building "
        "or installing the package."
    )


_LIB_PATH = _select_library_path()


def _add_windows_dll_search_paths() -> None:
    """Make DLL dependency lookup more reliable on Python 3.8+ for Windows."""
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


def _load_deepgpr_library() -> ctypes.CDLL:
    _add_windows_dll_search_paths()

    try:
        return ctypes.CDLL(str(_LIB_PATH.resolve()))
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load DeepGPR shared library:\n{_LIB_PATH}\n\n"
            "Possible causes:\n"
            "1. The DLL/SO was compiled for a different operating system or CPU architecture.\n"
            "2. CUDA runtime DLLs/shared libraries are missing or cannot be found.\n"
            "3. The NVIDIA driver or CUDA runtime version is incompatible.\n"
            "4. On Windows, CUDA_PATH\\bin or the required dependency directory is not in the DLL search path."
        ) from exc


def _require_exported_symbols(lib: ctypes.CDLL, symbols: tuple[str, ...]) -> None:
    missing = [name for name in symbols if not hasattr(lib, name)]
    if missing:
        raise RuntimeError(
            "The shared library was loaded, but the following exported C ABI symbols "
            f"were not found: {missing}. "
            "Check that the corresponding functions in deepgpr.cu are declared with DEEPGPR_API."
        )


c_lib = _load_deepgpr_library()
_require_exported_symbols(c_lib, ("forward", "backward"))


c_lib.forward.argtypes = [
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
c_lib.forward.restype = None


c_lib.backward.argtypes = [
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
c_lib.backward.restype = None


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