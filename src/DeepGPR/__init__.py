import ctypes
import os
import platform
import subprocess
from pathlib import Path


_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_INT_P = ctypes.POINTER(ctypes.c_int)

_PACKAGE_DIR = Path(__file__).resolve().parent
_LIB_DIR = _PACKAGE_DIR / "lib"
_CU_FILE = _LIB_DIR / "deepgpr.cu"
_SYSTEM_NAME = platform.system()

if _SYSTEM_NAME == "Windows":
    _LIB_EXTENSION = ".dll"
    _NVCC_CMD = [
        "nvcc",
        "-shared",
        "-o",
        str(_LIB_DIR / f"deepgpr{_LIB_EXTENSION}"),
        str(_CU_FILE),
    ]
else:
    _LIB_EXTENSION = ".so"
    _NVCC_CMD = [
        "nvcc",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-o",
        str(_LIB_DIR / f"deepgpr{_LIB_EXTENSION}"),
        str(_CU_FILE),
    ]

_LIB_PATH = _LIB_DIR / f"deepgpr{_LIB_EXTENSION}"


def _add_windows_dll_search_paths() -> None:
    """Make DLL dependency lookup more reliable on Python 3.8+ for Windows."""
    if _SYSTEM_NAME != "Windows" or not hasattr(os, "add_dll_directory"):
        return

    search_dirs = [_LIB_DIR]

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        search_dirs.append(Path(cuda_path) / "bin")

    cuda_path_v12 = os.environ.get("CUDA_PATH_V12_6")
    if cuda_path_v12:
        search_dirs.append(Path(cuda_path_v12) / "bin")

    for directory in search_dirs:
        try:
            if Path(directory).is_dir():
                os.add_dll_directory(str(directory))
        except OSError:
            pass


def _compile_cuda_extension_if_needed() -> None:
    """Compile deepgpr.cu only when the platform-specific shared library is absent."""
    if _LIB_PATH.is_file():
        return

    if not _CU_FILE.is_file():
        raise FileNotFoundError(
            f"CUDA source file was not found: {_CU_FILE}. "
            f"Expected precompiled library was also absent: {_LIB_PATH}."
        )

    print(f"Compiling CUDA extension for {_SYSTEM_NAME} directly via nvcc...")
    try:
        subprocess.run(_NVCC_CMD, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Compilation failed: 'nvcc' command not found. "
            "Please install NVIDIA CUDA Toolkit and ensure 'nvcc' is available in PATH. "
            "On Windows, Visual Studio C++ Build Tools must also be installed and cl.exe must be available."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Compilation failed with error code {exc.returncode}. "
            f"Command was: {' '.join(_NVCC_CMD)}"
        ) from exc

    if not _LIB_PATH.is_file():
        raise RuntimeError(f"Compilation finished but shared library was not generated: {_LIB_PATH}")


def _load_deepgpr_library() -> ctypes.CDLL:
    _compile_cuda_extension_if_needed()
    _add_windows_dll_search_paths()

    try:
        return ctypes.CDLL(str(_LIB_PATH.resolve()))
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load DeepGPR shared library: {_LIB_PATH}. "
            "If you are using Windows, ensure CUDA runtime DLLs are available in PATH "
            "or located under CUDA_PATH\\bin."
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
    "subprocess",
    "Path",
}

__all__ = [
    name for name in globals()
    if not name.startswith("_") and name not in _EXCLUDED_FROM_ALL
]