"""Marshalling of tensors to the native ``forward``/``backward`` entry points.

Call sites pass a ``{parameter_name: value}`` mapping. The mapping is turned
into the positional ``ctypes`` argument list using the signature tables in
:mod:`.abi`, so argument order is defined in exactly one place.
"""

from __future__ import annotations

import ctypes
import threading
import weakref
from contextlib import contextmanager
from typing import Any, Iterator, List, Mapping, Optional, Tuple

from ..utils.exceptions import NativeLibraryError
from .abi import (
    FLOAT,
    FLOAT_PTR,
    INT,
    INT_PTR,
    LONGLONG,
    NATIVE_FUNCTIONS,
    VOID_PTR,
    Signature,
)
from .loader import set_library_fdtd_order

_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_INT_P = ctypes.POINTER(ctypes.c_int)


def _address(value: Any) -> int:
    """Return the raw memory address of a tensor-like object.

    Supports ``torch.Tensor`` (``data_ptr()``), NumPy arrays (``.ctypes.data``)
    and plain integers.
    """
    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        return int(data_ptr())
    array_interface = getattr(value, "ctypes", None)
    if array_interface is not None:
        return int(array_interface.data)
    if isinstance(value, int):
        return value
    raise TypeError(f"Cannot obtain a memory address from {type(value).__name__}.")


def _convert(kind: str, value: Any) -> Any:
    """Convert one Python value to the representation expected by ctypes."""
    if kind == FLOAT_PTR:
        return ctypes.cast(_address(value), _FLOAT_P)
    if kind == INT_PTR:
        return ctypes.cast(_address(value), _INT_P)
    if kind == VOID_PTR:
        return ctypes.c_void_p(_address(value))
    if kind in (INT, LONGLONG):
        return int(value)
    if kind == FLOAT:
        return float(value)
    raise ValueError(f"Unknown native parameter kind: {kind}")


def marshal_arguments(signature: Signature, arguments: Mapping[str, Any]) -> List[Any]:
    """Build the positional argument list for a native call.

    Args:
        signature: Ordered ``(name, kind)`` pairs from :mod:`.abi`.
        arguments: Values keyed by parameter name.

    Raises:
        KeyError: If a parameter is missing or an unknown name was supplied.
    """
    expected = [name for name, _ in signature]
    missing = [name for name in expected if name not in arguments]
    unexpected = sorted(set(arguments) - set(expected))
    if missing or unexpected:
        raise KeyError(f"Native argument mismatch; missing={missing}, unexpected={unexpected}")
    return [_convert(kind, arguments[name]) for name, kind in signature]


class FdtdOrderGate:
    """Serialise changes of a library's process-global FDTD order.

    ``set_fdtd_order`` writes a global inside the native library that the
    solver reads when it starts. Without coordination, two threads that use
    different orders can race and one of them silently runs with the wrong
    stencil. The gate lets any number of calls with the *same* order run
    concurrently (so multi-threaded, multi-GPU use is not serialised) and
    only makes a call with a different order wait until the others finish.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._order: Optional[int] = None
        self._active = 0

    @contextmanager
    def hold(self, lib: Any, fdtd_order: int) -> Iterator[None]:
        """Select ``fdtd_order`` in ``lib`` for the duration of the block."""
        with self._condition:
            while self._active and self._order != fdtd_order:
                self._condition.wait()
            if self._active == 0:
                set_library_fdtd_order(lib, fdtd_order)
                self._order = fdtd_order
            self._active += 1
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                if self._active == 0:
                    self._condition.notify_all()


_FALLBACK_GATES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_FALLBACK_GATES_LOCK = threading.Lock()


def order_gate(lib: Any) -> FdtdOrderGate:
    """Return the :class:`FdtdOrderGate` associated with ``lib``."""
    gate = getattr(lib, "_deepgpr_order_gate", None)
    if gate is not None:
        return gate
    with _FALLBACK_GATES_LOCK:
        try:
            gate = _FALLBACK_GATES.get(lib)
            if gate is None:
                gate = FdtdOrderGate()
                _FALLBACK_GATES[lib] = gate
            return gate
        except TypeError:  # object is not weak-referenceable
            return FdtdOrderGate()


def invoke(lib: Any, function_name: str, fdtd_order: int, arguments: Mapping[str, Any]) -> None:
    """Call a native solver entry point with the requested FDTD order.

    Args:
        lib: Loaded native library.
        function_name: ``"forward"`` or ``"backward"``.
        fdtd_order: Spatial finite-difference order.
        arguments: Values keyed by C parameter name.
    """
    signature, _ = NATIVE_FUNCTIONS[function_name]
    positional = marshal_arguments(signature, arguments)
    function = getattr(lib, function_name)
    reports_errors = supports_error_reporting(lib)
    if reports_errors:
        lib.deepgpr_clear_last_error()
    with order_gate(lib).hold(lib, fdtd_order):
        function(*positional)
    if reports_errors:
        raise_native_error(lib, function_name)


def supports_error_reporting(lib: Any) -> bool:
    """Whether ``lib`` exports the error-reporting API (native sources >= 0.1.0).

    Libraries built from older sources only print CUDA errors to stderr; with
    them the bridge cannot detect a failed call.
    """
    probe = getattr(lib, "deepgpr_supports_error_reporting", None)
    return (
        probe is not None
        and hasattr(lib, "deepgpr_last_error")
        and hasattr(lib, "deepgpr_clear_last_error")
        and int(probe()) == 1
    )


def last_native_error(lib: Any) -> str:
    """Return the calling thread's last native error message ("" if none)."""
    if not supports_error_reporting(lib):
        return ""
    message = lib.deepgpr_last_error()
    if not message:
        return ""
    if isinstance(message, bytes):
        return message.decode("utf-8", errors="replace")
    return str(message)


def raise_native_error(lib: Any, function_name: str) -> None:
    """Raise :class:`NativeLibraryError` if the last native call failed.

    The native message is cleared once it has been converted to an exception.
    """
    message = last_native_error(lib)
    if message:
        lib.deepgpr_clear_last_error()
        raise NativeLibraryError(f"Native DeepGPR {function_name} failed: {message}")


StreamState = Optional[Tuple[Any, Any, Optional[int]]]


def _begin_native_call(device: Any) -> StreamState:
    """Make the tensor's CUDA device current and order the default stream.

    The native CUDA code launches on the legacy default stream; if the caller
    is running on another stream, the default stream first waits for it.
    """
    if device.type != "cuda":
        return None

    import torch

    previous_device = torch.cuda.current_device()
    target_device = previous_device if device.index is None else device.index
    restore_device = None
    if target_device != previous_device:
        torch.cuda.set_device(target_device)
        restore_device = previous_device

    current_stream = torch.cuda.current_stream(device)
    default_stream = torch.cuda.default_stream(device)
    if current_stream.cuda_stream != default_stream.cuda_stream:
        default_stream.wait_stream(current_stream)
    return current_stream, default_stream, restore_device


def _end_native_call(native_state: StreamState) -> None:
    """Make the caller's stream wait for native work and restore the device."""
    if native_state is None:
        return
    import torch

    current_stream, default_stream, restore_device = native_state
    if current_stream.cuda_stream != default_stream.cuda_stream:
        current_stream.wait_stream(default_stream)
    if restore_device is not None:
        torch.cuda.set_device(restore_device)


@contextmanager
def native_stream_scope(device: Any) -> Iterator[None]:
    """Context manager bracketing a native call with CUDA stream ordering.

    The caller's device is restored even if the native call raises.
    """
    state = _begin_native_call(device)
    try:
        yield
    finally:
        _end_native_call(state)


def require_capability(lib: Any, capability: str, message: str) -> None:
    """Raise :class:`NativeLibraryError` unless ``lib`` reports ``capability``."""
    probe = getattr(lib, capability, None)
    if probe is None or int(probe()) != 1:
        raise NativeLibraryError(message)


__all__ = [
    "FdtdOrderGate",
    "invoke",
    "last_native_error",
    "raise_native_error",
    "supports_error_reporting",
    "marshal_arguments",
    "native_stream_scope",
    "order_gate",
    "require_capability",
]
