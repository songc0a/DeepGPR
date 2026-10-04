"""Declarative description of the DeepGPR native C ABI (``lib/deepgpr.h``).

The Python bindings used to spell out ~90 positional ``ctypes`` arguments by
hand twice (once for ``argtypes`` and once at each call site). This module
describes each exported function exactly once, as an ordered tuple of
``(parameter_name, kind)`` pairs that mirrors the C prototype. Both the
``argtypes`` configuration and the call-argument marshalling are generated
from it, and ``tests/test_native_abi.py`` checks the tables against the header
so a mismatch can never go unnoticed.

This module is intentionally free of PyTorch imports.
"""

from __future__ import annotations

import ctypes
from typing import Dict, Tuple

from ..config.constants import CPML_FACE_NAMES

#: Parameter kinds and their ctypes representation.
FLOAT_PTR = "float*"
INT_PTR = "int*"
VOID_PTR = "void*"
FLOAT = "float"
INT = "int"
LONGLONG = "long long"
CHAR_PTR = "char*"

CTYPES_BY_KIND: Dict[str, object] = {
    FLOAT_PTR: ctypes.POINTER(ctypes.c_float),
    INT_PTR: ctypes.POINTER(ctypes.c_int),
    VOID_PTR: ctypes.c_void_p,
    FLOAT: ctypes.c_float,
    INT: ctypes.c_int,
    LONGLONG: ctypes.c_longlong,
    CHAR_PTR: ctypes.c_char_p,
}

Signature = Tuple[Tuple[str, str], ...]


def _phi_names(prefix: str = "") -> Tuple[str, ...]:
    """Return the 24 CPML auxiliary-array parameter names in native order."""
    names = []
    for face in CPML_FACE_NAMES:
        for field in ("e", "h"):
            for index in (1, 2):
                names.append(f"{prefix}{face}_{field}_phi{index}")
    return tuple(names)


def _pml_thickness_params() -> Signature:
    return tuple((f"pml_{face}", INT) for face in CPML_FACE_NAMES)


def _pml_coefficient_params() -> Signature:
    electric = tuple((f"{face}_e_coeff", FLOAT_PTR) for face in CPML_FACE_NAMES)
    magnetic = tuple((f"{face}_h_coeff", FLOAT_PTR) for face in CPML_FACE_NAMES)
    return electric + magnetic


_MATERIALS: Signature = (
    ("eps_r_pad", FLOAT_PTR),
    ("sigma_pad", FLOAT_PTR),
    ("mu_r_pad", FLOAT_PTR),
)

_UPDATE_COEFFICIENTS: Signature = tuple(
    (name, FLOAT_PTR) for name in ("ce_hist", "ce_curl", "ce_rhs", "ch_hist", "ch_curl", "ch_rhs")
)

_TIME_AND_GRID: Signature = (
    ("dt", FLOAT),
    ("nt", INT),
    ("nshot", INT),
    ("nreceiver", INT),
    ("dx", FLOAT),
    ("dy", FLOAT),
    ("dz", FLOAT),
)

_FIELD_SIZES: Signature = (
    ("nx_fields", INT),
    ("ny_fields", INT),
    ("nz_fields", INT),
)

#: ``void forward(...)`` in native order.
FORWARD_SIGNATURE: Signature = (
    _MATERIALS
    + (("E_saved", VOID_PTR), ("R_saved", VOID_PTR))
    + tuple((name, FLOAT_PTR) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
    + _UPDATE_COEFFICIENTS
    + tuple((name, FLOAT_PTR) for name in _phi_names())
    + _pml_thickness_params()
    + _pml_coefficient_params()
    + _TIME_AND_GRID
    + (
        ("receiver_location", INT_PTR),
        ("receiver_data", FLOAT_PTR),
        ("receiver_component", INT),
    )
    + _FIELD_SIZES
    + (
        ("nsource", INT),
        ("source_location", INT_PTR),
        ("source_waveform", FLOAT_PTR),
        ("source_component", INT),
        ("sampling_interval", INT),
        ("fwi_mode", INT),
        ("storage_type", INT),
        ("save_model_history", INT),
        ("use_async_offload", INT),
    )
)

#: ``void backward(...)`` in native order.
BACKWARD_SIGNATURE: Signature = (
    _MATERIALS
    + (("E_saved", VOID_PTR), ("R_saved", VOID_PTR))
    + tuple(
        (name, FLOAT_PTR)
        for name in (
            "lambda_ex",
            "lambda_ey",
            "lambda_ez",
            "lambda_hx",
            "lambda_hy",
            "lambda_hz",
        )
    )
    + _UPDATE_COEFFICIENTS
    + tuple((name, FLOAT_PTR) for name in _phi_names("lambda_"))
    + _pml_thickness_params()
    + _pml_coefficient_params()
    + _TIME_AND_GRID
    + _FIELD_SIZES
    + (
        ("ndata_source", INT),
        ("receiver_location", INT_PTR),
        ("data_grad", FLOAT_PTR),
        ("receiver_component", INT),
        ("nsource", INT),
        ("source_location", INT_PTR),
        ("source_component", INT),
        ("grad_source", FLOAT_PTR),
        ("source_requires_grad", INT),
        ("grad_eps_r", FLOAT_PTR),
        ("grad_sigma", FLOAT_PTR),
        ("eps_r_requires_grad", INT),
        ("sigma_requires_grad", INT),
        ("sampling_interval", INT),
        ("fwi_mode", INT),
        ("storage_type", INT),
        ("use_async_offload", INT),
    )
)

#: Every exported symbol with its parameters and return kind (``None`` = void).
NATIVE_FUNCTIONS: Dict[str, Tuple[Signature, object]] = {
    "deepgpr_supports_error_reporting": ((), INT),
    "deepgpr_last_error": ((), CHAR_PTR),
    "deepgpr_clear_last_error": ((), None),
    "deepgpr_deterministic_adjoint": ((), INT),
    "deepgpr_abi_version": ((), INT),
    "deepgpr_supports_external_pml": ((), INT),
    "deepgpr_supports_int8_wavefield": ((), INT),
    "deepgpr_supports_conversion_backends": ((), INT),
    "deepgpr_supports_int8_reduction_backends": ((), INT),
    "deepgpr_supports_rhs_reconstruction": ((), INT),
    "deepgpr_test_wavefield_conversion": (
        (
            ("input", FLOAT_PTR),
            ("encoded", VOID_PTR),
            ("decoded", FLOAT_PTR),
            ("count", LONGLONG),
            ("storage_kind", INT),
            ("conversion_backend", INT),
        ),
        None,
    ),
    "set_fdtd_order": ((("order", INT),), None),
    "forward": (FORWARD_SIGNATURE, None),
    "backward": (BACKWARD_SIGNATURE, None),
}

#: Symbols that every usable library must export.
REQUIRED_SYMBOLS: Tuple[str, ...] = ("forward", "backward", "deepgpr_abi_version")

#: Parameter names of the 24 CPML auxiliary arrays (forward naming).
PHI_PARAMETER_NAMES: Tuple[str, ...] = _phi_names()

#: Parameter names of the 24 CPML adjoint arrays (backward naming).
LAMBDA_PHI_PARAMETER_NAMES: Tuple[str, ...] = _phi_names("lambda_")


def ctypes_signature(signature: Signature) -> list:
    """Return the ``argtypes`` list for a signature.

    Args:
        signature: Ordered ``(name, kind)`` pairs.
    """
    return [CTYPES_BY_KIND[kind] for _, kind in signature]


__all__ = [
    "CHAR_PTR",
    "BACKWARD_SIGNATURE",
    "CTYPES_BY_KIND",
    "FLOAT",
    "FLOAT_PTR",
    "FORWARD_SIGNATURE",
    "INT",
    "INT_PTR",
    "LAMBDA_PHI_PARAMETER_NAMES",
    "LONGLONG",
    "NATIVE_FUNCTIONS",
    "PHI_PARAMETER_NAMES",
    "REQUIRED_SYMBOLS",
    "VOID_PTR",
    "ctypes_signature",
]
