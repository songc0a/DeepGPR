"""Backward-compatibility shim for the pre-0.1 ``DeepGPR.common`` module.

The implementation now lives in :mod:`DeepGPR.preprocessing`,
:mod:`DeepGPR.solver`, :mod:`DeepGPR.inversion` and :mod:`DeepGPR.utils`.
Every public (and test-visible private) name is re-exported here so existing
imports keep working. New code should import from the new modules.
"""

from __future__ import annotations

from .config.constants import SPEED_OF_LIGHT as c
from .inversion.regularization import TVRegularization
from .preprocessing.grid import normalize_grid_spacing as _normalize_grid_spacing
from .preprocessing.grid import pmlthick_revert
from .preprocessing.model_setup import _ExtendModel, _shift_locations, initialization
from .preprocessing.stability import check_cfl
from .solver.fields import checkpoint_initial_field, create_or_separate, zero_field
from .solver.pml import (
    CFS,
    CFSParameter,
    build_pml_coeffs,
    build_pml_phi,
    buildpmlcoeffs,
    calculate_pml_update_coeffs,
    e0,
    m0,
)
from .utils.tensor_checks import check_tensors_for_nan_inf

__all__ = [
    "CFS",
    "CFSParameter",
    "TVRegularization",
    "build_pml_coeffs",
    "build_pml_phi",
    "buildpmlcoeffs",
    "c",
    "calculate_pml_update_coeffs",
    "check_cfl",
    "check_tensors_for_nan_inf",
    "checkpoint_initial_field",
    "create_or_separate",
    "e0",
    "initialization",
    "m0",
    "pmlthick_revert",
    "zero_field",
]

# Private helpers that the test-suite and some user scripts import directly.
_COMPAT_PRIVATE = (_ExtendModel, _normalize_grid_spacing, _shift_locations)
