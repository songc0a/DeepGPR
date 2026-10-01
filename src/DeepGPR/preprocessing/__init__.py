"""Pre-processing: input validation, grid/PML normalisation, CFL and wavelets."""

from .grid import normalize_grid_spacing, pmlthick_revert
from .model_setup import (
    NATIVE_FLOAT_DTYPE,
    PreparedModel,
    extended_shape_of,
    initialization,
    physical_shape_of,
)
from .stability import check_cfl, max_stable_time_step
from .wavelets import gaussian, gaussian_derivative, morlet, ricker, sine_burst

__all__ = [
    "NATIVE_FLOAT_DTYPE",
    "PreparedModel",
    "check_cfl",
    "extended_shape_of",
    "gaussian",
    "gaussian_derivative",
    "initialization",
    "max_stable_time_step",
    "morlet",
    "normalize_grid_spacing",
    "physical_shape_of",
    "pmlthick_revert",
    "ricker",
    "sine_burst",
]
