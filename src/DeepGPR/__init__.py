"""DeepGPR: differentiable GPR wave propagation (FDTD) for PyTorch.

Quick start::

    import torch
    import DeepGPR

    wavelet = DeepGPR.wavelet.ricker(2e8, 2000, 3e-11, 5e-9)
    E_saved, E, H, pml, data = DeepGPR.compute(
        device="cpu", dx=0.02, dt=3e-11,
        source_amplitudes=wavelet.reshape(1, -1, 1),
        source_location=torch.tensor([[[10, 10, 0]]]),
        receiver_location=torch.tensor([[[10, 90, 0]]]),
        eps_r=eps_r, sigma=sigma,
    )

Package layout
--------------
``config``          physical constants, native ABI codes, API defaults
``native``          native library loading, ABI table, ctypes bridge
``preprocessing``   input validation, grid/PML normalisation, CFL, wavelets
``solver``          CPML, field state, history storage, autograd, ``compute``
``postprocessing``  receiver-data filters, envelopes, output files
``inversion``       FWI building blocks (TV regularisation)
``visualization``   matplotlib helpers
``utils``           logging, validation, exceptions

The historical modules ``DeepGPR.common``, ``DeepGPR.compute2``,
``DeepGPR.multiscale`` and ``DeepGPR.wavelet`` remain importable.
"""

from __future__ import annotations

from . import config, utils
from ._version import __version__
from .native.loader import (
    get_deepgpr_lib,
    get_deepgpr_library_path,
    set_library_fdtd_order,
)
from .utils.logger import configure_logging, get_logger

from . import native, preprocessing, solver, postprocessing, inversion, visualization
from . import common, compute2, multiscale, wavelet
from .common import (
    CFS,
    CFSParameter,
    TVRegularization,
    build_pml_coeffs,
    build_pml_phi,
    buildpmlcoeffs,
    c,
    calculate_pml_update_coeffs,
    check_cfl,
    check_tensors_for_nan_inf,
    checkpoint_initial_field,
    create_or_separate,
    e0,
    initialization,
    m0,
    pmlthick_revert,
    zero_field,
)
from .multiscale import apply_filter, design_fir_filter, hilbert_transform
from .preprocessing.stability import max_stable_time_step
from .solver.autograd import DeepGPR
from .solver.modeling import compute
from .solver.memory import estimate_compute_memory
from .solver.sampling import recommended_sampling_interval, source_max_frequency
from .solver.storage import decompress_wavefield_history
from .utils.exceptions import (
    CFLConditionError,
    DeepGPRError,
    NativeLibraryError,
    NativeLibraryNotFoundError,
    NonFiniteTensorError,
)
from .wavelet import gaussian, gaussian_derivative, morlet, ricker, sine_burst

__all__ = [
    "CFLConditionError",
    "CFS",
    "CFSParameter",
    "DeepGPR",
    "DeepGPRError",
    "NativeLibraryError",
    "NativeLibraryNotFoundError",
    "NonFiniteTensorError",
    "TVRegularization",
    "__version__",
    "apply_filter",
    "build_pml_coeffs",
    "build_pml_phi",
    "buildpmlcoeffs",
    "c",
    "calculate_pml_update_coeffs",
    "check_cfl",
    "check_tensors_for_nan_inf",
    "checkpoint_initial_field",
    "common",
    "compute",
    "compute2",
    "config",
    "configure_logging",
    "create_or_separate",
    "decompress_wavefield_history",
    "design_fir_filter",
    "e0",
    "estimate_compute_memory",
    "gaussian",
    "gaussian_derivative",
    "get_deepgpr_lib",
    "get_deepgpr_library_path",
    "get_logger",
    "hilbert_transform",
    "initialization",
    "inversion",
    "m0",
    "max_stable_time_step",
    "morlet",
    "multiscale",
    "native",
    "pmlthick_revert",
    "postprocessing",
    "preprocessing",
    "recommended_sampling_interval",
    "ricker",
    "set_library_fdtd_order",
    "sine_burst",
    "solver",
    "source_max_frequency",
    "utils",
    "visualization",
    "wavelet",
    "zero_field",
]
