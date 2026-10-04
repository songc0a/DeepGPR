"""FDTD solver: CPML, field state, history storage, autograd bridge, ``compute``."""

from .autograd import DeepGPR, SolverConfig, SolverTensors
from .modeling import compute
from .fields import checkpoint_initial_field, create_or_separate, zero_field
from .memory import estimate_compute_memory, format_compute_preview
from .pml import (
    CFS,
    CFSParameter,
    build_pml_coeffs,
    build_pml_phi,
    buildpmlcoeffs,
    calculate_pml_update_coeffs,
    pml_face_descriptors,
    pml_phi_element_count,
)
from .sampling import recommended_sampling_interval, source_max_frequency
from .storage import (
    WavefieldStorageConfig,
    decompress_wavefield_history,
    int8_history_layout,
)

__all__ = [
    "CFS",
    "CFSParameter",
    "DeepGPR",
    "SolverConfig",
    "SolverTensors",
    "WavefieldStorageConfig",
    "build_pml_coeffs",
    "build_pml_phi",
    "buildpmlcoeffs",
    "calculate_pml_update_coeffs",
    "checkpoint_initial_field",
    "compute",
    "create_or_separate",
    "decompress_wavefield_history",
    "estimate_compute_memory",
    "format_compute_preview",
    "int8_history_layout",
    "pml_face_descriptors",
    "pml_phi_element_count",
    "recommended_sampling_interval",
    "source_max_frequency",
    "zero_field",
]
