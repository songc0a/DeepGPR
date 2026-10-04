"""Physical, numerical and native-ABI constants used across DeepGPR.

Every value in this module was previously an inline literal somewhere in the
package. Centralising them documents their meaning without changing any
arithmetic: each constant keeps the exact expression it replaced, so results
remain bit-for-bit identical.

This module deliberately has no third-party imports so that it can be used by
tooling (for example the ABI consistency test) without PyTorch installed.
"""

from __future__ import annotations

from typing import Dict, Final, Mapping, Tuple

# ---------------------------------------------------------------------------
# Physical constants (SI units)
# ---------------------------------------------------------------------------

#: Speed of light in vacuum [m/s].
SPEED_OF_LIGHT: Final[float] = 299792458.0

#: Vacuum permeability ``mu_0`` [H/m], CODATA 2018 (``DEEPGPR_MU0`` in
#: ``lib/deepgpr.h``). Before 0.1.0 the Python side used the pre-2019 exact
#: value ``4 pi 1e-7`` while the native solvers used CODATA 2018; both now use
#: the same number (relative change ~5e-10, below float32 resolution).
VACUUM_PERMEABILITY: Final[float] = 1.25663706212e-06

#: Vacuum permittivity ``epsilon_0`` [F/m], CODATA 2018 (``DEEPGPR_EPSILON0``).
VACUUM_PERMITTIVITY: Final[float] = 8.8541878128e-12

# ---------------------------------------------------------------------------
# FDTD discretisation
# ---------------------------------------------------------------------------

#: Spatial finite-difference orders implemented by the native solvers.
SUPPORTED_FDTD_ORDERS: Final[Tuple[int, ...]] = (2, 4, 8)

#: Sum of the absolute staggered-stencil coefficients for each order. It
#: scales the CFL limit ``dt <= sqrt(eps_r mu_r) / (c * S * sqrt(sum 1/d^2))``.
FDTD_STENCIL_COEFFICIENT_SUMS: Final[Mapping[int, float]] = {
    2: 1.0,
    4: 9.0 / 8.0 + 1.0 / 24.0,
    8: 1225.0 / 1024.0 + 245.0 / 3072.0 + 49.0 / 5120.0 + 5.0 / 7168.0,
}

#: Model-gradient modes: 2 accumulates Ez only (2D TM), 3 uses Ex, Ey and Ez.
SUPPORTED_GRADIENT_MODES: Final[Tuple[int, ...]] = (2, 3)

#: Electric-field component indices accepted for sources and receivers.
FIELD_COMPONENTS: Final[Tuple[int, ...]] = (0, 1, 2)

#: Cells whose conductivity exceeds this value [S/m] are treated as perfect
#: electric conductors by both native backends (their E update coefficients are
#: zero and they receive no material gradient). Mirrors
#: ``DEEPGPR_PEC_SIGMA_THRESHOLD`` in ``lib/deepgpr.h`` (checked by the tests).
PEC_CONDUCTIVITY_THRESHOLD: Final[float] = 100.0

#: Field-sized weight buffers per shot used by the deterministic CPU adjoint
#: (``ADJOINT_CHANNELS`` in ``lib/deepgpr_cpu.c``).
CPU_ADJOINT_WEIGHT_CHANNELS: Final[int] = 6

#: Minimum relative permittivity accepted for a physical model.
MIN_RELATIVE_PERMITTIVITY: Final[float] = 1.0

# ---------------------------------------------------------------------------
# CPML (complex-frequency-shifted PML) profile parameters
# ---------------------------------------------------------------------------

#: Polynomial grading orders understood by :class:`DeepGPR.solver.pml.CFSParameter`.
CPML_SCALING_PROFILES: Final[Dict[str, int]] = {
    "constant": 0,
    "linear": 1,
    "quadratic": 2,
    "cubic": 3,
    "quartic": 4,
    "quintic": 5,
    "sextic": 6,
    "septic": 7,
    "octic": 8,
}

#: Numerator factor of the optimal CPML sigma_max, ``0.8 (m + 1) / (eta d sqrt(eps mu))``.
CPML_SIGMA_MAX_FACTOR: Final[float] = 0.8

#: Number of CFS parameter sets per boundary cell (single-pole CPML).
CPML_POLE_COUNT: Final[int] = 1

#: Number of update coefficients stored per CPML cell (RA, RB, RE, RF).
CPML_COEFFICIENTS_PER_CELL: Final[int] = 4

#: Number of CPML auxiliary (phi) tensors per boundary face.
CPML_PHI_TENSORS_PER_FACE: Final[int] = 4

#: Number of CPML faces (x0, xm, y0, ym, z0, zm).
CPML_FACE_COUNT: Final[int] = 6

#: Total number of CPML auxiliary tensors exchanged with the native solver.
CPML_PHI_TENSOR_COUNT: Final[int] = CPML_FACE_COUNT * CPML_PHI_TENSORS_PER_FACE

#: Human readable CPML face names in native order.
CPML_FACE_NAMES: Final[Tuple[str, ...]] = ("x0", "xm", "y0", "ym", "z0", "zm")

# ---------------------------------------------------------------------------
# Temporal sampling of the gradient history
# ---------------------------------------------------------------------------

#: Relative amplitude at which a source spectrum counts as decayed when
#: ``model_gradient_sampling_interval="auto"`` locates ``f_max``. For a Ricker
#: wavelet ``|W(f)| / |W(f_peak)| = (f/f_peak)^2 exp(1 - (f/f_peak)^2)`` equals
#: 3.0e-3 at ``f ~= 3 f_peak``, so the recommended interval is
#: ``floor(1 / (12 f_peak dt))``.
SOURCE_SPECTRUM_CUTOFF_FRACTION: Final[float] = 3.0e-3

# ---------------------------------------------------------------------------
# Native library ABI
# ---------------------------------------------------------------------------

#: C ABI version the Python bindings are written against (see ``lib/deepgpr.h``).
NATIVE_ABI_VERSION: Final[int] = 6

#: CUDA grid-dimension limit on shots per native call.
CUDA_MAX_SHOTS_PER_CALL: Final[int] = 65535

#: Largest signed 32-bit integer (native indices are ``int``).
INT32_MAX: Final[int] = 2**31 - 1

# Saved-wavefield storage codes packed into the native ``storage_type`` argument.
WAVEFIELD_STORAGE_FLOAT32: Final[int] = 0
WAVEFIELD_STORAGE_FLOAT16: Final[int] = 1
WAVEFIELD_STORAGE_BFLOAT16: Final[int] = 2
WAVEFIELD_STORAGE_INT8: Final[int] = 3

#: Bit offsets and mask of the INT8 block shape inside ``storage_type``.
INT8_BLOCK_X_SHIFT: Final[int] = 8
INT8_BLOCK_Y_SHIFT: Final[int] = 14
INT8_BLOCK_Z_SHIFT: Final[int] = 20
INT8_BLOCK_DIM_MASK: Final[int] = 0x3F

#: Flag telling the native forward solver not to write E/R histories.
WAVEFIELD_HISTORY_DISABLED: Final[int] = 1 << 26

#: Bit offset of the FP16/BF16 conversion backend selector.
WAVEFIELD_CONVERSION_SHIFT: Final[int] = 27

#: Native FP16/BF16 conversion backend codes.
WAVEFIELD_CONVERSION_BACKENDS: Final[Mapping[str, int]] = {
    "legacy": 0,
    "native_scalar": 1,
    "native_vec2": 2,
}

#: Bit offset of the INT8 tile-reduction backend selector.
INT8_REDUCTION_SHIFT: Final[int] = 29

#: Native INT8 tile-maximum reduction backend codes.
INT8_REDUCTION_BACKENDS: Final[Mapping[str, int]] = {
    "current": 0,
    "cub_block": 1,
    "warp_shuffle": 2,
}

#: Largest INT8 compression block volume supported by the fused kernels.
INT8_MAX_BLOCK_VOLUME: Final[int] = 256

#: Tile volume for which the CUB / warp-shuffle reductions are compiled.
INT8_SPECIALISED_REDUCTION_VOLUME: Final[int] = 64

#: Default INT8 compression blocks for 2D and 3D models.
INT8_DEFAULT_BLOCK_2D: Final[Tuple[int, int, int]] = (8, 8, 1)
INT8_DEFAULT_BLOCK_3D: Final[Tuple[int, int, int]] = (4, 4, 4)

#: Byte alignment of the INT8 payload before its FP32 scale array.
INT8_SCALE_ALIGNMENT_BYTES: Final[int] = 4

#: Size of one FP32 scale in the packed INT8 history.
FLOAT32_BYTES: Final[int] = 4

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

#: Safety margin applied to memory estimates when recommending capacity.
MEMORY_SAFETY_MARGIN: Final[float] = 1.20

#: Binary memory units used by the memory report.
MEMORY_UNITS: Final[Tuple[str, ...]] = ("B", "KiB", "MiB", "GiB", "TiB")

# ---------------------------------------------------------------------------
# Signal processing and regularisation
# ---------------------------------------------------------------------------

#: Hamming window coefficients ``a0 - a1 cos(2 pi n / (N - 1))``.
HAMMING_ALPHA: Final[float] = 0.54
HAMMING_BETA: Final[float] = 0.46

#: Stabiliser added inside the square root of the "isotropic" TV variant.
TV_ISOTROPIC_EPSILON: Final[float] = 1e-8

__all__ = [
    "CPML_COEFFICIENTS_PER_CELL",
    "CPML_FACE_COUNT",
    "CPML_FACE_NAMES",
    "CPML_PHI_TENSORS_PER_FACE",
    "CPML_PHI_TENSOR_COUNT",
    "CPML_POLE_COUNT",
    "CPML_SCALING_PROFILES",
    "CPU_ADJOINT_WEIGHT_CHANNELS",
    "CPML_SIGMA_MAX_FACTOR",
    "CUDA_MAX_SHOTS_PER_CALL",
    "FDTD_STENCIL_COEFFICIENT_SUMS",
    "FIELD_COMPONENTS",
    "FLOAT32_BYTES",
    "HAMMING_ALPHA",
    "HAMMING_BETA",
    "INT32_MAX",
    "INT8_BLOCK_DIM_MASK",
    "INT8_BLOCK_X_SHIFT",
    "INT8_BLOCK_Y_SHIFT",
    "INT8_BLOCK_Z_SHIFT",
    "INT8_DEFAULT_BLOCK_2D",
    "INT8_DEFAULT_BLOCK_3D",
    "INT8_MAX_BLOCK_VOLUME",
    "INT8_REDUCTION_BACKENDS",
    "INT8_REDUCTION_SHIFT",
    "INT8_SCALE_ALIGNMENT_BYTES",
    "INT8_SPECIALISED_REDUCTION_VOLUME",
    "MEMORY_SAFETY_MARGIN",
    "MEMORY_UNITS",
    "MIN_RELATIVE_PERMITTIVITY",
    "NATIVE_ABI_VERSION",
    "PEC_CONDUCTIVITY_THRESHOLD",
    "SOURCE_SPECTRUM_CUTOFF_FRACTION",
    "SPEED_OF_LIGHT",
    "SUPPORTED_FDTD_ORDERS",
    "SUPPORTED_GRADIENT_MODES",
    "TV_ISOTROPIC_EPSILON",
    "VACUUM_PERMEABILITY",
    "VACUUM_PERMITTIVITY",
    "WAVEFIELD_CONVERSION_BACKENDS",
    "WAVEFIELD_CONVERSION_SHIFT",
    "WAVEFIELD_HISTORY_DISABLED",
    "WAVEFIELD_STORAGE_BFLOAT16",
    "WAVEFIELD_STORAGE_FLOAT16",
    "WAVEFIELD_STORAGE_FLOAT32",
    "WAVEFIELD_STORAGE_INT8",
]
