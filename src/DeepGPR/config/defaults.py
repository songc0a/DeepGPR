"""Default values for the public DeepGPR API.

The defaults are the literals that previously appeared directly in function
signatures. Keeping them here makes the configuration discoverable and lets
callers build option dictionaries without copying magic numbers.
"""

from __future__ import annotations

from typing import Final

#: External CPML thickness in cells on every active face.
DEFAULT_PML_THICKNESS: Final[int] = 10

#: Spatial FDTD order (2, 4 or 8).
DEFAULT_FDTD_ORDER: Final[int] = 2

#: Model-gradient mode (2: Ez-only 2D TM, 3: full 3-component).
DEFAULT_GRADIENT_MODE: Final[int] = 2

#: Electric-field component excited by the sources (0: x, 1: y, 2: z).
DEFAULT_SOURCE_COMPONENT: Final[int] = 2

#: Electric-field component recorded by the receivers (0: x, 1: y, 2: z).
DEFAULT_RECEIVER_COMPONENT: Final[int] = 2

#: Temporal sampling interval of the forward history used for gradients.
DEFAULT_GRADIENT_SAMPLING_INTERVAL: Final[int] = 1

#: Saved-history compression mode.
DEFAULT_WAVEFIELD_COMPRESSION: Final[str] = "none"

#: CUDA FP16/BF16 conversion backend selector.
DEFAULT_CONVERSION_BACKEND: Final[str] = "auto"

#: CUDA INT8 tile-reduction backend selector.
DEFAULT_INT8_REDUCTION_BACKEND: Final[str] = "auto"

#: Number of FIR taps per ``fs / cutoff`` ratio used by :func:`apply_filter`.
FIR_TAPS_PER_PERIOD: Final[int] = 1

#: Default number of cycles for Morlet and sine-burst wavelets.
DEFAULT_WAVELET_CYCLES: Final[float] = 3.0

#: Default weights of :class:`DeepGPR.TVRegularization`.
DEFAULT_TV_WEIGHT_PERMITTIVITY: Final[float] = 1
DEFAULT_TV_WEIGHT_CONDUCTIVITY: Final[float] = 0.001
DEFAULT_TV_METHOD: Final[str] = "anisotropic"

#: Name of the package logger hierarchy root.
LOGGER_NAME: Final[str] = "DeepGPR"

#: Filename pattern of saved forward wavefields (``strftime`` time label).
FORWARD_WAVEFIELD_TIME_FORMAT: Final[str] = "%H-%M"
FORWARD_WAVEFIELD_PREFIX: Final[str] = "forward_wavefield"

__all__ = [
    "DEFAULT_CONVERSION_BACKEND",
    "DEFAULT_FDTD_ORDER",
    "DEFAULT_GRADIENT_MODE",
    "DEFAULT_GRADIENT_SAMPLING_INTERVAL",
    "DEFAULT_INT8_REDUCTION_BACKEND",
    "DEFAULT_PML_THICKNESS",
    "DEFAULT_RECEIVER_COMPONENT",
    "DEFAULT_SOURCE_COMPONENT",
    "DEFAULT_TV_METHOD",
    "DEFAULT_TV_WEIGHT_CONDUCTIVITY",
    "DEFAULT_TV_WEIGHT_PERMITTIVITY",
    "DEFAULT_WAVEFIELD_COMPRESSION",
    "DEFAULT_WAVELET_CYCLES",
    "FIR_TAPS_PER_PERIOD",
    "FORWARD_WAVEFIELD_PREFIX",
    "FORWARD_WAVEFIELD_TIME_FORMAT",
    "LOGGER_NAME",
]
