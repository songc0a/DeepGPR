"""Source wavelets (public ``DeepGPR.wavelet`` namespace).

The implementation lives in :mod:`DeepGPR.preprocessing.wavelets`; this
module keeps the documented ``DeepGPR.wavelet.ricker(...)`` access path.
"""

from __future__ import annotations

from .preprocessing.wavelets import (
    DeviceLike,
    _time_axis,
    gaussian,
    gaussian_derivative,
    morlet,
    ricker,
    sine_burst,
)

__all__ = ["gaussian", "gaussian_derivative", "morlet", "ricker", "sine_burst"]

_COMPAT_PRIVATE = (DeviceLike, _time_axis)
