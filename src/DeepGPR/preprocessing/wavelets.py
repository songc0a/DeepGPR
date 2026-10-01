"""Source wavelets for GPR modelling.

Every function returns a one-dimensional tensor of ``length`` samples taken at
``t_n = n * dt - peak_time`` and accepts optional ``dtype`` / ``device``.
Arithmetic is unchanged from the original ``DeepGPR.wavelet`` module.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch

from ..config.defaults import DEFAULT_WAVELET_CYCLES
from ..utils.validators import require_finite, require_positive_finite, require_positive_index

DeviceLike = Optional[Union[str, torch.device]]


def _time_axis(
    freq: float,
    length: int,
    dt: float,
    peak_time: float,
    dtype: Optional[torch.dtype],
    device: DeviceLike,
) -> Tuple[torch.Tensor, float]:
    """Build a validated time axis centred on the wavelet peak.

    Returns:
        ``(time, frequency)`` where ``time`` has ``length`` samples.
    """
    frequency = require_positive_finite("freq", freq)
    time_step = require_positive_finite("dt", dt)
    sample_count = require_positive_index("length", length)
    center_time = require_finite("peak_time", peak_time)
    if dtype is not None and not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype.")

    output_dtype = torch.get_default_dtype() if dtype is None else dtype
    time = torch.arange(sample_count, dtype=output_dtype, device=device)
    return time * time_step - center_time, frequency


def ricker(
    freq: float,
    length: int,
    dt: float,
    peak_time: float,
    dtype: Optional[torch.dtype] = None,
    device: DeviceLike = None,
) -> torch.Tensor:
    """Zero-phase Ricker (Mexican-hat) wavelet with unit peak amplitude.

    ``w(t) = (1 - 2 (pi f t)^2) exp(-(pi f t)^2)``

    Args:
        freq: Peak frequency [Hz].
        length: Number of samples.
        dt: Sample interval [s].
        peak_time: Time of the peak [s].
        dtype: Floating-point dtype (default: ``torch.get_default_dtype()``).
        device: Output device.
    """
    time, frequency = _time_axis(freq, length, dt, peak_time, dtype, device)
    phase_squared = (math.pi * frequency * time).square()
    return (1.0 - 2.0 * phase_squared) * torch.exp(-phase_squared)


def gaussian(
    freq: float,
    length: int,
    dt: float,
    peak_time: float,
    dtype: Optional[torch.dtype] = None,
    device: DeviceLike = None,
) -> torch.Tensor:
    """Unit-amplitude Gaussian pulse ``exp(-(pi f t)^2)``.

    Args:
        freq: Characteristic frequency [Hz].
        length: Number of samples.
        dt: Sample interval [s].
        peak_time: Time of the peak [s].
        dtype: Floating-point dtype.
        device: Output device.
    """
    time, frequency = _time_axis(freq, length, dt, peak_time, dtype, device)
    phase = math.pi * frequency * time
    return torch.exp(-phase.square())


def gaussian_derivative(
    freq: float,
    length: int,
    dt: float,
    peak_time: float,
    dtype: Optional[torch.dtype] = None,
    device: DeviceLike = None,
) -> torch.Tensor:
    """First-derivative Gaussian normalised to unit peak magnitude.

    Args:
        freq: Characteristic frequency [Hz].
        length: Number of samples.
        dt: Sample interval [s].
        peak_time: Time of the zero crossing [s].
        dtype: Floating-point dtype.
        device: Output device.
    """
    time, frequency = _time_axis(freq, length, dt, peak_time, dtype, device)
    phase = math.pi * frequency * time
    return -math.sqrt(2.0 * math.e) * phase * torch.exp(-phase.square())


def morlet(
    freq: float,
    length: int,
    dt: float,
    peak_time: float,
    cycles: float = DEFAULT_WAVELET_CYCLES,
    dtype: Optional[torch.dtype] = None,
    device: DeviceLike = None,
) -> torch.Tensor:
    """Gaussian-windowed cosine with unit centre amplitude.

    Args:
        freq: Carrier frequency [Hz].
        length: Number of samples.
        dt: Sample interval [s].
        peak_time: Centre time [s].
        cycles: Envelope width in carrier cycles.
        dtype: Floating-point dtype.
        device: Output device.
    """
    time, frequency = _time_axis(freq, length, dt, peak_time, dtype, device)
    cycle_count = require_positive_finite("cycles", cycles)
    carrier_phase = 2.0 * math.pi * frequency * time
    envelope_phase = math.pi * frequency * time / cycle_count
    return torch.cos(carrier_phase) * torch.exp(-envelope_phase.square())


def sine_burst(
    freq: float,
    length: int,
    dt: float,
    peak_time: float,
    cycles: float = DEFAULT_WAVELET_CYCLES,
    dtype: Optional[torch.dtype] = None,
    device: DeviceLike = None,
) -> torch.Tensor:
    """Finite ``cycles``-cycle cosine burst with a Hann envelope.

    Args:
        freq: Carrier frequency [Hz].
        length: Number of samples.
        dt: Sample interval [s].
        peak_time: Centre time [s].
        cycles: Burst length in carrier cycles.
        dtype: Floating-point dtype.
        device: Output device.
    """
    time, frequency = _time_axis(freq, length, dt, peak_time, dtype, device)
    cycle_count = require_positive_finite("cycles", cycles)
    duration = cycle_count / frequency
    inside = time.abs() <= 0.5 * duration
    window = 0.5 * (1.0 + torch.cos(2.0 * math.pi * time / duration))
    pulse = window * torch.cos(2.0 * math.pi * frequency * time)
    return torch.where(inside, pulse, torch.zeros_like(pulse))


__all__ = ["gaussian", "gaussian_derivative", "morlet", "ricker", "sine_burst"]
