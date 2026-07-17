from __future__ import annotations

import math
import operator
from typing import Optional, Union

import torch


DeviceLike = Optional[Union[str, torch.device]]

__all__ = [
    "ricker",
    "gaussian",
    "gaussian_derivative",
    "morlet",
    "sine_burst",
]


def _positive_scalar(name: str, value: float) -> float:
    """Return a finite positive scalar."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite positive scalar.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar.")
    return result


def _time_axis(
    freq: float,
    length: int,
    dt: float,
    peak_time: float,
    dtype: Optional[torch.dtype],
    device: DeviceLike,
) -> tuple[torch.Tensor, float]:
    """Build a validated time axis centered on the wavelet peak."""
    frequency = _positive_scalar("freq", freq)
    time_step = _positive_scalar("dt", dt)
    if isinstance(length, bool):
        raise TypeError("length must be a positive integer.")
    try:
        sample_count = operator.index(length)
    except TypeError as exc:
        raise TypeError("length must be a positive integer.") from exc
    if sample_count <= 0:
        raise ValueError("length must be a positive integer.")
    try:
        center_time = float(peak_time)
    except (TypeError, ValueError) as exc:
        raise TypeError("peak_time must be a finite scalar.") from exc
    if not math.isfinite(center_time):
        raise ValueError("peak_time must be a finite scalar.")
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
    """Return a zero-phase Ricker wavelet with unit peak amplitude."""
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
    """Return a unit-amplitude Gaussian pulse."""
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
    """Return a first-derivative Gaussian pulse normalized to unit magnitude."""
    time, frequency = _time_axis(freq, length, dt, peak_time, dtype, device)
    phase = math.pi * frequency * time
    return -math.sqrt(2.0 * math.e) * phase * torch.exp(-phase.square())


def morlet(
    freq: float,
    length: int,
    dt: float,
    peak_time: float,
    cycles: float = 3.0,
    dtype: Optional[torch.dtype] = None,
    device: DeviceLike = None,
) -> torch.Tensor:
    """Return a Gaussian-windowed cosine pulse with unit center amplitude."""
    time, frequency = _time_axis(freq, length, dt, peak_time, dtype, device)
    cycle_count = _positive_scalar("cycles", cycles)
    carrier_phase = 2.0 * math.pi * frequency * time
    envelope_phase = math.pi * frequency * time / cycle_count
    return torch.cos(carrier_phase) * torch.exp(-envelope_phase.square())


def sine_burst(
    freq: float,
    length: int,
    dt: float,
    peak_time: float,
    cycles: float = 3.0,
    dtype: Optional[torch.dtype] = None,
    device: DeviceLike = None,
) -> torch.Tensor:
    """Return a finite-cycle cosine burst with a Hann envelope."""
    time, frequency = _time_axis(freq, length, dt, peak_time, dtype, device)
    cycle_count = _positive_scalar("cycles", cycles)
    duration = cycle_count / frequency
    inside = time.abs() <= 0.5 * duration
    window = 0.5 * (1.0 + torch.cos(2.0 * math.pi * time / duration))
    pulse = window * torch.cos(2.0 * math.pi * frequency * time)
    return torch.where(inside, pulse, torch.zeros_like(pulse))
