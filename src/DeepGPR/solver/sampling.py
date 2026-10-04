"""Temporal sampling interval of the forward history used for model gradients.

``model_gradient_sampling_interval = S`` stores every ``S``-th forward step and
weights each stored step by the length of its sampling block. ``S > 1`` is an
approximation of the exact discrete gradient. Its error stays small while the
history is sampled at least four times per period of the highest frequency
carried by the forward wavefield, which gives the empirical bound

``S <= floor(1 / (4 f_max dt))``,

where ``f_max`` is the highest frequency at which the source amplitude
spectrum still exceeds :data:`SOURCE_SPECTRUM_CUTOFF_FRACTION` of its peak
(about three times the peak frequency for a Ricker wavelet). The bound only
looks at the *forward* source spectrum: the adjoint source of a non-smooth
misfit such as L1 is broader band, and its sampling error is correspondingly
larger (see ``docs/API.md``).
"""

from __future__ import annotations

import math
from typing import Any, Tuple, Union

import torch

from ..config.constants import SOURCE_SPECTRUM_CUTOFF_FRACTION

#: String accepted by ``model_gradient_sampling_interval`` to select the bound.
AUTO_SAMPLING_INTERVAL = "auto"

#: Minimum zero-padding factor of the spectrum used to locate ``f_max``.
_SPECTRUM_OVERSAMPLING = 4


def _waveform_rows(source_amplitudes: Any) -> torch.Tensor:
    """Return the waveforms as a float64 CPU tensor of shape ``(n, nt)``."""
    waveforms = torch.as_tensor(source_amplitudes).detach()
    if waveforms.ndim == 3:
        if waveforms.shape[-1] != 1:
            raise ValueError("source_amplitudes must have shape (n, nt, 1), (n, nt) or (nt,).")
        waveforms = waveforms[..., 0]
    elif waveforms.ndim == 1:
        waveforms = waveforms.unsqueeze(0)
    elif waveforms.ndim != 2:
        raise ValueError("source_amplitudes must have shape (n, nt, 1), (n, nt) or (nt,).")
    if waveforms.shape[-1] < 2:
        raise ValueError("source_amplitudes needs at least two time samples.")
    return waveforms.to(device="cpu", dtype=torch.float64)


def _time_step(dt: Any) -> float:
    value = float(dt)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("dt must be a positive finite number.")
    return value


def source_max_frequency(
    source_amplitudes: Any,
    dt: Any,
    *,
    cutoff_fraction: float = SOURCE_SPECTRUM_CUTOFF_FRACTION,
) -> float:
    """Highest frequency at which a source spectrum exceeds a peak fraction.

    The amplitude spectrum of every waveform is computed with zero padding;
    ``f_max`` is the highest frequency where it is still at least
    ``cutoff_fraction`` times that waveform's own spectral peak (linearly
    interpolated between bins). Several waveforms return the largest value.

    Args:
        source_amplitudes: ``(n, nt, 1)``, ``(n, nt)`` or ``(nt,)`` waveforms.
        dt: Sample interval [s].
        cutoff_fraction: Relative amplitude threshold in ``(0, 1)``.

    Returns:
        ``f_max`` in Hz, or ``0.0`` if every waveform is identically zero.
    """
    if not 0.0 < float(cutoff_fraction) < 1.0:
        raise ValueError("cutoff_fraction must lie in (0, 1).")
    time_step = _time_step(dt)
    waveforms = _waveform_rows(source_amplitudes)
    if not bool(torch.isfinite(waveforms).all()):
        raise ValueError("source_amplitudes must be finite.")
    length = waveforms.shape[-1]
    fft_length = 1 << max(0, (_SPECTRUM_OVERSAMPLING * length - 1).bit_length())
    spectrum = torch.fft.rfft(waveforms, n=fft_length, dim=-1).abs()
    frequency_step = 1.0 / (fft_length * time_step)

    maximum_frequency = 0.0
    for row in spectrum:
        peak = float(row.max())
        if peak <= 0.0:
            continue
        threshold = float(cutoff_fraction) * peak
        above = torch.nonzero(row >= threshold).flatten()
        last = int(above[-1])
        position = float(last)
        if last + 1 < row.numel():
            upper, lower = float(row[last]), float(row[last + 1])
            if upper > lower:
                position += (upper - threshold) / (upper - lower)
        maximum_frequency = max(maximum_frequency, position * frequency_step)
    return maximum_frequency


def recommended_sampling_interval(
    source_amplitudes: Any,
    dt: Any,
    *,
    cutoff_fraction: float = SOURCE_SPECTRUM_CUTOFF_FRACTION,
) -> int:
    """Largest ``model_gradient_sampling_interval`` inside the empirical bound.

    Returns ``max(1, floor(1 / (4 f_max dt)))`` with ``f_max`` from
    :func:`source_max_frequency`; several waveforms give the most conservative
    (smallest) interval. ``model_gradient_sampling_interval="auto"`` uses this
    value. Segmented or checkpointed runs should evaluate it once on the
    *complete* waveform and pass the integer to every segment, because a short
    segment of a waveform has a different spectrum.

    Args:
        source_amplitudes: ``(n, nt, 1)``, ``(n, nt)`` or ``(nt,)`` waveforms.
        dt: Time step [s].
        cutoff_fraction: Spectrum threshold relative to the peak.

    Returns:
        A positive interval; 1 for an identically zero source.
    """
    time_step = _time_step(dt)
    frequency = source_max_frequency(source_amplitudes, time_step, cutoff_fraction=cutoff_fraction)
    if frequency <= 0.0:
        return 1
    return max(1, int(math.floor(1.0 / (4.0 * frequency * time_step))))


def resolve_sampling_interval(
    value: Union[int, str], source_amplitudes: Any, dt: Any
) -> Tuple[int, bool]:
    """Validate ``model_gradient_sampling_interval``.

    Args:
        value: A positive integer or ``"auto"``.
        source_amplitudes: Source waveforms used for ``"auto"``.
        dt: Time step [s].

    Returns:
        ``(interval, automatic)``.

    Raises:
        ValueError: For anything other than a positive ``int`` or ``"auto"``.
    """
    if isinstance(value, str):
        if value.lower() != AUTO_SAMPLING_INTERVAL:
            raise ValueError(
                "model_gradient_sampling_interval must be a positive integer or 'auto'."
            )
        return recommended_sampling_interval(source_amplitudes, dt), True
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("model_gradient_sampling_interval must be a positive integer or 'auto'.")
    return value, False


__all__ = [
    "AUTO_SAMPLING_INTERVAL",
    "recommended_sampling_interval",
    "resolve_sampling_interval",
    "source_max_frequency",
]
