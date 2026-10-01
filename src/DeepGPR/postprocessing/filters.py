"""Signal-processing helpers for receiver data (multi-scale FWI, envelopes).

Two defects of DeepGPR <= 0.0.20 are corrected here (docs/AUDIT.md P-17/P-18):

* ``design_fir_filter`` replaced a regular tap by the sinc peak value when the
  number of taps was even;
* ``hilbert_transform`` did not double the highest positive-frequency bin when
  the trace length was odd.

Pass ``legacy=True`` to reproduce results computed with DeepGPR <= 0.0.20.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..config.constants import HAMMING_ALPHA, HAMMING_BETA
from ..config.defaults import FIR_TAPS_PER_PERIOD
from ..utils.validators import require_positive_finite


def design_fir_filter(
    cutoff: float,
    fs: float,
    numtaps: int,
    *,
    device: Optional[Any] = None,
    dtype: torch.dtype = torch.float32,
    legacy: bool = False,
) -> torch.Tensor:
    """Design a Hamming-windowed sinc low-pass FIR filter with unit DC gain.

    The ideal impulse response ``sin(2 pi fc t) / (pi t)`` is sampled at
    ``t = n - (numtaps - 1) / 2``. Only for an odd number of taps does a sample
    fall on ``t = 0``, where the 0/0 is replaced by its limit ``2 fc / fs``.

    Args:
        cutoff: Cut-off frequency [Hz].
        fs: Sampling frequency [Hz].
        numtaps: Number of taps (>= 2).
        device: Output device.
        dtype: Output dtype.
        legacy: Reproduce DeepGPR <= 0.0.20, which also overwrote tap
            ``(numtaps - 1) // 2`` with ``2 fc / fs`` for an even number of
            taps (where no sample lies at ``t = 0``).

    Returns:
        1D tensor of ``numtaps`` coefficients normalised to sum to one.

    Raises:
        ValueError: If ``numtaps < 2``.
    """
    if numtaps < 2:
        raise ValueError(f"numtaps must be at least 2, got {numtaps}.")
    n = torch.arange(numtaps, dtype=dtype, device=device)
    window = HAMMING_ALPHA - HAMMING_BETA * torch.cos(2 * torch.pi * n / (numtaps - 1))
    sinc = torch.sin(2 * torch.pi * (cutoff / fs) * (n - (numtaps - 1) / 2)) / (
        torch.pi * (n - (numtaps - 1) / 2)
    )
    if legacy or numtaps % 2 == 1:
        center = (numtaps - 1) // 2
        sinc[center] = 2 * cutoff / fs
    h = window * sinc
    return h / h.sum()


def apply_filter(
    data: torch.Tensor, fs: float, cutoff: float, *, legacy: bool = False
) -> torch.Tensor:
    """Low-pass filter a trace or receiver gather along time.

    The filter has ``int(fs / cutoff)`` taps and is applied causally after
    reflect-padding ``numtaps - 1`` samples at the start.

    Args:
        data: 1D trace ``(nt,)`` or receiver data ``(nstep, nt, nrx)``.
        fs: Sampling frequency [Hz] (``1 / dt``).
        cutoff: Cut-off frequency [Hz].
        legacy: Use the DeepGPR <= 0.0.20 filter design (see
            :func:`design_fir_filter`).

    Returns:
        Filtered tensor with the shape of ``data``.

    Raises:
        ValueError: For invalid frequencies, too few taps, a filter longer
            than the trace, or unsupported dimensionality.
    """
    # Validate without replacing the caller's objects: fs/cutoff may be 0-dim
    # tensors, and converting them to Python floats would change rounding.
    fs_value = require_positive_finite("fs", fs)
    cutoff_value = require_positive_finite("cutoff", cutoff)
    numtaps = int(FIR_TAPS_PER_PERIOD * (fs / cutoff))
    if numtaps < 2:
        raise ValueError(
            f"cutoff={cutoff_value:g} Hz is too high for fs={fs_value:g} Hz: the FIR "
            f"filter would have {numtaps} tap(s); use cutoff <= fs / 2."
        )
    if data.ndim not in (1, 3):
        raise ValueError(f"Data dimension: {data.ndim}. Expected a 1D or 3D tensor.")
    nt = data.shape[-1] if data.ndim == 1 else data.shape[1]
    if numtaps - 1 >= nt:
        raise ValueError(
            f"The FIR filter needs {numtaps - 1} padding samples, but the trace has only "
            f"{nt} samples; increase cutoff or the trace length."
        )

    fir_coeff = design_fir_filter(
        cutoff, fs, numtaps, device=data.device, dtype=data.dtype, legacy=legacy
    )
    kernel = fir_coeff.view(1, 1, -1)

    if data.ndim == 1:
        padded = torch.nn.functional.pad(data.view(1, 1, -1), (numtaps - 1, 0), mode="reflect")
        return torch.nn.functional.conv1d(padded, kernel, padding=0).view(-1)

    step, iterations, nrx = data.shape
    reshaped = data.permute(0, 2, 1).reshape(-1, 1, iterations)
    padded = torch.nn.functional.pad(reshaped, (numtaps - 1, 0), mode="reflect")
    filtered = torch.nn.functional.conv1d(padded, kernel, padding=0)
    return filtered.view(step, nrx, iterations).permute(0, 2, 1)


def hilbert_transform(
    data_in: torch.Tensor, p: float = 1, *, remove_dc: bool = True, legacy: bool = False
) -> torch.Tensor:
    """Envelope ``|analytic(data)| ** p`` along the time axis (dim 1).

    The analytic signal keeps the DC bin, doubles every positive-frequency bin
    ``1 .. ceil(nt / 2) - 1``, keeps the Nyquist bin of an even-length trace
    and zeroes the negative frequencies (as ``scipy.signal.hilbert``).

    Args:
        data_in: Receiver data ``(ns, nt, nr)``.
        p: Power applied to the envelope.
        remove_dc: Zero the DC bin first, i.e. take the envelope of the
            zero-mean trace (the behaviour of all previous releases).
        legacy: Reproduce DeepGPR <= 0.0.20, which did not double bin
            ``nt // 2`` for odd ``nt``.

    Returns:
        Real tensor with the shape of ``data_in``.
    """
    if data_in.ndim != 3:
        raise ValueError(f"hilbert_transform expects (ns, nt, nr) data, got {data_in.ndim}D.")
    nt = data_in.shape[1]
    doubled_end = nt // 2 if legacy else (nt + 1) // 2
    transforms = torch.fft.fft(data_in, dim=1)
    transforms[:, 1:doubled_end, :] *= 2.0
    transforms[:, nt // 2 + 1 : nt, :] = 0 + 0j
    if remove_dc:
        transforms[:, 0, :] = 0
    return torch.abs(torch.fft.ifft(transforms, dim=1)) ** p


__all__ = ["apply_filter", "design_fir_filter", "hilbert_transform"]
