"""Model-error metrics shared by the representation and FWI experiments."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _gaussian_window(size: int, sigma: float, dims: int, device, dtype) -> torch.Tensor:
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    window = g
    for _ in range(dims - 1):
        window = window[..., None] * g
    return window


def ssim(estimate: torch.Tensor, reference: torch.Tensor, data_range: Optional[float] = None, size: int = 7, sigma: float = 1.5) -> float:
    """Mean SSIM (Wang et al. 2004) with a Gaussian window, 2D or 3D, float64."""
    x = estimate.detach().double()[None, None]
    y = reference.detach().double()[None, None]
    dims = reference.dim()
    if data_range is None:
        data_range = float(reference.max() - reference.min()) or 1.0
    window = _gaussian_window(size, sigma, dims, x.device, x.dtype)[None, None]
    conv = F.conv3d if dims == 3 else F.conv2d
    mu_x, mu_y = conv(x, window), conv(y, window)
    sxx = conv(x * x, window) - mu_x**2
    syy = conv(y * y, window) - mu_y**2
    sxy = conv(x * y, window) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    value = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return float(value.mean())


def model_errors(estimate: torch.Tensor, reference: torch.Tensor, region=None, log: bool = False) -> Dict[str, float]:
    """RMSE, relative L2 and SSIM on an optional region (``log``: compare log10 values)."""
    e, r = estimate.detach(), reference.detach()
    if log:
        e, r = torch.log10(e.clamp_min(1e-9)), torch.log10(r.clamp_min(1e-9))
    if region is not None:
        e, r = e[region], r[region]
    diff = (e - r).double()
    return dict(
        rmse=float(diff.square().mean().sqrt()),
        rel_l2=float(diff.norm() / r.double().norm().clamp_min(1e-30)),
        ssim=ssim(e, r),
    )


__all__ = ["model_errors", "ssim"]
