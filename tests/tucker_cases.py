"""Small 2D/3D FDTD cases for the Tucker integration tests and benchmarks.

Both are reduced versions of the repository examples:

* 2D: ``examples/2DFWImodel.npy`` (the 2D FWI notebook), every second cell,
  transmission geometry (sources on the first row, receivers on the last),
  ``sigma = 0.002 eps`` with the deep part mirrored as in the notebook.
* 3D: the OverThrust section of the 3D FWI notebook (``OverThrust.npy``),
  coarsened, extruded along x, with an air layer and surface antennas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"


@dataclass
class FdtdCase:
    """Geometry, true/initial models and solver keywords of one case."""

    name: str
    eps_true: torch.Tensor
    sigma_true: torch.Tensor
    eps_init: torch.Tensor
    sigma_init: torch.Tensor
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def compute(self, eps, sigma, **extra):
        import DeepGPR

        options = dict(self.kwargs)
        options.update(extra)
        return DeepGPR.compute(eps_r=eps, sigma=sigma, **options)


def _smooth(model: torch.Tensor, passes: int = 3) -> torch.Tensor:
    x = model[None, None]
    pool = F.avg_pool2d if model.dim() == 2 else F.avg_pool3d
    for _ in range(passes):
        x = pool(x, 5, stride=1, padding=2, count_include_pad=False)
    return x[0, 0]


def case_2d(device: Any, nt: int = 400, shots: int = 3, stride: int = 2) -> FdtdCase:
    """Reduced 2D FWI example (Ez-TM, CUDA 2D fast path when on GPU)."""
    eps = torch.tensor(np.load(EXAMPLES / "2DFWImodel.npy")[::stride, ::stride], dtype=torch.float32)
    sigma = eps * 0.002
    half = eps.shape[0] * 95 // 200
    sigma[half:, :] = torch.flip(sigma[half:, :], dims=[1])
    nx, ny = eps.shape
    dx, dt, freq = 0.05 * stride, 2.0e-10 * stride / 2, 2.0e8 / stride
    wavelet = __import__("DeepGPR").wavelet.ricker(freq, nt, dt, 1.0 / freq)
    shot_x = torch.linspace(nx * 0.1, nx * 0.9, shots).round().to(torch.int32)
    source_location = torch.zeros((shots, 1, 3), dtype=torch.int32)
    source_location[:, 0, 0] = shot_x
    receivers = torch.arange(2, nx - 2, 4, dtype=torch.int32)
    receiver_location = torch.zeros((shots, receivers.numel(), 3), dtype=torch.int32)
    receiver_location[:, :, 0] = receivers
    receiver_location[:, :, 1] = ny - 1
    eps_init = _smooth(eps)
    sigma_init = _smooth(sigma)
    kwargs = dict(
        device=device,
        dx=dx,
        dt=dt,
        source_amplitudes=wavelet.reshape(1, nt, 1).to(device),
        source_location=source_location.to(device),
        receiver_location=receiver_location.to(device),
        pmlthick=10,
        mode=2,
    )
    to = dict(device=device)
    return FdtdCase("2d_fwi_example", eps.to(**to), sigma.to(**to), eps_init.to(**to), sigma_init.to(**to), kwargs)


def case_3d(
    device: Any,
    nt: int = 220,
    shots: int = 2,
    extrude: int = 12,
    stride: int = 10,
    air: int = 3,
    lateral_dip: float = 0.0,
) -> FdtdCase:
    """Reduced 3D FWI example (OverThrust section, mode 3, Ey antennas).

    Args:
        lateral_dip: Optional depth shift (cells) varying linearly along x,
            which makes the extruded model genuinely 3D.
    """
    section = torch.tensor(np.load(EXAMPLES / "OverThrust.npy")[::stride, ::stride], dtype=torch.float32)
    depth, width = section.shape
    eps = section[None].repeat(extrude, 1, 1)
    if lateral_dip:
        rows = torch.arange(depth, dtype=torch.float32)
        for ix in range(extrude):
            shift = lateral_dip * (ix / max(extrude - 1, 1) - 0.5)
            source_rows = (rows - shift).clamp(0, depth - 1)
            low = source_rows.floor().long()
            high = (low + 1).clamp(max=depth - 1)
            weight = (source_rows - low.float())[:, None]
            eps[ix] = section[low] * (1 - weight) + section[high] * weight
    eps = torch.cat([torch.ones(extrude, air, width), eps], dim=1)
    sigma = eps * 0.0005
    sigma[:, :air, :] = 0.0
    nx, ny, nz = eps.shape
    dx, dt, freq = 0.1, 1.5e-10, 1.0e8
    wavelet = __import__("DeepGPR").wavelet.ricker(freq, nt, dt, 1.2 / freq)
    shot_z = torch.linspace(nz * 0.3, nz * 0.7, shots).round().to(torch.int32)
    source_location = torch.zeros((shots, 1, 3), dtype=torch.int32)
    source_location[:, 0, 0] = nx // 2
    source_location[:, 0, 1] = air
    source_location[:, 0, 2] = shot_z
    receivers = torch.arange(3, nz - 3, 4, dtype=torch.int32)
    receiver_location = torch.zeros((shots, receivers.numel(), 3), dtype=torch.int32)
    receiver_location[:, :, 0] = nx // 2 - 2
    receiver_location[:, :, 1] = air
    receiver_location[:, :, 2] = receivers
    eps_init = _smooth(eps)
    eps_init[:, :air] = 1.0
    sigma_init = _smooth(sigma)
    sigma_init[:, :air] = 0.0
    kwargs = dict(
        device=device,
        dx=dx,
        dt=dt,
        source_amplitudes=wavelet.reshape(1, nt, 1).to(device),
        source_location=source_location.to(device),
        receiver_location=receiver_location.to(device),
        pmlthick=5,
        mode=3,
        source_direction=1,
        receiver_component=1,
    )
    to = dict(device=device)
    return FdtdCase("3d_overthrust_example", eps.to(**to), sigma.to(**to), eps_init.to(**to), sigma_init.to(**to), kwargs)


def observed(case: FdtdCase) -> torch.Tensor:
    """Receiver data of the true model (no history)."""
    with torch.no_grad():
        return case.compute(case.eps_true, case.sigma_true, save_wavefield_history=False)[-1]


def l2_gradients(case: FdtdCase, data_obs: torch.Tensor, eps=None, sigma=None, **extra):
    """``0.5 ||d - d_obs||^2 / ||d_obs||^2`` and its eps/sigma gradients at a model."""
    eps = (case.eps_init if eps is None else eps).detach().clone().requires_grad_(True)
    sigma = (case.sigma_init if sigma is None else sigma).detach().clone().requires_grad_(True)
    outputs = case.compute(eps, sigma, **extra)
    data = outputs[-1]
    loss = 0.5 * (data - data_obs).square().sum() / data_obs.square().sum()
    loss.backward()
    return {
        "loss": float(loss),
        "data": data.detach(),
        "grad_eps": eps.grad.detach(),
        "grad_sigma": sigma.grad.detach(),
        "e_saved": outputs[0],
    }


__all__ = ["FdtdCase", "case_2d", "case_3d", "l2_gradients", "observed"]
