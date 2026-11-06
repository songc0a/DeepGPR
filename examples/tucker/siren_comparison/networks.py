"""External SIREN and Tucker-SIREN material parameterizations (not part of DeepGPR).

All parameterizations produce ``(eps_r, sigma)`` on the full grid through the
same property mapping as the existing SIREN-IFWI experiments
(``~/sirenifwi3d``, ``MultiScaleSiren3D.get_params``)::

    eps_r   = eps_min + (eps_max - eps_min) * sigmoid(t_eps)
    log10 s = log_min + (log_max - log_min) * sigmoid(t_sigma),  sigma = 10**log10 s

and differ only in how the unconstrained fields ``t_eps``, ``t_sigma`` are
generated:

* :class:`SirenField`       - one SIREN ``(x, y[, z]) -> (t_eps, t_sigma)`` evaluated at every cell.
* :class:`TuckerSirenField` - per property, one 1D SIREN per axis ``x -> U_x[x, :]`` (Tucker
  factor rows) and a trainable core; ``t = G x_1 U_x x_2 U_y x_3 U_z`` via
  :func:`DeepGPR.tucker.tucker_reconstruct` (no re-implementation).
* :class:`DirectTuckerField` - per property, a :class:`DeepGPR.tucker.TuckerTensor` (core and
  factors optimised directly).
* :class:`DenseField`       - per property, a voxel grid.

The SIREN layers follow Sitzmann et al. (2020): ``sin(omega_0 (W x + b))`` with
``W ~ U(-1/n, 1/n)`` in the first layer and ``U(-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0)``
afterwards (as ``SineLayer`` in the existing notebooks).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from DeepGPR.tucker import TuckerTensor, hosvd, tucker_reconstruct


# ---------------------------------------------------------------------------
# Property mapping (identical for all methods)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropertyMapping:
    """Sigmoid maps of the existing SIREN-IFWI code (eps linear, sigma log10)."""

    eps_min: float = 1.0
    eps_max: float = 10.0
    sigma_min: float = 0.0
    sigma_max: float = 5e-3

    @property
    def log_min(self) -> float:
        return math.log10(self.sigma_min + 1e-9)

    @property
    def log_max(self) -> float:
        return math.log10(self.sigma_max)

    def eps(self, t: torch.Tensor) -> torch.Tensor:
        return self.eps_min + (self.eps_max - self.eps_min) * torch.sigmoid(t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.pow(10.0, self.log_min + (self.log_max - self.log_min) * torch.sigmoid(t))

    def eps_inverse(self, eps: torch.Tensor, margin: float = 1e-4) -> torch.Tensor:
        p = ((eps - self.eps_min) / (self.eps_max - self.eps_min)).clamp(margin, 1 - margin)
        return torch.log(p) - torch.log1p(-p)

    def sigma_inverse(self, sigma: torch.Tensor, margin: float = 1e-4) -> torch.Tensor:
        log_sigma = torch.log10(sigma.clamp_min(0) + 1e-9)
        p = ((log_sigma - self.log_min) / (self.log_max - self.log_min)).clamp(margin, 1 - margin)
        return torch.log(p) - torch.log1p(-p)


# ---------------------------------------------------------------------------
# SIREN
# ---------------------------------------------------------------------------


class SineLayer(nn.Module):
    """``sin(omega_0 * linear(x))`` with the SIREN initialisation."""

    def __init__(self, in_features: int, out_features: int, is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            bound = 1.0 / in_features if is_first else math.sqrt(6.0 / in_features) / omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(nn.Module):
    """``layers`` sine layers of width ``hidden`` and a linear output layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: int = 128,
        layers: int = 4,
        omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        self.in_features, self.out_features, self.hidden, self.layers = in_features, out_features, hidden, layers
        blocks: List[nn.Module] = [SineLayer(in_features, hidden, is_first=True, omega_0=omega_0)]
        blocks += [SineLayer(hidden, hidden, omega_0=hidden_omega_0) for _ in range(layers - 1)]
        final = nn.Linear(hidden, out_features)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden) / hidden_omega_0
            final.weight.uniform_(-bound, bound)
        blocks.append(final)
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def siren_parameter_count(in_features: int, out_features: int, hidden: int, layers: int) -> int:
    first = in_features * hidden + hidden
    middle = (layers - 1) * (hidden * hidden + hidden)
    last = hidden * out_features + out_features
    return first + middle + last


def coordinate_grid(shape: Sequence[int], device=None) -> torch.Tensor:
    """``(prod(shape), len(shape))`` coordinates in [-1, 1] (``indexing='ij'``)."""
    axes = [torch.linspace(-1, 1, n, device=device) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, len(shape))


# ---------------------------------------------------------------------------
# Material fields
# ---------------------------------------------------------------------------


class MaterialField(nn.Module):
    """Base: ``unconstrained() -> (t_eps, t_sigma)``; ``forward() -> (eps_r, sigma)``."""

    mapping: PropertyMapping

    def unconstrained(self) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        t_eps, t_sigma = self.unconstrained()
        return self.mapping.eps(t_eps), self.mapping.sigma(t_sigma)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> Dict[str, int]:
        return {"total": self.trainable_parameters()}


class SirenField(MaterialField):
    """Standard SIREN over all grid coordinates.

    ``chunk`` > 0 evaluates the grid in chunks under ``torch.utils.checkpoint``
    (the activations of a chunk are recomputed in backward), which keeps the
    gradient exact while bounding activation memory; ``chunk = 0`` evaluates
    all coordinates at once.
    """

    def __init__(self, shape, mapping, hidden=128, layers=4, omega_0=30.0, chunk=0, device=None):
        super().__init__()
        self.shape, self.mapping, self.chunk = tuple(shape), mapping, int(chunk)
        self.net = Siren(len(shape), 2, hidden, layers, omega_0, omega_0)
        self.register_buffer("coords", coordinate_grid(shape, device), persistent=False)

    def _evaluate(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)

    def logits(self) -> torch.Tensor:
        if self.chunk <= 0 or self.chunk >= self.coords.shape[0]:
            return self._evaluate(self.coords)
        # Reentrant checkpoint (as in the existing SIREN-IFWI notebooks): each
        # chunk runs without saving activations and is recomputed and
        # back-propagated on its own in backward, so only one chunk's
        # activations exist at a time. The coordinates require grad only
        # so that the reentrant checkpoint records the chunk in the graph;
        # their gradient is discarded. (torch 1.11's non-reentrant variant
        # kept the recomputed activations of all chunks alive.)
        pieces = []
        for start in range(0, self.coords.shape[0], self.chunk):
            chunk = self.coords[start : start + self.chunk].detach().requires_grad_(True)
            pieces.append(checkpoint(self._evaluate, chunk))
        return torch.cat(pieces, dim=0)

    def unconstrained(self):
        out = self.logits()
        return out[:, 0].reshape(self.shape), out[:, 1].reshape(self.shape)

    def parameter_breakdown(self):
        return {"siren": self.trainable_parameters(), "total": self.trainable_parameters()}


class TuckerFactorNet(nn.Module):
    """1D SIREN ``x -> U[x, :]`` producing an ``(n, rank)`` factor matrix."""

    def __init__(self, n, rank, hidden=32, layers=2, omega_0=30.0, device=None):
        super().__init__()
        self.net = Siren(1, rank, hidden, layers, omega_0, omega_0)
        self.register_buffer("x", torch.linspace(-1, 1, n, device=device)[:, None], persistent=False)

    def forward(self) -> torch.Tensor:  # type: ignore[override]
        return self.net(self.x)


class TuckerSirenProperty(nn.Module):
    """One property: three (or two) factor SIRENs and a trainable core."""

    def __init__(self, shape, ranks, hidden=32, layers=2, omega_0=30.0, device=None):
        super().__init__()
        self.shape, self.ranks = tuple(shape), tuple(ranks)
        self.factor_nets = nn.ModuleList(
            [TuckerFactorNet(n, r, hidden, layers, omega_0, device) for n, r in zip(shape, ranks)]
        )
        self.core = nn.Parameter(torch.zeros(ranks, device=device))

    def factors(self) -> List[torch.Tensor]:
        return [net() for net in self.factor_nets]

    def forward(self) -> torch.Tensor:  # type: ignore[override]
        # DeepGPR's differentiable reconstruction (gradients reach the core
        # and every factor network).
        return tucker_reconstruct(self.core, self.factors())


class TuckerSirenField(MaterialField):
    """Independent Tucker-SIRENs for eps and sigma (six 1D SIRENs + two cores in 3D)."""

    def __init__(self, shape, mapping, ranks, hidden=32, layers=2, omega_0=30.0, sigma_ranks=None, device=None):
        super().__init__()
        self.shape, self.mapping = tuple(shape), mapping
        ranks = (ranks,) * len(shape) if isinstance(ranks, int) else tuple(ranks)
        sigma_ranks = ranks if sigma_ranks is None else ((sigma_ranks,) * len(shape) if isinstance(sigma_ranks, int) else tuple(sigma_ranks))
        self.eps_field = TuckerSirenProperty(shape, ranks, hidden, layers, omega_0, device)
        self.sigma_field = TuckerSirenProperty(shape, sigma_ranks, hidden, layers, omega_0, device)

    def unconstrained(self):
        return self.eps_field(), self.sigma_field()

    def factor_outputs(self):
        return self.eps_field.factors(), self.sigma_field.factors()

    def parameter_breakdown(self):
        nets = sum(p.numel() for f in (self.eps_field, self.sigma_field) for p in f.factor_nets.parameters())
        cores = self.eps_field.core.numel() + self.sigma_field.core.numel()
        return {"factor_networks": nets, "cores": cores, "total": nets + cores}


class DirectTuckerField(MaterialField):
    """DeepGPR ``TuckerTensor`` (core + factors optimised directly) per property."""

    def __init__(self, t_eps0: torch.Tensor, t_sigma0: torch.Tensor, mapping, ranks, sigma_ranks=None):
        super().__init__()
        self.mapping = mapping
        self.eps_tucker = TuckerTensor.from_tensor(t_eps0, ranks)
        self.sigma_tucker = TuckerTensor.from_tensor(t_sigma0, sigma_ranks or ranks)

    def unconstrained(self):
        return self.eps_tucker(), self.sigma_tucker()

    def parameter_breakdown(self):
        cores = self.eps_tucker.core_param.numel() + self.sigma_tucker.core_param.numel()
        total = self.trainable_parameters()
        return {"cores": cores, "factors": total - cores, "total": total}


class DenseField(MaterialField):
    """Voxel grids of the unconstrained values."""

    def __init__(self, t_eps0: torch.Tensor, t_sigma0: torch.Tensor, mapping):
        super().__init__()
        self.mapping = mapping
        self.t_eps = nn.Parameter(t_eps0.clone())
        self.t_sigma = nn.Parameter(t_sigma0.clone())

    def unconstrained(self):
        return self.t_eps, self.t_sigma

    def parameter_breakdown(self):
        return {"voxels": self.trainable_parameters(), "total": self.trainable_parameters()}


class FixedTopField(MaterialField):
    """A field over the subsurface plus a fixed (known) air layer on top (axis 1)."""

    def __init__(self, field: MaterialField, eps_top: torch.Tensor, sigma_top: torch.Tensor):
        super().__init__()
        self.field = field
        self.mapping = field.mapping
        self.register_buffer("eps_top", eps_top.clone())
        self.register_buffer("sigma_top", sigma_top.clone())

    def forward(self):  # type: ignore[override]
        eps, sigma = self.field()
        return torch.cat([self.eps_top, eps], dim=1), torch.cat([self.sigma_top, sigma], dim=1)

    def parameter_breakdown(self):
        return self.field.parameter_breakdown()


def split_parameters(field: MaterialField):
    """``(eps parameters, sigma parameters, shared parameters)`` of a field."""
    inner = field.field if isinstance(field, FixedTopField) else field
    if isinstance(inner, DenseField):
        return [inner.t_eps], [inner.t_sigma], []
    if isinstance(inner, DirectTuckerField):
        return list(inner.eps_tucker.parameters()), list(inner.sigma_tucker.parameters()), []
    if isinstance(inner, TuckerSirenField):
        return list(inner.eps_field.parameters()), list(inner.sigma_field.parameters()), []
    return [], [], list(inner.parameters())  # one network for both properties


def gaussian_smooth(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian filter, edge-replicating (as skimage.filters.gaussian, mode 'nearest')."""
    radius = int(4.0 * sigma + 0.5)
    t = torch.arange(-radius, radius + 1, dtype=torch.float64, device=x.device)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = (kernel / kernel.sum()).to(x.dtype)
    y = x
    for axis in range(x.dim()):
        moved = torch.movedim(y, axis, -1)
        shape = moved.shape
        flat = moved.reshape(-1, 1, shape[-1])
        flat = torch.nn.functional.pad(flat, (radius, radius), mode="replicate")
        flat = torch.nn.functional.conv1d(flat, kernel.view(1, 1, -1))
        y = torch.movedim(flat.reshape(shape), -1, axis)
    return y


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def tucker_targets(t: torch.Tensor, ranks) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """HOSVD of ``t`` in the balanced gauge: unit-RMS factor columns, core in field units."""
    core, factors = hosvd(t.double(), ranks)
    scaled = [f * math.sqrt(f.shape[0]) for f in factors]
    for mode, f in enumerate(factors):
        shape = [1] * core.dim()
        shape[mode] = -1
        core = core / math.sqrt(f.shape[0])
    return core.float(), [f.float() for f in scaled]


def prefit_tucker_siren(prop: TuckerSirenProperty, target: torch.Tensor, steps: int = 1500, lr: float = 1e-3, joint_steps: int = 500) -> Dict[str, float]:
    """Stage 1: fit each factor SIREN to the balanced HOSVD factors (core fixed);
    stage 2: joint fit of the reconstruction to ``target``."""
    core, factors = tucker_targets(target, prop.ranks)
    with torch.no_grad():
        prop.core.copy_(core.to(prop.core.device))
    for net, f in zip(prop.factor_nets, factors):
        f = f.to(prop.core.device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = (net() - f).square().mean()
            loss.backward()
            opt.step()
    opt = torch.optim.Adam(prop.parameters(), lr=lr * 0.3)
    for _ in range(joint_steps):
        opt.zero_grad(set_to_none=True)
        loss = (prop() - target).square().mean()
        loss.backward()
        opt.step()
    return {"final_mse": float(loss)}


def prefit_field(field: MaterialField, t_eps: torch.Tensor, t_sigma: torch.Tensor, steps: int, lr: float, cosine: bool = True) -> List[float]:
    """Full-grid MSE fit of the unconstrained fields (generic, any parameterization).

    Adam with cosine decay to ``lr / 100``: for the 3D SIREN 4x128 this
    reaches 0.17 % relative eps error to the initial model in 3000 steps
    (a constant 1e-4 left 2 %, which changed the initial data loss 10x).
    """
    opt = torch.optim.Adam(field.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.01) if cosine else None
    history = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        a, b = field.unconstrained()
        loss = (a - t_eps).square().mean() + (b - t_sigma).square().mean()
        loss.backward()
        opt.step()
        if schedule is not None:
            schedule.step()
        if step % 50 == 0 or step == steps - 1:
            history.append(float(loss))
    return history


__all__ = [
    "DenseField",
    "DirectTuckerField",
    "FixedTopField",
    "MaterialField",
    "gaussian_smooth",
    "PropertyMapping",
    "SineLayer",
    "Siren",
    "SirenField",
    "TuckerFactorNet",
    "TuckerSirenField",
    "TuckerSirenProperty",
    "coordinate_grid",
    "prefit_field",
    "prefit_tucker_siren",
    "siren_parameter_count",
    "tucker_targets",
]
