"""Regularisation terms for full-waveform inversion."""

from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn as nn

from ..config.constants import TV_ISOTROPIC_EPSILON
from ..config.defaults import (
    DEFAULT_TV_METHOD,
    DEFAULT_TV_WEIGHT_CONDUCTIVITY,
    DEFAULT_TV_WEIGHT_PERMITTIVITY,
)

_Number = Union[int, float]

#: Accepted ``method`` names and the variant they select.
TV_METHODS = {
    "anisotropic": "anisotropic",
    "isotropic": "isotropic",
    "iso": "isotropic",
    "legacy_isotropic": "legacy_isotropic",
}


class TVRegularization(nn.Module):
    """Total-variation regularisation of permittivity and conductivity.

    Variants (``method``):

    * ``"anisotropic"`` (default): ``sum |d_x m| + sum |d_y m| (+ sum |d_z m|)``.
    * ``"isotropic"``: the standard isotropic TV
      ``sum_cells sqrt(d_x m^2 + d_y m^2 (+ d_z m^2) + epsilon)`` with forward
      differences and a zero difference across the far boundary (Neumann).
    * ``"legacy_isotropic"``: the variant that DeepGPR <= 0.0.20 selected for
      every ``method`` other than ``"anisotropic"``:
      ``sqrt(Lx^2 + Ly^2 + Lz^2 + epsilon)`` where ``Lx = sum |d_x m|`` over
      the whole model. Use it only to reproduce earlier results.

    2D models may be given as ``(nx, ny)`` or ``(nx, ny, 1)``; 3D models as
    ``(nx, ny, nz)``; 4D input is treated as a batch of 2D models (differences
    along the last two axes).

    Args:
        weight_ep: Weight of the permittivity term (skipped if <= 0).
        weight_sigma: Weight of the conductivity term (skipped if <= 0).
        method: ``"anisotropic"``, ``"isotropic"`` (alias ``"iso"``) or
            ``"legacy_isotropic"``.
        epsilon: Smoothing constant inside the square root of the isotropic
            variants (makes the penalty differentiable at zero gradient).

    Raises:
        ValueError: For an unknown ``method``.
    """

    def __init__(
        self,
        weight_ep: _Number = DEFAULT_TV_WEIGHT_PERMITTIVITY,
        weight_sigma: _Number = DEFAULT_TV_WEIGHT_CONDUCTIVITY,
        method: str = DEFAULT_TV_METHOD,
        epsilon: float = TV_ISOTROPIC_EPSILON,
    ) -> None:
        super().__init__()
        key = method.lower() if isinstance(method, str) else method
        if key not in TV_METHODS:
            raise ValueError(f"Unknown TV method {method!r}; expected one of {sorted(TV_METHODS)}.")
        self.weight_ep = weight_ep
        self.weight_sigma = weight_sigma
        self.method = method
        self.variant = TV_METHODS[key]
        self.epsilon = epsilon

    @staticmethod
    def _spatial_axes(data: torch.Tensor) -> List[int]:
        """Axes along which differences are taken (after squeezing 2D models)."""
        if data.dim() == 2:
            return [0, 1]
        if data.dim() == 3:
            return [0, 1, 2]
        return [data.dim() - 2, data.dim() - 1]

    def _isotropic_tv(self, data: torch.Tensor) -> torch.Tensor:
        """Standard isotropic TV with forward differences (Neumann boundary)."""
        squared = torch.zeros_like(data)
        for axis in self._spatial_axes(data):
            size = data.shape[axis]
            if size < 2:
                continue
            difference = data.narrow(axis, 1, size - 1) - data.narrow(axis, 0, size - 1)
            padding = [0, 0] * data.dim()
            padding[2 * (data.dim() - 1 - axis) + 1] = 1  # pad the far end with 0
            squared = squared + torch.nn.functional.pad(difference, padding).square()
        return torch.sqrt(squared + self.epsilon).sum()

    def _compute_tv(self, data: torch.Tensor) -> torch.Tensor:
        """Total variation of one tensor (see the class docstring)."""
        if data.dim() == 3 and data.shape[-1] == 1:
            data = data.squeeze(-1)
        if data.dim() not in (2, 3, 4):
            return torch.tensor(0.0, device=data.device)

        if self.variant == "isotropic":
            return self._isotropic_tv(data)

        if data.dim() == 2:
            loss_x = torch.sum(torch.abs(data[1:, :] - data[:-1, :]))
            loss_y = torch.sum(torch.abs(data[:, 1:] - data[:, :-1]))
            loss_z = 0.0
        elif data.dim() == 3:
            loss_x = torch.sum(torch.abs(data[1:, :, :] - data[:-1, :, :]))
            loss_y = torch.sum(torch.abs(data[:, 1:, :] - data[:, :-1, :]))
            loss_z = torch.sum(torch.abs(data[:, :, 1:] - data[:, :, :-1]))
        else:
            loss_x = torch.sum(torch.abs(data[..., 1:, :] - data[..., :-1, :]))
            loss_y = torch.sum(torch.abs(data[..., :, 1:] - data[..., :, :-1]))
            loss_z = 0.0

        if self.variant == "anisotropic":
            return loss_x + loss_y + loss_z
        return (loss_x**2 + loss_y**2 + loss_z**2 + self.epsilon).sqrt()

    def forward(
        self, ep: Optional[torch.Tensor] = None, sigma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return the weighted TV penalty.

        Args:
            ep: Relative permittivity or ``None`` to skip.
            sigma: Conductivity or ``None`` to skip.

        Raises:
            ValueError: If both ``ep`` and ``sigma`` are ``None``.
        """
        if ep is None and sigma is None:
            raise ValueError("TVRegularization needs at least one of ep or sigma.")
        reference = ep if ep is not None else sigma
        assert reference is not None
        loss = torch.tensor(0.0, device=reference.device)
        if ep is not None and self.weight_ep > 0:
            loss += self.weight_ep * self._compute_tv(ep)
        if sigma is not None and self.weight_sigma > 0:
            loss += self.weight_sigma * self._compute_tv(sigma)
        return loss


__all__ = ["TVRegularization", "TV_METHODS"]
