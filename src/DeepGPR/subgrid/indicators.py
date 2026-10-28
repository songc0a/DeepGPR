r"""Adjoint-weighted (goal-oriented) error indicators for the main grid.

Dual-weighted residual idea (Becker and Rannacher, Acta Numerica 10, 1-102,
2001, doi:10.1017/S0962492901000010; for parameter identification Becker and
Vexler, Numer. Math. 96, 435-459, 2004, doi:10.1007/s00211-003-0482-9) applied
to the discrete Yee scheme: the change of the objective ``J`` when the main
grid's update is replaced by a more accurate one is, to first order,

.. math::  \Delta J \approx \sum_n \langle g_E^n, \rho_E^n\rangle + \langle g_H^n, \rho_H^n\rangle

where ``g^n`` is the exact discrete adjoint (cotangent) of a perturbation added
to the main E (H) update of step ``n`` and ``rho`` is the residual of the
computed solution in the more accurate scheme (leading truncation terms)::

    rho_E = cb (curl4 - curl2) H^(n+1/2) + (1/24) (E^(n+2) - 3 E^(n+1) + 3 E^n - E^(n-1))
    rho_H = -(dt/mu) (curl4 - curl2) E^n + (1/24) (H^(n+3/2) - 3 H^(n+1/2) + 3 H^(n-1/2) - H^(n-3/2))

``curl4`` uses the fourth-order staggered difference ``(9/8, -1/24)``; the
second term is the leapfrog's ``dt^3/24 d^3/dt^3`` truncation. The cell-wise
contributions, accumulated with sign over time and taken in absolute value per
main node, are the indicator ``eta``; ``sum(eta_signed)`` estimates ``J`` on a
grid without the main grid's leading truncation error minus ``J`` on the main
grid (the effectivity is checked against a uniformly refined grid in
``tests/test_subgrid_indicators.py``). The cotangents ``g`` come from the
reverse sweep of :mod:`.adjoint` (``SubgridPropagator(..., indicator=True)``).

Nodes in the main CPML (and two cells around it) and inside the outer surfaces
of existing subgrids (where the main grid carries only the incident field) are
excluded. Marking: :func:`doerfler_mark` (Doerfler, SIAM J. Numer. Anal. 33(3),
1106-1124, 1996, doi:10.1137/0733054) selects the smallest set of nodes whose
indicators sum to a fraction ``theta`` of the total.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from ..config.constants import VACUUM_PERMEABILITY
from . import hsg

TERMS = hsg.TERMS


def _shift(f: torch.Tensor, axis: int, k: int) -> torch.Tensor:
    """``f[i + k]`` along spatial ``axis`` (zero outside)."""
    dim = axis + 1
    n = f.shape[dim]
    out = torch.zeros_like(f)
    if k >= 0:
        out.narrow(dim, 0, n - k).copy_(f.narrow(dim, k, n - k))
    else:
        out.narrow(dim, -k, n + k).copy_(f.narrow(dim, 0, n + k))
    return out


def curl_correction(fields, spacing, kind: str) -> List[torch.Tensor]:
    """``(curl4 - curl2)`` of ``fields`` at the nodes of the other kind.

    ``kind="e"``: the E-update curl of H (backward differences), ``"h"``: the
    H-update curl of E (forward differences)."""
    out = [0.0, 0.0, 0.0]
    for c, a, f, sign in TERMS:
        x = fields[f]
        h = spacing[a]
        if kind == "e":     # d f[i] = f[i] - f[i-1] (staggered at i - 1/2 .. i + 1/2)
            d1 = x - _shift(x, a, -1)
            d3 = _shift(x, a, 1) - _shift(x, a, -2)
        else:               # d f[i] = f[i+1] - f[i]
            d1 = _shift(x, a, 1) - x
            d3 = _shift(x, a, 2) - _shift(x, a, -1)
        out[c] = out[c] + sign * (d1 / 8.0 - d3 / 24.0) / h
    return out


def third_difference(f_m1, f0, f1, f2):
    return f2 - 3.0 * f1 + 3.0 * f0 - f_m1


class DWRIndicator:
    """Accumulates the adjoint-weighted residual of the main grid during a reverse sweep."""

    def __init__(self, solver, margin: int = 2):
        st = solver.static
        y = st.main
        self.spacing = y.spacing
        self.dt = solver.dt
        self.pml = solver.pml
        shape = y.shape
        mask = torch.zeros(shape, dtype=torch.bool, device=st.device)
        sl = tuple(slice(self.pml[2 * a] + margin, shape[a] - 1 - self.pml[2 * a + 1] - margin)
                   for a in range(3))
        mask[sl] = True
        for sg in st.subs:
            box = tuple(slice(max(lo - margin, 0), hi + margin + 1)
                        for lo, hi in zip(sg.os_lo, sg.os_hi))
            mask[box] = False
        self.mask = mask.to(st.dtype)
        self.eta_space = torch.zeros(shape, dtype=torch.float64, device=st.device)
        self.eta_time = torch.zeros_like(self.eta_space)
        self.chr = self.dt / VACUUM_PERMEABILITY
        self.steps = 0

    def accumulate(self, cb, gE, gH, E_n, H_half, E_next=None, H_prev=None, time_terms=None):
        """Add one step: cotangents ``gE``/``gH`` (3 tensors ``(S, NX, NY, NZ)``) of the
        perturbations of the main E/H updates, the fields ``E^n``, ``H^(n+1/2)`` and, for
        the temporal part, ``time_terms = (dE3, dH3)`` third differences (or ``None``)."""
        rE = curl_correction(H_half, self.spacing, "e")
        rH = curl_correction(E_n, self.spacing, "h")
        sp = 0.0
        for c in range(3):
            if gE[c] is not None:
                sp = sp + (gE[c] * (cb * rE[c])).sum(0)
            if gH[c] is not None:
                sp = sp - (gH[c] * (self.chr * rH[c])).sum(0)
        self.eta_space += (sp * self.mask).double()
        if time_terms is not None:
            dE3, dH3 = time_terms
            tm = 0.0
            for c in range(3):
                if gE[c] is not None and dE3 is not None:
                    tm = tm + (gE[c] * dE3[c]).sum(0) / 24.0
                if gH[c] is not None and dH3 is not None:
                    tm = tm + (gH[c] * dH3[c]).sum(0) / 24.0
            self.eta_time += (tm * self.mask).double()
        self.steps += 1

    def signed(self) -> torch.Tensor:
        return self.eta_space + self.eta_time

    def eta(self) -> np.ndarray:
        """``|eta|`` per main node of the model interior (CPML stripped)."""
        e = self.signed().abs()
        sl = tuple(slice(self.pml[2 * a], e.shape[a] - 1 - self.pml[2 * a + 1]) for a in range(3))
        return e[sl].cpu().numpy()

    def estimate(self) -> float:
        """``sum(eta_signed)``: estimated ``J(accurate) - J(main grid)``."""
        return float(self.signed().sum())


def doerfler_mark(eta: np.ndarray, theta: float = 0.5) -> np.ndarray:
    """Smallest set of nodes whose ``eta`` sum to ``theta`` of the total (bool mask)."""
    flat = np.asarray(eta, dtype=np.float64).reshape(-1)
    total = flat.sum()
    out = np.zeros(flat.shape, dtype=bool)
    if total <= 0:
        return out.reshape(np.shape(eta))
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    k = int(np.searchsorted(csum, theta * total)) + 1
    out[order[:k]] = True
    return out.reshape(np.shape(eta))


__all__ = ["DWRIndicator", "curl_correction", "doerfler_mark", "third_difference"]
