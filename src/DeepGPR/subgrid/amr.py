r"""Adjoint-error-driven dynamic refinement for FWI (AMR-FWI).

The inversion variables stay on the model grid; only the solver's layout (the
subgrid boxes on the fixed main grid) changes between *cycles* of the
optimiser. Refinement is persistent: the boxes of the current layout stay
marked (the main grid's indicator is not defined inside them), new marks add or
enlarge boxes. After each cycle the main grid's adjoint-weighted residual
(:class:`~DeepGPR.subgrid.indicators.DWRIndicator`) is evaluated for the current
model and the current data residual, marked with Doerfler's rule inside the
admissible region, merged with the a priori criteria of
:func:`~DeepGPR.subgrid.adapt.plan_refinement` on the current model, and
clustered into new boxes. The optimiser's history is reset when the layout
changes (the discrete objective changes with it).

Two objectives are supported by :func:`indicator`:

* ``"total"``: ``J = 1/2 ||d(m) - d_obs||^2``; the indicator is that of ``d(m)``.
* ``"differential"``: ``J = 1/2 ||(d(m) - d(m_0)) - s_obs||^2`` with a known
  background ``m_0`` (FWI of the anomaly response, the setting of
  ``examples/subgrid/fwi3d_subgrid.py``); the indicator is
  ``eta(m) - eta(m_0)`` for the same adjoint source, i.e. the DWR estimate of
  the change of the differential objective. It vanishes at ``m = m_0`` and
  concentrates where refinement changes the predicted anomaly response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .adapt import RefinementPlan, plan_refinement
from .indicators import doerfler_mark


def indicator(prop, eps, sig, w, residual, *, eps0=None, sig0=None, weight=None) -> np.ndarray:
    """``|eta|`` per main interior node for the data residual ``residual`` (``dJ/dd``).

    ``prop`` must have been built with ``indicator=True``. With ``eps0``/``sig0``
    the differential indicator ``eta(m) - eta(m_0)`` is returned."""
    def signed(e, s):
        e = e.detach().clone().requires_grad_(True)
        d = prop(e, s.detach(), w)
        g = residual if weight is None else residual * weight
        d.backward(g)
        return prop.indicator_obj.signed().clone()

    eta = signed(eps, sig)
    if eps0 is not None:
        eta = eta - signed(eps0, sig0)
    ind = prop.indicator_obj
    e = eta.abs()
    sl = tuple(slice(ind.pml[2 * a], e.shape[a] - 1 - ind.pml[2 * a + 1]) for a in range(3))
    return e[sl].cpu().numpy()


@dataclass
class RegridResult:
    plan: RefinementPlan
    eta: np.ndarray
    marked: np.ndarray
    changed: bool
    info: dict = field(default_factory=dict)


def regrid(eps, sig, h: float, coarsen: int, eta: np.ndarray, *, allowed: Optional[np.ndarray],
           theta: float = 0.5, f_max: float, ppw: float = 10.0, keep_points=None,
           current: Optional[list] = None, use_model_criteria: bool = True,
           **plan_kw) -> RegridResult:
    """New subgrid layout from the indicator ``eta`` (main interior nodes)."""
    e = np.asarray(eta, dtype=np.float64)
    if allowed is not None:
        e = np.where(allowed, e, 0.0)
    marked = doerfler_mark(e, theta)
    # refinement persists: the main grid's indicator is not evaluated inside existing
    # subgrids (it carries only the incident field there), so their boxes stay marked
    dx = coarsen * h
    margin = int(plan_kw.get("margin", 1))
    for sg in current or []:
        # shrunk by the planner's padding, so that an unchanged box is reproduced
        box = np.asarray(sg.box, dtype=np.float64).reshape(3, 2)
        sl = tuple(slice(int(round(box[a, 0] / dx)) + margin,
                         int(round(box[a, 1] / dx)) - margin + 1) for a in range(3))
        marked[sl] = True
    if use_model_criteria:
        plan = plan_refinement(eps, sig, h, f_max, ppw=ppw, coarsen=coarsen, extra_flags=marked,
                               keep_points=keep_points, **plan_kw)
    else:
        z = np.zeros_like(eps)
        plan = plan_refinement(z + 1.0, z, h, f_max, ppw=1e-9, coarsen=coarsen,
                               extra_flags=marked, keep_points=keep_points, **plan_kw)
    if allowed is not None:
        # model criteria outside the admissible region are ignored
        keep = []
        for sg, cells in zip(plan.subgrids, plan.boxes_cells):
            sub = allowed[cells[0][0]:cells[0][1] + 1, cells[1][0]:cells[1][1] + 1,
                          cells[2][0]:cells[2][1] + 1]
            if sub.any():
                keep.append((sg, cells))
        plan.subgrids = [k[0] for k in keep]
        plan.boxes_cells = [k[1] for k in keep]
    boxes = sorted(tuple(map(tuple, np.round(np.asarray(sg.box), 6))) for sg in plan.subgrids)
    old = sorted(tuple(map(tuple, np.round(np.asarray(sg.box), 6))) for sg in (current or []))
    return RegridResult(plan, e, marked, boxes != old,
                        {"marked_nodes": int(marked.sum()), "eta_total": float(e.sum())})


__all__ = ["RegridResult", "indicator", "regrid"]
