r"""Dynamic mesh and time-step scheduling for adaptive FWI.

The forward model of a cycle of the inversion is a *layout*: the Yee main grid
(spacing ``dx``), a body-fitted DGTD band on the surface (optional,
:class:`~DeepGPR.hybrid.HybridMaxwell3D`) and Huygens subgrids
(:class:`~DeepGPR.subgrid.Subgrid`). Each part has its own stability limit, and
the solvers already run each at its own rate:

* the Yee grid at the macro step ``dt`` (its CFL limit times ``lts_cfl``);
* the DG band at ``dt / p`` by the local time stepping
  (:func:`DeepGPR.hybrid.functional3d.lts_h_half_step`);
* a subgrid of ratio ``r`` at ``dt / r`` (its own leapfrog, Huygens coupling).

:class:`TimeStepLadder` collects this *time-step ladder* of a solver and the
unknown counts of its parts; :class:`CostModel` predicts the wall-clock cost of
a forward run or a gradient from them (per-unknown rates, calibrated once on
the machine at hand); :class:`Scheduler` decides between optimisation cycles
whether the layout should change and plans the new one from the adjoint-weighted
error indicator (:mod:`.indicators`) under the time budget:

* **error balancing**: the dual-weighted-residual estimate ``eta = J(accurate) -
  J(layout)`` of the modelling error is compared with the data misfit; the
  layout is refined only while the modelling error is a sizeable fraction of
  the residual (``balance``), and the optimiser is left alone while the data
  residual still falls quickly (``plateau`` test on the misfit history), unless
  the modelling error is most of the misfit (``urgent``: then the layout is
  refined at once, even before the first iteration);
* **cost control**: the fine-node budget of the next layout follows from the
  remaining wall time, the iterations still planned and the cost model; the
  marking threshold of the maximum strategy is raised until the planned boxes
  fit, and the boxes with the least indicator mass are dropped if they still
  do not;
* **hysteresis**: a re-plan that would barely change the layout is skipped
  (:func:`DeepGPR.subgrid.amr.regrid`), because every change restarts the
  quasi-Newton history.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .amr import RegridResult, regrid


# --------------------------------------------------------------------------- ladder
@dataclass
class TimeStepLadder:
    """Time steps and unknown counts of the parts of a layout.

    Attributes:
        dt: Macro (Yee main grid) time step [s].
        dg_substeps: Sub-steps ``p`` of the DG band per macro step (1 = none).
        yee_nodes: Yee grid nodes (each with up to six field unknowns).
        dg_nodes: DG nodes (elements times nodes per element; six fields each).
        sub_nodes: Fine nodes per subgrid.
        sub_time_ratios: Sub-steps per macro step of every subgrid.
        dg_dt_limit, yee_dt_limit: Stability limits of the parts, when known.
        dg_damping: DG-step equivalents of the split penalty damping per macro step
            (``lts_pen_k`` with a penalty flux under LTS, else 0).
    """

    dt: float
    dg_substeps: int = 1
    yee_nodes: int = 0
    dg_nodes: int = 0
    sub_nodes: List[int] = field(default_factory=list)
    sub_time_ratios: List[int] = field(default_factory=list)
    dg_dt_limit: Optional[float] = None
    yee_dt_limit: Optional[float] = None
    dg_damping: int = 0

    @property
    def dg_dt(self) -> float:
        return self.dt / max(self.dg_substeps, 1)

    def steps(self, duration: float) -> int:
        """Macro steps covering ``duration`` [s]."""
        return int(math.ceil(duration / self.dt))

    def rates(self) -> Dict[str, float]:
        """Steps per second of simulated time of every part (``1/dt`` of the part)."""
        out = {"yee": 1.0 / self.dt, "dg": self.dg_substeps / self.dt}
        for g, r in enumerate(self.sub_time_ratios):
            out[f"subgrid{g}"] = r / self.dt
        return out

    def describe(self) -> str:
        parts = [f"Yee {self.yee_nodes} nodes at dt = {self.dt:.3e} s"]
        if self.dg_nodes:
            parts.append(f"DG {self.dg_nodes} nodes at dt/{self.dg_substeps} = {self.dg_dt:.3e} s")
        for g, (n, r) in enumerate(zip(self.sub_nodes, self.sub_time_ratios)):
            parts.append(f"subgrid {g}: {n} fine nodes at dt/{r} = {self.dt / r:.3e} s")
        return "; ".join(parts)

    @classmethod
    def from_solver(cls, solver) -> "TimeStepLadder":
        """The ladder of a :class:`~DeepGPR.hybrid.HybridMaxwell3D`,
        :class:`~DeepGPR.hybrid.HybridTM2D`, :class:`~DeepGPR.subgrid.SubgridMaxwell3D`
        or :class:`~DeepGPR.subgrid.hybrid.HybridSubgridMaxwell3D`."""
        hyb = getattr(solver, "hyb", None)          # hybrid with subgrids
        base = hyb if hyb is not None else solver
        st = base.static
        yee = getattr(st, "yee", None) or getattr(st, "main", None)
        if yee is None:
            raise TypeError("The solver has no Yee part.")
        yee_nodes = int(np.prod(yee.shape))
        dg = getattr(st, "dg", None)
        dg_nodes = int(dg.K * dg.Np) if dg is not None else 0
        p = int(getattr(dg, "lts", 1)) if dg is not None else 1
        subs = list(getattr(solver.static, "subs", []))
        sub_nodes = [int(np.prod(sg.yee.shape)) for sg in subs]
        ratios = [int(sg.time_ratio) for sg in subs]
        dg_limit = None
        lam = getattr(base, "lambda_max", None)
        if lam:
            dg_limit = 2.0 / math.sqrt(lam)
        yee_limit = base.yee_dt_limit() if hasattr(base, "yee_dt_limit") else None
        damping = (int(getattr(dg, "lts_pen_k", 1)) if (dg is not None and p > 1
                                                        and getattr(dg, "lts_split", False)) else 0)
        return cls(float(solver.dt), p, yee_nodes, dg_nodes, sub_nodes, ratios, dg_limit,
                   yee_limit, damping)


# --------------------------------------------------------------------------- cost
@dataclass
class CostModel:
    """Wall-clock cost per macro step as a linear model in the unknown counts.

    ``seconds/step = yee * yee_nodes + dg * dg_nodes * (p + k)
    + sub * sum_g sub_nodes_g * r_g + overhead``: a DG step evaluates two curl
    operators; a macro step of the local time stepping evaluates ``2 p`` (``p - 1``
    z-updates and ``p`` H-type evaluations in the recursion, one E update), plus ``2 k``
    for the split penalty damping (``ladder.dg_damping``); a subgrid of time ratio
    ``r`` takes ``r`` fine steps per macro step. The defaults
    are RTX 4090 measurements of the fused kernels (`docs/HYBRID_DGTD.md` 11.4,
    `docs/SUBGRID.md` 2); :meth:`calibrate` rescales them to the machine and
    code path at hand. ``gradient_factor`` is the cost of a gradient (forward with
    checkpoints, recomputation and reverse sweep) relative to a forward run.
    """

    yee: float = 2.4e-10
    dg: float = 4.2e-10
    sub: float = 2.5e-10
    overhead: float = 1.0e-4
    gradient_factor: float = 5.0

    def seconds_per_step(self, ladder: TimeStepLadder) -> float:
        p = max(ladder.dg_substeps, 1)
        dg_evals = float(p + ladder.dg_damping)
        sub = sum(n * r for n, r in zip(ladder.sub_nodes, ladder.sub_time_ratios))
        return (self.yee * ladder.yee_nodes + self.dg * ladder.dg_nodes * dg_evals
                + self.sub * sub + self.overhead)

    def predict(self, ladder: TimeStepLadder, duration: float, *, shots: int = 1,
                gradient: bool = False) -> float:
        """Seconds of a forward run (or a gradient) over ``duration`` [s] of
        simulated time for ``shots`` shots."""
        cost = self.seconds_per_step(ladder) * ladder.steps(duration) * shots
        return cost * (self.gradient_factor if gradient else 1.0)

    def calibrate(self, solver, duration: float, measured_seconds: float,
                  shots: int = 1) -> float:
        """Rescale all rates so that the model reproduces ``measured_seconds`` of a
        forward run of ``solver`` over ``duration``; returns the scale factor."""
        ladder = TimeStepLadder.from_solver(solver)
        predicted = self.predict(ladder, duration, shots=shots)
        scale = measured_seconds / max(predicted, 1e-30)
        self.yee *= scale
        self.dg *= scale
        self.sub *= scale
        self.overhead *= scale
        return scale

    def measure(self, solver, source_amplitudes, source_positions, receiver_positions,
                shots: int = 1) -> float:
        """Time one forward run of ``solver`` and :meth:`calibrate` on it."""
        res = solver.run(source_amplitudes, source_positions, receiver_positions)
        nt = int(np.asarray(source_amplitudes).shape[-1])
        return self.calibrate(solver, nt * solver.dt, float(res["seconds"]), shots)

    def fine_node_budget(self, ladder: TimeStepLadder, duration: float, seconds_per_gradient: float,
                         *, shots: int = 1, time_ratio: int = 3) -> int:
        """Largest total number of subgrid fine nodes (time ratio ``time_ratio``) whose
        gradient over ``duration`` stays within ``seconds_per_gradient``, given the
        Yee and DG parts of ``ladder``."""
        base = TimeStepLadder(ladder.dt, ladder.dg_substeps, ladder.yee_nodes, ladder.dg_nodes,
                              dg_damping=ladder.dg_damping)
        fixed = self.predict(base, duration, shots=shots, gradient=True)
        per_node = (self.sub * time_ratio * base.steps(duration) * shots * self.gradient_factor)
        return int(max(0.0, (seconds_per_gradient - fixed) / max(per_node, 1e-300)))


def subgrid_fine_nodes(boxes_cells: Sequence[Sequence[Sequence[int]]], ratio: int = 3,
                       is_os_sep: int = 3, pml_cells: int = 6,
                       pml_sep: Optional[int] = None) -> List[int]:
    """Fine nodes of subgrids given their inner boxes in main cells ``((i0, i1), ...)``
    (the layout of :attr:`RefinementPlan.boxes_cells`)."""
    sep = ratio // 2 + 2 if pml_sep is None else int(pml_sep)
    nb = is_os_sep * ratio + sep + pml_cells          # fine cells beyond the inner surface
    out = []
    for cells in boxes_cells:
        n = 1
        for a, b in cells:
            n *= (int(b) - int(a)) * ratio + 2 * nb + 1
        out.append(int(n))
    return out


# --------------------------------------------------------------------------- scheduler
@dataclass
class Decision:
    action: str                  # "keep", "refine" or "coarsen"
    reason: str
    misfit: float
    estimate: float
    plateau: bool
    balance: float


class Scheduler:
    """Decides between optimisation cycles whether the layout changes, and plans it.

    Args:
        cost: :class:`CostModel` (calibrated).
        budget_seconds: Wall-clock budget of the whole inversion.
        iterations: Iterations planned in total (sets the affordable cost per gradient
            together with the remaining time).
        balance: Refine while ``|eta| >= balance * misfit`` (the modelling error is a
            sizeable part of the residual); coarsen when ``|eta| < coarsen_factor *
            balance * misfit`` and boxes exist.
        plateau: Relative misfit decrease over ``window`` iterations below which the
            optimiser is considered stalled on the current layout.
        window, min_cycle: Iterations of the plateau test (counted on the current
            layout only) and the least number of iterations between two layout
            changes (a quasi-Newton restart needs several iterations before its
            misfit history says anything; default 10).
        coarsen_factor: See ``balance``.
        end_fraction: No layout change once less than this fraction of the budget
            remains (a change restarts the optimiser, which the remaining time
            cannot amortise).
        urgent: Refine without waiting for a plateau or ``min_cycle`` iterations when
            ``|eta| >= urgent * misfit``: the layout's own modelling error is most of
            the residual, so iterating on it would mostly fit that error (a check
            before the first iteration then refines at once). ``None`` (default)
            disables it; 0.5 is a sensible value.
    """

    def __init__(self, cost: CostModel, budget_seconds: float, iterations: int, *,
                 balance: float = 0.25, plateau: float = 0.02, window: int = 5,
                 min_cycle: int = 10, coarsen_factor: float = 0.1, end_fraction: float = 0.1,
                 urgent: Optional[float] = None):
        self.cost = cost
        self.budget = float(budget_seconds)
        self.iterations = int(iterations)
        self.balance = float(balance)
        self.plateau = float(plateau)
        self.window = int(window)
        self.min_cycle = int(min_cycle)
        self.coarsen_factor = float(coarsen_factor)
        self.end_fraction = float(end_fraction)
        self.urgent = None if urgent is None else float(urgent)
        self.t0 = time.perf_counter()
        self.misfits: List[float] = []
        self.last_change = 0
        self.log: List[dict] = []

    # ---------------------------------------------------------------- bookkeeping
    def record(self, misfit: float) -> None:
        self.misfits.append(float(misfit))

    @property
    def iteration(self) -> int:
        return len(self.misfits)

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def remaining_seconds(self) -> float:
        return max(0.0, self.budget - self.elapsed())

    def remaining_iterations(self) -> int:
        return max(1, self.iterations - self.iteration)

    def seconds_per_gradient(self, evaluations_per_iteration: float = 1.5) -> float:
        """Affordable seconds per gradient for the iterations still planned."""
        return self.remaining_seconds() / (self.remaining_iterations() * evaluations_per_iteration)

    def is_plateau(self) -> bool:
        """Relative misfit decrease over the last ``window`` iterations below ``plateau``;
        only iterations on the current layout count (a layout change alters the
        discrete objective, so the history before it is not comparable)."""
        m = self.misfits[self.last_change:]
        if len(m) < self.window + 1:
            return False
        a, b = m[-self.window - 1], m[-1]
        return (a - b) / max(abs(a), 1e-300) < self.plateau

    # ---------------------------------------------------------------- decision
    def decide(self, estimate: float, has_boxes: bool, force: bool = False) -> Decision:
        """``estimate``: the dual-weighted-residual estimate of the modelling error of
        the current layout (same units as the misfit); ``has_boxes``: whether the
        layout holds subgrids."""
        misfit = self.misfits[-1] if self.misfits else float("inf")
        plateau = self.is_plateau()
        ratio = abs(float(estimate)) / max(misfit, 1e-300)
        since = self.iteration - self.last_change
        urgent = self.urgent is not None and ratio >= max(self.urgent, self.balance)
        if not force and self.remaining_seconds() < self.end_fraction * self.budget:
            d = Decision("keep", "the budget is nearly spent", misfit, estimate, plateau, ratio)
        elif not force and not urgent and since < self.min_cycle:
            d = Decision("keep", f"only {since} iteration(s) since the last change", misfit,
                         estimate, plateau, ratio)
        elif not force and not urgent and not plateau:
            d = Decision("keep", "the data residual still decreases on this layout", misfit,
                         estimate, plateau, ratio)
        elif urgent and not force and (not plateau or since < self.min_cycle):
            d = Decision("refine", f"modelling error {ratio:.2f} of the misfit >= {self.urgent} "
                         "(urgent: the layout's own error dominates the residual)",
                         misfit, estimate, plateau, ratio)
        elif ratio >= self.balance:
            d = Decision("refine", f"modelling error {ratio:.2f} of the misfit >= {self.balance}",
                         misfit, estimate, plateau, ratio)
        elif has_boxes and ratio < self.coarsen_factor * self.balance:
            d = Decision("coarsen", f"modelling error {ratio:.3f} of the misfit is negligible",
                         misfit, estimate, plateau, ratio)
        else:
            d = Decision("keep", f"modelling error {ratio:.2f} of the misfit is balanced",
                         misfit, estimate, plateau, ratio)
        self.log.append(dict(iteration=self.iteration, **d.__dict__))
        return d

    # ---------------------------------------------------------------- planning
    def plan(self, eps, sig, h: float, coarsen: int, eta: np.ndarray, *, ladder: TimeStepLadder,
             duration: float, shots: int = 1, allowed: Optional[np.ndarray] = None,
             current: Optional[list] = None, f_max: float, ratio: int = 3, gamma: float = 0.2,
             hysteresis: Optional[float] = 0.9, max_fine_nodes: Optional[int] = None,
             evaluations_per_iteration: float = 1.5, **regrid_kw) -> RegridResult:
        """New layout from the indicator under the fine-node budget of the remaining
        time (see the module docstring). ``max_fine_nodes`` overrides the budget."""
        if max_fine_nodes is None:
            max_fine_nodes = self.cost.fine_node_budget(
                ladder, duration, self.seconds_per_gradient(evaluations_per_iteration),
                shots=shots, time_ratio=ratio)
        kw = dict(allowed=allowed, f_max=f_max, current=current, persistent=False,
                  use_model_criteria=False, hysteresis=hysteresis, min_ratio=ratio, **regrid_kw)
        g = float(gamma)
        rg = regrid(eps, sig, h, coarsen, eta, gamma=g, **kw)
        pml = int(regrid_kw.get("pml_cells", 6))
        sep = int(regrid_kw.get("is_os_sep", 3))
        nodes = subgrid_fine_nodes(rg.plan.boxes_cells, ratio, sep, pml)
        tries = 0
        while sum(nodes) > max_fine_nodes and tries < 8 and rg.plan.subgrids:
            g = min(0.95, 1.0 - (1.0 - g) * 0.6)         # raise the marking threshold
            rg = regrid(eps, sig, h, coarsen, eta, gamma=g, **kw)
            nodes = subgrid_fine_nodes(rg.plan.boxes_cells, ratio, sep, pml)
            tries += 1
        if sum(nodes) > max_fine_nodes and rg.plan.subgrids:
            # drop the boxes with the least indicator mass until the rest fits
            mass = []
            e = np.asarray(eta, dtype=np.float64)
            for cells in rg.plan.boxes_cells:
                sl = tuple(slice(a, b + 1) for a, b in cells)
                mass.append(float(e[sl].sum()))
            order = np.argsort(mass)[::-1]
            keep, total = [], 0
            for i in order:
                if total + nodes[i] <= max_fine_nodes:
                    keep.append(int(i))
                    total += nodes[i]
            keep.sort()
            rg.plan.subgrids = [rg.plan.subgrids[i] for i in keep]
            rg.plan.boxes_cells = [rg.plan.boxes_cells[i] for i in keep]
            nodes = [nodes[i] for i in keep]
            boxes = sorted(tuple(map(tuple, np.round(np.asarray(sg.box), 6)))
                           for sg in rg.plan.subgrids)
            old = sorted(tuple(map(tuple, np.round(np.asarray(sg.box), 6)))
                         for sg in (current or []))
            rg.changed = boxes != old
        rg.info.update(fine_nodes=nodes, max_fine_nodes=int(max_fine_nodes), gamma=g,
                       seconds_per_gradient=self.seconds_per_gradient(evaluations_per_iteration))
        if rg.changed:
            self.last_change = self.iteration
        return rg


__all__ = ["CostModel", "Decision", "Scheduler", "TimeStepLadder", "subgrid_fine_nodes"]
