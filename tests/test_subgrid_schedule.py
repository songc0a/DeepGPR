"""Dynamic mesh and time-step scheduling (``DeepGPR.subgrid.schedule``).

* the time-step ladder of the solvers (Yee macro step, DG sub-steps, subgrid
  sub-cycling) and the cost model (calibration, monotone in the unknowns);
* the fine-node budget of the remaining time;
* the error-balancing decision rule;
* planning under a budget: the marking threshold rises until the boxes fit,
  the least important boxes are dropped otherwise.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from DeepGPR.hybrid import HybridMaxwell3D, HybridTM2D, Surface, Surface3D  # noqa: E402
from DeepGPR.subgrid import CostModel, Scheduler, Subgrid, SubgridMaxwell3D, TimeStepLadder  # noqa: E402
from DeepGPR.subgrid.hybrid import HybridSubgridMaxwell3D  # noqa: E402
from DeepGPR.subgrid.schedule import subgrid_fine_nodes  # noqa: E402


class LadderAndCostTest(unittest.TestCase):
    def test_ladder_of_hybrid_solvers(self):
        nx, ny, nz = 14, 12, 11
        eps = torch.full((nx, ny, nz), 4.0, dtype=torch.float64)
        sig = torch.zeros_like(eps)
        sol = HybridMaxwell3D(eps, sig, 0.02, surface=Surface3D.sine(0.14, 0.02, 0.24, 0.2),
                              dg_band=(0.06, 0.22), lts="auto", flux="central")
        lad = TimeStepLadder.from_solver(sol)
        self.assertEqual(lad.dg_substeps, sol.lts)
        self.assertGreater(lad.dg_substeps, 1)
        self.assertAlmostEqual(lad.dg_dt, sol.dt_fine)
        self.assertGreater(lad.dg_nodes, 0)
        self.assertEqual(lad.yee_nodes, sol.NX * sol.NY * sol.NZ)
        self.assertIn("DG", lad.describe())
        rates = lad.rates()
        self.assertAlmostEqual(rates["dg"] / rates["yee"], sol.lts)
        e2 = torch.full((30, 40), 4.0, dtype=torch.float64)
        sol2 = HybridTM2D(e2, torch.zeros_like(e2), 0.02, surface=Surface.sine(0.24, 0.04, 0.5),
                          dg_band=(0.15, 0.4), mesh="terrain", lts=3, flux="central")
        lad2 = TimeStepLadder.from_solver(sol2)
        self.assertEqual(lad2.dg_substeps, 3)
        self.assertEqual(lad2.yee_nodes, sol2.NX * sol2.NY)

    def test_ladder_of_subgrid_solvers(self):
        shape = (61, 61, 61)
        eps = torch.full(shape, 4.0, dtype=torch.float64)
        sig = torch.zeros_like(eps)
        box = ((0.27, 0.33), (0.27, 0.33), (0.27, 0.33))
        sol = SubgridMaxwell3D(eps, sig, 0.01, coarsen=3, subgrids=[Subgrid(box, pml_cells=4)],
                               pmlthick=4)
        lad = TimeStepLadder.from_solver(sol)
        self.assertEqual(lad.sub_time_ratios, [3])
        self.assertEqual(lad.sub_nodes, [int(np.prod(sol.static.subs[0].yee.shape))])
        self.assertEqual(lad.dg_substeps, 1)
        shape = (79, 55, 55)
        eps = torch.full(shape, 4.0, dtype=torch.float64)
        sig = torch.zeros_like(eps)
        hyb = HybridSubgridMaxwell3D(eps, sig, 0.01, coarsen=3, surface=Surface3D.sine(0.09, 0.02, 0.3, 0.3),
                                     dg_band=(0.06, 0.15), pmlthick=4, lts="auto", flux="central",
                                     mesh_options=dict(h_y=0.03, h_z=0.03),
                                     subgrids=[Subgrid(((0.39, 0.45), (0.24, 0.30), (0.24, 0.30)),
                                                       is_os_sep=2, pml_cells=4)])
        lad = TimeStepLadder.from_solver(hyb)
        self.assertEqual(lad.dg_substeps, hyb.hyb.lts)
        self.assertGreater(lad.dg_substeps, 1)
        self.assertEqual(lad.sub_time_ratios, [3])
        self.assertIn("subgrid 0", lad.describe())

    def test_cost_model(self):
        cm = CostModel()
        lad = TimeStepLadder(1e-11, 3, 100000, 50000, [20000], [3])
        c = cm.seconds_per_step(lad)
        self.assertGreater(c, cm.seconds_per_step(TimeStepLadder(1e-11, 1, 100000, 50000)))
        self.assertGreater(c, cm.seconds_per_step(TimeStepLadder(1e-11, 3, 100000, 50000)))
        self.assertAlmostEqual(cm.predict(lad, 1e-8, shots=2, gradient=True),
                               2 * lad.steps(1e-8) * c * cm.gradient_factor)
        # calibration reproduces a measurement
        shape = (31, 25, 25)
        eps = torch.full(shape, 4.0, dtype=torch.float64)
        sol = SubgridMaxwell3D(eps, torch.zeros_like(eps), 0.01, coarsen=3, pmlthick=4)
        scale = cm.calibrate(sol, 100 * sol.dt, 2.0)
        self.assertGreater(scale, 0)
        self.assertAlmostEqual(cm.predict(TimeStepLadder.from_solver(sol), 100 * sol.dt), 2.0)
        # fine-node budget: more time buys more nodes; nothing when the base does not fit
        base = TimeStepLadder.from_solver(sol)
        b1 = cm.fine_node_budget(base, 100 * sol.dt, 20.0)
        b2 = cm.fine_node_budget(base, 100 * sol.dt, 40.0)
        self.assertGreater(b2, b1)
        self.assertEqual(cm.fine_node_budget(base, 100 * sol.dt, 1e-9), 0)
        self.assertEqual(subgrid_fine_nodes([((0, 2), (0, 2), (0, 2))], 3, 3, 6),
                         [(2 * 3 + 2 * (3 * 3 + 3 + 6) + 1) ** 3])


class DecisionTest(unittest.TestCase):
    def test_rules(self):
        sch = Scheduler(CostModel(), budget_seconds=1e6, iterations=100, balance=0.25,
                        plateau=0.02, window=3, min_cycle=2)
        for m in (1.0, 0.8, 0.6, 0.5):
            sch.record(m)
        self.assertEqual(sch.decide(0.3, has_boxes=False).action, "keep")     # still falling
        for m in (0.499, 0.498, 0.497):
            sch.record(m)
        self.assertTrue(sch.is_plateau())
        self.assertEqual(sch.decide(0.3, has_boxes=False).action, "refine")   # error dominates
        self.assertEqual(sch.decide(0.001, has_boxes=True).action, "coarsen")
        self.assertEqual(sch.decide(0.001, has_boxes=False).action, "keep")
        self.assertEqual(sch.decide(0.05, has_boxes=True).action, "keep")     # balanced
        sch.last_change = sch.iteration
        self.assertEqual(sch.decide(0.3, has_boxes=False).action, "keep")     # just changed
        self.assertEqual(sch.decide(0.3, has_boxes=False, force=True).action, "refine")
        self.assertEqual(len(sch.log), 7)

    def test_urgent_refinement(self):
        """A modelling error of at least ``urgent`` of the misfit refines at once:
        no plateau and no ``min_cycle`` wait (a check before the first iteration)."""
        sch = Scheduler(CostModel(), budget_seconds=1e6, iterations=100, balance=0.01,
                        plateau=0.02, window=3, min_cycle=4, urgent=0.5)
        sch.record(1.0)                                          # initial misfit only
        d = sch.decide(-0.8, has_boxes=False)
        self.assertEqual(d.action, "refine")
        self.assertIn("urgent", d.reason)
        self.assertEqual(sch.decide(0.3, has_boxes=False).action, "keep")   # wait as before
        off = Scheduler(CostModel(), budget_seconds=1e6, iterations=100, balance=0.01,
                        plateau=0.02, window=3, min_cycle=4, urgent=None)
        off.record(1.0)
        self.assertEqual(off.decide(-0.8, has_boxes=False).action, "keep")
        # the budget rule still wins
        late = Scheduler(CostModel(), budget_seconds=1.0, iterations=100, balance=0.01, urgent=0.5)
        late.t0 -= 10.0                                          # budget spent
        late.record(1.0)
        self.assertEqual(late.decide(0.9, has_boxes=False).action, "keep")


class PlanningTest(unittest.TestCase):
    def setUp(self):
        self.shape = (91, 91, 91)
        self.h = 0.01
        self.eps = np.full(self.shape, 4.0)
        self.sig = np.zeros(self.shape)
        n = (self.shape[0] - 1) // 3 + 1
        self.eta = np.zeros((n, n, n))
        self.eta[10:14, 8:11, 8:11] = 1.0       # a strong compact target
        self.eta[18:21, 19:22, 19:22] = 0.3     # a weaker one, well separated
        self.eta += 1e-3
        self.ladder = TimeStepLadder(5e-11, 1, n ** 3, 0)

    def plan(self, seconds_per_gradient=None, max_fine_nodes=None):
        cm = CostModel(yee=1e-9, sub=1e-9, overhead=0.0, gradient_factor=5.0)
        sch = Scheduler(cm, budget_seconds=1e9, iterations=1)
        if seconds_per_gradient is not None:
            sch.seconds_per_gradient = lambda *_: seconds_per_gradient   # fix the budget
        return sch.plan(self.eps, self.sig, self.h, 3, self.eta, ladder=self.ladder,
                        duration=1e-8, f_max=5e8, pml_cells=4, is_os_sep=2, pmlthick=4,
                        gamma=0.2, hysteresis=None, theta=0.9, max_fine_nodes=max_fine_nodes)

    def test_budget_shrinks_the_layout(self):
        rich = self.plan(1e9)
        self.assertEqual(len(rich.plan.subgrids), 2)
        total_rich = sum(rich.info["fine_nodes"])
        self.assertEqual(rich.info["fine_nodes"],
                         subgrid_fine_nodes(rich.plan.boxes_cells, 3, 2, 4))
        tight = self.plan(max_fine_nodes=int(0.6 * total_rich))
        self.assertLess(sum(tight.info["fine_nodes"]), total_rich)
        self.assertEqual(len(tight.plan.subgrids), 1)           # the strong target survives
        self.assertLessEqual(sum(tight.info["fine_nodes"]), tight.info["max_fine_nodes"])
        self.assertGreater(self.eta[tuple(slice(a, b + 1) for a, b in tight.plan.boxes_cells[0])].max(), 0.9)
        # the budget of the remaining time: zero time buys no box
        none = self.plan(1e-12)
        self.assertEqual(none.plan.subgrids, [])
        self.assertEqual(none.info["max_fine_nodes"], 0)
        # a generous time budget keeps both boxes
        both = self.plan(1e-9 * 5 * 200 * (3 * 2 * total_rich + self.ladder.yee_nodes))
        self.assertEqual(len(both.plan.subgrids), 2)


if __name__ == "__main__":
    unittest.main()
