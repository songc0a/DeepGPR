"""A priori refinement plan (``DeepGPR.subgrid.adapt``)."""

import unittest

import numpy as np

from DeepGPR.subgrid import SubgridMaxwell3D
from DeepGPR.subgrid.adapt import berger_rigoutsos, merge_boxes, plan_refinement, required_ratio
from DeepGPR.subgrid.amr import regrid


def _model(shape=(91, 121, 121), h=0.01, air=True):
    eps = np.full(shape, 6.0)
    sig = np.full(shape, 0.003)
    if air:
        eps[:13] = 1.0
        sig[:13] = 0.0
    X, Y, Z = np.meshgrid(*[np.arange(n) * h for n in shape], indexing="ij")
    eps[((X - 0.45) ** 2 + (Y - 0.35) ** 2 + (Z - 0.6) ** 2) <= 0.03 ** 2] = 20.0
    sig[((X - 0.45) ** 2 + (Y - 0.85) ** 2 + (Z - 0.6) ** 2) <= 0.03 ** 2] = 1e8
    return eps, sig


class BergerRigoutsosTest(unittest.TestCase):
    def test_boxes_cover_flags_and_separate_clusters(self):
        f = np.zeros((30, 30, 30), dtype=bool)
        f[3:6, 3:7, 4:6] = True
        f[20:25, 18:22, 10:14] = True
        f[21, 2, 25] = True
        boxes = berger_rigoutsos(f, efficiency=0.7)
        cover = np.zeros_like(f)
        for b in boxes:
            cover[b[0][0]:b[0][1] + 1, b[1][0]:b[1][1] + 1, b[2][0]:b[2][1] + 1] = True
        self.assertTrue(np.all(cover[f]))
        self.assertGreaterEqual(len(boxes), 3)
        self.assertLess(cover.sum(), 4 * f.sum())

    def test_merge_boxes(self):
        boxes = [((0, 2), (0, 2), (0, 2)), ((5, 7), (0, 2), (0, 2)), ((20, 22), (20, 22), (20, 22))]
        merged = merge_boxes(boxes, gap=4)
        self.assertEqual(len(merged), 2)
        self.assertIn(((0, 7), (0, 2), (0, 2)), merged)


class PlanTest(unittest.TestCase):
    def test_targets_refined_surface_reported(self):
        eps, sig = _model()
        plan = plan_refinement(eps, sig, 0.01, f_max=4e8, ppw=10)
        self.assertEqual(plan.coarsen, 3)
        self.assertEqual(len(plan.subgrids), 2)
        self.assertGreater(plan.info["unrefinable"], 0)         # the air-soil surface
        centres = sorted(np.mean(np.asarray(sg.box), axis=1)[1] for sg in plan.subgrids)
        np.testing.assert_allclose(centres, [0.35, 0.85], atol=0.04)
        for sg in plan.subgrids:
            self.assertEqual(sg.ratio, 3)
        sol = SubgridMaxwell3D(eps, sig, 0.01, coarsen=plan.coarsen, subgrids=plan.subgrids,
                               pmlthick=10, cuda_kernels=False)
        self.assertEqual(len(sol.static.subs), 2)

    def test_dispersion_criterion_and_coarsen_choice(self):
        eps, sig = _model(air=False)
        # 700 MHz: eps 6 needs 1.6 cm at 10 points per wavelength -> no coarsening
        self.assertEqual(plan_refinement(eps, sig, 0.01, f_max=7e8).coarsen, 1)
        r, info = required_ratio(eps, sig, 0.01, 3, 4e8, 10, 0.15, 0.15)
        self.assertGreater(info["dispersion_nodes"], 0)          # eps 20 sphere: lambda/10 < 3 cm
        self.assertEqual(int(r.max()), 3)

    def test_antennas_are_not_trapped_between_surfaces(self):
        eps, sig = _model(air=False)
        pts = [[0.30, 0.35, 0.6], [0.45, 0.35, 0.48]]
        plan = plan_refinement(eps, sig, 0.01, f_max=4e8, keep_points=pts)
        sol = SubgridMaxwell3D(eps, sig, 0.01, coarsen=plan.coarsen, subgrids=plan.subgrids,
                               pmlthick=10, cuda_kernels=False)
        for p in pts:
            sol.classify(p)                     # raises when a point lies between surfaces


class RegridTest(unittest.TestCase):
    def test_marks_become_a_box_and_refinement_persists(self):
        eps, sig = np.full((91, 91, 91), 4.0), np.full((91, 91, 91), 0.002)
        eta = np.zeros((31, 31, 31))
        eta[15:17, 14:16, 15:17] = 1.0               # main nodes around (0.47, 0.44, 0.47) m
        eta[2, 2, 2] = 0.01                          # small, outside the admissible region
        r1 = regrid(eps, sig, 0.01, 3, eta, allowed=None, theta=0.9, f_max=4e8)
        self.assertEqual(len(r1.plan.subgrids), 1)
        box = np.asarray(r1.plan.subgrids[0].box)
        self.assertTrue(np.all(box[:, 0] <= [0.45, 0.42, 0.45]) and np.all(box[:, 1] >= [0.48, 0.45, 0.48]))
        # no new marks: the existing box stays (the indicator is not defined inside it)
        r2 = regrid(eps, sig, 0.01, 3, np.zeros_like(eta), allowed=None, f_max=4e8,
                    current=r1.plan.subgrids)
        self.assertEqual(len(r2.plan.subgrids), 1)
        self.assertFalse(r2.changed)

    def test_maximum_strategy_non_persistent_and_hysteresis(self):
        eps, sig = np.full((91, 91, 91), 4.0), np.full((91, 91, 91), 0.002)
        eta = np.full((31, 31, 31), 1e-3)                # diffuse remainder
        eta[15:17, 14:16, 15:17] = 1.0
        allowed = np.zeros(eta.shape, dtype=bool)
        allowed[9:22, 9:22, 9:22] = True
        wide = regrid(eps, sig, 0.01, 3, eta, allowed=allowed, theta=0.9, f_max=4e8,
                      use_model_criteria=False)
        tight = regrid(eps, sig, 0.01, 3, eta, allowed=allowed, theta=0.9, f_max=4e8,
                       use_model_criteria=False, gamma=0.2)
        self.assertGreater(wide.info["marked_nodes"], tight.info["marked_nodes"])
        self.assertEqual(tight.info["marked_nodes"], 8)
        size = lambda r: np.prod(np.diff(np.asarray(r.plan.subgrids[0].box), axis=1))  # noqa: E731
        self.assertLess(size(tight), size(wide))
        # the target moves: a persistent layout keeps the old box, a non-persistent one moves
        moved = np.zeros_like(eta)
        moved[11:13, 18:20, 11:13] = 1.0
        keep = regrid(eps, sig, 0.01, 3, moved, allowed=allowed, f_max=4e8, use_model_criteria=False,
                      gamma=0.2, current=tight.plan.subgrids)
        move = regrid(eps, sig, 0.01, 3, moved, allowed=allowed, f_max=4e8, use_model_criteria=False,
                      gamma=0.2, current=tight.plan.subgrids, persistent=False)
        self.assertEqual(len(keep.plan.subgrids), 1)
        self.assertGreater(size(keep), size(tight))       # merged with the old box
        self.assertTrue(move.changed)
        self.assertLessEqual(size(move), size(tight) * 1.01)
        # hysteresis: marks inside the current box keep the layout
        same = regrid(eps, sig, 0.01, 3, eta, allowed=allowed, f_max=4e8, use_model_criteria=False,
                      gamma=0.5, current=tight.plan.subgrids, persistent=False, hysteresis=0.9)
        self.assertFalse(same.changed)
        self.assertTrue(same.info.get("kept"))


if __name__ == "__main__":
    unittest.main()
