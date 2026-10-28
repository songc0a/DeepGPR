"""Adjoint-weighted error indicator of the main grid (``DeepGPR.subgrid.indicators``).

* Doerfler marking selects the smallest set reaching the requested fraction;
* the DWR estimate (sum of the signed indicator) predicts the change of a
  data functional between the 3 cm main grid and a uniform 1 cm grid
  (``<g, d_fine - d_coarse> * 9/8``, the 1 cm grid still carrying 1/9 of the
  second-order error) within 20 % (effectivity 0.98-1.01 measured at 200-300 MHz).
"""

import unittest

import numpy as np
import torch

from DeepGPR.subgrid import SubgridMaxwell3D, fine_source_samples
from DeepGPR.subgrid.adjoint import SubgridPropagator
from DeepGPR.subgrid.indicators import doerfler_mark

CUDA = torch.cuda.is_available()


class DoerflerTest(unittest.TestCase):
    def test_marks_smallest_set(self):
        eta = np.array([0.1, 5.0, 0.2, 3.0, 1.0, 0.7])
        m = doerfler_mark(eta, 0.8)          # 5 + 3 = 8 >= 0.8 * 10
        np.testing.assert_array_equal(m, [False, True, False, True, False, False])
        self.assertFalse(doerfler_mark(np.zeros(4)).any())


@unittest.skipUnless(CUDA, "CUDA not available")
class EffectivityTest(unittest.TestCase):
    def test_dwr_estimate_matches_refinement(self):
        torch.set_default_dtype(torch.float64)
        try:
            shape = (61, 61, 61)
            eps = torch.full(shape, 4.0)
            sig = torch.zeros(shape)
            src = [[0.15, 0.30, 0.315]]
            rec = [[0.45, 0.30, 0.315], [0.30, 0.45, 0.315], [0.30, 0.30, 0.465]]
            nt, f0 = 260, 2.5e8
            sol = SubgridMaxwell3D(eps, sig, 0.01, coarsen=3, pmlthick=10, device="cuda")
            t = np.arange(nt) * sol.dt - 1.2 / f0
            a = (np.pi * f0 * t) ** 2
            w = torch.as_tensor((1 - 2 * a) * np.exp(-a)).cuda()
            prop = SubgridPropagator(sol, src, rec, indicator=True, checkpoint_every=16)
            e = eps.cuda().requires_grad_(True)
            d = prop(e, sig.cuda(), w)
            (0.5 * (d ** 2).sum()).backward()
            est = prop.indicator_obj.estimate()
            fine = SubgridMaxwell3D(eps, sig, 0.01, coarsen=1, pmlthick=30, dt=sol.dt / 3,
                                    dipole_length=0.03, device="cuda")
            wf = fine_source_samples(w[None].cpu(), 3)[0]
            wf = torch.cat([torch.zeros(1), wf, torch.zeros(3)])[:3 * nt]
            df = fine.run(wf, src, rec)["receivers"][:, 2::3]
            d = d.detach()
            ref = float((d * (df - d)).sum()) * 9.0 / 8.0
            self.assertGreater(est / ref, 0.8)
            self.assertLess(est / ref, 1.2)
            eta = prop.indicator_obj.eta()
            self.assertEqual(eta.shape, (21, 21, 21))
            self.assertTrue(np.all(eta >= 0))
        finally:
            torch.set_default_dtype(torch.float32)


if __name__ == "__main__":
    unittest.main()
