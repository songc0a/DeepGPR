"""DGTD rugged surface + FDTD with Huygens subgrids (``DeepGPR.subgrid.hybrid``).

* without subgrids the split step reproduces :class:`DeepGPR.hybrid.HybridMaxwell3D`
  on the block-averaged model to round-off;
* with a subgrid below the DG band the gradient of the discrete adjoint
  (:class:`DeepGPR.subgrid.adjoint.SubgridPropagator`) agrees with central finite
  differences (eps_r in the subgrid, in the DG band, everywhere; sigma).
"""

import unittest

import numpy as np
import torch

from DeepGPR.hybrid import HybridMaxwell3D, Surface3D
from DeepGPR.subgrid import Subgrid
from DeepGPR.subgrid.adjoint import SubgridPropagator
from DeepGPR.subgrid.hybrid import HybridSubgridMaxwell3D
from DeepGPR.subgrid.solver import block_average

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SHAPE = (79, 55, 55)
SURF = Surface3D.sine(0.09, 0.02, 0.3, 0.3)
BAND = (0.06, 0.15)
BOX = ((0.39, 0.45), (0.24, 0.30), (0.24, 0.30))
MESH = dict(h_y=0.03, h_z=0.03)
SRC = [[0.03, 0.27, 0.27]]
REC = [[0.03, 0.36, 0.27], [0.60, 0.27, 0.27]]


def model():
    g = torch.Generator().manual_seed(0)
    eps = 4.0 + torch.rand(SHAPE, generator=g, dtype=torch.float64)
    sig = 0.002 + 0.002 * torch.rand(SHAPE, generator=g, dtype=torch.float64)
    return eps, sig


def ricker(nt, dt, f0=8e8):
    t = np.arange(nt) * dt - 1.0 / f0
    a = (np.pi * f0 * t) ** 2
    return torch.as_tensor((1 - 2 * a) * np.exp(-a))


class HybridSubgridTest(unittest.TestCase):
    def test_without_subgrids_equals_hybrid(self):
        eps, sig = model()
        a = HybridSubgridMaxwell3D(eps, sig, 0.01, coarsen=3, surface=SURF, dg_band=BAND,
                                   pmlthick=4, device=DEV, mesh_options=MESH)
        w = ricker(150, a.dt)
        da = a.run(w, SRC, REC)["receivers"]
        b = HybridMaxwell3D(block_average(eps, 3), block_average(sig, 3), 0.03, surface=SURF,
                            dg_band=BAND, pmlthick=4, dt=a.dt, device=DEV, mesh_options=MESH)
        db = b.run(w, SRC, REC)["receivers"]
        self.assertLess(float((da - db).norm() / db.norm()), 1e-12)

    def test_time_ratio_one_under_the_band(self):
        """Under the DG band the main dt is below the subgrid's CFL limit: the subgrid at
        the main dt agrees with the sub-cycled one to the time-discretisation error."""
        eps, sig = model()
        out = []
        for tr in (3, 1):
            sol = HybridSubgridMaxwell3D(eps, sig, 0.01, coarsen=3, surface=SURF, dg_band=BAND,
                                         pmlthick=4, device=DEV, mesh_options=MESH,
                                         subgrids=[Subgrid(BOX, is_os_sep=2, pml_cells=4,
                                                           time_ratio=tr)])
            out.append(sol.run(ricker(400, sol.dt), SRC, REC)["receivers"])
        self.assertLess(float((out[0] - out[1]).norm() / out[0].norm()), 1e-3)

    def test_gradient_finite_differences(self):
        eps, sig = model()
        sol = HybridSubgridMaxwell3D(eps, sig, 0.01, coarsen=3, surface=SURF, dg_band=BAND,
                                     pmlthick=4, device=DEV, mesh_options=MESH,
                                     subgrids=[Subgrid(BOX, is_os_sep=2, pml_cells=4)])
        prop = SubgridPropagator(sol, SRC, REC, checkpoint_every=40)
        w = ricker(600, sol.dt).to(DEV)
        e = eps.to(DEV).requires_grad_(True)
        s = sig.to(DEV).requires_grad_(True)
        (0.5 * (prop(e, s, w) ** 2).sum()).backward()
        ge, gs = e.grad, s.grad
        e0, s0 = eps.to(DEV), sig.to(DEV)

        def J(de, ds):
            with torch.no_grad():
                return 0.5 * float((prop(e0 + de, s0 + ds, w) ** 2).sum())

        gen = torch.Generator().manual_seed(3)
        rnd = lambda: (torch.rand(SHAPE, generator=gen, dtype=torch.float64) - 0.5).to(DEV)  # noqa: E731
        sub = torch.zeros(SHAPE, dtype=torch.float64, device=DEV)
        sub[40:45, 25:30, 25:30] = 1.0
        band = torch.zeros_like(sub)
        band[6:15] = 1.0
        z = torch.zeros_like(sub)
        # J depends weakly on the subgrid (deep, one transmission receiver): a larger step
        # keeps the finite difference above the run-to-run rounding of the atomics
        cases = {"eps in the subgrid": (rnd() * sub, z, 1e-4),
                 "eps in the DG band": (rnd() * band, z, 1e-5),
                 "eps everywhere": (rnd(), z, 1e-5), "sigma": (z, rnd(), 1e-5)}
        for name, (ve, vs, step) in cases.items():
            he = step * 5.0 / max(float(ve.abs().max()), 1e-30)
            hs = 1e-5 * 0.004 / max(float(vs.abs().max()), 1e-30)
            fd = (J(he * ve, hs * vs) - J(-he * ve, -hs * vs)) / 2.0
            an = float((ge * he * ve).sum() + (gs * hs * vs).sum())
            with self.subTest(name):
                self.assertGreater(abs(an), 0.0)
                self.assertLess(abs(fd - an) / abs(an), 1e-6)


if __name__ == "__main__":
    unittest.main()
