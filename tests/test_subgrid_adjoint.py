"""Discrete adjoint of the Huygens subgridding (``DeepGPR.subgrid.adjoint``).

The gradient of ``J = 1/2 ||d||^2`` with respect to the model-grid ``eps_r``,
``sigma`` and the source wavelet, with sources and receivers both on the main
grid and inside the subgrid:

* agrees with central finite differences (float64);
* is identical on the fused CUDA kernels (hand-written transposes) and on the
  PyTorch step (autograd), and independent of the checkpoint interval.
"""

import unittest

import numpy as np
import torch

from DeepGPR.subgrid import Subgrid, SubgridMaxwell3D
from DeepGPR.subgrid.adjoint import SubgridPropagator

CUDA = torch.cuda.is_available()
SHAPE = (61, 61, 64)
BOX = ((0.21, 0.27), (0.24, 0.33), (0.21, 0.36))
SRC = [[0.08, 0.28, 0.28], [0.24, 0.285, 0.29]]                  # main + subgrid source
REC = [[0.08, 0.38, 0.30], [0.24, 0.27, 0.27], [0.40, 0.30, 0.28]]


def ricker(nt, dt, f0=5e8):
    t = np.arange(nt) * dt - 1.0 / f0
    a = (np.pi * f0 * t) ** 2
    return torch.as_tensor((1 - 2 * a) * np.exp(-a))


def model():
    g = torch.Generator().manual_seed(1)
    eps = 4.0 + torch.rand(SHAPE, generator=g, dtype=torch.float64)
    sig = 0.005 + 0.005 * torch.rand(SHAPE, generator=g, dtype=torch.float64)
    return eps, sig


def setup(device, fused, k=7, nt=60):
    eps, sig = model()
    sol = SubgridMaxwell3D(eps, sig, 0.01, coarsen=3, pmlthick=4, cuda_kernels=fused,
                           device=device, subgrids=[Subgrid(BOX, ratio=3, is_os_sep=2,
                                                            pml_cells=4)])
    prop = SubgridPropagator(sol, SRC, REC, checkpoint_every=k)
    return sol, prop, ricker(nt, sol.dt).to(device), eps.to(device), sig.to(device)


def gradient(prop, w, eps, sig):
    w = w.clone().requires_grad_(True)
    e = eps.clone().requires_grad_(True)
    s = sig.clone().requires_grad_(True)
    J = 0.5 * (prop(e, s, w) ** 2).sum()
    J.backward()
    return float(J), e.grad, s.grad, w.grad


class AdjointTest(unittest.TestCase):
    device = "cuda" if CUDA else "cpu"

    def test_finite_differences(self):
        sol, prop, w, eps, sig = setup(self.device, CUDA)
        _, ge, gs, gw = gradient(prop, w, eps, sig)

        def J(e, s, ww):
            with torch.no_grad():
                return 0.5 * float((prop(e, s, ww) ** 2).sum())

        gen = torch.Generator().manual_seed(2)
        rnd = lambda t: (torch.rand(t.shape, generator=gen, dtype=t.dtype) - 0.5).to(t.device)  # noqa: E731
        box = torch.zeros_like(eps)
        box[22:27, 25:33, 22:36] = 1.0
        cases = {"eps in the subgrid": (rnd(eps) * box, 0 * sig, 0 * w),
                 "eps everywhere": (rnd(eps), 0 * sig, 0 * w),
                 "sigma": (0 * eps, rnd(sig), 0 * w),
                 "wavelet": (0 * eps, 0 * sig, rnd(w))}
        for name, (ve, vs, vw) in cases.items():
            he = 1e-5 * float(eps.abs().max()) / max(float(ve.abs().max()), 1e-30)
            hs = 1e-5 * float(sig.abs().max()) / max(float(vs.abs().max()), 1e-30)
            hw = 1e-5 * float(w.abs().max()) / max(float(vw.abs().max()), 1e-30)
            fd = (J(eps + he * ve, sig + hs * vs, w + hw * vw)
                  - J(eps - he * ve, sig - hs * vs, w - hw * vw)) / 2.0
            an = float((ge * he * ve).sum() + (gs * hs * vs).sum() + (gw * hw * vw).sum())
            with self.subTest(name):
                self.assertLess(abs(fd - an) / abs(an), 1e-6)

    @unittest.skipUnless(CUDA, "CUDA not available")
    def test_fused_kernels_match_autograd_of_pytorch_step(self):
        _, p_cpu, w, eps, sig = setup("cpu", False)
        _, p_gpu, wg, eg, sg = setup("cuda", True)
        a = gradient(p_cpu, w, eps, sig)
        b = gradient(p_gpu, wg, eg, sg)
        self.assertAlmostEqual(a[0], b[0], delta=1e-12 * abs(a[0]))
        for x, y in zip(a[1:], b[1:]):
            self.assertLess(float((x - y.cpu()).norm() / x.norm()), 1e-12)

    def test_checkpoint_interval_does_not_change_the_gradient(self):
        _, p1, w, eps, sig = setup(self.device, CUDA, k=1, nt=40)
        _, p2, _, _, _ = setup(self.device, CUDA, k=40, nt=40)
        a, b = gradient(p1, w, eps, sig), gradient(p2, w, eps, sig)
        for x, y in zip(a[1:], b[1:]):
            self.assertLess(float((x - y).norm() / x.norm()), 1e-12)


if __name__ == "__main__":
    unittest.main()
