"""Staircase DGTD-FDTD interface following the surface (2D TM prototype).

* geometry of the Yee region: weights, ghost edges, concave corners, segments;
* the layer mesh tiles the layer and conforms to the surface;
* energy conservation of the coupled scheme to round-off (central fluxes, no
  absorbers) on a sine surface with many corners, for r = 1 and 3, on the CPU
  and through the fused CUDA kernels;
* discrete adjoint: source dot product and finite-difference directional
  derivatives of eps_r and sigma (float64);
* on CUDA: the hand-written step transpose (vjp2d with the staircase
  transposes) against autograd, per step and for the whole reverse sweep.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from DeepGPR.hybrid import HybridPropagator, Surface  # noqa: E402
from DeepGPR.hybrid.staircase import StaircaseHybridTM2D, yee_layer_structure  # noqa: E402
from test_hybrid_dgtd import gaussian_burst  # noqa: E402

CUDA = torch.cuda.is_available()


def closed_box(r=1, device="cpu", flux="central", pml=0, sig=0.0):
    nx, ny, dx = 50, 60, 0.02
    eps = torch.full((nx, ny), 2.0, dtype=torch.float64)
    eps[30:] = 4.0
    sigma = torch.full((nx, ny), sig, dtype=torch.float64)
    return StaircaseHybridTM2D(eps, sigma, dx, surface=Surface.sine(0.5, 0.12, 0.5),
                               layer=(0.06, 0.06), pmlthick=pml, dg_order=1, dg_h=dx, flux=flux,
                               cfl_safety=0.9, fdtd_refinement=r, device=device,
                               dtype=torch.float64)


class StructureTest(unittest.TestCase):
    def test_weights_and_corners(self):
        d = np.zeros((9, 12), dtype=bool)        # Yee cells of the layer: a step
        d[3:6, :6] = True
        d[4:7, 6:] = True
        s = yee_layer_structure(d)
        w = s["w"]
        self.assertEqual(w[3, 3], 0.5)            # straight piece
        self.assertEqual(w[4, 4], 0.0)            # inside the layer
        self.assertEqual(w[3, 6], 0.75)           # concave corner of R
        self.assertEqual(w[4, 6], 0.25)           # convex corner of R
        self.assertIn((3, 6), s["concave"])
        # every boundary node's dual sides are active edges or its own ghosts
        owned = {g[0] for g in s["ghosts"]["hy"]}
        self.assertEqual(len(owned), len(s["ghosts"]["hy"]))

    def test_layer_mesh_tiles_and_conforms(self):
        sol = closed_box()
        poly = sol.mesh.info["polygon"]
        xs, ys = poly[:, 0], poly[:, 1]
        area = 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
        p = sol.mesh.vertices[sol.mesh.triangles]
        tri = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                           - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
        self.assertAlmostEqual(tri.sum() / area, 1.0, places=12)
        self.assertGreater(len(sol._struct["concave"]), 10)


class EnergyTest(unittest.TestCase):
    def _check(self, sol):
        nt = 1500
        w = gaussian_burst(nt, sol.dt, 1e9, 1.5e-9, 300)
        res = sol.run(w, [[[0.25, 0.6]]], [[[0.45, 0.3]]], energy_every=5)
        steps = np.array(res["energy_steps"])
        e = res["energy"][:, 0].cpu().numpy()[steps > 320]
        self.assertGreater(e.mean(), 0)
        self.assertLess((e.max() - e.min()) / e.mean(), 1e-12)

    def test_energy_r1(self):
        self._check(closed_box(1))

    def test_energy_r3(self):
        self._check(closed_box(3))

    @unittest.skipUnless(CUDA, "CUDA not available")
    def test_energy_cuda_kernels(self):
        self._check(closed_box(3, device="cuda"))


class AdjointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nx, ny, dx = 30, 40, 0.02
        cls.eps = torch.full((nx, ny), 4.0, dtype=torch.float64)
        cls.eps[18:22, 15:20] = 8.0
        cls.sig = torch.full((nx, ny), 0.005, dtype=torch.float64)
        cls.surf = Surface.sine(0.25, 0.04, 0.5)
        sol = StaircaseHybridTM2D(cls.eps, cls.sig, dx, surface=cls.surf, layer=(0.06, 0.06),
                                  pmlthick=6, fdtd_refinement=3, nshot=2)
        src = np.array([[[0.18, 0.3]], [[0.2, 0.5]]])
        rec = np.array([[[0.18, 0.5], [0.45, 0.2], [0.05, 0.6]]] * 2)
        cls.prop = HybridPropagator(sol, src, rec, checkpoint_every=17)
        t = (np.arange(220) + 0.5) * sol.dt - 1e-9
        a = (np.pi * 1e9 * t) ** 2
        cls.w = torch.tensor((1 - 2 * a) * np.exp(-a))
        with torch.no_grad():
            cls.dobs = 0.9 * cls.prop(cls.eps, cls.sig, cls.w[None])

    def J(self, e, s):
        return 0.5 * ((self.prop(e, s, self.w[None]) - self.dobs) ** 2).sum()

    def test_source_dot_product(self):
        w = self.w.clone().requires_grad_(True)
        d = self.prop(self.eps, self.sig, w[None])
        r = torch.randn_like(d)
        gw, = torch.autograd.grad((d * r).sum(), w)
        dw = torch.randn_like(self.w)
        with torch.no_grad():
            lin = self.prop(self.eps, self.sig, (self.w + dw)[None]) - d.detach()
        self.assertLess(abs(float((lin * r).sum()) - float((gw * dw).sum()))
                        / abs(float((lin * r).sum())), 1e-12)

    def test_finite_differences(self):
        e = self.eps.clone().requires_grad_(True)
        s = self.sig.clone().requires_grad_(True)
        self.J(e, s).backward()
        nx, ny = self.eps.shape
        gen = torch.Generator().manual_seed(3)
        x = torch.arange(nx)[:, None] * 0.02
        above = x < torch.tensor(self.surf(np.arange(ny) * 0.02))[None]
        for par, g, scale in (("eps", e.grad, 4.0), ("sigma", s.grad, 0.005)):
            v = torch.randn((nx, ny), generator=gen, dtype=torch.float64)
            v[above] = 0
            adj = float((g * v).sum())
            errs = []
            for h in (1e-3, 1e-4, 1e-5):
                hh = h * scale
                with torch.no_grad():
                    if par == "eps":
                        fd = (self.J(self.eps + hh * v, self.sig) - self.J(self.eps - hh * v, self.sig)) / (2 * hh)
                    else:
                        fd = (self.J(self.eps, self.sig + hh * v) - self.J(self.eps, self.sig - hh * v)) / (2 * hh)
                errs.append(abs(float(fd) - adj) / abs(adj))
            with self.subTest(parameter=par):
                self.assertLess(min(errs), 1e-7)


@unittest.skipUnless(CUDA, "CUDA not available")
class HandWrittenAdjointTest(unittest.TestCase):
    """vjp2d.step_vjp with the staircase transposes against autograd of the step."""

    def test_step_vjp_matches_autograd(self):
        from DeepGPR.hybrid import functional as fn, vjp2d
        nx, ny, dx = 30, 40, 0.02
        gen = torch.Generator().manual_seed(0)
        eps = 3 + 3 * torch.rand(nx, ny, dtype=torch.float64, generator=gen)
        sig = 0.01 * torch.rand(nx, ny, dtype=torch.float64, generator=gen)
        sol = StaircaseHybridTM2D(eps, sig, dx, surface=Surface.sine(0.3, 0.06, 0.4),
                                  layer=(0.06, 0.06), pmlthick=6, fdtd_refinement=3, nshot=2,
                                  device="cuda", dtype=torch.float64)
        st, c = sol.static, sol.coefs
        self.assertTrue(vjp2d.supported(st))
        self.assertGreater(st.staircase.n_concave, 0)
        rnd = lambda t: torch.randn(t.shape, generator=gen, dtype=torch.float64).cuda() if t.numel() else t  # noqa: E731
        u = [rnd(t) for t in fn.zero_state(st)]
        lam = [rnd(t) for t in fn.zero_state(st)]
        src = sol.point_ops(np.array([[[0.25, 0.3]], [[0.5, 0.5]]]), source=True)
        rec = sol.point_ops(np.array([[[0.24, 0.2], [0.45, 0.5], [0.1, 0.6]]] * 2), source=False)
        amp = torch.randn(1, generator=gen, dtype=torch.float64).cuda()
        gd = torch.randn((2, rec.count), generator=gen, dtype=torch.float64).cuda()
        ul = [t.clone().requires_grad_(True) if t.numel() else t for t in u]
        cl = c.replaced([t.detach().clone().requires_grad_(True) for t in c.tensors()])
        al = amp.clone().requires_grad_(True)
        out = fn.step(st, cl, tuple(ul), src, al)
        act = [i for i, t in enumerate(u) if t.numel()]
        ref = torch.autograd.grad([out[i] for i in act] + [fn.sample(st, rec, out)],
                                  [ul[i] for i in act] + cl.tensors() + [al],
                                  [lam[i] for i in act] + [gd], allow_unused=True)
        with torch.no_grad():
            u1 = fn.step(st, c, tuple(u), src, amp)
            acc = {}
            lam_n, g_amp = vjp2d.step_vjp(st, c, u, u1, vjp2d.sample_adjoint(st, rec, gd, lam),
                                          src, amp, acc, True, True)
            acc = vjp2d.finalize(acc, c)
        names = [n for n in c.NAMES if getattr(c, n) is not None]
        mine = [lam_n[i] for i in act] + [acc.get(n) for n in names] + [g_amp]
        for k, (a, b) in enumerate(zip(mine, ref)):
            b = torch.zeros_like(a) if b is None else b
            a = torch.zeros_like(b) if a is None else a
            with self.subTest(item=k):
                self.assertLess(float((a - b).abs().max() / (b.abs().max() + 1e-300)), 1e-12)

    def test_sweep_matches_autograd_sweep(self):
        import os
        nx, ny, dx = 30, 40, 0.02
        eps = torch.full((nx, ny), 4.0, dtype=torch.float64)
        eps[18:22, 15:20] = 8.0
        sig = torch.full((nx, ny), 0.005, dtype=torch.float64)
        sol = StaircaseHybridTM2D(eps, sig, dx, surface=Surface.sine(0.25, 0.04, 0.5),
                                  layer=(0.06, 0.06), pmlthick=6, fdtd_refinement=3, nshot=2,
                                  device="cuda", dtype=torch.float64)
        prop = HybridPropagator(sol, np.array([[[0.18, 0.3]], [[0.2, 0.5]]]),
                                np.array([[[0.18, 0.5], [0.45, 0.2], [0.05, 0.6]]] * 2),
                                checkpoint_every=17)
        t = (np.arange(200) + 0.5) * sol.dt - 1e-9
        a = (np.pi * 1e9 * t) ** 2
        w0 = torch.tensor((1 - 2 * a) * np.exp(-a)).cuda()
        with torch.no_grad():
            dobs = 0.9 * prop(eps, sig, w0[None])
        grads = []
        for env in (None, "1"):
            if env:
                os.environ["DEEPGPR_HYBRID_AUTOGRAD_VJP"] = env
            try:
                e, s, w = (t.clone().requires_grad_(True) for t in (eps, sig, w0))
                (0.5 * ((prop(e, s, w[None]) - dobs) ** 2).sum()).backward()
                grads.append((e.grad, s.grad, w.grad))
            finally:
                os.environ.pop("DEEPGPR_HYBRID_AUTOGRAD_VJP", None)
        for a_, b_ in zip(*grads):
            self.assertLess(float((a_ - b_).norm() / b_.norm()), 1e-11)


if __name__ == "__main__":
    unittest.main()
