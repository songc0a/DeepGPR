"""Phase 3 tests: discrete adjoint of the hybrid DGTD-FDTD scheme.

* dot-product test of the (linear) source-to-data map, to round-off;
* gradients independent of the checkpoint interval;
* zero sensitivity above the surface (air is not a model parameter);
* finite-difference directional derivatives for eps_r and sigma with
  perturbations in the DG band, in the Yee region, across the interface rows
  and everywhere (float64, second-order convergence down to round-off);
* pure-Yee mode against DeepGPR's native discrete adjoint;
* full DGTD (DG CPML on all faces);
* on CUDA: the same checks through the fused kernels and the hand-written step
  transpose, the fused step against the PyTorch step (forward and VJP), the
  hand-written transpose against autograd (step and sweep) and CUDA Graph
  replay against the eager step.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import DeepGPR  # noqa: E402
from DeepGPR.hybrid import HybridPropagator, HybridTM2D, Surface  # noqa: E402
from DeepGPR.hybrid import functional as fn  # noqa: E402

CUDA = torch.cuda.is_available()


def ricker(nt, dt, f=1e9, t0=1.0e-9):
    t = (np.arange(nt) + 0.5) * dt - t0
    a = (np.pi * f * t) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a))


class HybridAdjointTest(unittest.TestCase):
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        nx, ny, dx = 30, 40, 0.02
        cls.dx = dx
        cls.eps = torch.full((nx, ny), 4.0, dtype=torch.float64)
        cls.eps[20:24, 15:20] = 8.0
        cls.sig = torch.full((nx, ny), 0.005, dtype=torch.float64)
        cls.surf = Surface.sine(0.25, 0.04, 0.5)
        sol = HybridTM2D(cls.eps, cls.sig, dx, surface=cls.surf, dg_band=(0.16, 0.38), pmlthick=6,
                         mesh="delaunay", fdtd_refinement=3, nshot=2, device=cls.device)
        src = np.array([[[0.18, 0.3]], [[0.5, 0.5]]])
        rec = np.array([[[0.18, 0.5], [0.45, 0.2], [0.05, 0.6]]] * 2)
        cls.prop = HybridPropagator(sol, src, rec, checkpoint_every=17)
        cls.w = ricker(220, sol.dt).to(cls.device)
        with torch.no_grad():
            cls.dobs = 0.9 * cls.prop(cls.eps, cls.sig, cls.w[None])
        e = cls.eps.clone().requires_grad_(True)
        s = cls.sig.clone().requires_grad_(True)
        cls.J(e, s).backward()
        cls.ge, cls.gs = e.grad.clone(), s.grad.clone()
        x = torch.arange(nx)[:, None] * dx
        cls.above = x < torch.tensor(cls.surf(np.arange(ny) * dx))[None]


    @classmethod
    def J(cls, e, s):
        return 0.5 * ((cls.prop(e, s, cls.w[None]) - cls.dobs) ** 2).sum()

    def test_source_dot_product(self):
        w = self.w.clone().requires_grad_(True)
        d = self.prop(self.eps, self.sig, w[None])
        r = torch.randn_like(d)
        gw, = torch.autograd.grad((d * r).sum(), w)
        dw = torch.randn_like(self.w)
        with torch.no_grad():
            lin = self.prop(self.eps, self.sig, (self.w + dw)[None]) - d.detach()
        lhs, rhs = float((lin * r).sum()), float((gw * dw).sum())
        self.assertLess(abs(lhs - rhs) / abs(lhs), 1e-12)

    def test_checkpoint_interval_does_not_change_gradient(self):
        self.prop.checkpoint_every = 1
        try:
            e = self.eps.clone().requires_grad_(True)
            s = self.sig.clone().requires_grad_(True)
            self.J(e, s).backward()
        finally:
            self.prop.checkpoint_every = 17
        self.assertLess(float((e.grad - self.ge).abs().max() / self.ge.abs().max()), 1e-13)
        self.assertLess(float((s.grad - self.gs).abs().max() / self.gs.abs().max()), 1e-13)

    def test_no_sensitivity_above_surface(self):
        self.assertEqual(float(self.ge[self.above].abs().max()), 0.0)
        self.assertEqual(float(self.gs[self.above].abs().max()), 0.0)
        self.assertGreater(float(self.ge[~self.above].abs().max()), 0.0)

    def test_finite_difference_directional_derivatives(self):
        nx, ny = self.eps.shape
        regions = {"DG band": (slice(13, 19), slice(5, 35)),
                   "Yee region": (slice(21, 27), slice(5, 35)),
                   "interface rows": (slice(18, 21), slice(5, 35))}
        gen = torch.Generator().manual_seed(3)
        for name, region in regions.items():
            v = torch.zeros(nx, ny, dtype=torch.float64)
            v[region] = torch.randn(v[region].shape, generator=gen, dtype=torch.float64)
            v[self.above] = 0
            for par, g, scale in (("eps", self.ge, 4.0), ("sigma", self.gs, 0.005)):
                adj = float((g * v).sum())
                errors = []
                for h in (1e-3, 1e-4, 1e-5):
                    hh = h * scale
                    with torch.no_grad():
                        if par == "eps":
                            jp, jm = self.J(self.eps + hh * v, self.sig), self.J(self.eps - hh * v, self.sig)
                        else:
                            jp, jm = self.J(self.eps, self.sig + hh * v), self.J(self.eps, self.sig - hh * v)
                    fd = float((jp - jm) / (2 * hh))
                    errors.append(abs(fd - adj) / max(abs(fd), abs(adj)))
                with self.subTest(region=name, parameter=par):
                    self.assertLess(min(errors), 1e-7)


@unittest.skipUnless(CUDA, "CUDA not available")
class HybridAdjointCudaTest(HybridAdjointTest):
    """The checks above through the fused kernels and the hand-written transpose."""

    device = "cuda"

    def test_fused_path_is_used(self):
        from DeepGPR.hybrid import vjp2d
        self.assertTrue(vjp2d.supported(self.prop.static))

    def test_hand_written_sweep_matches_autograd_sweep(self):
        import os
        grads = []
        for env in (None, "1"):
            if env:
                os.environ["DEEPGPR_HYBRID_AUTOGRAD_VJP"] = env
            try:
                e = self.eps.clone().requires_grad_(True)
                s = self.sig.clone().requires_grad_(True)
                w = self.w.clone().requires_grad_(True)
                (0.5 * ((self.prop(e, s, w[None]) - self.dobs) ** 2).sum()).backward()
                grads.append((e.grad, s.grad, w.grad))
            finally:
                os.environ.pop("DEEPGPR_HYBRID_AUTOGRAD_VJP", None)
        for a, b in zip(*grads):
            self.assertLess(float((a - b).norm() / b.norm()), 1e-12)


def _hybrid_cuda(**kw):
    torch.manual_seed(0)
    nx, ny, dx = 30, 36, 0.02
    eps = 3 + 3 * torch.rand(nx, ny, dtype=torch.float64)
    sig = 0.01 * torch.rand(nx, ny, dtype=torch.float64)
    args = dict(surface=Surface.sine(0.25, 0.04, 0.5), dg_band=(0.16, 0.38), pmlthick=6)
    args.update(kw)
    return HybridTM2D(eps, sig, dx, device="cuda", dtype=torch.float64, nshot=2, **args)


CONFIGS = (dict(), dict(mesh="delaunay", dg_order=3, fdtd_refinement=3),
           dict(dg_order=2, side_boundary="abc"), dict(flux="central"),
           dict(dg_band=(None, None)), dict(dg_band=None))


@unittest.skipUnless(CUDA, "CUDA not available")
class Cuda2DKernelTest(unittest.TestCase):
    def test_fused_step_matches_pytorch_step(self):
        import dataclasses
        for kw in CONFIGS:
            sol = _hybrid_cuda(**kw)
            st, c = sol.static, sol.coefs
            self.assertTrue(st.yee is None or st.yee.cuda is not None)
            self.assertTrue(st.dg is None or st.dg.cuda is not None)
            st_t = dataclasses.replace(
                st, yee=None if st.yee is None else dataclasses.replace(st.yee, cuda=None),
                dg=None if st.dg is None else dataclasses.replace(st.dg, cuda=None))
            gen = torch.Generator().manual_seed(1)
            rnd = lambda t: torch.randn(t.shape, generator=gen, dtype=torch.float64).cuda() if t.numel() else t  # noqa: E731
            s0 = [rnd(t) for t in fn.zero_state(st)]
            src = sol.point_ops(np.array([[[0.08, 0.3]], [[0.3, 0.4]]]), source=True)
            amp = torch.ones(1, dtype=torch.float64, device="cuda")
            outs, grads = [], []
            for static in (st, st_t):
                u = [t.clone().requires_grad_(True) if t.numel() else t for t in s0]
                cc = c.replaced([t.detach().clone().requires_grad_(True) for t in c.tensors()])
                out = fn.step(static, cc, tuple(u), src, amp)
                act = [i for i, t in enumerate(out) if t.numel()]
                g = torch.autograd.grad([out[i] for i in act], [u[i] for i in act] + cc.tensors(),
                                        [torch.ones_like(out[i]) * (i + 1) for i in act],
                                        allow_unused=True)
                outs.append([out[i] for i in act])
                grads.append(g)
            rel = lambda a, b: float((a - b).abs().max() / (b.abs().max() + 1e-300))  # noqa: E731
            for a, b in zip(*outs):
                with self.subTest(**kw):
                    self.assertLess(rel(a, b), 1e-14)
            for a, b in zip(*grads):
                if a is None and b is None:
                    continue
                a = torch.zeros_like(b) if a is None else a
                b = torch.zeros_like(a) if b is None else b
                with self.subTest(**kw):
                    self.assertLess(rel(a, b), 1e-13)

    def test_hand_written_step_vjp_matches_autograd(self):
        from DeepGPR.hybrid import vjp2d
        for kw in CONFIGS:
            sol = _hybrid_cuda(**kw)
            st, c = sol.static, sol.coefs
            self.assertTrue(vjp2d.supported(st))
            gen = torch.Generator().manual_seed(1)
            rnd = lambda t: torch.randn(t.shape, generator=gen, dtype=torch.float64).cuda() if t.numel() else t  # noqa: E731
            u = [rnd(t) for t in fn.zero_state(st)]
            lam = [rnd(t) for t in fn.zero_state(st)]
            src = sol.point_ops(np.array([[[0.08, 0.3]], [[0.3, 0.4]]]), source=True)
            rec = sol.point_ops(np.array([[[0.06, 0.2], [0.28, 0.5], [0.45, 0.3]]] * 2), source=False)
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
            for a, b in zip(mine, ref):
                b = torch.zeros_like(a) if b is None else b
                a = torch.zeros_like(b) if a is None else a
                with self.subTest(**kw):
                    self.assertLess(float((a - b).abs().max() / (b.abs().max() + 1e-300)), 1e-12)

    def test_cuda_graph_replay_matches_eager_step(self):
        import os
        from DeepGPR.hybrid import graphs
        sol = _hybrid_cuda()
        self.assertTrue(graphs.enabled(sol.static))
        w = ricker(300, sol.dt).cuda()
        src = np.array([[[0.08, 0.3]], [[0.08, 0.4]]])
        rec = np.array([[[0.06, 0.2], [0.45, 0.3]]] * 2)
        out = []
        for env in (None, "1"):
            if env:
                os.environ["DEEPGPR_HYBRID_DISABLE_GRAPHS"] = env
            try:
                out.append(sol.run(w, src, rec)["receivers"])
            finally:
                os.environ.pop("DEEPGPR_HYBRID_DISABLE_GRAPHS", None)
        self.assertGreater(float(out[1].abs().max()), 0.0)
        self.assertLess(float((out[0] - out[1]).abs().max() / out[1].abs().max()), 1e-13)


class NativeAndFullDGTest(unittest.TestCase):
    def test_pure_yee_gradient_matches_native_deepgpr(self):
        torch.manual_seed(1)
        nx, ny, dx, nt, p = 40, 50, 0.02, 300, 8
        eps = 2 + 4 * torch.rand(nx, ny)
        sig = 0.01 * torch.rand(nx, ny)
        dt = 0.9 * DeepGPR.max_stable_time_step(dx, nx + 2 * p, ny + 2 * p, 1)
        w = DeepGPR.wavelet.ricker(4e8, nt, dt, 1 / 4e8)
        src = torch.tensor([[[12, 20, 0]], [[25, 30, 0]]], dtype=torch.int)
        rec = torch.tensor([[[5, 5, 0], [30, 40, 0]], [[8, 45, 0], [35, 10, 0]]], dtype=torch.int)
        r = torch.randn(2, nt, 2)
        e1, s1 = eps.clone().requires_grad_(True), sig.clone().requires_grad_(True)
        d1 = DeepGPR.compute(device="cpu", dx=dx, dt=dt, source_amplitudes=w.reshape(1, -1, 1),
                             source_location=src, receiver_location=rec, eps_r=e1, sigma=s1,
                             pmlthick=p)[-1]
        (d1 * r).sum().backward()
        sol = HybridTM2D(eps, sig, dx, pmlthick=p, dt=dt, nshot=2, dtype=torch.float32)
        prop = HybridPropagator(sol, src[:, :, :2].double().numpy() * dx,
                                rec[:, :, :2].double().numpy() * dx)
        e2, s2 = eps.clone().requires_grad_(True), sig.clone().requires_grad_(True)
        d2 = prop(e2, s2, w)
        (d2 * r).sum().backward()
        rel = lambda a, b: float((a - b).norm() / b.norm())  # noqa: E731
        self.assertLess(rel(d2.detach(), d1.detach()), 1e-5)
        self.assertLess(rel(e2.grad, e1.grad), 1e-4)
        self.assertLess(rel(s2.grad, s1.grad), 1e-4)

    def test_full_dgtd_gradient(self):
        nx, ny, dx = 20, 24, 0.02
        eps = torch.full((nx, ny), 3.0, dtype=torch.float64)
        eps[12:15, 10:14] = 6.0
        sig = torch.full((nx, ny), 0.002, dtype=torch.float64)
        surf = Surface.flat(0.16)
        sol = HybridTM2D(eps, sig, dx, surface=surf, dg_band=(None, None), pmlthick=5,
                         mesh="delaunay")
        prop = HybridPropagator(sol, np.array([[[0.12, 0.2]]]), np.array([[[0.12, 0.3], [0.3, 0.4]]]))
        w = ricker(200, sol.dt)
        with torch.no_grad():
            dobs = 1.1 * prop(eps, sig, w[None])
        J = lambda e, s: 0.5 * ((prop(e, s, w[None]) - dobs) ** 2).sum()  # noqa: E731
        e = eps.clone().requires_grad_(True)
        J(e, sig).backward()
        v = torch.zeros_like(eps)
        v[9:16, 4:20] = torch.randn(7, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
        adj = float((e.grad * v).sum())
        hh = 1e-4 * 3.0
        with torch.no_grad():
            fd = float((J(eps + hh * v, sig) - J(eps - hh * v, sig)) / (2 * hh))
        self.assertLess(abs(fd - adj) / abs(adj), 1e-6)


if __name__ == "__main__":
    unittest.main()
