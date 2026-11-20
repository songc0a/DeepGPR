"""Local time stepping (LTS) of the hybrid DGTD-FDTD scheme (2D and 3D).

* with one sub-step the LTS code path reproduces the plain step bitwise;
* the three-level invariant of Diaz & Grote's leapfrog LTS,
  ``(|e^n|^2 - <e^(n+1), e^(n-1)>) / (2 dt^2)`` in the ``M_e`` inner product, is
  conserved to round-off for central fluxes without losses (p = 1, 2, 3);
* long runs with random initial data at the automatic schedule stay bounded;
* LTS and the global leapfrog agree to the time-discretisation error on a smooth
  problem (absorbers, penalties and conductivity on);
* the discrete adjoint of the LTS step (autograd transpose) agrees with central
  finite differences;
* the hybrid-with-subgrids step uses the same LTS (no subgrids: equal to the
  hybrid; a sub-cycled subgrid under the band runs with it);
* on CUDA: fused kernels against the PyTorch step, graph replay against eager,
  the hand-written transpose is bypassed.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from DeepGPR.hybrid import HybridMaxwell3D, HybridPropagator, HybridTM2D, Surface, Surface3D  # noqa: E402
from DeepGPR.hybrid import functional as fn  # noqa: E402
from DeepGPR.hybrid import functional3d as fn3  # noqa: E402

CUDA = torch.cuda.is_available()


def ricker(nt, dt, f0, t0=None):
    t0 = 1.0 / f0 if t0 is None else t0
    t = (np.arange(nt) + 0.5) * dt - t0
    a = (np.pi * f0 * t) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a))


def interp_rows(t_to, t_from, d):
    return np.stack([np.interp(t_to, t_from, d[:, k]) for k in range(d.shape[1])], 1)


# --------------------------------------------------------------------------- 2D
class LTS2DTest(unittest.TestCase):
    device = "cpu"
    cuda_kernels = True

    @classmethod
    def setUpClass(cls):
        nx, ny = 30, 40
        cls.dx = 0.02
        cls.eps = torch.full((nx, ny), 4.0, dtype=torch.float64)
        cls.eps[:12] = 1.0
        cls.eps[20:24, 15:20] = 7.0
        cls.sig = torch.full((nx, ny), 0.003, dtype=torch.float64)
        cls.surf = Surface.sine(0.24, 0.04, 0.5)

    def make(self, sig=None, **kw):
        kw.setdefault("mesh", "terrain")
        kw.setdefault("mesh_options", dict(h_y=0.019))
        kw.setdefault("dg_band", (0.15, 0.4))
        kw.setdefault("flux", "central")          # LTS needs central fluxes in the band
        return HybridTM2D(self.eps, self.sig if sig is None else sig, self.dx, surface=self.surf,
                          dg_order=1, dg_h=self.dx, device=self.device,
                          cuda_kernels=self.cuda_kernels, **kw)

    def conservative(self, **kw):
        return self.make(sig=torch.zeros_like(self.sig), pmlthick=0, **kw)

    def test_upwind_penalty_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make(flux=0.1, lts=3)
        self.make(flux=0.1)                       # the global scheme keeps the penalty

    def random_state(self, sol, seed=1):
        gen = torch.Generator().manual_seed(seed)
        st = list(fn.zero_state(sol.static))
        y = sol.static.yee
        act = (sol._node_weight_grid(sol.fd) > 0).to(self.device)
        act[:, 0] = act[:, -1] = act[0] = act[-1] = False
        rnd = lambda shape: torch.randn(shape, generator=gen, dtype=torch.float64).to(self.device)  # noqa: E731
        st[0] = rnd(st[0].shape) * act
        st[1] = rnd(st[1].shape) * (y.cHx != 0) / 377.0
        st[2] = rnd(st[2].shape) * (y.cHy != 0) / 377.0
        st[7] = rnd(st[7].shape)
        st[8] = rnd(st[8].shape) / 377.0
        st[9] = rnd(st[9].shape) / 377.0
        return tuple(st)

    def invariant(self, sol, state, steps, every=40):
        S, c = sol.static, sol.coefs
        ip = lambda a, b: float(sol._e_inner(sol.fd, sol.dg, [(a[0],), (a[7],)],  # noqa: E731
                                             [(b[0],), (b[7],)]))
        prev, vals = None, []
        for n in range(steps):
            nxt = fn.step(S, c, state)
            if prev is not None and n % every == 0:
                vals.append((ip(state, state) - ip(nxt, prev)) / (2 * sol.dt ** 2))
            prev, state = state, nxt
        v = np.array(vals)
        return (v.max() - v.min()) / abs(v.mean()), state

    def test_one_substep_matches_plain_step(self):
        a, b = self.conservative(), self.conservative(lts=1)
        self.assertEqual(b.lts, 1)
        st = self.random_state(a)
        ua, ub = fn.step(a.static, a.coefs, st), fn.step(b.static, b.coefs, st)
        diff = max(float((x - y).abs().max() / (1e-300 + y.abs().max())) for x, y in zip(ua, ub)
                   if x.numel())
        # the CUDA step is reproducible only to round-off (index_add atomics)
        self.assertLessEqual(diff, 1e-12 if self.device == "cuda" else 0.0)

    def test_schedule(self):
        sol = self.make(lts="auto")
        self.assertGreater(sol.lts, 1)
        self.assertAlmostEqual(sol.dt, sol.lts_cfl * sol.yee_dt_limit())
        self.assertLessEqual(sol.dt_fine, self.make().dt * (1 + 1e-12))
        self.assertEqual(sol.static.dg.lts, sol.lts)
        self.assertEqual(self.make(lts=2).lts, 2)
        self.assertEqual(self.make(dg_band=None, lts="auto").lts, 1)   # pure Yee: no LTS

    def test_modified_energy_invariant(self):
        for p in (1, 2, 3):
            sol = self.conservative(lts=p)
            var, state = self.invariant(sol, self.random_state(sol), 240)
            with self.subTest(p=p):
                self.assertLess(var, 1e-12)
                self.assertTrue(bool(torch.isfinite(state[7]).all()))

    def test_long_run_is_bounded(self):
        sol = self.conservative(lts="auto")
        self.assertGreater(sol.lts, 1)
        state = self.random_state(sol)
        S, c = sol.static, sol.coefs
        peaks = []
        for n in range(2000):
            state = fn.step(S, c, state)
            if n % 250 == 249:
                peaks.append(float(state[7].abs().max()))
        self.assertTrue(all(math.isfinite(v) for v in peaks))
        self.assertLess(max(peaks[4:]), 3.0 * max(peaks[:4]))

    def test_matches_global_leapfrog(self):
        src, rec = [[[0.1, 0.4]]], [[[0.5, 0.3], [0.2, 0.6], [0.3, 0.9]]]
        glob = self.make()
        nt = int(12e-9 / glob.dt)
        dg = glob.run(ricker(nt, glob.dt, 0.9e9, 2.5e-9), src, rec)["receivers"][0].cpu().numpy()
        tg = (np.arange(nt) + 1) * glob.dt
        for lts in ("auto", 2):
            sol = self.make(lts=lts)
            nt_l = int(12e-9 / sol.dt)
            dl = sol.run(ricker(nt_l, sol.dt, 0.9e9, 2.5e-9), src, rec)["receivers"][0].cpu().numpy()
            tl = (np.arange(nt_l) + 1) * sol.dt
            di = interp_rows(tl, tg, dg)
            err = np.linalg.norm(di - dl) / np.linalg.norm(di)
            with self.subTest(lts=lts, p=sol.lts):
                self.assertLess(err, 0.05)

    def test_gradient_finite_differences(self):
        sol = self.make(lts=3, pmlthick=6, fdtd_refinement=1)
        src = np.array([[[0.18, 0.3]]])
        rec = np.array([[[0.18, 0.5], [0.45, 0.2], [0.05, 0.6]]])
        prop = HybridPropagator(sol, src, rec, checkpoint_every=11)
        w = ricker(90, sol.dt, 1e9, 1e-9).to(self.device)
        eps, sig = self.eps.to(self.device), self.sig.to(self.device)
        with torch.no_grad():
            dobs = 0.9 * prop(eps, sig, w[None])
        J = lambda e, s: 0.5 * ((prop(e, s, w[None]) - dobs) ** 2).sum()  # noqa: E731
        e = eps.clone().requires_grad_(True)
        s = sig.clone().requires_grad_(True)
        J(e, s).backward()
        gen = torch.Generator().manual_seed(3)
        v = torch.zeros(self.eps.shape, dtype=torch.float64)
        v[13:27, 5:35] = torch.randn((14, 30), generator=gen, dtype=torch.float64)
        v = v.to(self.device)
        for par, g, scale in (("eps", e.grad, 4.0), ("sigma", s.grad, 0.003)):
            adj = float((g * v).sum())
            errs = []
            for h in (1e-3, 1e-4, 1e-5):
                hh = h * scale
                with torch.no_grad():
                    if par == "eps":
                        fd = (J(eps + hh * v, sig) - J(eps - hh * v, sig)) / (2 * hh)
                    else:
                        fd = (J(eps, sig + hh * v) - J(eps, sig - hh * v)) / (2 * hh)
                errs.append(abs(float(fd) - adj) / max(abs(adj), abs(float(fd))))
            with self.subTest(parameter=par):
                self.assertLess(min(errs), 1e-7)


@unittest.skipUnless(CUDA, "CUDA not available")
class LTS2DCudaTest(LTS2DTest):
    device = "cuda"

    def test_fused_step_matches_pytorch_step(self):
        a = self.make(lts=3)
        b = HybridTM2D(self.eps, self.sig, self.dx, surface=self.surf, dg_band=(0.15, 0.4),
                       dg_order=1, dg_h=self.dx, device=self.device, cuda_kernels=False, lts=3,
                       dt=a.dt, mesh="terrain", mesh_options=dict(h_y=0.019), flux="central")
        st = self.random_state(a)
        ua, ub = fn.step(a.static, a.coefs, st), fn.step(b.static, b.coefs, st)
        for x, y in zip(ua, ub):
            if x.numel():
                self.assertLess(float((x - y).abs().max() / (1e-300 + y.abs().max())), 1e-12)

    def test_fused_step_transpose_matches_pytorch_step(self):
        a = self.make(lts=3)
        self.cuda_kernels = False
        try:
            b = self.make(lts=3, dt=a.dt)
        finally:
            self.cuda_kernels = True
        st = self.random_state(a)
        assert_close_lists(self, step_vjp(fn, a, st), step_vjp(fn, b, st), 1e-9)

    def test_hand_written_transpose_is_bypassed(self):
        from DeepGPR.hybrid import vjp2d
        self.assertFalse(vjp2d.supported(self.make(lts=3).static))
        self.assertTrue(vjp2d.supported(self.make().static))

    def test_graph_replay_matches_eager(self):
        from DeepGPR.hybrid import graphs
        sol = self.make(lts=3)
        src = sol.point_ops(torch.tensor([[[0.1, 0.4]]], dtype=torch.float64), source=True)
        rec = sol.point_ops(torch.tensor([[[0.5, 0.3]]], dtype=torch.float64), source=False)
        g = graphs.make(sol.static, sol.coefs, src, rec, 1)
        self.assertIsNotNone(g)
        state = self.random_state(sol)
        g.set_state(state)
        amp = torch.tensor([0.7], dtype=torch.float64, device=self.device)
        for _ in range(3):
            g(amp)
            state = fn.step(sol.static, sol.coefs, state, src, amp)
        for a, b in zip(g.state, state):
            if a.numel():
                self.assertLess(float((a - b).abs().max() / (1e-300 + b.abs().max())), 1e-12)


def step_vjp(F, sol, state, seed=3):
    """Gradients of a random linear functional of one step with respect to the state
    and the coefficient tensors (autograd through ``F.step``)."""
    st = sol.static
    gen = torch.Generator().manual_seed(seed)
    u = [t.detach().clone() for t in state]
    active = [i for i, t in enumerate(u) if t.numel() and t.is_floating_point()]
    for i in active:
        u[i].requires_grad_(True)
    leaves = [t.detach().clone().requires_grad_(True) for t in sol.coefs.tensors()]
    out = F.step(st, sol.coefs.replaced(leaves), tuple(u))
    outs = [out[i] for i in active]
    cots = [torch.randn(o.shape, generator=gen, dtype=o.dtype).to(o.device) for o in outs]
    grads = torch.autograd.grad(outs, [u[i] for i in active] + leaves, cots, allow_unused=True)
    return [torch.zeros(1, dtype=torch.float64) if g is None else g.detach() for g in grads]


def assert_close_lists(tc, ga, gb, tol):
    for k, (x, y) in enumerate(zip(ga, gb)):
        scale = float(y.abs().max()) + float(x.abs().max())
        if scale == 0.0:
            continue
        tc.assertLess(float((x - y).abs().max()) / scale, tol, f"gradient {k}")


# --------------------------------------------------------------------------- 3D
def hybrid3d(device="cpu", **kw):
    nx, ny, nz, dx = 14, 12, 11, 0.02
    eps = 3 + 3 * torch.rand(nx, ny, nz, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    sig = kw.pop("sig", torch.zeros(nx, ny, nz, dtype=torch.float64))
    surf = Surface3D.sine(0.14, 0.02, 0.24, 0.2)
    kw.setdefault("flux", "central")              # LTS needs central fluxes in the band
    return HybridMaxwell3D(eps, sig, dx, surface=surf, dg_band=(0.06, 0.22), device=device, **kw)


def random_state3d(st, gen, device="cpu"):
    s = list(fn3.zero_state(st))
    for k in range(6):
        s[k] = torch.randn(s[k].shape, generator=gen, dtype=torch.float64).to(device) * st.yee.act[k]
    s[18] = torch.randn(s[18].shape, generator=gen, dtype=torch.float64).to(device)
    s[19] = torch.randn(s[19].shape, generator=gen, dtype=torch.float64).to(device) / 377.0
    return tuple(s)


class LTS3DTest(unittest.TestCase):
    device = "cpu"

    def conservative(self, **kw):
        return hybrid3d(self.device, pmlthick=0, interface_penalty=0.0, **kw)

    def invariant(self, sol, state, steps, every=20):
        st, c = sol.static, sol.coefs
        fields = sol._fields_of_model(sigma_zero=True)
        eps_rel = (fields[0], fields[2])
        e_of = lambda u: [u[0], u[1], u[2], u[18]]  # noqa: E731
        ip = lambda a, b: float(sol._e_inner(st, eps_rel, e_of(a), e_of(b)))  # noqa: E731
        prev, vals = None, []
        for n in range(steps):
            nxt = fn3.step(st, c, state)
            if prev is not None and n % every == 0:
                vals.append((ip(state, state) - ip(nxt, prev)) / (2 * sol.dt ** 2))
            prev, state = state, nxt
        v = np.array(vals)
        return (v.max() - v.min()) / abs(v.mean()), state

    def test_one_substep_matches_plain_step(self):
        a, b = self.conservative(), self.conservative(lts=1)
        st = random_state3d(a.static, torch.Generator().manual_seed(1), self.device)
        ua, ub = fn3.step(a.static, a.coefs, st), fn3.step(b.static, b.coefs, st)
        diff = max(float((x - y).abs().max() / (1e-300 + y.abs().max())) for x, y in zip(ua, ub)
                   if x.numel())
        self.assertLessEqual(diff, 1e-12 if self.device == "cuda" else 0.0)

    def test_schedule_and_summary(self):
        sol = hybrid3d(self.device, lts="auto")
        self.assertGreater(sol.lts, 1)
        self.assertAlmostEqual(sol.dt, sol.lts_cfl * sol.yee_dt_limit())
        info = sol.summary()
        self.assertEqual(info["lts"], sol.lts)
        self.assertAlmostEqual(info["dt_fine"], sol.dt / sol.lts)

    def test_modified_energy_invariant(self):
        for p, ov in ((1, 1), (2, 1), (3, 0), (3, 1)):
            sol = self.conservative(lts=p, lts_overlap=ov)
            state = random_state3d(sol.static, torch.Generator().manual_seed(1), self.device)
            var, state = self.invariant(sol, state, 120)
            with self.subTest(p=p, overlap=ov):
                self.assertLess(var, 1e-12)
                self.assertTrue(bool(torch.isfinite(state[18]).all()))

    def test_long_run_is_bounded(self):
        sol = self.conservative(lts="auto")
        self.assertGreater(sol.lts, 1)
        state = random_state3d(sol.static, torch.Generator().manual_seed(1), self.device)
        st, c = sol.static, sol.coefs
        peaks = []
        for n in range(1200):
            state = fn3.step(st, c, state)
            if n % 150 == 149:
                peaks.append(float(state[18].abs().max()))
        self.assertTrue(all(math.isfinite(v) for v in peaks))
        self.assertLess(max(peaks[4:]), 3.0 * max(peaks[:4]))

    def test_matches_global_leapfrog(self):
        src = [[[0.03, 0.12, 0.1]]]
        rec = [[[0.25, 0.1, 0.1], [0.12, 0.14, 0.08], [0.05, 0.05, 0.15]]]
        sig = torch.full((14, 12, 11), 0.003, dtype=torch.float64)
        glob = hybrid3d(self.device, sig=sig)
        nt = int(6e-9 / glob.dt)
        dg = glob.run(ricker(nt, glob.dt, 0.8e9, 2e-9), src, rec)["receivers"][0].cpu().numpy()
        tg = (np.arange(nt) + 1) * glob.dt
        sol = hybrid3d(self.device, sig=sig, lts="auto")
        self.assertGreater(sol.lts, 1)
        nt_l = int(6e-9 / sol.dt)
        dl = sol.run(ricker(nt_l, sol.dt, 0.8e9, 2e-9), src, rec)["receivers"][0].cpu().numpy()
        tl = (np.arange(nt_l) + 1) * sol.dt
        di = interp_rows(tl, tg, dg)
        self.assertLess(np.linalg.norm(di - dl) / np.linalg.norm(di), 0.05)

    def test_gradient_finite_differences(self):
        shape = (10, 11, 11)
        eps = torch.full(shape, 4.0, dtype=torch.float64)
        eps[7:9, 4:7, 4:7] = 6.0
        sig = torch.full(shape, 0.004, dtype=torch.float64)
        surf = Surface3D.sine(0.1, 0.015, 0.2, 0.2)
        sol = HybridMaxwell3D(eps, sig, 0.02, surface=surf, dg_band=(0.06, 0.14), pmlthick=3,
                              device=self.device, lts=3, flux="central")
        self.assertEqual(sol.static.dg.lts, 3)
        prop = HybridPropagator(sol, np.array([[[0.04, 0.10, 0.11]]]),
                                np.array([[[0.04, 0.16, 0.11], [0.18, 0.08, 0.13]]]),
                                checkpoint_every=9)
        w = ricker(70, sol.dt, 2e9, 0.6e-9).to(self.device)
        eps, sig = eps.to(self.device), sig.to(self.device)
        with torch.no_grad():
            dobs = 0.9 * prop(eps, sig, w[None])
        J = lambda e, s: 0.5 * ((prop(e, s, w[None]) - dobs) ** 2).sum()  # noqa: E731
        e = eps.clone().requires_grad_(True)
        s = sig.clone().requires_grad_(True)
        J(e, s).backward()
        gen = torch.Generator().manual_seed(4)
        for par, g, scale in (("eps", e.grad, 4.0), ("sigma", s.grad, 0.004)):
            v = torch.zeros(shape, dtype=torch.float64)
            v[4:10, 2:9, 2:9] = torch.randn((6, 7, 7), generator=gen, dtype=torch.float64)
            v = v.to(self.device)
            adj = float((g * v).sum())
            errs = []
            for h in (1e-3, 1e-4, 1e-5):
                hh = h * scale
                with torch.no_grad():
                    if par == "eps":
                        fd = (J(eps + hh * v, sig) - J(eps - hh * v, sig)) / (2 * hh)
                    else:
                        fd = (J(eps, sig + hh * v) - J(eps, sig - hh * v)) / (2 * hh)
                errs.append(abs(float(fd) - adj) / abs(adj))
            with self.subTest(parameter=par):
                self.assertLess(min(errs), 1e-6)


    def test_cost_schedule_picks_the_cheapest(self):
        """lts="cost": the chosen sub-steps minimise the modelled cost per simulated second
        over p = 1 (global stepping, explicit penalty) .. p_auto."""
        for flux in ("central", 0.1):
            sol = hybrid3d(self.device, lts="cost", flux=flux)
            table = sol.lts_cost_table
            self.assertEqual(table[0][0], 1)
            best = min(table, key=lambda r: r[2])
            self.assertEqual(sol.lts, best[0])
            self.assertAlmostEqual(sol.dt, best[1])
            self.assertEqual(sol.summary()["lts_cost_table"], table)

    # ---------------------------------------------------------------- split penalty
    def test_split_penalty_long_run_decays(self):
        """Penalty flux with LTS (split damping): random data in a closed PEC box decay
        for p = 3 (the explicit penalty inside or once per macro step was unstable)."""
        sol = hybrid3d(self.device, pmlthick=0, interface_penalty=0.0, flux=0.1, lts=3)
        self.assertTrue(sol.static.dg.lts_split)
        st, c = sol.static, sol.coefs
        state = random_state3d(st, torch.Generator().manual_seed(1), self.device)
        peaks = []
        with torch.no_grad():
            for n in range(900):
                state = fn3.step(st, c, state)
                if n % 150 == 149:
                    peaks.append(float(state[18].abs().max()))
        self.assertTrue(all(math.isfinite(v) for v in peaks))
        self.assertLess(peaks[-1], peaks[0])

    def test_split_penalty_matches_global_penalty(self):
        src = [[[0.03, 0.12, 0.1]]]
        rec = [[[0.25, 0.1, 0.1], [0.12, 0.14, 0.08], [0.05, 0.05, 0.15]]]
        sig = torch.full((14, 12, 11), 0.003, dtype=torch.float64)
        glob = hybrid3d(self.device, sig=sig, flux=0.1)
        nt = int(6e-9 / glob.dt)
        dg = glob.run(ricker(nt, glob.dt, 0.8e9, 2e-9), src, rec)["receivers"][0].cpu().numpy()
        tg = (np.arange(nt) + 1) * glob.dt
        sol = hybrid3d(self.device, sig=sig, flux=0.1, lts="auto")
        self.assertGreater(sol.lts, 1)
        nt_l = int(6e-9 / sol.dt)
        dl = sol.run(ricker(nt_l, sol.dt, 0.8e9, 2e-9), src, rec)["receivers"][0].cpu().numpy()
        tl = (np.arange(nt_l) + 1) * sol.dt
        di = interp_rows(tl, tg, dg)
        self.assertLess(np.linalg.norm(di - dl) / np.linalg.norm(di), 0.05)

    def test_split_penalty_gradient_finite_differences(self):
        """Exact discrete adjoint of the LTS step with the split penalty (the penalty
        coefficients depend on eps through the impedances)."""
        shape = (10, 11, 11)
        eps = torch.full(shape, 4.0, dtype=torch.float64)
        eps[7:9, 4:7, 4:7] = 6.0
        eps[3:5, 3:8, 3:8] = 5.0                  # inside the band
        sig = torch.full(shape, 0.004, dtype=torch.float64)
        surf = Surface3D.sine(0.1, 0.015, 0.2, 0.2)
        sol = HybridMaxwell3D(eps, sig, 0.02, surface=surf, dg_band=(0.06, 0.14), pmlthick=3,
                              device=self.device, lts=3, flux=0.1)
        self.assertTrue(sol.static.dg.lts_split)
        prop = HybridPropagator(sol, np.array([[[0.04, 0.10, 0.11]]]),
                                np.array([[[0.04, 0.16, 0.11], [0.18, 0.08, 0.13]]]),
                                checkpoint_every=9)
        w = ricker(70, sol.dt, 2e9, 0.6e-9).to(self.device)
        eps, sig = eps.to(self.device), sig.to(self.device)
        with torch.no_grad():
            dobs = 0.9 * prop(eps, sig, w[None])
        J = lambda e, s: 0.5 * ((prop(e, s, w[None]) - dobs) ** 2).sum()  # noqa: E731
        e = eps.clone().requires_grad_(True)
        s = sig.clone().requires_grad_(True)
        J(e, s).backward()
        gen = torch.Generator().manual_seed(5)
        for par, g, scale in (("eps", e.grad, 4.0), ("sigma", s.grad, 0.004)):
            v = torch.zeros(shape, dtype=torch.float64)
            v[2:10, 2:9, 2:9] = torch.randn((8, 7, 7), generator=gen, dtype=torch.float64)
            v = v.to(self.device)
            adj = float((g * v).sum())
            errs = []
            for h in (1e-3, 1e-4, 1e-5):
                hh = h * scale
                with torch.no_grad():
                    if par == "eps":
                        fd = (J(eps + hh * v, sig) - J(eps - hh * v, sig)) / (2 * hh)
                    else:
                        fd = (J(eps, sig + hh * v) - J(eps, sig - hh * v)) / (2 * hh)
                errs.append(abs(float(fd) - adj) / abs(adj))
            with self.subTest(parameter=par):
                self.assertLess(min(errs), 1e-6)


@unittest.skipUnless(CUDA, "CUDA not available")
class LTS3DCudaTest(LTS3DTest):
    device = "cuda"

    def test_fused_step_matches_pytorch_step(self):
        a = hybrid3d("cuda", lts=3)
        b = hybrid3d("cuda", lts=3, dt=a.dt, cuda_kernels=False)
        st = random_state3d(a.static, torch.Generator().manual_seed(1), "cuda")
        ua, ub = fn3.step(a.static, a.coefs, st), fn3.step(b.static, b.coefs, st)
        for x, y in zip(ua, ub):
            if x.numel():
                self.assertLess(float((x - y).abs().max() / (1e-300 + y.abs().max())), 1e-11)

    def test_hand_written_lts_transpose_matches_autograd(self):
        """vjp3d_lts.step_vjp (recompute + transposes of the fused LTS step) against autograd
        through the PyTorch step: state, coefficient and source cotangents."""
        from DeepGPR.hybrid import vjp3d_lts
        sig = torch.full((14, 12, 11), 0.003, dtype=torch.float64)
        for kw in (dict(lts=3), dict(lts=3, flux=0.1), dict(lts=2, flux=1.0)):
            a = hybrid3d("cuda", sig=sig, **kw)
            b = hybrid3d("cuda", sig=sig, dt=a.dt, cuda_kernels=False, **kw)
            self.assertTrue(vjp3d_lts.supported(a.static))
            gen = torch.Generator().manual_seed(5)
            u = list(random_state3d(a.static, gen, "cuda"))
            for i in (20, 21, 22, 23):
                if u[i].numel():
                    u[i] = torch.randn(u[i].shape, generator=gen, dtype=torch.float64).to("cuda") * 0.1
            u = tuple(u)
            pos = np.array([[[0.03, 0.12, 0.1]]])
            src_a, src_b = a.point_ops(pos, source=True), b.point_ops(pos, source=True)
            amp = torch.tensor([0.7], dtype=torch.float64, device="cuda")
            un = fn3.step(a.static, a.coefs, u, src_a, amp)
            lam = [torch.randn(t.shape, generator=gen, dtype=torch.float64).to("cuda") if t.numel() else t
                   for t in un]
            acc = {}
            out, g_amp = vjp3d_lts.step_vjp(a.static, a.coefs, u, un, lam, src_a, amp, acc, True, True)
            gc = vjp3d_lts.finalize(acc, a.coefs)
            act = [i for i, t in enumerate(u) if t.numel()]
            ul = list(u)
            for i in act:
                ul[i] = u[i].detach().clone().requires_grad_(True)
            leaves = [t.detach().clone().requires_grad_(True) for t in b.coefs.tensors()]
            al = amp.detach().clone().requires_grad_(True)
            nxt = fn3.step(b.static, b.coefs.replaced(leaves), tuple(ul), src_b, al)
            grads = torch.autograd.grad([nxt[i] for i in act], [ul[i] for i in act] + leaves + [al],
                                        [lam[i] for i in act], allow_unused=True)
            refs = [(out[i], grads[k]) for k, i in enumerate(act)]
            names = [n for n in b.coefs.NAMES if getattr(b.coefs, n) is not None]
            refs += [(gc.get(n), grads[len(act) + j]) for j, n in enumerate(names)]
            refs.append((g_amp, grads[-1]))
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                for got, ref in refs:
                    if ref is None or float(ref.abs().max()) == 0.0:
                        continue
                    got = torch.zeros_like(ref) if got is None else got
                    self.assertLess(float((got - ref).abs().max() / ref.abs().max()), 1e-8)

    def test_split_penalty_fused_matches_pytorch(self):
        a = hybrid3d("cuda", lts=3, flux=0.1)
        b = hybrid3d("cuda", lts=3, flux=0.1, dt=a.dt, cuda_kernels=False)
        self.assertEqual(a.static.dg.lts_pen_k, b.static.dg.lts_pen_k)
        st = random_state3d(a.static, torch.Generator().manual_seed(3), "cuda")
        ua, ub = fn3.step(a.static, a.coefs, st), fn3.step(b.static, b.coefs, st)
        for x, y in zip(ua, ub):
            if x.numel():
                self.assertLess(float((x - y).abs().max() / (1e-300 + y.abs().max())), 1e-11)
        assert_close_lists(self, step_vjp(fn3, a, st), step_vjp(fn3, b, st), 1e-9)

    def test_fused_step_transpose_matches_pytorch_step(self):
        a = hybrid3d("cuda", lts=3, sig=torch.full((14, 12, 11), 0.003, dtype=torch.float64))
        b = hybrid3d("cuda", lts=3, dt=a.dt, cuda_kernels=False,
                     sig=torch.full((14, 12, 11), 0.003, dtype=torch.float64))
        st = random_state3d(a.static, torch.Generator().manual_seed(2), "cuda")
        assert_close_lists(self, step_vjp(fn3, a, st), step_vjp(fn3, b, st), 1e-9)

    def test_hand_written_transpose_is_bypassed(self):
        from DeepGPR.hybrid import vjp3d
        self.assertFalse(vjp3d.supported(hybrid3d("cuda", lts=3).static))

    def test_graph_replay_matches_eager(self):
        from DeepGPR.hybrid import graphs
        sol = hybrid3d("cuda", lts=3)
        src = sol.point_ops(np.array([[[0.03, 0.12, 0.1]]]), source=True)
        rec = sol.point_ops(np.array([[[0.25, 0.1, 0.1]]]), source=False)
        g = graphs.make(sol.static, sol.coefs, src, rec, 1)
        self.assertIsNotNone(g)
        state = random_state3d(sol.static, torch.Generator().manual_seed(1), "cuda")
        g.set_state(state)
        amp = torch.tensor([0.7], dtype=torch.float64, device="cuda")
        for _ in range(3):
            g(amp)
            state = fn3.step(sol.static, sol.coefs, state, src, amp)
        for a, b in zip(g.state, state):
            if a.numel():
                self.assertLess(float((a - b).abs().max() / (1e-300 + b.abs().max())), 1e-11)


class LTSSubgridHybridTest(unittest.TestCase):
    """The hybrid-with-subgrids step shares the LTS half step."""

    def test_without_subgrids_equals_hybrid_with_lts(self):
        from DeepGPR.subgrid.hybrid import HybridSubgridMaxwell3D
        from DeepGPR.subgrid.solver import block_average
        shape = (31, 25, 25)
        g = torch.Generator().manual_seed(0)
        eps = 4.0 + torch.rand(shape, generator=g, dtype=torch.float64)
        sig = 0.002 + 0.002 * torch.rand(shape, generator=g, dtype=torch.float64)
        surf = Surface3D.sine(0.09, 0.02, 0.3, 0.3)
        kw = dict(surface=surf, dg_band=(0.06, 0.15), pmlthick=3, mesh_options=dict(h_y=0.03, h_z=0.03),
                  flux="central")
        a = HybridSubgridMaxwell3D(eps, sig, 0.01, coarsen=3, lts="auto", **kw)
        self.assertGreater(a.hyb.lts, 1)
        w = ricker(60, a.dt, 8e8)
        src, rec = [[0.03, 0.12, 0.12]], [[0.03, 0.18, 0.12], [0.25, 0.12, 0.12]]
        da = a.run(w, src, rec)["receivers"]
        b = HybridMaxwell3D(block_average(eps, 3), block_average(sig, 3), 0.03, dt=a.dt,
                            lts=a.hyb.lts, **kw)
        db = b.run(w, src, rec)["receivers"]
        self.assertLess(float((da - db).norm() / db.norm()), 1e-12)


if __name__ == "__main__":
    unittest.main()
