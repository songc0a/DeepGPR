"""Tests of the 2D DGTD-Yee-FDTD hybrid solver (DeepGPR.hybrid).

Fast CPU checks: reference-element operators, the PyTorch Yee mirror against
the native DeepGPR solver, exact discrete energy conservation of the coupled
scheme (P1/P2, structured and Delaunay meshes, non-matching interfaces), the
mortar identities, the coupled CFL limit, and accuracy against pure FDTD.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import DeepGPR  # noqa: E402
from DeepGPR.hybrid import HybridTM2D, Surface  # noqa: E402
from DeepGPR.hybrid.reference import reference_triangle  # noqa: E402

C0 = 299792458.0


def ricker(f, nt, dt, t0, dtype=torch.float64):
    t = torch.arange(nt, dtype=torch.float64) * dt - t0
    a = (math.pi * f * t) ** 2
    return ((1 - 2 * a) * torch.exp(-a)).to(dtype)


def gaussian_burst(nt, dt, f, t0, off_after):
    t = torch.arange(nt, dtype=torch.float64) * dt
    w = torch.exp(-((t - t0) * f * 2) ** 2) * torch.cos(2 * math.pi * f * (t - t0))
    w[off_after:] = 0
    return w


class ReferenceElementTest(unittest.TestCase):
    def test_operators_are_exact_for_polynomials(self):
        for order in (1, 2, 3):
            ref = reference_triangle(order)
            r, s = ref.r, ref.s
            f = r**order + 0.5 * s**order + r * s ** (order - 1)
            dfr = order * r ** (order - 1) + s ** (order - 1)
            dfs = 0.5 * order * s ** (order - 1) + (order - 1) * r * s ** max(order - 2, 0)
            self.assertLess(np.abs(ref.Dr @ f - dfr).max(), 1e-12)
            self.assertLess(np.abs(ref.Ds @ f - dfs).max(), 1e-12)
            self.assertAlmostEqual(ref.M.sum(), 2.0, places=12)
            # Integration by parts: M Dr + (M Dr)^T = boundary term (LIFT form).
            ones = np.ones(ref.np_)
            self.assertAlmostEqual(ones @ ref.M @ (ref.Dr @ r), 2.0, places=12)


class PureYeeModeTest(unittest.TestCase):
    def test_matches_native_deepgpr(self):
        torch.manual_seed(0)
        nx, ny, dx, nt = 40, 50, 0.02, 300
        eps = 1 + 5 * torch.rand(nx, ny)
        sig = 0.01 * torch.rand(nx, ny)
        dt = 0.9 * DeepGPR.max_stable_time_step(dx, nx + 16, ny + 16, 1)
        w = DeepGPR.wavelet.ricker(3e8, nt, dt, 1 / 3e8)
        src = torch.tensor([[[15, 20, 0]]], dtype=torch.int)
        rec = torch.tensor([[[5, 5, 0], [30, 40, 0]]], dtype=torch.int)
        ref = DeepGPR.compute(device="cpu", dx=dx, dt=dt, source_amplitudes=w.reshape(1, -1, 1),
                              source_location=src, receiver_location=rec, eps_r=eps,
                              sigma=sig, pmlthick=8, save_wavefield_history=False)[-1]
        sol = HybridTM2D(eps, sig, dx, pmlthick=8, dt=dt, dtype=torch.float32)
        out = sol.run(w, src[:, :, :2].double() * dx, rec[:, :, :2].double() * dx)
        err = (out["receivers"] - ref).abs().max() / ref.abs().max()
        self.assertLess(float(err), 2e-6)


def _closed_box(order, mesh, opts, surface=None, device="cpu"):
    nx, ny, dx = 50, 60, 0.02
    eps = torch.full((nx, ny), 2.0)
    eps[30:] = 4.0
    sig = torch.zeros(nx, ny)
    return HybridTM2D(eps, sig, dx, surface=surface, dg_band=(0.3, 0.8), pmlthick=0,
                      dg_order=order, dg_h=dx * order, mesh=mesh, mesh_options=opts,
                      flux="central", cfl_safety=0.9, device=device)


class EnergyConservationTest(unittest.TestCase):
    """Central fluxes + mortar coupling: the leapfrog energy is invariant."""

    device = "cpu"

    def _check(self, sol):
        nt = 1500
        w = gaussian_burst(nt, sol.dt, 1e9, 1.5e-9, 300)
        res = sol.run(w, [[[0.25, 0.6]]], [[[0.45, 0.3]]], energy_every=5)
        steps = np.array(res["energy_steps"])
        e = res["energy"][:, 0].cpu().numpy()[steps > 320]
        self.assertGreater(e.mean(), 0)
        self.assertLess((e.max() - e.min()) / e.mean(), 1e-12)

    def test_p1_terrain_jitter_nonmatching(self):
        self._check(_closed_box(1, "terrain", dict(h_y=0.0173, jitter=0.3), device=self.device))

    def test_p2_terrain(self):
        self._check(_closed_box(2, "terrain", dict(h_y=0.037), device=self.device))

    def test_p1_delaunay_nonmatching(self):
        self._check(_closed_box(1, "delaunay", dict(h_interface=0.0231), device=self.device))

    def test_p1_with_topography(self):
        surf = Surface.sine(0.55, 0.08, 0.6)
        self._check(_closed_box(1, "terrain", dict(h_y=0.019), surface=surf, device=self.device))

    def test_full_dgtd(self):
        nx, ny, dx = 30, 30, 0.02
        sol = HybridTM2D(torch.full((nx, ny), 3.0), torch.zeros(nx, ny), dx, pmlthick=0,
                         dg_band=(None, None), dg_order=1, dg_h=dx, mesh="delaunay",
                         flux="central", device=self.device)
        nt = 800
        w = gaussian_burst(nt, sol.dt, 1e9, 1.5e-9, 300)
        res = sol.run(w, [[[0.3, 0.3]]], [[[0.2, 0.2]]], energy_every=5)
        steps = np.array(res["energy_steps"])
        e = res["energy"][:, 0].cpu().numpy()[steps > 320]
        self.assertLess((e.max() - e.min()) / e.mean(), 1e-12)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class EnergyConservationCudaTest(EnergyConservationTest):
    """The same invariants through the fused 2D CUDA kernels."""

    device = "cuda"


class MortarTest(unittest.TestCase):
    def test_partition_of_unity_and_adjointness(self):
        sol = _closed_box(2, "delaunay", dict(h_interface=0.0231))
        for row, cp in sol.couplings.values():
            ny = sol.NY
            ones = torch.ones((1, ny), dtype=torch.float64)
            # A constant FDTD trace projects to a constant on every DG face.
            self.assertLess(float((cp.project_to_dg(ones) - 1).abs().max()), 1e-12)
            # Hy^T C E = dy E^T H_b for random data (the energy cancellation).
            e = torch.randn((1, ny), dtype=torch.float64)
            hy = torch.randn((1, sol.dg.K * sol.dg.Np), dtype=torch.float64)
            lhs = (hy[:, cp.nodes] * cp.C(e)).sum()
            rhs = sol.dy * (e * cp.trace_to_fd(hy)).sum()
            self.assertLess(abs(float(lhs - rhs)), 1e-12 * float(lhs.abs() + rhs.abs()))


class CFLTest(unittest.TestCase):
    def test_lanczos_limit_is_sharp(self):
        """Stable just below 2/sqrt(lambda_max), unstable just above."""
        nx, ny, dx = 30, 30, 0.02
        eps = torch.full((nx, ny), 1.0)
        base = HybridTM2D(eps, torch.zeros(nx, ny), dx, dg_band=(0.2, 0.4), pmlthick=0,
                          dg_order=1, dg_h=0.7 * dx, flux="central")
        dt_max = 2 / math.sqrt(base.lambda_max)
        growth = {}
        for factor in (0.98, 1.03):
            sol = HybridTM2D(eps, torch.zeros(nx, ny), dx, dg_band=(0.2, 0.4), pmlthick=0,
                             dg_order=1, dg_h=0.7 * dx, dt=factor * dt_max, flux="central")
            torch.manual_seed(1)
            sol.dg.E = torch.randn_like(sol.dg.E)
            sol.fd.Ez[:, 2:-2, 2:-2] = torch.randn_like(sol.fd.Ez[:, 2:-2, 2:-2]) * (
                sol.fd.node_weights()[2:-2, None] > 0)
            for _ in range(3000):
                sol.step()
            growth[factor] = float(sol.dg.E.abs().max())
        self.assertLess(growth[0.98], 1e2)
        # Above the limit the solution grows without bound (it may overflow).
        self.assertFalse(math.isfinite(growth[1.03]) and growth[1.03] < 1e6)


class AccuracyTest(unittest.TestCase):
    """Hybrid and pure Yee against a 4x finer Yee reference (homogeneous medium)."""

    @classmethod
    def setUpClass(cls):
        cls.T = 8e-9

        def model(dx, **kw):
            nx, ny = int(round(1.0 / dx)), int(round(1.2 / dx))
            eps = torch.full((nx, ny), 2.0)
            sig = torch.full((nx, ny), 0.002)
            return HybridTM2D(eps, sig, dx, pmlthick=int(round(0.15 / dx)), **kw)

        cls.fine = model(0.0025)
        cls.fd = model(0.01)
        cls.hy = model(0.01, dg_band=(0.3, 0.6), dg_order=1, dg_h=0.01,
                       mesh_options=dict(h_y=0.009))

    def _trace(self, sol, src, rec):
        nt = int(self.T / sol.dt)
        d = sol.run(ricker(1e9, nt, sol.dt, 1.2e-9), src, rec)["receivers"][0].numpy()
        return np.arange(nt) * sol.dt, d

    def _errors(self, src, rec):
        tf, ref = self._trace(self.fine, src, rec)
        out = []
        for sol in (self.fd, self.hy):
            t, d = self._trace(sol, src, rec)
            r = np.stack([np.interp(t, tf, ref[:, k]) for k in range(ref.shape[1])], 1)
            out.append(np.linalg.norm(d - r) / np.linalg.norm(r))
        return out

    def test_fd_to_dg_to_fd(self):
        # Source below the band, receivers above it: the wave crosses both interfaces.
        err_fd, err_hy = self._errors([[[0.75, 0.6]]], [[[0.15, 0.6], [0.15, 0.9]]])
        self.assertLess(err_hy, 0.04)
        self.assertLess(err_hy, err_fd)

    def test_dg_source(self):
        # Source inside the band, receivers in both Yee regions and in the band.
        err_fd, err_hy = self._errors([[[0.45, 0.55]]], [[[0.8, 0.55], [0.1, 0.8], [0.4, 0.3]]])
        self.assertLess(err_hy, 0.04)
        self.assertLess(err_hy, 1.5 * err_fd)


if __name__ == "__main__":
    unittest.main()
