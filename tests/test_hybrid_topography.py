"""Phase 2 tests of DeepGPR.hybrid: topography input, body-fitted meshes,
material mapping, point sources/receivers on element boundaries, the DG CPML
and long-run stability of full DGTD."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from DeepGPR.hybrid import HybridTM2D, Surface  # noqa: E402
from DeepGPR.hybrid.mesh import (  # noqa: E402
    AIR, GROUND, delaunay_band_mesh, signed_areas, terrain_following_band_mesh,
)


def ricker(nt, dt, f, t0):
    t = (np.arange(nt) + 0.5) * dt - t0
    a = (np.pi * f * t) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a))


class SurfaceTest(unittest.TestCase):
    def test_elevation_interpolation(self):
        y = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        z = np.array([0.0, 0.1, 0.4, 0.45, 0.2, 0.2])
        s = Surface.from_elevation(y, z, datum=1.0)
        np.testing.assert_allclose(s(y), 1.0 - z, atol=1e-14)
        # Monotone cubic: no overshoot between samples.
        fine = np.linspace(0, 2.5, 501)
        d = s(fine)
        for a, b, za, zb in zip(y[:-1], y[1:], z[:-1], z[1:]):
            seg = d[(fine >= a) & (fine <= b)]
            self.assertGreaterEqual(seg.min(), 1.0 - max(za, zb) - 1e-12)
            self.assertLessEqual(seg.max(), 1.0 - min(za, zb) + 1e-12)

    def test_slope_is_flat_outside_ramp(self):
        s = Surface.slope(1.5, -0.3, y0=2.0, y_start=2.0, y_end=4.0)
        np.testing.assert_allclose(s([0.0, 2.0, 3.0, 4.0, 6.0]), [1.5, 1.5, 1.2, 0.9, 0.9])


class MeshTest(unittest.TestCase):
    surface = staticmethod(lambda y: 1.0 + 0.2 * np.sin(2 * np.pi * y / 1.3))

    def _check_conforming(self, mesh, x_top, x_bottom, width):
        area = signed_areas(mesh.vertices, mesh.triangles)
        self.assertTrue(np.all(area > 0))
        self.assertAlmostEqual(area.sum(), (x_bottom - x_top) * width, places=9)
        # Surface vertices are mesh vertices and every surface segment is an edge.
        dist = np.linalg.norm(mesh.surface[:, None] - mesh.vertices[None], axis=2)
        idx = dist.argmin(axis=1)
        self.assertLess(dist[np.arange(len(idx)), idx].max(), 1e-9)
        tri = mesh.triangles
        edges = {tuple(sorted(e)) for e in np.concatenate(
            [tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]).tolist()}
        for a, b in zip(idx[:-1], idx[1:]):
            self.assertIn(tuple(sorted((int(a), int(b)))), edges)
        # Region tags agree with the side of the surface polyline.
        c = mesh.vertices[mesh.triangles].mean(axis=1)
        above = c[:, 0] < mesh.surface_depth(c[:, 1])
        self.assertTrue(np.all((mesh.region == AIR) == above))
        self.assertTrue(np.all((mesh.region == GROUND) == ~above))

    def test_delaunay_mesh_conforms_to_surface(self):
        m = delaunay_band_mesh(2.0, 0.6, 1.5, self.surface, 0.03, h_interface=0.027)
        self._check_conforming(m, 0.6, 1.5, 2.0)
        self.assertGreater(m.quality()["min_angle_deg"], 20)

    def test_delaunay_local_refinement(self):
        m = delaunay_band_mesh(2.0, 0.6, 1.5, self.surface, 0.04, h_surface=0.015,
                               refine_width=0.05)
        self._check_conforming(m, 0.6, 1.5, 2.0)
        p = m.vertices[m.triangles]
        edge = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
        near = np.abs(p.mean(1)[:, 0] - m.surface_depth(p.mean(1)[:, 1])) < 0.02
        far = np.abs(p.mean(1)[:, 0] - m.surface_depth(p.mean(1)[:, 1])) > 0.2
        self.assertLess(np.median(edge[near]), 0.6 * np.median(edge[far]))
        self.assertGreater(m.quality()["min_angle_deg"], 20)

    def test_terrain_mesh_with_grading_and_jitter(self):
        y = np.linspace(0, 2.0, 81)
        m = terrain_following_band_mesh(y, 0.6, 1.5, self.surface, 0.03, surface_refinement=0.5,
                                        jitter=0.3)
        self._check_conforming(m, 0.6, 1.5, 2.0)


class MaterialAndPointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nx, ny, dx = 60, 80, 0.02
        eps = torch.full((nx, ny), 6.0, dtype=torch.float64)
        sig = torch.full((nx, ny), 0.01, dtype=torch.float64)
        x = torch.arange(nx)[:, None] * dx
        y = torch.arange(ny)[None, :] * dx
        cls.anomaly = ((x - 0.85) ** 2 + (y - 0.8) ** 2) <= 0.1 ** 2
        eps[cls.anomaly] = 15.0
        cls.surf = Surface.sine(0.5, 0.08, 0.9)
        cls.sol = HybridTM2D(eps, sig, dx, surface=cls.surf, dg_band=(0.3, 1.0), pmlthick=8,
                             mesh="delaunay")

    def test_air_ground_and_anomaly_materials(self):
        s = self.sol
        eps_e, sig_e, _ = (t.numpy() for t in s.dg_materials)
        air = s.mesh.region == AIR
        self.assertTrue(np.all(eps_e[air] == 1.0) and np.all(sig_e[air] == 0.0))
        c = s.mesh.vertices[s.mesh.triangles].mean(1) - np.array([s.ox, s.oy])
        inside = (c[:, 0] - 0.85) ** 2 + (c[:, 1] - 0.8) ** 2 < 0.08 ** 2
        outside = ((c[:, 0] - 0.85) ** 2 + (c[:, 1] - 0.8) ** 2 > 0.13 ** 2) & ~air
        self.assertTrue(np.all(eps_e[inside] == 15.0))
        self.assertTrue(np.all(eps_e[outside] == 6.0))
        # Yee grid: air exactly above the surface, ground below, no extra
        # material at the interface rows.
        p0 = s.pml[0]
        eps_fd = s.eps_fd_pad.numpy()[p0:p0 + s.nx, s.pml[2]:s.pml[2] + s.ny]
        xs = np.arange(s.nx)[:, None] * s.dx
        above = xs < self.surf(np.arange(s.ny) * s.dy)[None]
        self.assertTrue(np.all(eps_fd[above] == 1.0))
        self.assertTrue(np.all(eps_fd[~above & ~self.anomaly.numpy()] == 6.0))

    def test_receivers_on_edges_and_vertices_interpolate_linear_fields(self):
        s = self.sol
        f = lambda x, y: 2.0 * x - 3.0 * y + 0.5  # noqa: E731
        s.dg.E = torch.as_tensor(f(s.geometry.x, s.geometry.y))[None]
        xs, ys = np.meshgrid(np.arange(s.NX) * s.dx, np.arange(s.NY) * s.dy, indexing="ij")
        s.fd.Ez = torch.as_tensor(f(xs, ys))[None].clone()
        v = s.mesh.vertices
        tri = s.mesh.triangles[len(s.mesh.triangles) // 2]
        # A DG vertex, an edge midpoint, an interior point and a Yee point.
        points = [v[tri[0]], 0.5 * (v[tri[0]] + v[tri[1]]), v[tri].mean(0),
                  [0.0542 + s.ox, 0.37 + s.oy]]
        phys = np.array(points) - np.array([s.ox, s.oy])
        ops = s._point_ops(torch.as_tensor(phys[None]), source=False)
        values = s._sample(ops, len(points))[0].numpy()
        np.testing.assert_allclose(values, f(np.array(points)[:, 0], np.array(points)[:, 1]),
                                   rtol=0, atol=1e-11)

    def test_source_on_edge_injects_unit_charge(self):
        s = self.sol
        v = s.mesh.vertices
        c = v[s.mesh.triangles].mean(1)
        # A ground triangle well inside the band and away from the surface,
        # so every element touching its vertices has the same material.
        ok = ((s.mesh.region == GROUND) & (c[:, 0] > s.mesh.surface_depth(c[:, 1]) + 0.1)
              & (c[:, 0] < s.x_bottom - 0.1) & (np.abs(c[:, 1] - s.width / 2) < 0.2))
        tri = s.mesh.triangles[np.nonzero(ok)[0][0]]
        for p in (v[tri[0]], 0.5 * (v[tri[1]] + v[tri[2]]), v[tri].mean(0)):
            phys = (p - np.array([s.ox, s.oy]))[None, None]
            ops = s._point_ops(torch.as_tensor(phys), source=True)
            s.dg.E = torch.zeros_like(s.dg.E)
            s._inject(ops, torch.ones(1, dtype=s.dtype))
            charge = (torch.matmul(s.dg.E[0], s.dg.mass) * s.dg.J).sum(-1)
            # sum_K 1^T M_K dE_K = -cb (all containing elements share cb here).
            k = int(ops.dg_elem[0])
            self.assertAlmostEqual(float(charge.sum()), -float(s.dg.cb[k, 0]),
                                   delta=1e-9 * float(s.dg.cb[k, 0]))


class DGCPMLTest(unittest.TestCase):
    def test_full_dgtd_cpml_reflection(self):
        """Full DGTD in vacuum: domain 1 m vs the same with the right CPML 2 m further out."""
        dx = 0.02
        src, rec = [0.5, 0.6], [[0.5, 0.8], [0.25, 0.9]]
        out = []
        dt = None
        for ny in (50, 150):
            s = HybridTM2D(torch.ones(50, ny), torch.zeros(50, ny), dx, pmlthick=20,
                           dg_band=(None, None), dg_order=1, dg_h=dx, mesh="terrain", dt=dt)
            dt = s.dt
            nt = int(8e-9 / dt)
            out.append(s.run(ricker(nt, dt, 5e8, 2.4e-9), [[src]], [rec])["receivers"][0].numpy())
        refl = np.abs(out[0] - out[1]).max(0) / np.abs(out[1]).max(0)
        self.assertLess(refl.max(), 1.5e-3)


class FullDGTDStabilityTest(unittest.TestCase):
    def test_energy_decays_with_cpml_and_topography(self):
        nx, ny, dx = 40, 50, 0.02
        s = HybridTM2D(torch.full((nx, ny), 6.0), torch.zeros(nx, ny), dx,
                       surface=Surface.sine(0.4, 0.06, 0.5), pmlthick=10, dg_band=(None, None),
                       mesh="delaunay")
        nt = 12000
        w = ricker(nt, s.dt, 1e9, 1.2e-9)
        res = s.run(w, [[[0.3, 0.5]]], [[[0.3, 0.7]]], energy_every=500)
        e = res["energy"][:, 0].numpy()
        self.assertTrue(np.isfinite(e).all())
        self.assertLess(e[-1], 1e-6 * e.max())


class FDTDRefinementTest(unittest.TestCase):
    def test_refined_yee_grid(self):
        nx, ny, dx = 40, 50, 0.02
        eps = torch.full((nx, ny), 6.0, dtype=torch.float64)
        eps[30:33, 20:24] = 12.0
        surf = Surface.sine(0.4, 0.05, 0.6)
        # Band edges on rows of both grids (0.24 = 12 * 0.02 = 36 * 0.02 / 3).
        common = dict(surface=surf, dg_band=(0.24, 0.54), pmlthick=6, mesh="delaunay",
                      flux="central", cfl_safety=0.9)
        s1 = HybridTM2D(eps, torch.zeros_like(eps), dx, **common)
        s3 = HybridTM2D(eps, torch.zeros_like(eps), dx, fdtd_refinement=3, **common)
        self.assertEqual(s3.NX - 1, 3 * (s1.NX - 1))
        self.assertAlmostEqual(s3.ox, s1.ox)
        # DG materials come from the model pixels, independent of the Yee grid.
        for a, b in zip(s1.dg_materials, s3.dg_materials):
            self.assertTrue(torch.equal(a, b))
        # Each model pixel becomes a 3x3 block of Yee nodes (odd r: no ties).
        p = s3.pml[0]
        block = s3.eps_fd_pad[p + 90:p + 99, s3.pml[2] + 60:s3.pml[2] + 72]
        self.assertTrue(torch.all(block[1:-1, 1:-1] == 12.0))
        # The refined coupled scheme still conserves the discrete energy and
        # For P1 with h = dx the DG limit is about the Yee limit at dx/3, so r = 3
        # costs at most ~20 % more steps (none for r = 2).
        self.assertLess(abs(s3.dt / s1.dt - 1), 0.2)
        closed = HybridTM2D(eps, torch.zeros_like(eps), dx, fdtd_refinement=3,
                            **dict(common, pmlthick=0))
        nt = 1200
        t = (np.arange(nt) + 0.5) * closed.dt
        w = torch.tensor(np.exp(-((t - 1.5e-9) * 2e9) ** 2) * np.cos(2 * np.pi * 1e9 * (t - 1.5e-9)))
        w[300:] = 0
        res = closed.run(w, [[[0.42, 0.5]]], [[[0.15, 0.3]]], energy_every=10)
        steps = np.array(res["energy_steps"])
        e = res["energy"][:, 0].numpy()[steps > 320]
        self.assertLess((e.max() - e.min()) / e.mean(), 1e-12)


if __name__ == "__main__":
    unittest.main()
