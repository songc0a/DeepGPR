"""Huygens subgridding of the 3D Yee scheme (``DeepGPR.subgrid``): forward model.

* the surface corrections and precursors (linear fields are interpolated exactly,
  OS sources coincide with the main nodes they replace);
* transparency: a subgrid in a homogeneous medium reflects < 1 % of the direct wave;
* antennas inside a subgrid: data match a uniform fine grid;
* a target inside the subgrid: scattered field much closer to the fine grid than
  the coarse grid's;
* fused CUDA coupling kernels = PyTorch step, CUDA Graph replay = eager step;
* long-run stability; argument validation.
"""

import math
import unittest

import numpy as np
import torch

from DeepGPR.subgrid import Subgrid, SubgridMaxwell3D, fine_source_samples
from DeepGPR.subgrid import hsg

CUDA = torch.cuda.is_available()


def ricker(nt, dt, f0):
    t = np.arange(nt) * dt - 1.2 / f0
    a = (np.pi * f0 * t) ** 2
    return torch.as_tensor((1 - 2 * a) * np.exp(-a))


def homogeneous(shape, eps=4.0):
    return torch.full(shape, eps, dtype=torch.float64), torch.zeros(shape, dtype=torch.float64)


def fine_reference(eps, sig, h, w, dt, src, rec, device, r=3, pml=24):
    """Uniform grid at the model spacing (the subgrid spacing), same source current."""
    sol = SubgridMaxwell3D(eps, sig, h, coarsen=1, pmlthick=pml, dt=dt / r, dipole_length=r * h,
                           device=device)
    wf = fine_source_samples(w[None], r)[0]
    shift = (r - 1) // 2      # the main grid's half-step source delay on the fine time axis
    wf = torch.cat([torch.zeros(shift, dtype=wf.dtype), wf, torch.zeros(r, dtype=wf.dtype)])
    nt = w.shape[0]
    return sol.run(wf[:r * nt], src, rec)["receivers"][0][r - 1::r].cpu()


class OperatorTest(unittest.TestCase):
    def setUp(self):
        eps, sig = homogeneous((61, 61, 64))
        self.sol = SubgridMaxwell3D(eps, sig, 0.01, coarsen=3, pmlthick=4, cuda_kernels=False,
                                    subgrids=[Subgrid(((0.21, 0.27), (0.24, 0.33), (0.21, 0.36)),
                                                      ratio=3, is_os_sep=2, pml_cells=4)])
        self.sg = self.sol.static.subs[0]

    def test_precursors_interpolate_linear_fields_exactly(self):
        main = self.sol.static.main
        rng = np.random.default_rng(0)
        for kind, ops, comps, corr in (("e", self.sg.pe, self.sg.pe_comp, self.sg.is_h),
                                       ("h", self.sg.ph, self.sg.ph_comp, self.sg.is_e)):
            coef = rng.normal(size=4)
            fields = []
            for c in range(3):
                off = hsg.offsets(kind, c)
                g = np.meshgrid(*[np.arange(n) + o for n, o in zip(main.shape, off)], indexing="ij")
                fields.append(torch.as_tensor(coef[0] + sum(coef[1 + a] * g[a] for a in range(3)))[None])
            vals = hsg.precursors(ops, comps, fields)[0].numpy()
            # expected: the linear field at the fine positions of the differenced nodes
            okind = "e" if kind == "e" else "h"
            o = 0
            for t in hsg.TERM_ORDER:
                c, a, f, _ = hsg.TERMS[t]
                for side in (-1, 1):
                    lo = (self.sg.nb,) * 3
                    hi = tuple(self.sg.nb + (h - l) * 3 for l, h in zip(self.sg.lo, self.sg.hi))
                    _, q = hsg.face_entries(lo, hi, "h" if kind == "e" else "e", t, side)
                    off = np.array(hsg.offsets(okind, f))
                    X = np.asarray(self.sg.lo)[None] + (q + off[None] - self.sg.nb) / 3.0
                    exp = coef[0] + X @ coef[1:]
                    np.testing.assert_allclose(vals[o:o + len(q)], exp, rtol=0, atol=1e-10)
                    o += len(q)
            self.assertEqual(o, vals.size)
            self.assertEqual(o, corr.size)

    def test_outer_surface_sources_coincide_with_main_nodes(self):
        sg, r = self.sg, self.sg.ratio
        for corr, kind in ((sg.os_h, "h"), (sg.os_e, "e")):
            okind = "e" if kind == "h" else "h"
            for t, f, sidx in zip(hsg.TERM_ORDER, corr.src_comp, corr.src_idx):
                c, a, ff, _ = hsg.TERMS[t]
                self.assertEqual(f, ff)
                off = np.array(hsg.offsets(okind, f))
                qs = [hsg.face_entries(sg.os_lo, sg.os_hi, kind, t, s)[1] for s in (-1, 1)]
                q = np.concatenate(qs)
                fine = np.stack(np.unravel_index(sidx.numpy(), sg.yee.shape), 1)
                x_main = q + off[None]
                x_fine = np.asarray(sg.lo)[None] + (fine + off[None] - sg.nb) / r
                np.testing.assert_allclose(x_fine, x_main, atol=1e-12)

    def test_validation(self):
        eps, sig = homogeneous((61, 61, 64))
        with self.assertRaises(ValueError):       # even ratio
            SubgridMaxwell3D(eps, sig, 0.01, coarsen=3, pmlthick=4, cuda_kernels=False,
                             subgrids=[Subgrid(((0.21, 0.27), (0.24, 0.33), (0.21, 0.36)), ratio=2)])
        with self.assertRaises(ValueError):       # subgrid CPML reaches the main CPML
            SubgridMaxwell3D(eps, sig, 0.01, coarsen=3, pmlthick=4, cuda_kernels=False,
                             subgrids=[Subgrid(((0.06, 0.12), (0.24, 0.33), (0.21, 0.36)))])
        with self.assertRaises(ValueError):       # main dt above the subgrid's CFL limit
            SubgridMaxwell3D(eps, sig, 0.01, coarsen=3, pmlthick=4, cuda_kernels=False,
                             subgrids=[Subgrid(((0.21, 0.27), (0.24, 0.33), (0.21, 0.36)),
                                               is_os_sep=2, pml_cells=4, time_ratio=1)])
        with self.assertRaises(ValueError):       # point between the inner and outer surface
            self.sol.point_ops([[0.18, 0.28, 0.28]], source=True)
        self.assertEqual(self.sol.classify([0.24, 0.285, 0.285]), 0)
        self.assertEqual(self.sol.classify([0.05, 0.05, 0.05]), -1)

    def test_model_gradient_reaches_both_grids(self):
        eps, sig = homogeneous((61, 61, 64))
        e = eps.clone().requires_grad_(True)
        me, ms, se, ss = self.sol.material_fields(e, sig)
        (me.sum() + 10.0 * se[0].sum()).backward()
        g = e.grad
        self.assertGreater(float(g[24, 28, 28]), 10.0)       # inside the box: subgrid nodes
        self.assertGreater(float(g[3, 3, 3]), 0.0)           # main grid only
        self.assertAlmostEqual(float(g.sum()), me.numel() + 10.0 * se[0].numel(), places=6)


def _transparency(device):
    eps, sig = homogeneous((76, 73, 73))
    box = ((0.33, 0.42), (0.27, 0.45), (0.27, 0.45))
    sol = SubgridMaxwell3D(eps, sig, 0.01, coarsen=3, subgrids=[Subgrid(box)], pmlthick=8,
                           device=device)
    w = ricker(260, sol.dt, 4e8)
    src, rec = [[0.09, 0.36, 0.36]], [[0.09, 0.45, 0.36], [0.15, 0.36, 0.45]]
    d_sg = sol.run(w, src, rec)["receivers"][0].cpu()
    coarse = SubgridMaxwell3D(eps, sig, 0.01, coarsen=3, pmlthick=8, dt=sol.dt, device=device)
    d_c = coarse.run(w, src, rec)["receivers"][0].cpu()
    return (d_sg - d_c).abs().max(0).values / d_c.abs().max(0).values


class TransparencyTest(unittest.TestCase):
    def test_reflection_below_one_percent_cpu(self):
        ratio = _transparency("cpu")
        self.assertLess(float(ratio.max()), 1e-2)


@unittest.skipUnless(CUDA, "CUDA not available")
class CudaTest(unittest.TestCase):
    shape = (76, 73, 73)
    box = ((0.30, 0.48), (0.27, 0.45), (0.27, 0.45))

    def model(self, eps_ball=None):
        h = 0.01
        eps, sig = homogeneous(self.shape)
        if eps_ball is not None:
            X, Y, Z = torch.meshgrid(*[torch.arange(n) * h for n in self.shape], indexing="ij")
            ball = ((X - 0.39) ** 2 + (Y - 0.36) ** 2 + (Z - 0.36) ** 2) <= 0.03 ** 2
            eps = torch.where(ball, torch.full_like(eps, eps_ball), eps)
        return eps, sig

    def test_fused_kernels_match_pytorch_step(self):
        g = torch.Generator().manual_seed(0)
        eps = 4.0 + 2.0 * torch.rand(self.shape, generator=g, dtype=torch.float64)
        sig = 0.01 * torch.rand(self.shape, generator=g, dtype=torch.float64)
        src = [[0.12, 0.3, 0.3], [0.39, 0.36, 0.37]]
        rec = [[0.12, 0.42, 0.3], [0.40, 0.34, 0.36], [0.63, 0.3, 0.3]]
        out = []
        for dev, fused in (("cpu", False), ("cuda", True)):
            sol = SubgridMaxwell3D(eps, sig, 0.01, coarsen=3, subgrids=[Subgrid(self.box)],
                                   pmlthick=8, cuda_kernels=fused, device=dev)
            out.append(sol.run(ricker(120, sol.dt, 4e8), src, rec)["receivers"].cpu())
        err = float((out[0] - out[1]).norm() / out[0].norm())
        self.assertLess(err, 1e-12)

    def test_graph_replay_matches_eager(self):
        eps, sig = self.model(16.0)
        sol = SubgridMaxwell3D(eps, sig, 0.01, coarsen=3, subgrids=[Subgrid(self.box)],
                               pmlthick=8, device="cuda")
        w = ricker(150, sol.dt, 4e8)
        src = [[0.12, 0.36, 0.36], [0.33, 0.36, 0.37]]
        rec = [[0.12, 0.45, 0.36], [0.40, 0.36, 0.36]]
        a = sol.run(w, src, rec, use_graph=False)["receivers"]
        b = sol.run(w, src, rec, use_graph=True)["receivers"]
        self.assertLess(float((a - b).norm() / a.norm()), 1e-12)

    def test_antennas_inside_match_uniform_fine_grid(self):
        eps, sig = self.model(20.0)
        sol = SubgridMaxwell3D(eps, sig, 0.01, coarsen=3, subgrids=[Subgrid(self.box)],
                               pmlthick=8, device="cuda")
        w = ricker(300, sol.dt, 4e8)
        src = [[0.33, 0.36, 0.36]]
        rec = [[0.44, 0.36, 0.36], [0.39, 0.42, 0.36]]
        d = sol.run(w, src, rec)["receivers"][0].cpu()
        ref = fine_reference(eps, sig, 0.01, w, sol.dt, src, rec, "cuda")
        err = (d - ref).norm(dim=0) / ref.norm(dim=0)
        self.assertLess(float(err.max()), 1e-2)

    def test_scattered_field_of_unresolved_target(self):
        """200 MHz, a 3 cm sphere (eps 20) in the box, antennas outside: the subgrid's
        scattered field is within 8 % of the uniform fine grid, the coarse grid's off by
        more than 40 %."""
        eps_t, sig = self.model(20.0)
        eps_b, _ = self.model(None)
        src = [[0.12, 0.36, 0.36]]
        rec = [[0.12, 0.45, 0.36], [0.63, 0.36, 0.36]]
        out = {}
        for kind in ("sub", "coarse"):
            sgs = [Subgrid(self.box)] if kind == "sub" else []
            scat = 0
            for e, s in ((eps_t, 1.0), (eps_b, -1.0)):
                sol = SubgridMaxwell3D(e, sig, 0.01, coarsen=3, subgrids=sgs, pmlthick=8,
                                       device="cuda")
                w = ricker(450, sol.dt, 2e8)
                scat = scat + s * sol.run(w, src, rec)["receivers"][0].cpu()
            out[kind] = scat
        ref = (fine_reference(eps_t, sig, 0.01, w, sol.dt, src, rec, "cuda")
               - fine_reference(eps_b, sig, 0.01, w, sol.dt, src, rec, "cuda"))
        err = {k: (v - ref).norm(dim=0) / ref.norm(dim=0) for k, v in out.items()}
        self.assertLess(float(err["sub"].max()), 0.08)
        self.assertGreater(float(err["coarse"].min()), 0.4)

    def test_long_run_is_stable(self):
        eps, sig = self.model(20.0)
        sol = SubgridMaxwell3D(eps, sig, 0.01, coarsen=3, subgrids=[Subgrid(self.box)],
                               pmlthick=8, device="cuda")
        nt = 6000
        w = ricker(nt, sol.dt, 4e8)
        rec = [[0.12, 0.45, 0.36], [0.40, 0.36, 0.36]]
        d = sol.run(w, [[0.12, 0.36, 0.36], [0.33, 0.36, 0.37]], rec)["receivers"][0].cpu()
        peak = float(d.abs().max())
        self.assertTrue(math.isfinite(peak))
        self.assertLess(float(d[-1000:].abs().max()), 1e-4 * peak)

    def test_transparency_cuda(self):
        self.assertLess(float(_transparency("cuda").max()), 1e-2)


if __name__ == "__main__":
    unittest.main()
