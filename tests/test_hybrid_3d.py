"""Phase 4 tests: 3D hybrid DGTD (tetrahedra) - Yee-FDTD.

* reference tetrahedron: exact derivatives, lifting / integration by parts;
* terrain-following tetrahedral mesh: conformity, volume, fast point location;
* pure-Yee 3D mode against DeepGPR's native 3D solver (all components);
* energy conservation of the coupled conservative scheme (P1, P2, refined Yee);
* exactness of the 3D mortar projection;
* the dissipative interface penalty and the side UPML are stable;
* fused CUDA DG operator and its transpose against the PyTorch operator;
* discrete adjoint: dot product, finite differences, native pure-Yee gradient.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import DeepGPR  # noqa: E402
from DeepGPR.hybrid import HybridMaxwell3D, HybridPropagator, Surface3D  # noqa: E402
from DeepGPR.hybrid import functional3d as fn3  # noqa: E402
from DeepGPR.hybrid.coupling3d import InterfaceCoupling3D, horizontal_faces  # noqa: E402
from DeepGPR.hybrid.dgtd3d import build_geometry_3d  # noqa: E402
from DeepGPR.hybrid.mesh3d import (adaptive_column_tet_mesh, locate_points_3d,  # noqa: E402
                                   locate_points_columns, signed_volumes,
                                   terrain_following_tet_mesh)
from DeepGPR.hybrid.reference3d import FACE_VERTICES, reference_tetrahedron  # noqa: E402

CUDA = torch.cuda.is_available()


def small_mesh(jitter=0.3):
    y = np.linspace(0, 0.3, 7)
    z = np.linspace(0, 0.24, 6)
    surf = lambda Y, Z: 0.2 + 0.03 * np.sin(2 * np.pi * Y / 0.3) * np.cos(2 * np.pi * Z / 0.24)  # noqa: E731
    return terrain_following_tet_mesh(y, z, 0.1, 0.3, surf, 0.04, jitter=jitter)


class ReferenceAndMeshTest(unittest.TestCase):
    def test_reference_operators(self):
        for order in (1, 2):
            ref = reference_tetrahedron(order)
            r, s, t = ref.r, ref.s, ref.t
            f = r ** order + 2 * s * t ** (order - 1) - t
            self.assertLess(np.abs(ref.Dr @ f - order * r ** (order - 1)).max(), 1e-12)
            self.assertAlmostEqual(float(np.ones(ref.np_) @ ref.M @ np.ones(ref.np_)), 4.0 / 3.0)
            for fm in ref.face_mass:
                self.assertAlmostEqual(float(fm.sum()), 2.0)

    def test_mesh_and_geometry(self):
        m = small_mesh()
        self.assertAlmostEqual(float(signed_volumes(m.vertices, m.tets).sum()), 0.2 * 0.3 * 0.24)
        f = np.sort(np.concatenate([m.tets[:, list(fv)] for fv in FACE_VERTICES]), axis=1)
        self.assertLessEqual(int(np.unique(f, axis=0, return_counts=True)[1].max()), 2)
        for order in (1, 2):
            ref = reference_tetrahedron(order)
            g = build_geometry_3d(m, ref)
            # D_x of a degree-N polynomial is exact; lifting reproduces int_dK f n_x.
            f = g.x ** order + g.y * g.z ** (order - 1)
            R = g.rst_x
            dx = sum(R[:, q, 0, None] * (f @ D.T) for q, D in enumerate((ref.Dr, ref.Ds, ref.Dt)))
            self.assertLess(np.abs(dx - order * g.x ** (order - 1)).max(), 1e-10)
            fs = g.sJ / g.J[:, None]
            flux = np.concatenate([f[:, ref.fmask[k]] * g.normals[:, k, 0, None] * fs[:, k, None]
                                   for k in range(4)], axis=1)
            lhs = (g.J[:, None] * ((flux @ ref.LIFT.T) @ ref.M)).sum(1)
            rhs = (g.J[:, None] * (dx @ ref.M)).sum(1)
            self.assertLess(np.abs(lhs - rhs).max(), 1e-12 * np.abs(rhs).max() + 1e-18)

    def test_adaptive_column_mesh(self):
        y = np.linspace(0, 0.3, 16)
        z = np.linspace(0, 0.24, 13)
        surf = lambda Y, Z: 0.2 + 0.05 * np.sin(2 * np.pi * Y / 0.3) * np.cos(2 * np.pi * Z / 0.24)  # noqa: E731
        m = adaptive_column_tet_mesh(y, z, 0.1, 0.3, surf, 0.02)
        vol = signed_volumes(m.vertices, m.tets)
        self.assertTrue(np.all(vol > 0))
        self.assertAlmostEqual(float(vol.sum()), 0.2 * 0.3 * 0.24, places=14)
        f = np.sort(np.concatenate([m.tets[:, list(fv)] for fv in FACE_VERTICES]), axis=1)
        faces, counts = np.unique(f, axis=0, return_counts=True)
        self.assertLessEqual(int(counts.max()), 2)
        # conforming: faces of a single tetrahedron lie on the band's boundary only
        p = m.vertices[faces[counts == 1]]                     # (n, 3 vertices, 3)
        on = lambda a, v: np.all(np.abs(p[:, :, a] - v) < 1e-12, axis=1)  # noqa: E731
        self.assertTrue(np.all(on(0, 0.1) | on(0, 0.3) | on(1, 0.0) | on(1, 0.3) | on(2, 0.0)
                               | on(2, 0.24)))
        # air volume = integral of (surface - x_top) for the piecewise-linear surface
        Y, Z = np.meshgrid(y, z, indexing="ij")
        s = surf(Y, Z) - 0.1
        cell = 0.25 * (s[:-1, :-1] + s[1:, :-1] + s[:-1, 1:] + s[1:, 1:])
        air = vol[m.region == 0].sum()
        self.assertLess(abs(air - cell.sum() * 0.02 * 0.02) / air, 0.02)
        terrain = terrain_following_tet_mesh(y, z, 0.1, 0.3, surf, 0.02)
        self.assertGreater(m.quality()["min_inradius"], 1.3 * terrain.quality()["min_inradius"])
        self.assertLess(m.num_elements, terrain.num_elements)

    def test_fast_point_location(self):
        m = small_mesh()
        rng = np.random.default_rng(0)
        P = np.stack([rng.uniform(0.1, 0.3, 300), rng.uniform(0, 0.3, 300),
                      rng.uniform(0, 0.24, 300)], 1)
        elem, _ = locate_points_columns(m, P)
        self.assertTrue(np.all(elem >= 0))
        for i in range(0, 300, 10):
            self.assertIn(elem[i], locate_points_3d(m, P[i:i + 1])[0][0])

    def test_mortar_projection_exact_for_linear_fields(self):
        m = small_mesh(jitter=0.0)
        for order in (1, 2):
            g = build_geometry_3d(m, reference_tetrahedron(order))
            faces = horizontal_faces(g, 0.1, 1e-9)
            for dyz in (0.02, 0.013):
                NY, NZ = int(round(0.3 / dyz)) + 2, int(round(0.24 / dyz)) + 2
                cp = InterfaceCoupling3D(m, g, faces, 0.1, dyz, dyz, NY, NZ)
                yn = g.y.reshape(-1)[cp.nodes.numpy()]
                zn = g.z.reshape(-1)[cp.nodes.numpy()]
                # faces whose nodes are all at least two lattice cells from the walls
                k_of = cp.slots.numpy() // (4 * g.ref.nfp)
                ok = np.ones(len(yn), bool)
                for kk in np.unique(k_of):
                    sel = k_of == kk
                    ok[sel] = np.all((yn[sel] > 2 * dyz) & (yn[sel] < 0.3 - 2 * dyz)
                                     & (zn[sel] > 2 * dyz) & (zn[sel] < 0.24 - 2 * dyz))
                for c, (oy, oz) in enumerate(((0.5 * dyz, 0.0), (0.0, 0.5 * dyz))):
                    J, K = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="ij")
                    f = 0.3 + 2 * (oy + J * dyz) - 1.5 * (oz + K * dyz)
                    proj = cp.to_dg[c](torch.tensor(f.reshape(1, -1)))[0].numpy()
                    self.assertLess(np.abs(proj - (0.3 + 2 * yn - 1.5 * zn))[ok].max(), 1e-12)


class YeeMirror3DTest(unittest.TestCase):
    def test_pure_yee_matches_native(self):
        torch.manual_seed(1)
        nx, ny, nz, dx, nt, p = 12, 13, 14, 0.02, 90, 4
        eps = 2 + 4 * torch.rand(nx, ny, nz)
        sig = 0.01 * torch.rand(nx, ny, nz)
        dt = 0.9 * DeepGPR.max_stable_time_step(dx, nx + 2 * p, ny + 2 * p, nz + 2 * p)
        w = DeepGPR.wavelet.ricker(2e9, nt, dt, 1 / 2e9)
        src = torch.tensor([[[6, 6, 7]]], dtype=torch.int)
        rec = torch.tensor([[[2, 3, 3], [9, 10, 11]]], dtype=torch.int)
        for comp in (0, 1, 2):
            d1 = DeepGPR.compute(device="cpu", dx=dx, dt=dt, source_amplitudes=w.reshape(1, -1, 1),
                                 source_location=src, receiver_location=rec, eps_r=eps, sigma=sig,
                                 pmlthick=p, source_direction=comp, receiver_component=comp)[-1]
            sol = HybridMaxwell3D(eps, sig, dx, pmlthick=p, dt=dt, component=comp)
            off = np.zeros(3)
            off[comp] = 0.5
            d2 = sol.run(w.double(), (src.double().numpy() + off) * dx,
                         (rec.double().numpy() + off) * dx)["receivers"]
            with self.subTest(component=comp):
                self.assertLess(float((d2.float() - d1).norm() / d1.norm()), 2e-6)


def hybrid(device="cpu", **kw):
    nx, ny, nz, dx = 14, 12, 11, 0.02
    eps = 3 + 3 * torch.rand(nx, ny, nz, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    sig = kw.pop("sig", torch.zeros(nx, ny, nz, dtype=torch.float64))
    surf = Surface3D.sine(0.14, 0.02, 0.24, 0.2)
    if kw.get("mesh") == "terrain":
        kw.setdefault("mesh_options", dict(jitter=0.2))
    return HybridMaxwell3D(eps, sig, dx, surface=surf, dg_band=(0.06, 0.22), device=device, **kw)


def random_state(st, gen, yee=True):
    s = list(fn3.zero_state(st))
    if yee:
        for k in range(6):
            s[k] = torch.randn(s[k].shape, generator=gen, dtype=torch.float64).to(s[k].device) * st.yee.act[k]
    s[18] = torch.randn(s[18].shape, generator=gen, dtype=torch.float64).to(s[18].device)
    s[19] = torch.randn(s[19].shape, generator=gen, dtype=torch.float64).to(s[19].device) / 377.0
    return tuple(s)


class Energy3DTest(unittest.TestCase):
    def test_energy_conservation(self):
        for kw in (dict(dg_order=1), dict(dg_order=2), dict(fdtd_refinement=3),
                   dict(mesh="terrain")):
            sol = hybrid(pmlthick=0, flux="central", interface_penalty=0.0, **kw)
            st, c = sol.static, sol.coefs
            state = random_state(st, torch.Generator().manual_seed(1))
            E = []
            for n in range(150):
                nxt = fn3.step(st, c, state)
                if n % 30 == 0:
                    E.append(float(sol.energy(state, nxt)[0]))
                state = nxt
            with self.subTest(**kw):
                self.assertLess(max(abs(e - E[0]) for e in E) / E[0], 1e-12)

    def test_interface_penalty_is_dissipative(self):
        sol = hybrid(pmlthick=0, flux="central", interface_penalty=0.1)
        st, c = sol.static, sol.coefs
        state = random_state(st, torch.Generator().manual_seed(2))
        prev_e = None
        for n in range(120):
            nxt = fn3.step(st, c, state)
            e = float(sol.energy(state, nxt)[0])
            if prev_e is not None:
                self.assertLessEqual(e, prev_e * (1 + 1e-12))
            prev_e, state = e, nxt

    def test_side_upml_long_run_stable(self):
        # Physical excitation (zero-mean Ricker): the energy decays by many orders
        # and does not grow afterwards. (Random initial data with static content
        # shows the known weak, linear-in-time drift of the UPML; see the docs.)
        sol = hybrid(pmlthick=4, device="cuda" if CUDA else "cpu")
        st, c = sol.static, sol.coefs
        nt = 6000 if CUDA else 2500
        t = (np.arange(nt) + 0.5) * sol.dt - 0.6e-9
        a = (np.pi * 2e9 * t) ** 2
        w = torch.tensor((1 - 2 * a) * np.exp(-a), device=sol.device)
        src = sol.point_ops(np.array([[[0.04, 0.12, 0.11]]]), source=True)
        state, energy = fn3.zero_state(st), []
        with torch.no_grad():
            for n in range(nt):
                nxt = fn3.step(st, c, state, src, w[n:n + 1])
                if n % 50 == 0:
                    energy.append(float(sol.energy(state, nxt)[0]))
                state = nxt
        energy = np.array(energy)
        late = energy[len(energy) // 2:]
        self.assertLess(late.max() / energy.max(), 1e-4)
        self.assertLessEqual(late[-1], 1.001 * late[:5].max())


@unittest.skipUnless(CUDA, "CUDA not available")
class CudaKernel3DTest(unittest.TestCase):
    def test_fused_operator_and_transpose(self):
        for order in (1, 2):
            sol = hybrid(device="cuda", dg_order=order, pmlthick=2, side_boundary="abc",
                         dt=1e-12, nshot=2)
            d, c = sol.static.dg, sol.coefs
            self.assertIsNotNone(d.cuda)
            gen = torch.Generator().manual_seed(order)
            mk = lambda *s: torch.randn(s, generator=gen, dtype=torch.float64).cuda().requires_grad_(True)  # noqa: E731
            F, G = mk(2, 3, d.K, d.Np), mk(2, 3, d.K, d.Np)
            # external input: compact on the interface slots (kernels), dense for PyTorch
            X = mk(2, 2, d.cuda.nX)
            slots = torch.cat([cp.slots for cp in sol.static.couplings])
            Xd = torch.zeros((2, 2, d.K * 4 * d.Nfp), dtype=X.dtype, device=X.device).index_copy(
                2, slots, X).view(2, 2, d.K, 4 * d.Nfp)
            for b, x, xd in (([c.e0, c.e1, c.e2, c.e3], X, Xd), ([c.h0, c.h1, c.h2, c.h3], None, None)):
                b = [t.detach().clone().requires_grad_(True) for t in b]
                out_c = d.cuda.terms(F, G, x, *b)
                out_t = torch.stack(fn3.dg_terms(d, F, fn3.dg_flux(d, b, F, G, xd)), 1)
                self.assertLess(float((out_c - out_t).norm() / out_t.norm()), 1e-13)
                gT = torch.randn_like(out_t)
                ins = [F, G] + ([x] if x is not None else []) + b
                g_c = torch.autograd.grad(out_c, ins, gT)
                g_t = torch.autograd.grad(out_t, ins, gT, allow_unused=True)
                for a, bb in zip(g_c, g_t):
                    if bb is not None:
                        self.assertLess(float((a - bb).norm() / bb.norm()), 1e-13)

    def test_cuda_graph_replay_matches_eager_step(self):
        # Same kernels and arguments; agreement to round-off (the eager step is not
        # bitwise reproducible either: index_add in the coupling uses atomics).
        import os
        sol = hybrid(device="cuda", pmlthick=3, nshot=2)
        nt = 300
        t = (np.arange(nt) + 0.5) * sol.dt - 0.6e-9
        a = (np.pi * 2e9 * t) ** 2
        w = (1 - 2 * a) * np.exp(-a)
        src = np.array([[[0.04, 0.12, 0.11]], [[0.04, 0.10, 0.12]]])
        rec = np.array([[[0.04, 0.16, 0.11], [0.18, 0.08, 0.13]]] * 2)
        out = []
        for env in (None, "1"):
            if env:
                os.environ["DEEPGPR_HYBRID_DISABLE_GRAPHS"] = env
            try:
                out.append((sol.run(w, src, rec)["receivers"], sol.state))
            finally:
                os.environ.pop("DEEPGPR_HYBRID_DISABLE_GRAPHS", None)
        rel = lambda x, y: float((x - y).abs().max() / y.abs().max())  # noqa: E731
        self.assertGreater(float(out[1][0].abs().max()), 0.0)
        self.assertLess(rel(out[0][0], out[1][0]), 1e-13)
        for a_, b_ in zip(out[0][1], out[1][1]):
            if b_.numel() and float(b_.abs().max()) > 0:
                self.assertLess(rel(a_, b_), 1e-13)

    def test_hand_written_step_vjp_matches_autograd(self):
        # vjp3d.step_vjp (transpose kernels + explicit coupling transposes) against
        # reverse-mode differentiation of the same fused step, all inputs/coefficients.
        from DeepGPR.hybrid import vjp3d
        for kw in (dict(), dict(dg_order=2), dict(fdtd_refinement=3, mesh="terrain"),
                   dict(side_boundary="abc", interface_penalty=0.0)):
            sol = hybrid(device="cuda", pmlthick=3, nshot=2, dt=1.5e-12,
                         sig=0.01 * torch.rand(14, 12, 11, dtype=torch.float64,
                                               generator=torch.Generator().manual_seed(3)), **kw)
            st, c = sol.static, sol.coefs
            self.assertTrue(vjp3d.supported(st))
            gen = torch.Generator().manual_seed(5)
            rnd = lambda t: torch.randn(t.shape, generator=gen, dtype=torch.float64).cuda() if t.numel() else t  # noqa: E731
            u = [rnd(t) for t in fn3.zero_state(st)]
            lam = [rnd(t) for t in fn3.zero_state(st)]
            src = sol.point_ops(np.array([[[0.05, 0.11, 0.12]], [[0.17, 0.13, 0.10]]]), source=True)
            rec = sol.point_ops(np.array([[[0.03, 0.10, 0.10], [0.15, 0.12, 0.12]]] * 2), source=False)
            self.assertTrue(src.fd_shot.numel() and src.dg_shot.numel() and rec.dg_shot.numel())
            amp = torch.randn(int(max(src.fd_item.max(), src.dg_item.max())) + 1, generator=gen,
                              dtype=torch.float64).cuda()
            gd = torch.randn((2, rec.count), generator=gen, dtype=torch.float64).cuda()
            ul = [t.clone().requires_grad_(True) if t.numel() else t for t in u]
            cl = c.replaced([t.detach().clone().requires_grad_(True) for t in c.tensors()])
            al = amp.clone().requires_grad_(True)
            out = fn3.step(st, cl, tuple(ul), src, al)
            act = [i for i, t in enumerate(u) if t.numel()]
            ref = torch.autograd.grad([out[i] for i in act] + [fn3.sample(st, rec, out)],
                                      [ul[i] for i in act] + cl.tensors() + [al],
                                      [lam[i] for i in act] + [gd], allow_unused=True)
            with torch.no_grad():
                u1 = fn3.step(st, c, tuple(u), src, amp)
                acc = {}
                lam_n, g_amp = vjp3d.step_vjp(st, c, u, u1, vjp3d.sample_adjoint(st, rec, gd, lam),
                                              src, amp, acc, True, True)
            names = [n for n in c.NAMES if getattr(c, n) is not None]
            acc = vjp3d.finalize(acc, c)
            mine = [lam_n[i] for i in act] + [acc.get(n) for n in names] + [g_amp]
            for a, b in zip(mine, ref):
                b = torch.zeros_like(a) if b is None else b
                a = torch.zeros_like(b) if a is None else a
                with self.subTest(**kw):
                    self.assertLess(float((a - b).abs().max() / (b.abs().max() + 1e-300)), 1e-12)


class Adjoint3DTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if CUDA else "cpu"
        shape = (10, 11, 11)
        cls.eps = torch.full(shape, 4.0, dtype=torch.float64)
        cls.eps[7:9, 4:7, 4:7] = 6.0
        cls.sig = torch.full(shape, 0.004, dtype=torch.float64)
        surf = Surface3D.sine(0.1, 0.015, 0.2, 0.2)
        sol = HybridMaxwell3D(cls.eps, cls.sig, 0.02, surface=surf, dg_band=(0.06, 0.14), pmlthick=3,
                              device=cls.device)
        cls.prop = HybridPropagator(sol, np.array([[[0.04, 0.10, 0.11]]]),
                                    np.array([[[0.04, 0.16, 0.11], [0.18, 0.08, 0.13]]]),
                                    checkpoint_every=13)
        nt = 200
        t = (np.arange(nt) + 0.5) * sol.dt - 0.6e-9
        a = (np.pi * 2e9 * t) ** 2
        cls.w = torch.tensor((1 - 2 * a) * np.exp(-a)).to(cls.device)
        cls.eps, cls.sig = cls.eps.to(cls.device), cls.sig.to(cls.device)
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
        lhs, rhs = float((lin * r).sum()), float((gw * dw).sum())
        self.assertLess(abs(lhs - rhs) / abs(lhs), 1e-11)

    def test_finite_difference_directional_derivative(self):
        e = self.eps.clone().requires_grad_(True)
        s = self.sig.clone().requires_grad_(True)
        self.J(e, s).backward()
        gen = torch.Generator().manual_seed(4)
        for par, g, scale in (("eps", e.grad, 4.0), ("sigma", s.grad, 0.004)):
            v = torch.zeros(self.eps.shape, dtype=torch.float64)
            v[5:10, 2:9, 2:9] = torch.randn((5, 7, 7), generator=gen, dtype=torch.float64)
            v = v.to(self.device)
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
                self.assertLess(min(errs), 1e-6)

    @unittest.skipUnless(CUDA, "CUDA not available")
    def test_hand_written_sweep_matches_autograd_sweep(self):
        import os
        from DeepGPR.hybrid import vjp3d
        self.assertTrue(vjp3d.supported(self.prop.static))
        grads = []
        for env in (None, "1"):
            if env:
                os.environ["DEEPGPR_HYBRID_AUTOGRAD_VJP"] = env
            try:
                e = self.eps.clone().requires_grad_(True)
                s = self.sig.clone().requires_grad_(True)
                w = self.w.clone().requires_grad_(True)
                d = self.prop(e, s, w[None])
                (0.5 * ((d - self.dobs) ** 2).sum()).backward()
                grads.append((e.grad, s.grad, w.grad))
            finally:
                os.environ.pop("DEEPGPR_HYBRID_AUTOGRAD_VJP", None)
        for a, b in zip(*grads):
            self.assertLess(float((a - b).norm() / b.norm()), 1e-11)

    def test_pure_yee_gradient_matches_native(self):
        torch.manual_seed(1)
        nx, ny, nz, dx, nt, p = 10, 11, 12, 0.02, 90, 4
        eps = 2 + 4 * torch.rand(nx, ny, nz)
        sig = 0.01 * torch.rand(nx, ny, nz)
        dt = 0.9 * DeepGPR.max_stable_time_step(dx, nx + 2 * p, ny + 2 * p, nz + 2 * p)
        w = DeepGPR.wavelet.ricker(2e9, nt, dt, 1 / 2e9)
        src = torch.tensor([[[5, 5, 6]]], dtype=torch.int)
        rec = torch.tensor([[[2, 3, 3], [8, 8, 9]]], dtype=torch.int)
        r = torch.randn(1, nt, 2)
        e1, s1 = eps.clone().requires_grad_(True), sig.clone().requires_grad_(True)
        d1 = DeepGPR.compute(device="cpu", dx=dx, dt=dt, source_amplitudes=w.reshape(1, -1, 1),
                             source_location=src, receiver_location=rec, eps_r=e1, sigma=s1,
                             pmlthick=p, mode=3)[-1]
        (d1 * r).sum().backward()
        sol = HybridMaxwell3D(eps, sig, dx, pmlthick=p, dt=dt, dtype=torch.float32)
        off = np.array([0, 0, 0.5])
        prop = HybridPropagator(sol, (src.double().numpy() + off) * dx, (rec.double().numpy() + off) * dx)
        e2, s2 = eps.clone().requires_grad_(True), sig.clone().requires_grad_(True)
        d2 = prop(e2, s2, w)
        (d2 * r).sum().backward()
        rel = lambda a, b: float((a - b).norm() / b.norm())  # noqa: E731
        self.assertLess(rel(d2.detach(), d1.detach()), 1e-5)
        self.assertLess(rel(e2.grad, e1.grad), 1e-4)
        self.assertLess(rel(s2.grad, s1.grad), 1e-4)


if __name__ == "__main__":
    unittest.main()
