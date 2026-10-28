"""Phase 4 gradient verification of the 3D hybrid discrete adjoint.

1. Dot-product test of the linear source-to-data map.
2. Gradients independent of the checkpoint interval.
3. Zero sensitivity above the surface.
4. Finite-difference directional derivatives of ``J = 1/2 ||d(m) - d_obs||^2``
   for eps_r and sigma with perturbations in the DG band, the Yee region below
   it, the interface planes and the side UPML columns; central differences over
   a sweep of relative steps. Float64 on the GPU (fused CUDA DG kernels and
   their transpose) and on the CPU (PyTorch operator).
5. Pure-Yee 3D mode against DeepGPR's native discrete adjoint (float32).

    python examples/hybrid/phase4_gradient_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import DeepGPR  # noqa: E402
from DeepGPR.hybrid import HybridPropagator  # noqa: E402
from DeepGPR.hybrid.solver3d import HybridMaxwell3D  # noqa: E402
from DeepGPR.hybrid.topography import Surface3D  # noqa: E402

OUT = ROOT / "examples" / "hybrid" / "results"
STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
SHAPE, DX, PML = (14, 16, 16), 0.02, 4
SURF = Surface3D.sine(0.14, 0.02, 0.3, 0.3)


def ricker(nt, dt, f=1.5e9, t0=0.7e-9):
    t = (np.arange(nt) + 0.5) * dt - t0
    a = (np.pi * f * t) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a))


def setup(device, nt=320):
    eps = torch.full(SHAPE, 4.0, dtype=torch.float64)
    eps[9:12, 6:10, 6:10] = 7.0
    sig = torch.full(SHAPE, 0.004, dtype=torch.float64)
    sol = HybridMaxwell3D(eps, sig, DX, surface=SURF, dg_band=(0.08, 0.20), pmlthick=PML,
                          nshot=2, device=device, dtype=torch.float64)
    src = np.array([[[0.06, 0.14, 0.15]], [[0.12, 0.16, 0.17]]])
    rec = np.array([[[0.06, 0.22, 0.15], [0.24, 0.12, 0.15], [0.12, 0.08, 0.25]]] * 2)
    prop = HybridPropagator(sol, src, rec, checkpoint_every=23)
    w = ricker(nt, sol.dt).to(device)
    x = torch.arange(SHAPE[0], dtype=torch.float64)[:, None, None] * DX
    Y, Z = np.meshgrid(np.arange(SHAPE[1]) * DX, np.arange(SHAPE[2]) * DX, indexing="ij")
    above = (x < torch.as_tensor(SURF(Y, Z))[None]).to(device)
    return sol, prop, eps.to(device), sig.to(device), w, above


def regions():
    sl = slice
    return {"DG band": (sl(6, 10), sl(3, 13), sl(3, 13)),
            "Yee region": (sl(11, 14), sl(3, 13), sl(3, 13)),
            "bottom interface plane": (sl(9, 12), sl(3, 13), sl(3, 13)),
            "side UPML columns": (sl(5, 12), sl(0, 3), sl(0, 16))}


def checks(device, nt=320, steps=STEPS, region_names=None):
    sol, prop, eps, sig, w, above = setup(device, nt)
    res = {"dt": sol.dt, "cuda_kernels": sol.static.dg.cuda is not None}
    wl = w.clone().requires_grad_(True)
    d = prop(eps, sig, wl[None])
    r = torch.randn_like(d)
    gw, = torch.autograd.grad((d * r).sum(), wl)
    dw = torch.randn_like(w)
    with torch.no_grad():
        lin = prop(eps, sig, (w + dw)[None]) - d.detach()
    lhs, rhs = float((lin * r).sum()), float((gw * dw).sum())
    res["source_dot_product"] = abs(lhs - rhs) / abs(lhs)
    with torch.no_grad():
        dobs = 0.9 * prop(eps, sig, w[None])
    J = lambda e, s: 0.5 * ((prop(e, s, w[None]) - dobs) ** 2).sum()  # noqa: E731
    grads = {}
    for k in (23, 1, 1000):
        prop.checkpoint_every = k
        e, s = eps.clone().requires_grad_(True), sig.clone().requires_grad_(True)
        J(e, s).backward()
        grads[k] = (e.grad, s.grad)
    prop.checkpoint_every = 23
    ge, gs = grads[23]
    res["checkpoint_max_rel_diff"] = max(
        float((grads[k][i] - grads[23][i]).abs().max() / grads[23][i].abs().max())
        for k in (1, 1000) for i in (0, 1))
    res["max_abs_grad_above_surface"] = float(max(ge[above].abs().max(), gs[above].abs().max()))
    gen = torch.Generator().manual_seed(3)
    fd = {}
    for name, reg in regions().items():
        if region_names is not None and name not in region_names:
            continue
        v = torch.zeros(SHAPE, dtype=torch.float64)
        v[reg] = torch.randn(v[reg].shape, generator=gen, dtype=torch.float64)
        v = v.to(device)
        v[above] = 0
        for par, g, scale in (("eps_r", ge, 4.0), ("sigma", gs, 0.004)):
            adj = float((g * v).sum())
            errs = []
            for h in steps:
                hh = h * scale
                with torch.no_grad():
                    if par == "eps_r":
                        jp, jm = J(eps + hh * v, sig), J(eps - hh * v, sig)
                    else:
                        jp, jm = J(eps, sig + hh * v), J(eps, sig - hh * v)
                f = float((jp - jm) / (2 * hh))
                errs.append(abs(f - adj) / max(abs(f), abs(adj), 1e-300))
            fd[f"{name} / {par}"] = {"adjoint": adj, "errors": dict(zip(map(str, steps), errs))}
    res["fd"] = fd
    return res


def check_native():
    torch.manual_seed(1)
    nx, ny, nz, dx, nt, p = 16, 18, 20, 0.02, 160, 5
    eps = 2 + 4 * torch.rand(nx, ny, nz)
    sig = 0.01 * torch.rand(nx, ny, nz)
    dt = 0.9 * DeepGPR.max_stable_time_step(dx, nx + 2 * p, ny + 2 * p, nz + 2 * p)
    w = DeepGPR.wavelet.ricker(1.5e9, nt, dt, 1 / 1.5e9)
    src = torch.tensor([[[8, 9, 10]], [[5, 12, 6]]], dtype=torch.int)
    rec = torch.tensor([[[3, 4, 5], [12, 15, 16]], [[2, 2, 3], [14, 6, 15]]], dtype=torch.int)
    r = torch.randn(2, nt, 2)
    e1, s1 = eps.clone().requires_grad_(True), sig.clone().requires_grad_(True)
    d1 = DeepGPR.compute(device="cpu", dx=dx, dt=dt, source_amplitudes=w.reshape(1, -1, 1),
                         source_location=src, receiver_location=rec, eps_r=e1, sigma=s1,
                         pmlthick=p, mode=3)[-1]
    (d1 * r).sum().backward()
    sol = HybridMaxwell3D(eps, sig, dx, pmlthick=p, dt=dt, nshot=2, dtype=torch.float32)
    off = np.array([0, 0, 0.5])
    prop = HybridPropagator(sol, (src.double().numpy() + off) * dx, (rec.double().numpy() + off) * dx)
    e2, s2 = eps.clone().requires_grad_(True), sig.clone().requires_grad_(True)
    d2 = prop(e2, s2, w)
    (d2 * r).sum().backward()
    rel = lambda a, b: float((a - b).norm() / b.norm())  # noqa: E731
    return {"data": rel(d2.detach(), d1.detach()), "grad_eps_r": rel(e2.grad, e1.grad),
            "grad_sigma": rel(s2.grad, s1.grad)}


def report(tag, r):
    print(f"\n[{tag}] CUDA kernels: {r['cuda_kernels']}  dt {r['dt']:.3e}")
    print(f"  source dot product rel {r['source_dot_product']:.2e}")
    print(f"  checkpoint interval max rel diff {r['checkpoint_max_rel_diff']:.2e}")
    print(f"  max |grad| above surface {r['max_abs_grad_above_surface']:.1e}")
    steps = next(iter(r["fd"].values()))["errors"].keys()
    print("  FD directional derivatives, relative error per relative step h", ", ".join(steps))
    for name, row in r["fd"].items():
        print(f"    {name:34s} " + " ".join(f"{e:.1e}" for e in row["errors"].values()))
    sys.stdout.flush()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"native_pure_yee_float32": check_native()}
    print("native pure-Yee 3D:", res["native_pure_yee_float32"], flush=True)
    if torch.cuda.is_available():
        res["float64_cuda"] = checks("cuda")
        report("cuda", res["float64_cuda"])
    # The CPU (PyTorch operator) check is smaller: fewer steps, two regions.
    res["float64_cpu"] = checks("cpu", nt=160, steps=(1e-3, 1e-4, 1e-5),
                                region_names=("DG band", "bottom interface plane"))
    report("cpu", res["float64_cpu"])
    with open(OUT / "phase4_gradient_check.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
