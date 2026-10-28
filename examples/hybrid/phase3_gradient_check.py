"""Phase 3 gradient verification of the hybrid discrete adjoint.

1. Dot-product test of the linear source-to-data map: <F dw, r> vs <dw, F^T r>.
2. Gradients independent of the checkpoint interval.
3. Pure-Yee mode against DeepGPR's native discrete adjoint (float32).
4. Finite-difference directional derivatives of J = 1/2 ||d(m) - d_obs||^2 for
   eps_r and sigma, with perturbations confined to the DG band, the Yee region,
   the interface rows, and everywhere; central differences over a sweep of
   relative steps h. Float64 (CPU) and float32 (GPU, if available).

    python examples/hybrid/phase3_gradient_check.py
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
from DeepGPR.hybrid import HybridPropagator, HybridTM2D, Surface  # noqa: E402

OUT = ROOT / "examples" / "hybrid" / "results"
STEPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)


def ricker(nt, dt, f=1e9, t0=1.0e-9):
    t = (np.arange(nt) + 0.5) * dt - t0
    a = (np.pi * f * t) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a))


def setup(dtype, device):
    nx, ny, dx = 30, 40, 0.02
    eps = torch.full((nx, ny), 4.0, dtype=torch.float64)
    eps[20:24, 15:20] = 8.0
    sig = torch.full((nx, ny), 0.005, dtype=torch.float64)
    surf = Surface.sine(0.25, 0.04, 0.5)
    sol = HybridTM2D(eps, sig, dx, surface=surf, dg_band=(0.16, 0.38), pmlthick=6,
                     mesh="delaunay", fdtd_refinement=3, nshot=2, dtype=dtype, device=device)
    src = np.array([[[0.18, 0.3]], [[0.5, 0.5]]])
    rec = np.array([[[0.18, 0.5], [0.45, 0.2], [0.05, 0.6]]] * 2)
    prop = HybridPropagator(sol, src, rec, checkpoint_every=17)
    w = ricker(250, sol.dt).to(device, dtype)
    eps, sig = eps.to(device, dtype), sig.to(device, dtype)
    x = torch.arange(nx)[:, None] * dx
    above = (x < torch.tensor(surf(np.arange(ny) * dx))[None]).to(device)
    return prop, eps, sig, w, above


def checks_float64():
    prop, eps, sig, w, above = setup(torch.float64, "cpu")
    res = {}
    wl = w.clone().requires_grad_(True)
    d = prop(eps, sig, wl[None])
    r = torch.randn_like(d)
    gw, = torch.autograd.grad((d * r).sum(), wl)
    dw = torch.randn_like(w)
    with torch.no_grad():
        lin = prop(eps, sig, (w + dw)[None]) - d.detach()
    lhs, rhs = float((lin * r).sum()), float((gw * dw).sum())
    res["source_dot_product"] = {"lhs": lhs, "rhs": rhs, "rel": abs(lhs - rhs) / abs(lhs)}
    with torch.no_grad():
        dobs = 0.9 * prop(eps, sig, w[None])
    J = lambda e, s: 0.5 * ((prop(e, s, w[None]) - dobs) ** 2).sum()  # noqa: E731
    grads = {}
    for k in (17, 1, 250):
        prop.checkpoint_every = k
        e, s = eps.clone().requires_grad_(True), sig.clone().requires_grad_(True)
        J(e, s).backward()
        grads[k] = (e.grad, s.grad)
    prop.checkpoint_every = 17
    ge, gs = grads[17]
    res["checkpoint_max_rel_diff"] = max(
        float((grads[k][i] - grads[17][i]).abs().max() / grads[17][i].abs().max())
        for k in (1, 250) for i in (0, 1))
    res["max_abs_grad_above_surface"] = float(max(ge[above].abs().max(), gs[above].abs().max()))
    res["fd"] = fd_sweep(J, eps, sig, ge, gs, above, STEPS)
    return res


def fd_sweep(J, eps, sig, ge, gs, above, steps):
    nx, ny = eps.shape
    regions = {"DG band": (slice(13, 19), slice(5, 35)), "Yee region": (slice(21, 27), slice(5, 35)),
               "interface rows": (slice(18, 21), slice(5, 35)), "everywhere": (slice(0, nx), slice(0, ny))}
    gen = torch.Generator().manual_seed(3)
    out = {}
    for name, region in regions.items():
        v = torch.zeros(nx, ny, dtype=torch.float64)
        v[region] = torch.randn(v[region].shape, generator=gen, dtype=torch.float64)
        v = v.to(eps.device, eps.dtype)
        v[above] = 0
        for par, g, scale in (("eps_r", ge, 4.0), ("sigma", gs, 0.005)):
            adj = float((g * v).sum())
            errs = []
            for h in steps:
                hh = h * scale
                with torch.no_grad():
                    if par == "eps_r":
                        jp, jm = J(eps + hh * v, sig), J(eps - hh * v, sig)
                    else:
                        jp, jm = J(eps, sig + hh * v), J(eps, sig - hh * v)
                fd = float((jp - jm) / (2 * hh))
                errs.append(abs(fd - adj) / max(abs(fd), abs(adj), 1e-300))
            out[f"{name} / {par}"] = {"adjoint": adj, "errors": dict(zip(map(str, steps), errs))}
    return out


def check_float32_gpu():
    if not torch.cuda.is_available():
        return None
    prop, eps, sig, w, above = setup(torch.float32, "cuda")
    with torch.no_grad():
        dobs = 0.9 * prop(eps, sig, w[None])
    J = lambda e, s: 0.5 * ((prop(e, s, w[None]) - dobs).double() ** 2).sum()  # noqa: E731
    e, s = eps.clone().requires_grad_(True), sig.clone().requires_grad_(True)
    J(e, s).backward()
    return fd_sweep(J, eps, sig, e.grad, s.grad, above, (1e-1, 1e-2, 1e-3, 1e-4))


def check_native():
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
    return {"data": rel(d2.detach(), d1.detach()), "grad_eps_r": rel(e2.grad, e1.grad),
            "grad_sigma": rel(s2.grad, s1.grad)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"float64_cpu": checks_float64(), "native_pure_yee_float32": check_native(),
           "float32_gpu_fd": check_float32_gpu()}
    with open(OUT / "phase3_gradient_check.json", "w") as fh:
        json.dump(res, fh, indent=1)
    r = res["float64_cpu"]
    print(f"source dot product: rel {r['source_dot_product']['rel']:.2e}")
    print(f"checkpoint interval: max rel diff {r['checkpoint_max_rel_diff']:.2e}")
    print(f"max |grad| above surface: {r['max_abs_grad_above_surface']:.1e}")
    print("native pure-Yee:", res["native_pure_yee_float32"])
    for tag, table in (("float64", r["fd"]), ("float32 GPU", res["float32_gpu_fd"] or {})):
        print(f"\nFD directional derivatives ({tag}), relative error per relative step h")
        for name, row in table.items():
            print(f"  {name:28s} " + " ".join(f"{e:.1e}" for e in row["errors"].values()))


if __name__ == "__main__":
    main()
