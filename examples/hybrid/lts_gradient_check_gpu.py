"""LTS vs global gradients on the GPU: isolates float32, fused kernels and the transpose path.

Small 3D model (the one of tests/test_hybrid_lts.py); gradient of J = 1/2 |d|^2 dt over 6 ns.
"""
import os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from DeepGPR.hybrid import HybridMaxwell3D, HybridPropagator, Surface3D
shape, dx = (14, 12, 11), 0.02
eps = 3 + 3 * torch.rand(shape, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
sig = torch.full(shape, 0.003, dtype=torch.float64)
surf = Surface3D.sine(0.14, 0.02, 0.24, 0.2)
src = np.array([[[0.03, 0.12, 0.1]]]); rec = np.array([[[0.25, 0.1, 0.1], [0.12, 0.14, 0.08], [0.05, 0.05, 0.15]]])
def ricker(nt, dt, f0, t0):
    t = (np.arange(nt) + 0.5) * dt - t0; a = (np.pi * f0 * t) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a))
def grad(lts, dtype, cuda_kernels, autograd=False, T=6e-9):
    if autograd: os.environ["DEEPGPR_HYBRID_AUTOGRAD_VJP"] = "1"
    try:
        sol = HybridMaxwell3D(eps, sig, dx, surface=surf, dg_band=(0.06, 0.22), pmlthick=6, flux="central",
                              lts=lts, device="cuda", dtype=dtype, cuda_kernels=cuda_kernels)
        prop = HybridPropagator(sol, src, rec)
        nt = int(T / sol.dt); w = ricker(nt, sol.dt, 0.8e9, 2e-9).to("cuda", dtype)
        e = eps.to("cuda", dtype).requires_grad_(True); s = sig.to("cuda", dtype).requires_grad_(True)
        d = prop(e, s, w[None]); (0.5 * (d ** 2).sum() * sol.dt).backward()
        return e.grad.double().cpu().numpy(), s.grad.double().cpu().numpy(), sol.lts
    finally:
        os.environ.pop("DEEPGPR_HYBRID_AUTOGRAD_VJP", None)
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(a))
ref_e, ref_s, _ = grad(None, torch.float64, True)                     # global, f64, fused, hand-written
for name, kw in (("global f64 fused autograd", dict(lts=None, dtype=torch.float64, cuda_kernels=True, autograd=True)),
                 ("global f64 pytorch", dict(lts=None, dtype=torch.float64, cuda_kernels=False)),
                 ("lts f64 pytorch", dict(lts="auto", dtype=torch.float64, cuda_kernels=False)),
                 ("lts f64 fused", dict(lts="auto", dtype=torch.float64, cuda_kernels=True)),
                 ("global f32 fused", dict(lts=None, dtype=torch.float32, cuda_kernels=True)),
                 ("lts f32 fused", dict(lts="auto", dtype=torch.float32, cuda_kernels=True)),
                 ("lts f32 pytorch", dict(lts="auto", dtype=torch.float32, cuda_kernels=False))):
    ge, gs, p = grad(**kw)
    print(f"{name:28s} p={p}: eps {rel(ref_e, ge):.4f} sigma {rel(ref_s, gs):.4f}  (vs global f64 fused hand-written)", flush=True)
