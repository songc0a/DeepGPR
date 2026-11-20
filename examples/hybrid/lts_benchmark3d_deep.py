"""LTS cost on a Yee-dominated layout: deep domain, thin band (forward only, float32).

2 cm grid, 1.6 m deep, 1.2 x 1.2 m wide (``--lateral``), sine surface 0.26 +- 0.03 m, DG
band 0.20-0.32 m (6 layers of 2 cm columns), central fluxes, side UPML strength as given. Reports the
macro-step cost and the forward speed-up of lts="auto" against the global leapfrog
for the same 16 ns, plus the receiver difference.

    python examples/hybrid/lts_benchmark3d_deep.py [--strength 0.4] [--depth 1.6] [--lateral 1.2]
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from DeepGPR.hybrid import HybridMaxwell3D, Surface3D

ap = argparse.ArgumentParser()
ap.add_argument("--strength", type=float, default=0.4)
ap.add_argument("--depth", type=float, default=1.6)
ap.add_argument("--band", type=float, nargs=2, default=[0.20, 0.32])
ap.add_argument("--tmax", type=float, default=16e-9)
ap.add_argument("--lateral", type=float, default=1.2)
args = ap.parse_args()
DX = 0.02
nx = int(round(args.depth / DX)); ny = nz = int(round(args.lateral / DX))
eps = np.full((nx, ny, nz), 6.0); sig = np.full((nx, ny, nz), 0.003)
surf = Surface3D.sine(0.26, 0.03, 0.6, 0.8)
L = args.lateral
src = np.array([[[0.16, L / 2, L / 2 + 0.01]]])
rec = np.array([[[0.16, y, L / 2 + 0.01] for y in np.arange(0.1, L - 0.09, 0.1)]
                + [[1.0, y, L / 2 + 0.01] for y in np.arange(0.2, L - 0.19, 0.2)]])
def wavelet(nt, dt, f=4e8):
    t = (np.arange(nt) + 0.5) * dt - 1.2 / f; a = (np.pi * f * t) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a))
out = {}
for name, kw in (("global", {}), ("lts", dict(lts="auto"))):
    sol = HybridMaxwell3D(eps, sig, DX, surface=surf, dg_band=tuple(args.band), pmlthick=10, flux="central",
                          dg_pml_strength=args.strength, device="cuda", dtype=torch.float32, **kw)
    nt = int(np.ceil(args.tmax / sol.dt)) + 1
    w = wavelet(nt, sol.dt)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    res = sol.run(w, src, rec)
    torch.cuda.synchronize(); secs = time.perf_counter() - t0
    info = sol.summary()
    d = res["receivers"][0].double().cpu().numpy(); t = (np.arange(nt) + 1) * sol.dt
    out[name] = (d, t, secs)
    print(f"{name}: p={sol.lts} dt={sol.dt:.3e} nt={nt} {secs:.2f} s ({1e3 * secs / nt:.2f} ms/step); Yee {info['grid']} "
          f"({info['fdtd_dofs'] // 6} nodes), DG {info['dg_dofs'] // 6} nodes, K={sol.geometry.K}", flush=True)
dg, tg, sg = out["global"]; dl, tl, sl = out["lts"]
di = np.stack([np.interp(tl, tg, dg[:, k]) for k in range(dg.shape[1])], 1)
print(f"LTS vs global receivers {np.linalg.norm(di - dl) / np.linalg.norm(di):.4f}; forward speed-up {sg / sl:.2f}x")
