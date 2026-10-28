"""Phase 1 verification: flat DGTD-FDTD interfaces in a homogeneous medium.

Measures, with the hybrid solver of ``DeepGPR.hybrid``:

1. discrete energy conservation of the coupled scheme (closed PEC box,
   central fluxes, non-matching interface meshes, P1/P2, two mesh types);
2. the artificial reflection of the interface, ``R = A_reflected / A_incident``,
   versus incidence angle, grid spacing (convergence), DG mesh ratio, order
   and flux, against pure Yee-FDTD with the same time step;
3. waveform accuracy (relative L2) of the hybrid, pure Yee and pure DGTD
   against a 8x finer Yee reference, for a wave crossing both interfaces;
4. long-time stability with CPML (energy history over 60000 steps).

Run from the repository root (CUDA recommended)::

    python examples/hybrid/phase1_flat_interface.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from DeepGPR.hybrid import HybridTM2D, Surface  # noqa: E402

C0 = 299792458.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = ROOT / "examples" / "hybrid" / "results"


def ricker(f, nt, dt, t0):
    t = np.arange(nt) * dt - t0
    a = (np.pi * f * t) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a))


def burst(nt, dt, f, t0, off_after):
    t = torch.arange(nt, dtype=torch.float64) * dt
    w = torch.exp(-((t - t0) * f * 2) ** 2) * torch.cos(2 * math.pi * f * (t - t0))
    w[off_after:] = 0
    return w


# ------------------------------------------------------------ 1. energy
def energy_conservation():
    out = {}
    nx, ny, dx = 60, 70, 0.02
    eps = torch.full((nx, ny), 2.0)
    eps[35:] = 4.0
    for mesh, opts in (("terrain", dict(h_y=0.0173, jitter=0.3)),
                       ("delaunay", dict(h_interface=0.0231))):
        for order in (1, 2):
            sol = HybridTM2D(eps, torch.zeros(nx, ny), dx, dg_band=(0.4, 0.9), pmlthick=0,
                             dg_order=order, dg_h=dx * order, mesh=mesh, mesh_options=opts,
                             flux="central", cfl_safety=0.9, device=DEVICE)
            nt = 4000
            res = sol.run(burst(nt, sol.dt, 1e9, 1.5e-9, 400), [[[0.3, 0.7]]], [[[0.5, 0.3]]],
                          energy_every=10)
            steps = np.array(res["energy_steps"])
            e = res["energy"][:, 0].cpu().numpy()[steps > 450]
            out[f"{mesh}_P{order}"] = float((e.max() - e.min()) / e.mean())
    return out


# ------------------------------------------------------------ 2. reflection
def reflection(dx):
    L, D = 3.0, 2.0
    nx, ny = int(round(D / dx)), int(round(L / dx))
    eps, sig = torch.ones(nx, ny), torch.zeros(nx, ny)
    band = (0.5, 1.0)
    xs, ys, xI = 1.3, 1.5, band[1]
    offs = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    rec = [[[1.15, ys + o] for o in offs]]
    configs = {
        "P1 h=dx central": dict(dg_order=1, dg_h=dx, flux="central"),
        "P1 h=dx alpha=0.1": dict(dg_order=1, dg_h=dx),
        "P1 h_y=0.83dx (non-matching)": dict(dg_order=1, dg_h=dx, mesh_options=dict(h_y=0.83 * dx)),
        **({"P1 h=0.5dx": dict(dg_order=1, dg_h=0.5 * dx)} if dx >= 0.01 else {}),
        "P2 h=2dx": dict(dg_order=2, dg_h=2 * dx),
        "P1 delaunay h=dx": dict(dg_order=1, dg_h=dx, mesh="delaunay"),
    }
    probes = {k: HybridTM2D(eps, sig, dx, dg_band=band, pmlthick=20, device=DEVICE, **kw)
              for k, kw in configs.items()}
    dt = min(p.dt for p in probes.values())
    nt = int(14e-9 / dt)
    w = ricker(1e9, nt, dt, 1.2e-9)
    t = np.arange(nt) * dt
    ref = HybridTM2D(eps, sig, dx, pmlthick=20, dt=dt, device=DEVICE).run(
        w, [[[xs, ys]]], rec)["receivers"][0].cpu().numpy()
    table = {}
    for name, kw in configs.items():
        sol = HybridTM2D(eps, sig, dx, dg_band=band, pmlthick=20, dt=dt, device=DEVICE, **kw)
        d = sol.run(w, [[[xs, ys]]], rec)["receivers"][0].cpu().numpy() - ref
        row = {}
        for k, (xr, yr) in enumerate(rec[0]):
            t_dir = math.hypot(xr - xs, yr - ys) / C0 + 1.2e-9
            path = math.hypot((xs - xI) + (xr - xI), yr - ys)
            t_ref = path / C0 + 1.2e-9
            angle = math.degrees(math.atan2(abs(yr - ys), (xs - xI) + (xr - xI)))
            incident = np.abs(ref[np.abs(t - t_dir) < 1e-9, k]).max()
            spreading = math.sqrt(math.hypot(xr - xs, yr - ys) / path)  # 2D line source
            row[f"{angle:.0f}deg"] = float(np.abs(d[np.abs(t - t_ref) < 1e-9, k]).max()
                                           / (incident * spreading))
        table[name] = row
    return {"dx": dx, "dt": dt, "R": table}


# ------------------------------------------------------------ 3. accuracy
def accuracy():
    src, rec = [[[0.75, 0.6]]], [[[0.15, 0.6], [0.15, 0.9]]]
    T = 8e-9

    def model(dx, **kw):
        nx, ny = int(round(1.0 / dx)), int(round(1.2 / dx))
        return HybridTM2D(torch.full((nx, ny), 2.0), torch.full((nx, ny), 0.002), dx,
                          pmlthick=int(round(0.15 / dx)), device=DEVICE, **kw)

    def trace(sol):
        nt = int(T / sol.dt)
        d = sol.run(ricker(1e9, nt, sol.dt, 1.2e-9), src, rec)["receivers"][0].cpu().numpy()
        return np.arange(nt) * sol.dt, d

    tf, ref = trace(model(0.00125))
    out = {}
    for dx in (0.02, 0.01, 0.005):
        hy = model(dx, dg_band=(0.3, 0.6), dg_order=1, dg_h=dx)
        runs = {
            "hybrid P1 h=dx": hy,
            "hybrid P2 h=2dx": model(dx, dg_band=(0.3, 0.6), dg_order=2, dg_h=2 * dx),
            "Yee, dt of hybrid": model(dx, dt=hy.dt),
            "Yee, own CFL dt": model(dx),
            "full DGTD P1 h=dx": model(dx, dg_band=(None, None), dg_order=1, dg_h=dx),
        }
        row = {}
        for name, sol in runs.items():
            t, d = trace(sol)
            r = np.stack([np.interp(t, tf, ref[:, k]) for k in range(ref.shape[1])], 1)
            row[name] = float(np.linalg.norm(d - r) / np.linalg.norm(r))
        out[f"dx={dx}"] = row
    return out


# ------------------------------------------------------------ 4. long run
def long_run():
    nx, ny, dx = 100, 150, 0.02
    surf = Surface.sine(0.9, 0.15, 1.2)
    out = {}
    for label, kw in (("alpha=0.1 delaunay", dict(mesh="delaunay")),
                      ("central delaunay", dict(mesh="delaunay", flux="central")),
                      ("alpha=0.1 terrain, abc sides", dict(mesh="terrain", side_boundary="abc"))):
        sol = HybridTM2D(torch.full((nx, ny), 6.0), torch.zeros(nx, ny), dx, surface=surf,
                         dg_band=(0.5, 1.3), pmlthick=15, device=DEVICE, **kw)
        nt = 60000
        w = burst(nt, sol.dt, 3e8, 4e-9, int(1e-8 / sol.dt))
        t0 = time.perf_counter()
        res = sol.run(w, [[[0.6, 1.5]]], [[[0.6, 1.0]]], energy_every=500)
        e = res["energy"][:, 0].cpu().numpy()
        peak = int(np.argmax(e))
        out[label] = {"steps": nt, "simulated_ns": nt * sol.dt * 1e9,
                      "final_over_peak_energy": float(e[-1] / e[peak]),
                      "finite": bool(np.isfinite(e).all()), "seconds": time.perf_counter() - t0}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"device": str(DEVICE)}
    summary["energy_conservation_rel_variation"] = energy_conservation()
    print(json.dumps(summary, indent=1), flush=True)
    summary["artificial_reflection"] = [reflection(dx) for dx in (0.02, 0.01, 0.005)]
    print(json.dumps(summary["artificial_reflection"], indent=1), flush=True)
    summary["accuracy_vs_fine_yee"] = accuracy()
    print(json.dumps(summary["accuracy_vs_fine_yee"], indent=1), flush=True)
    summary["long_run"] = long_run()
    print(json.dumps(summary["long_run"], indent=1), flush=True)
    with open(OUT / "phase1_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
