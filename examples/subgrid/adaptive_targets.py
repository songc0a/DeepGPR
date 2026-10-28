"""Adaptive subgridding (a priori plan) on a 3D GPR model: accuracy, time, memory.

Model (model grid h = 1 cm, 1.5 x 2.1 x 2.1 m): 0.21 m of air over soil
(eps_r 6, 3 mS/m) with two targets the 3 cm main grid cannot represent: a
sphere (radius 4 cm, eps_r 20) and an air-filled pipe segment (radius 5 cm,
0.4 m long). :func:`DeepGPR.subgrid.adapt.plan_refinement` (f_max 400 MHz,
10 points per wavelength) chooses the main spacing and the subgrid boxes;
the scattered field of the targets (total minus the target-free background,
each solver with its own background) is compared with a uniform grid at the
model spacing for

* ``coarse``:  the main grid alone (3 cm);
* ``subgrid``: the main grid with the planned subgrids (1 cm inside);
* ``fine``:    the uniform 1 cm grid (reference).

    python examples/subgrid/adaptive_targets.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from DeepGPR.subgrid import Subgrid, SubgridMaxwell3D, fine_source_samples
from DeepGPR.subgrid.adapt import plan_refinement

OUT = Path(__file__).resolve().parent / "results"
H = 0.01
SHAPE = (151, 211, 211)
F0 = float(os.environ.get("F0", "1.6e8"))
NT = int(os.environ.get("NT", "480"))
AIR = os.environ.get("AIR", "1") == "1"
PEC = os.environ.get("PEC", "0") == "1"
TAG = os.environ.get("TAG", "")
DTYPE = torch.float32


def model(targets=True):
    eps = np.full(SHAPE, 6.0)
    sig = np.full(SHAPE, 0.003)
    if AIR:
        eps[:21] = 1.0
        sig[:21] = 0.0
    if targets:
        X, Y, Z = np.meshgrid(*[np.arange(n) * H for n in SHAPE], indexing="ij")
        ball = ((X - 0.6) ** 2 + (Y - 0.7) ** 2 + (Z - 1.05) ** 2) <= 0.04 ** 2
        pipe = ((X - 0.9) ** 2 + (Z - 1.05) ** 2 <= 0.05 ** 2) & (Y > 1.1) & (Y < 1.5)
        if PEC:
            sig[ball] = 1e8
            sig[pipe] = 1e8
        else:
            eps[ball] = 20.0
            eps[pipe] = 1.0
            sig[pipe] = 0.0
    return eps, sig


def ricker(nt, dt):
    t = np.arange(nt) * dt - 1.2 / F0
    a = (np.pi * F0 * t) ** 2
    return torch.as_tensor((1 - 2 * a) * np.exp(-a))


SRC = [[0.15, 0.85, 1.05]]
REC = [[0.15, y, 1.05] for y in np.arange(0.45, 1.66, 0.15)]


def run(kind, eps, sig, plan, dt=None):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    e = torch.as_tensor(eps, dtype=DTYPE)
    s = torch.as_tensor(sig, dtype=DTYPE)
    t0 = time.perf_counter()
    if kind == "fine":
        R = plan.coarsen
        sol = SubgridMaxwell3D(e, s, H, coarsen=1, pmlthick=10 * R, dt=dt / R,
                               dipole_length=R * H, device="cuda", dtype=DTYPE)
        w = ricker(NT, dt)
        wf = fine_source_samples(w[None], R)[0]
        shift = (R - 1) // 2
        wf = torch.cat([torch.zeros(shift, dtype=wf.dtype), wf, torch.zeros(R, dtype=wf.dtype)])
        out = sol.run(wf[:R * NT], SRC, REC)
        data = out["receivers"][0][R - 1::R].cpu().double()
    else:
        sgs = plan.subgrids if kind == "subgrid" else []
        sol = SubgridMaxwell3D(e, s, H, coarsen=plan.coarsen, subgrids=sgs, pmlthick=10,
                               device="cuda", dtype=DTYPE)
        out = sol.run(ricker(NT, sol.dt), SRC, REC)
        data = out["receivers"][0].cpu().double()
    total = time.perf_counter() - t0
    mem = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
    summ = sol.summary()
    info = dict(seconds=out["seconds"], total_seconds=total, peak_GB=mem,
                node_updates_per_main_step=summ["node_updates_per_step"] * (plan.coarsen if kind == "fine" else 1),
                dt=sol.dt)
    del sol
    torch.cuda.empty_cache()
    return data, info


def main():
    eps, sig = model(True)
    eps0, sig0 = model(False)
    t0 = time.perf_counter()
    plan = plan_refinement(eps, sig, H, f_max=4e8, ppw=10, keep_points=SRC + REC)
    plan_s = time.perf_counter() - t0
    print(plan.summary(), f"(planned in {plan_s:.2f} s)", flush=True)
    rows = {}
    dt = None
    for kind in ("subgrid", "coarse", "fine"):
        d1, info = run(kind, eps, sig, plan, dt)
        d0, _ = run(kind, eps0, sig0, plan, dt)
        dt = info["dt"] if kind != "fine" else dt
        rows[kind] = dict(info, scattered=(d1 - d0).numpy(), total=d1.numpy())
        print(f"{kind:8s} {info['seconds']:7.2f} s  peak {info['peak_GB']:.2f} GB  "
              f"node updates / main step {info['node_updates_per_main_step'] / 1e6:.1f} M", flush=True)
    ref = rows["fine"]["scattered"]
    res = {"plan": plan.summary(), "plan_seconds": plan_s}
    for kind in ("subgrid", "coarse"):
        e = np.linalg.norm(rows[kind]["scattered"] - ref) / np.linalg.norm(ref)
        et = np.linalg.norm(rows[kind]["total"] - rows["fine"]["total"]) / np.linalg.norm(rows["fine"]["total"])
        res[kind] = dict(scattered_error=float(e), total_error=float(et),
                         **{k: v for k, v in rows[kind].items() if k not in ("scattered", "total")})
        print(f"{kind:8s} scattered-field error {e:.3f}, total-field error {et:.3f}")
    res["fine"] = {k: v for k, v in rows["fine"].items() if k not in ("scattered", "total")}
    OUT.mkdir(parents=True, exist_ok=True)
    res["config"] = dict(F0=F0, NT=NT, AIR=AIR, PEC=PEC)
    (OUT / f"adaptive_targets{TAG}.json").write_text(json.dumps(res, indent=1))
    np.savez(OUT / f"adaptive_targets{TAG}_traces.npz", **{f"{k}_{q}": rows[k][q] for k in rows
                                                            for q in ("scattered", "total")})
    print("wrote", OUT / f"adaptive_targets{TAG}.json")


if __name__ == "__main__":
    sys.exit(main())
