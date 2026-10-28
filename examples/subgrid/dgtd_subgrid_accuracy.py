"""DGTD rugged surface + FDTD with a Huygens subgrid: consistency, transparency, accuracy.

Model (model grid h = 1 cm, 1.2 x 0.9 x 0.9 m): air over a rugged surface
(``d = 0.2 + 0.04 sin(2 pi y / 0.6) cos(2 pi z / 0.6)`` m), soil eps_r 6,
2 mS/m; a 3 cm sphere (eps_r 20) at 0.66 m depth. Ricker 250 MHz, source and
two receivers 5 cm above the surface, one receiver below the target. Solvers:

* ``hybrid``:     ``HybridMaxwell3D`` on the 3 cm block averages (DG band 0.09-0.33 m);
* ``subgrid``:    the same with a ratio-3 subgrid over the sphere
  (``HybridSubgridMaxwell3D``), sub-cycled (``time_ratio=3``) and at the main dt
  (``time_ratio=1``);
* reference:      the same DG band and elements, Yee at 1 cm everywhere with the
  model itself (``HybridMaxwell3D`` at h, ``dg_h`` = 3 cm; 3 cm dipole).

Reported: the split step against ``HybridMaxwell3D`` without subgrids, the
subgrid's transparency in the target-free model, the target response (with
minus without the sphere) against the reference, run times.

    python examples/subgrid/dgtd_subgrid_accuracy.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from DeepGPR.hybrid import HybridMaxwell3D, Surface3D
from DeepGPR.subgrid import Subgrid
from DeepGPR.subgrid.hybrid import HybridSubgridMaxwell3D
from DeepGPR.subgrid.solver import block_average

OUT = Path(__file__).resolve().parent / "results" / "dgtd_subgrid_accuracy.json"
H = 0.01
SHAPE = (121, 91, 91)
SURF = Surface3D.sine(0.2, 0.04, 0.6, 0.6)
BAND = (0.09, 0.33)
BOX = ((0.60, 0.72), (0.39, 0.51), (0.39, 0.51))
F0, NT = 2.5e8, 2200
SRC = [[0.05, 0.45, 0.45]]
REC = [[0.05, 0.60, 0.45], [0.05, 0.45, 0.30], [0.95, 0.45, 0.45]]


def ricker(nt, dt):
    t = np.arange(nt) * dt - 1.2 / F0
    a = (np.pi * F0 * t) ** 2
    return torch.as_tensor((1 - 2 * a) * np.exp(-a))


def models():
    eps = torch.full(SHAPE, 6.0, dtype=torch.float64)
    sig = torch.full(SHAPE, 0.002, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(*[torch.arange(n, dtype=torch.float64) * H for n in SHAPE],
                             indexing="ij")
    ball = ((X - 0.66) ** 2 + (Y - 0.45) ** 2 + (Z - 0.45) ** 2) <= 0.03 ** 2
    return eps, torch.where(ball, torch.full_like(eps, 20.0), eps), sig


def rel(a, b):
    return ((a - b).norm(dim=1) / b.norm(dim=1)).cpu().numpy().ravel().tolist()


def main():
    torch.set_default_dtype(torch.float64)
    eps_b, eps_t, sig = models()
    res = {}
    runs = {}
    for name, sgs in (("hybrid", []), ("subgrid_tr3", [Subgrid(BOX)]),
                      ("subgrid_tr1", [Subgrid(BOX, time_ratio=1)])):
        out = []
        for e in (eps_t, eps_b):
            sol = HybridSubgridMaxwell3D(e, sig, H, coarsen=3, surface=SURF, dg_band=BAND,
                                         pmlthick=8, device="cuda", subgrids=sgs)
            o = sol.run(ricker(NT, sol.dt), SRC, REC)
            out.append(o["receivers"])
            res.setdefault(name, {})["seconds"] = o["seconds"]
            dt = sol.dt
        runs[name] = out
        print(f"{name}: {res[name]['seconds']:.1f} s per run ({NT} steps, dt {dt:.3e} s)", flush=True)
    ref = HybridMaxwell3D(block_average(eps_b, 3), block_average(sig, 3), 3 * H, surface=SURF,
                          dg_band=BAND, pmlthick=8, dt=dt, device="cuda")
    d = ref.run(ricker(NT, dt), SRC, REC)["receivers"]
    res["split_step_vs_HybridMaxwell3D"] = rel(runs["hybrid"][1], d)
    res["transparency"] = rel(runs["subgrid_tr3"][1], runs["hybrid"][1])
    fine = []
    for e in (eps_t, eps_b):
        sol = HybridMaxwell3D(e, sig, H, surface=SURF, dg_band=BAND, pmlthick=24, dg_h=3 * H,
                              dt=dt, device="cuda")
        o = sol.run(ricker(NT, dt), SRC, REC)
        fine.append(3.0 * o["receivers"])           # 3 cm dipole, as on the 3 cm grids
        res["reference_seconds"] = o["seconds"]
    resp_ref = fine[0] - fine[1]
    for name, (t, b) in runs.items():
        res[name]["target_response_error"] = rel(t - b, resp_ref)
        res[name]["total_field_error"] = rel(t, fine[0])
    res["time_ratio_1_vs_3"] = rel(runs["subgrid_tr1"][0] - runs["subgrid_tr1"][1],
                                   runs["subgrid_tr3"][0] - runs["subgrid_tr3"][1])
    print(json.dumps(res, indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
