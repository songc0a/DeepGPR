"""2D speed of the hybrid DGTD-FDTD step: forward and gradient wall time per simulated ns,
against pure Yee FDTD on the same grid (same kernels, same adjoint machinery). Timing and
profiling are those of ``glue_benchmark3d.py``.

Models:

* ``F_*``: the sand-pit field profile of the 2D field FWI (``realfwi_dzg/hyb2d.py``,
  run ``H_l1_v2``): 1 cm grid 172 x 973 with 20-cell PML, flat sand surface 20 cm below
  the top, P1 band 1 cm above to 5 cm below it (default mesh, flux 0.1), 10 neighbouring
  training shots of the 500 MHz common-offset profile (0.18 m offset, antennas 2 cm
  above the sand), 32 ns. ``F_hybrid``, ``F_fdtd`` (pure Yee 1 cm).
* ``R*_*``: the rugged model of ``rugged2d_study.py`` with the settings of
  ``lts_benchmark2d.py`` (terrain-following P1 band h = 1.25 cm, central fluxes),
  14 shots, 24 ns. ``R3_*``: Yee 2.5/3 cm (the study's setting), ``R1_*``: Yee 2.5 cm
  (the band's step is far below the Yee limit). ``*_global``, ``*_lts`` (``lts="auto"``),
  ``R3_fdtd`` (pure Yee 2.5/3 cm, staircase).

    python examples/hybrid/glue_benchmark2d.py wall --configs F_hybrid F_fdtd --tsim 32e-9 \\
        --out results.json [--save DIR]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import glue_benchmark3d as G

DEV = G.DEV


def config_F(kind):
    from DeepGPR.hybrid import HybridTM2D, Surface
    h, nx, ny, na, arow, pml = 0.01, 172, 973, 20, 18, 20
    dtr, sep, y0 = 0.04943, 0.18, -0.61
    eps = torch.full((nx, ny), 4.6, dtype=torch.float32, device=DEV)
    sig = torch.full((nx, ny), 1e-3, dtype=torch.float32, device=DEV)
    eps[:na], sig[:na] = 1.0, 0.0
    traces = np.arange(80, 100, 2)                   # one batch of 10 training traces
    tx = np.round((traces * dtr - sep / 2 - y0) / h)
    rx = np.round((traces * dtr + sep / 2 - y0) / h)
    src = np.stack([np.full(len(tx), arow * h), tx * h], 1)[:, None, :]
    rec = np.stack([np.full(len(rx), arow * h), rx * h], 1)[:, None, :]
    x_s = na * h
    band = None if kind == "fdtd" else (x_s - 0.01, x_s + 0.05)
    sol = HybridTM2D(eps, sig, h, surface=Surface.flat(x_s), dg_band=band, pmlthick=pml,
                     flux=0.1, fdtd_refinement=1, nshot=len(traces), device=DEV,
                     dtype=torch.float32)

    def wav(nt, dt):
        t = np.arange(nt) * dt - 3.0e-9
        a = (np.pi * 5e8 * t) ** 2
        return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=torch.float32, device=DEV)[None]
    return sol, src, rec, eps, sig, wav


def config_R(r, kind):
    import lts_benchmark2d as L
    import rugged2d_study as R
    from DeepGPR.hybrid import HybridTM2D
    model = R.true_model()
    src, rec = R.antennas()
    if kind == "fdtd":
        sol = HybridTM2D(model[0], model[1], R.DX, surface=R.SURFACE, dg_band=None,
                         pmlthick=int(round(R.PML_M / R.DX)), fdtd_refinement=r,
                         nshot=len(R.SRC_Y), device=DEV, dtype=torch.float32)
    else:
        kw = dict(lts="auto") if kind == "lts" else {}
        sol = L.build(model, r, nshot=len(R.SRC_Y), **kw)
    eps = torch.tensor(model[0], dtype=torch.float32, device=DEV)
    sig = torch.tensor(model[1], dtype=torch.float32, device=DEV)

    def wav(nt, dt):
        return R.wavelet(nt, dt).to(DEV, torch.float32)[None]
    return sol, src[:, None, :], rec, eps, sig, wav


G.CONFIGS.update({
    "F_hybrid": lambda: config_F("hybrid"), "F_fdtd": lambda: config_F("fdtd"),
    "R3_global": lambda: config_R(3, "global"), "R3_lts": lambda: config_R(3, "lts"),
    "R3_fdtd": lambda: config_R(3, "fdtd"), "R1_global": lambda: config_R(1, "global"),
    "R1_lts": lambda: config_R(1, "lts")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("wall", "prof"))
    ap.add_argument("--configs", nargs="+", default=["F_hybrid", "F_fdtd"])
    ap.add_argument("--tsim", type=float, default=32e-9)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--forward-only", action="store_true")
    args = ap.parse_args()
    res = (G.wall(args.configs, args.tsim, args.reps, args.save, args.forward_only)
           if args.mode == "wall" else G.prof(args.configs, args.tsim))
    res["_meta"] = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                        time=time.strftime("%Y-%m-%d %H:%M:%S"), tsim=args.tsim, reps=args.reps)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
