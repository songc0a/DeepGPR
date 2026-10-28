"""Phase 4: artificial reflection and transmission error of the 3D DGTD band.

Homogeneous ground (eps_r 6, no surface): a DGTD band 0.20-0.44 m (P1, 2 cm)
in the middle of the model. Any difference between the hybrid and the pure Yee
scheme on the *same* Yee grid at receivers above the band is a spurious
reflection from the band; below the band it is the transmission error (DG
dispersion and both interfaces). Ez dipole 0.10 m above the band top, 400 MHz
Ricker (15 Yee cells per wavelength at 2 cm, 45 with ``fdtd_refinement=3``).
Both runs use the hybrid's time step. The side absorbers differ (UPML in the
hybrid, DeepGPR's CPML in pure-Yee mode), which adds their difference in
absorption to the numbers.

    python examples/hybrid/phase4_interface_reflection.py
"""

from __future__ import annotations

import json

import numpy as np
import torch

import models3d as M
from DeepGPR.hybrid.solver3d import HybridMaxwell3D

OUT = M.ROOT / "examples" / "hybrid" / "results"
SRC = np.array([[[0.10, 0.60, 0.61]]])
YS = np.round(np.arange(0.30, 0.91, 0.06), 6)
YS = YS[np.abs(YS - 0.60) > 0.1]          # skip the source near field
ABOVE = [[0.10, y, 0.61] for y in YS]
BELOW = [[0.62, y, 0.61] for y in YS]


def gather(r, band, dt):
    eps = np.full(M.SHAPE, M.EPS_SOIL)
    sig = np.full(M.SHAPE, M.SIG_SOIL)
    sol = HybridMaxwell3D(eps, sig, M.DX, dg_band=band, pmlthick=M.PML, fdtd_refinement=r,
                          device="cuda", dtype=torch.float32, dt=dt)
    nt = int(np.ceil(M.T_MAX / sol.dt)) + 1
    res = sol.run(torch.tensor(M.wavelet(nt, sol.dt)), SRC, np.array([ABOVE + BELOW]))
    return res["receivers"][0].double().cpu().numpy(), sol.dt


def main():
    out = {}
    for r in (1, 3):
        hyb, dt = gather(r, M.BAND, None)
        # Pure Yee with the hybrid's dt (same time discretisation, same Yee grid);
        # pure-Yee mode keeps DeepGPR's CPML on all faces.
        yee, _ = gather(r, None, dt)
        n = YS.size
        a = (slice(None), slice(0, n))
        b = (slice(None), slice(n, 2 * n))
        rel = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))  # noqa: E731
        peak = lambda x, y: float(np.abs(x - y).max() / np.abs(y).max())  # noqa: E731
        out[f"r{r}"] = {"above_rel_l2": rel(hyb[a], yee[a]), "above_peak": peak(hyb[a], yee[a]),
                        "below_rel_l2": rel(hyb[b], yee[b]), "below_peak": peak(hyb[b], yee[b]),
                        "dt": dt}
        print(f"r = {r}: above the band (reflection) rel L2 {out[f'r{r}']['above_rel_l2']:.2e}, "
              f"peak {out[f'r{r}']['above_peak']:.2e}; below (transmission) rel L2 "
              f"{out[f'r{r}']['below_rel_l2']:.2e}, peak {out[f'r{r}']['below_peak']:.2e}",
              flush=True)
        torch.cuda.empty_cache()
    with open(OUT / "phase4_interface_reflection.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
