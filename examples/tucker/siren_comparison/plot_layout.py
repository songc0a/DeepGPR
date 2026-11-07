"""Layout of the 3D FWI case: model, 2D/3D antenna positions, Fig. 7 slice.

Usage: ``python plot_layout.py`` -> ``results/full/figures/model3d_layout.png``
(positions as in benchmark_tucker.build_case and bench_fwi.grid_shots_3d);
``python plot_layout.py --protocol v5`` -> the round-5 model
(``bench_fwi.build_case_large``), ``results/full_v5/figures/model3d_layout.png``.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import benchmark_tucker as bt  # noqa: E402
import bench_fwi  # noqa: E402
from bench_fwi import VALIDATION_SHOTS  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--protocol", default="v1", choices=["v1", "v5"])
large = parser.parse_args().protocol == "v5"
case = bench_fwi.build_case_large(False, torch.device("cpu")) if large else bt.build_case(False, torch.device("cpu"))
eps, air, c = case["eps"].numpy(), case["air"], 0.299792458
dx = case["kwargs"]["dx"]
nx, _, nz = eps.shape
if large:
    print(f"two-way vertical time to 2 m: <= {case['two_way_ns']:.1f} ns; record {case['nt'] * case['kwargs']['dt'] * 1e9:.1f} ns")
    grid = [(int(x), int(z)) for x, _, z in case["source_location"][:, 0].tolist()]
    diagonal = []
    rec = case["receiver_location"][0].numpy()
    rx, rz = rec[:, 0], rec[:, 2]
    held, zone = set(VALIDATION_SHOTS["3d_v5"]), case["zone_depth"]
else:
    two_way = 2 * np.cumsum(np.sqrt(eps[:, air:, :]) * dx / c, axis=1)  # ns
    reach = (two_way < 18.0 - 2.4).sum(axis=1)  # record minus wavelet delay
    print(f"two-way vertical reach: {reach.min()}-{reach.max()} cells")
    grid = [(int(x), int(z)) for x in torch.linspace(nx * 0.2, nx * 0.8, 4).round() for z in torch.linspace(nz * 0.2, nz * 0.8, 4).round()]
    diagonal = list(zip(torch.linspace(nx * 0.3, nx * 0.7, 4).round().int().tolist(), torch.linspace(nz * 0.3, nz * 0.7, 4).round().int().tolist()))
    rx, rz = np.meshgrid(np.arange(4, nx - 4, nx // 10), np.arange(4, nz - 4, nz // 10), indexing="ij")
    held, zone = set(VALIDATION_SHOTS["3d"]), 40
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
a = ax[0]
a.scatter(np.ravel(rz) * dx, np.ravel(rx) * dx, s=6 if large else 12, c="0.45", label=f"{np.size(rx)} receivers (Ez), all shots")
for i, (x, z) in enumerate(grid):
    a.scatter(z * dx, x * dx, marker="*", s=150, c="none" if i in held else "tab:red", edgecolors="tab:red", linewidths=1.5)
a.scatter([], [], marker="*", s=150, c="tab:red", label="round 5: 36 sources (6 x 6)" if large else "rounds 3-4: 16 sources (4 x 4)")
a.scatter([], [], marker="*", s=150, c="none", edgecolors="tab:red", label="search: held-out shots")
if diagonal:
    a.scatter([z * dx for _, z in diagonal], [x * dx for x, _ in diagonal], marker="^", s=60, c="tab:blue", label="rounds 1-2: 4 sources (diagonal)")
a.axhline(nx // 2 * dx, color="k", ls="--", lw=1)
a.text(0.02, nx // 2 * dx + 0.03, f"Fig. 7 slice (x = {nx // 2})", fontsize=8)
a.set(xlim=(0, nz * dx), ylim=(0, nx * dx), aspect="equal", xlabel=f"z (m), {nz} cells", ylabel=f"x (m), {nx} cells")
a.set_title(f"Plan view of the ground surface (y = {air} cells)", fontsize=9)
a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
b = ax[1]
im = b.imshow(eps[nx // 2], cmap="viridis", extent=(0, nz * dx, eps.shape[1] * dx, 0), aspect="equal")
b.axhline(air * dx, color="w", lw=1)
b.axhline((air + zone) * dx, color="w", ls="--", lw=1)
b.text(0.03, air * dx - 0.03, f"air ({air} cells), antennas at the surface", color="w", fontsize=8)
b.text(0.03, (air + zone) * dx - 0.03, f"zone: top {zone} subsurface cells", color="w", fontsize=8)
b.set(xlabel="z (m)", ylabel=f"y = depth (m), {eps.shape[1]} cells")
b.set_title(f"True eps_r at x = {nx // 2} (the Fig. 7 slice)", fontsize=9)
plt.colorbar(im, ax=b, shrink=0.8, label="eps_r")
kw = case["kwargs"]
freq = case["config"]["freq"] if large else 5.0e8
fig.suptitle(f"3D model {nx} x {eps.shape[1]} x {nz} cells, dx = {dx * 100:g} cm (+10-cell CPML outside), {freq / 1e6:g} MHz Ricker, dt {kw['dt'] * 1e12:g} ps, nt {case['nt']} ({case['nt'] * kw['dt'] * 1e9:.1f} ns)", fontsize=9)
fig.tight_layout()
out = HERE / "results" / ("full_v5" if large else "full") / "figures" / "model3d_layout.png"
fig.savefig(out, dpi=150)
print(out)
