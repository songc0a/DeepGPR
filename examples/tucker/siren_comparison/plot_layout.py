"""Layout of the 3D FWI case: model, 2D/3D antenna positions, Fig. 7 slice.

Usage: ``python plot_layout.py`` -> ``results/full/figures/model3d_layout.png``
(positions as in benchmark_tucker.build_case and bench_fwi.grid_shots_3d).
"""

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
from bench_fwi import VALIDATION_SHOTS  # noqa: E402

case = bt.build_case(False, torch.device("cpu"))
eps, air, dx, c = case["eps"].numpy(), case["air"], 0.02, 0.299792458
nx, _, nz = eps.shape
two_way = 2 * np.cumsum(np.sqrt(eps[:, air:, :]) * dx / c, axis=1)  # ns
reach = (two_way < 18.0 - 2.4).sum(axis=1)  # record minus wavelet delay
print(f"two-way vertical reach: {reach.min()}-{reach.max()} cells")
grid = [(int(x), int(z)) for x in torch.linspace(nx * 0.2, nx * 0.8, 4).round() for z in torch.linspace(nz * 0.2, nz * 0.8, 4).round()]
diagonal = list(zip(torch.linspace(nx * 0.3, nx * 0.7, 4).round().int().tolist(), torch.linspace(nz * 0.3, nz * 0.7, 4).round().int().tolist()))
rx, rz = np.meshgrid(np.arange(4, nx - 4, nx // 10), np.arange(4, nz - 4, nz // 10), indexing="ij")
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
a = ax[0]
a.scatter(rz.ravel() * dx, rx.ravel() * dx, s=12, c="0.45", label="100 receivers (Ez), all shots")
held = set(VALIDATION_SHOTS["3d"])
for i, (x, z) in enumerate(grid):
    a.scatter(z * dx, x * dx, marker="*", s=150, c="none" if i in held else "tab:red", edgecolors="tab:red", linewidths=1.5)
a.scatter([], [], marker="*", s=150, c="tab:red", label="rounds 3-4: 16 sources (4 x 4)")
a.scatter([], [], marker="*", s=150, c="none", edgecolors="tab:red", label="round-4 search: held-out shots")
a.scatter([z * dx for _, z in diagonal], [x * dx for x, _ in diagonal], marker="^", s=60, c="tab:blue", label="rounds 1-2: 4 sources (diagonal)")
a.axhline(nx // 2 * dx, color="k", ls="--", lw=1)
a.text(0.02, nx // 2 * dx + 0.03, f"Fig. 7 slice (x = {nx // 2})", fontsize=8)
a.set(xlim=(0, nz * dx), ylim=(0, nx * dx), aspect="equal", xlabel=f"z (m), {nz} cells", ylabel=f"x (m), {nx} cells")
a.set_title(f"Plan view of the ground surface (y = {air} cells)", fontsize=9)
a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
b = ax[1]
im = b.imshow(eps[nx // 2], cmap="viridis", extent=(0, nz * dx, eps.shape[1] * dx, 0), aspect="equal")
b.axhline(air * dx, color="w", lw=1)
b.axhline((air + 40) * dx, color="w", ls="--", lw=1)
b.text(0.03, air * dx - 0.03, f"air ({air} cells), antennas at the surface", color="w", fontsize=8)
b.text(0.03, (air + 40) * dx - 0.03, "zone: top 40 subsurface cells", color="w", fontsize=8)
b.set(xlabel="z (m)", ylabel=f"y = depth (m), {eps.shape[1]} cells")
b.set_title(f"True eps_r at x = {nx // 2} (the Fig. 7 slice)", fontsize=9)
plt.colorbar(im, ax=b, shrink=0.8, label="eps_r")
fig.suptitle(f"3D model {nx} x {eps.shape[1]} x {nz} cells, dx = 2 cm (+10-cell CPML outside), 500 MHz Ricker, dt 30 ps, nt 600 (18 ns)", fontsize=9)
fig.tight_layout()
out = HERE / "results" / "full" / "figures" / "model3d_layout.png"
fig.savefig(out, dpi=150)
print(out)
