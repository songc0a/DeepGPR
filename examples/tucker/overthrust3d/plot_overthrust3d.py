"""Slices and convergence of the Overthrust FWI (``fwi_overthrust3d.py``).

Reads ``<run>/slices/it*.npz`` (it-002 = true model, it-001 = start),
``<run>/history.json`` and ``<run>/meta.json``; writes ``<run>/overthrust_fwi.png``.
Vertical slices are drawn depth down (air rows cut off); depth slices as maps.
The last model row is the update (FWI - start, diverging scale) with the shots
marked, since it is hard to see on the permittivity scale.

    python plot_overthrust3d.py <run dir> [iteration] [--method "exact gradient, 1 GPU"]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("iteration", nargs="?", type=int, default=None)
parser.add_argument("--method", default="Tucker r16 history, 2 GPUs", help="gradient / GPU text for the title")
args = parser.parse_args()
run = Path(args.run)
meta = json.loads((run / "meta.json").read_text())
history = json.loads((run / "history.json").read_text()) if (run / "history.json").exists() else []
files = sorted(run.glob("slices/it*.npz"), key=lambda p: int(p.stem[2:]))
done = [p for p in files if int(p.stem[2:]) >= 0]
if args.iteration is not None:
    done = [p for p in done if int(p.stem[2:]) == args.iteration]
latest = done[-1] if done else None
air = meta["shape"][1] - meta["ground_shape"][1]
nx, depth, nz = meta["ground_shape"]
lx, ld, lz = meta["size_m"]
rows = [("true", run / "slices" / "it-002.npz"), ("start", run / "slices" / "it-001.npz")]
if latest is not None:
    rows.append((f"FWI, iteration {int(latest.stem[2:]) + 1}", latest))
data = [(name, dict(np.load(path)), dict(cmap="viridis", vmin=3.2, vmax=8.0)) for name, path in rows]
depth_idx = data[0][1]["depth_index"]
acq = json.loads((run / "acquisition.json").read_text()) if (run / "acquisition.json").exists() else {}
if latest is not None:
    fwi, start = data[2][1], data[1][1]
    update = {k: fwi[k] - start[k] for k in ("x_slice", "z_slice", "depth_slices")}
    rms = float(np.sqrt(np.mean(update["x_slice"][air:] ** 2)))
    lim = max(0.5, float(np.ceil(3 * rms * 2) / 2))  # +-0.5 unless the update is large
    data.append(("update (FWI - start)", update, dict(cmap="RdBu_r", vmin=-lim, vmax=lim)))

fig = plt.figure(figsize=(16, 3.6 * len(data) + (2.6 if history else 0)))
grid = fig.add_gridspec(len(data) + (1 if history else 0), 5, width_ratios=[1.6, 1.6, 1, 1, 1], hspace=0.45, wspace=0.32, top=0.955)
dxm = lx / nx
for r, (name, d, kw) in enumerate(data):
    ax = fig.add_subplot(grid[r, 0])
    ax.imshow(d["x_slice"][air:], extent=(0, lz, ld, 0), aspect="auto", **kw)  # (depth, z)
    ax.set_title(f"{name}: x = {lx / 2:.1f} m", fontsize=9)
    ax.set_ylabel("depth (m)")
    ax.set_xlabel("z (m)")
    ax = fig.add_subplot(grid[r, 1])
    image = ax.imshow(d["z_slice"][:, air:].T, extent=(0, lx, ld, 0), aspect="auto", **kw)
    ax.set_title(f"{name}: z = {lz / 2:.1f} m", fontsize=9)
    ax.set_xlabel("x (m)")
    for k in range(3):
        ax = fig.add_subplot(grid[r, 2 + k])
        ax.imshow(d["depth_slices"][k].T, extent=(0, lx, lz, 0), aspect="equal", **kw)
        ax.set_title(f"depth {(depth_idx[k] - air + 0.5) * ld / depth:.2f} m", fontsize=9)
        ax.set_xlabel("x (m)")
        if k == 0:
            ax.set_ylabel("z (m)")
        if kw["cmap"] == "RdBu_r" and acq.get("sources"):
            src = np.array(acq["sources"], dtype=float)
            ax.plot((src[:, 0] + 0.5) * dxm, (src[:, 2] + 0.5) * dxm, "k+", ms=6, mew=1.2)
    fig.colorbar(image, ax=fig.axes[-5:], shrink=0.85, pad=0.01,
                 label="relative permittivity" if kw["cmap"] == "viridis" else "change in permittivity")
if history:
    it = np.array([h["iteration"] + 1 for h in history])
    ax = fig.add_subplot(grid[len(data), 0:2])
    ax.plot(it, [h["loss"] for h in history], color="#2a6fdb", lw=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("MS-SSIM misfit (0 = perfect)")
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(grid[len(data), 2:5])
    for key, label, color in (("rmse", "all", "#222222"), ("rmse_upper", "upper half", "#2a6fdb"), ("rmse_lower", "lower half", "#d9822b")):
        ax.plot(it, [h[key] for h in history], label=label, color=color, lw=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("eps RMSE")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.grid(alpha=0.3)
n_shots = len(acq.get("sources", [None] * 10))
layout = f"{acq['shot_grid']} x {acq['shot_grid']}" if acq.get("shot_grid") else "3-4-3"
lines = acq.get("receiver_lines", [78, 78])
fig.suptitle(f"3D Overthrust GPR FWI: {lx:.1f} x {lz:.1f} x {ld:.2f} m, dx {100 * dxm:.0f} cm, {meta['cells'] / 1e6:.1f} M cells, nt {meta['nt']}, "
             f"{n_shots} shots ({layout}), {lines[0] * lines[1]} receivers, {args.method}", fontsize=11, y=0.985)
fig.savefig(run / "overthrust_fwi.png", dpi=110, bbox_inches="tight")
print(run / "overthrust_fwi.png")
