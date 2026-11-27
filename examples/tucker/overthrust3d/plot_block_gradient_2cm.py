"""Figure of the full-size (2 cm) single-shot gradient check: block Tucker with / without the tight source region.

    python plot_block_gradient_2cm.py gradcheck_block_2cm
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TOLS = ("0.03", "0.02", "0.015")
COLORS = {"0.03": "#d9822b", "0.02": "#2a6fdb", "0.015": "#2e9e6a"}
MODELS = {"full_init_shot5": "start model (it 0)", "full_it30_shot5": "exact-FWI model after 30 its"}


def main() -> None:
    folder = Path(sys.argv[1])
    acq = json.loads((Path(folder).parent / "run16_250MHz" / "acquisition.json").read_text())
    src = acq["sources"][5]
    dx = 0.02
    fig = plt.figure(figsize=(17, 11.5))
    g = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9], hspace=0.32, wspace=0.28, left=0.05, right=0.985, top=0.9,
                         bottom=0.06)
    summary = {}
    sections = {}
    for k, (tag, title) in enumerate(MODELS.items()):
        ref = np.load(folder / f"{tag}_shot05_exact.npy").astype(np.float64)
        depth = (np.arange(ref.shape[1]) + 0.5) * dx
        refrow = np.sqrt((ref ** 2).sum(axis=(0, 2)))
        ax = fig.add_subplot(g[0, k])
        summary[tag] = {}
        for tol in TOLS:
            for region, ls in (("", "--"), ("_src0.1", "-")):
                name = f"block{tol}{region}"
                path = folder / (f"{tag}_shot05_{name}.npy" if not region else f"{tag}{region}_shot05_{name}.npy")
                gb = np.load(path).astype(np.float64)
                diff = gb - ref
                rel = float(np.linalg.norm(diff) / np.linalg.norm(ref))
                summary[tag][name] = rel
                err = np.sqrt((diff ** 2).sum(axis=(0, 2))) / np.maximum(refrow, 1e-300)
                ax.plot(err * 100, depth, color=COLORS[tol], ls=ls, lw=1.5 if region else 1.1,
                        label=f"tol {float(tol) * 100:g} %{' + source region' if region else ''}: {rel * 100:.2f} %")
                if tag == "full_it30_shot5" and tol == "0.03":
                    sections[name] = diff[:, :, int(src[2])].copy()
                del gb, diff
        if tag == "full_it30_shot5":
            sections["exact"] = ref[:, :, int(src[2])].copy()
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_ylim(depth[-1], 0)
        ax.set_xlabel("relative error of the gradient per depth row (%)")
        ax.set_ylabel("depth (m)")
        ax.set_title(f"{title}, shot 5 (2 cm, 250 MHz)", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        ax2 = ax.twiny()
        ax2.plot(refrow / refrow.max(), depth, color="#999999", lw=1, ls=":")
        ax2.set_xlim(0, 1.05)
        ax2.tick_params(labelsize=7, colors="#777777")
        ax2.set_xlabel("exact gradient norm per row (dotted, max = 1)", fontsize=8, color="#777777")

    # bars: total error per tolerance, with/without region, both models
    ax = fig.add_subplot(g[0, 2:])
    width = 0.2
    for j, (tag, title) in enumerate(MODELS.items()):
        for i, (region, hatch) in enumerate((("", ""), ("_src0.1", "//"))):
            vals = [summary[tag][f"block{t}{region}"] * 100 for t in TOLS]
            x = np.arange(len(TOLS)) + (2 * j + i - 1.5) * width
            bars = ax.bar(x, vals, width * 0.92, color=["#c0392b", "#8a4fd3"][j], alpha=0.45 if not region else 0.9,
                          hatch=hatch, edgecolor="white", label=f"{title}{', source region at tol / 10' if region else ''}")
            for xi, v in zip(x, vals):
                ax.annotate(f"{v:.1f}", (xi, v), ha="center", va="bottom", fontsize=8)
    ax.axhline(3, color="k", lw=0.8, ls=":")
    ax.set_xticks(range(len(TOLS)))
    ax.set_xticklabels([f"block tolerance {float(t) * 100:g} %" for t in TOLS])
    ax.set_ylabel("gradient relative error, shot 5 (%)")
    ax.set_title("full-size MS-SSIM eps gradient vs exact (multi-level checkpointing); dotted = 3 %", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # sections through the source (it-30 model)
    ex = sections["exact"]
    c = float(np.percentile(np.abs(ex), 99.5))
    nx, ny = ex.shape
    extent = (0, nx * dx, ny * dx, 0)
    for j, (name, title, amp) in enumerate((("exact", "exact gradient", 1),
                                             ("block0.03", "tol 3 % - exact (x5)", 5),
                                             ("block0.03_src0.1", "tol 3 % + source region - exact (x5)", 5))):
        ax = fig.add_subplot(g[1, j])
        ax.imshow(sections[name].T * amp, extent=extent, aspect="auto", cmap="RdBu_r", vmin=-c, vmax=c, interpolation="nearest")
        ax.plot([src[0] * dx], [0], marker="v", color="k", ms=6, clip_on=False)
        ax.set_title(f"exact-FWI it-30 model: {title}\nx-depth section through the source (z = {src[2] * dx:.2f} m)", fontsize=9)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("depth (m)", fontsize=8)
        ax.tick_params(labelsize=7)
    ax = fig.add_subplot(g[1, 3])
    for name, label in (("block0.03", "tol 3 % (hidden under the other curve away from the source)"),
                        ("block0.03_src0.1", "tol 3 % + source region")):
        d = sections[name]
        rows = np.sqrt((d[:, :25] ** 2).sum(axis=1))  # top 0.5 m
        ax.plot(np.arange(nx) * dx, rows / np.sqrt((ex[:, :25] ** 2).sum(axis=1)).max(), label=label,
                color="#d9822b" if name == "block0.03" else "#8a4fd3")
    ax.plot(np.arange(nx) * dx, np.sqrt((ex[:, :25] ** 2).sum(axis=1)) / np.sqrt((ex[:, :25] ** 2).sum(axis=1)).max(),
            color="#999999", ls=":", label="exact gradient")
    ax.axvline(src[0] * dx, color="k", lw=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_title("top 0.5 m of that section: column norm of the error\n(normalised by the max column norm of the exact gradient)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.suptitle("Overthrust full size (2 cm, 138.9 M cells, 2819 steps), one shot: block-wise Tucker gradient vs exact gradient.\n"
                 "Source region = blocks within 0.6 m horizontally and 0.6 m below the surface around the source, compressed with "
                 "1/10 of the tolerance (it holds the near-singular direct wave)", fontsize=10.5)
    path = folder / "gradcheck_2cm_shot5.png"
    fig.savefig(path, dpi=105)
    (folder / "summary_2cm.json").write_text(json.dumps(summary, indent=1))
    print(path, json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
