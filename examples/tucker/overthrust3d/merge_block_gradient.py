"""Combine the parts written by check_block_gradient.py and draw the gradient-check figures.

    python merge_block_gradient.py gradcheck_block_4cm [--models init it30 it50]

Per model (parts <model>_*.json/npz): per-shot relative error / cosine of each compressed gradient against the
exact one, the same for the gradient summed over all shots (what the FWI uses), and depth profiles of the error.
Writes summary.json, gradcheck_summary.png and gradcheck_sections_<model>.png into the folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"block0.01": "#2e9e6a", "block0.02": "#2a6fdb", "block0.03": "#d9822b", "r16": "#c0392b"}
TITLES = {"init": "start model (smoothed true, it 0)", "it30": "exact-FWI model after 30 its",
          "it50": "exact-FWI model after 50 its (final)"}


def label(n: str) -> str:
    return f"block Tucker tol {float(n[5:]) * 100:g} %" if n.startswith("block") else f"global Tucker {n} (FWI setting)"


def load(folder: Path, model: str):
    parts = sorted(folder.glob(f"{model}_*.json"))
    infos = [json.loads(p.read_text()) for p in parts]
    arrays = [np.load(p.with_suffix(".npz")) for p in parts]
    names = list(infos[0]["per_shot"])
    per_shot = {n: [r for i in infos for r in i["per_shot"][n]] for n in names}
    shots = [s for i in infos for s in i["shots"]]
    sums = {n: sum(a[f"sum_{n}"].astype(np.float64) for a in arrays) for n in names}
    rows = {n: np.concatenate([a[f"rows_{n}"] for a in arrays]) for n in names}
    return infos[0], names, shots, per_shot, sums, rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--models", nargs="+", default=["init", "it30", "it50"])
    p.add_argument("--run", default=None, help="run dir with eps_true.npy (default: the parts' run)")
    a = p.parse_args()
    folder = Path(a.folder)
    models = [m for m in a.models if list(folder.glob(f"{m}_*.json"))]
    summary = {}
    data = {}
    for m in models:
        info, names, shots, per_shot, sums, rows = load(folder, m)
        ref = sums["exact"]
        dx, air = info["dx"], info["air"]
        depth = (np.arange(ref.shape[1]) + 0.5) * dx
        res = dict(model=info["model"], shots=sorted(shots), methods={})
        for n in names:
            if n == "exact":
                continue
            g = sums[n]
            diff = g - ref
            rel = float(np.linalg.norm(diff) / np.linalg.norm(ref))
            cos = float((g * ref).sum() / (np.linalg.norm(g) * np.linalg.norm(ref)))
            ps = per_shot[n]
            res["methods"][n] = dict(
                summed_rel=rel, summed_cos=cos,
                summed_rel_below_1m=float(np.linalg.norm(diff[:, depth > 1.0]) / np.linalg.norm(ref[:, depth > 1.0])),
                summed_rel_below_2m=float(np.linalg.norm(diff[:, depth > 2.0]) / np.linalg.norm(ref[:, depth > 2.0])),
                shot_rel_median=float(np.median([r["rel"] for r in ps])), shot_rel_max=float(max(r["rel"] for r in ps)),
                shot_rel_min=float(min(r["rel"] for r in ps)),
                shot_cos_min=float(min(r["cos"] for r in ps)),
                corr_mean=float(np.mean([r["corr"] for r in ps])),
                seconds_mean=float(np.mean([r["seconds"] for r in ps])),
                peak_gib_max=float(max(r["peak_gib"] for r in ps)),
                stored_gib_mean=(float(np.mean([r["stored_bytes"] for r in ps])) / 2**30 if ps[0]["stored_bytes"] else None),
                ratio_mean=(float(np.mean([r["raw_bytes"] / r["stored_bytes"] for r in ps])) if ps[0]["stored_bytes"] else None),
            )
        ex = per_shot["exact"]
        res["exact"] = dict(corr_mean=float(np.mean([r["corr"] for r in ex])), seconds_mean=float(np.mean([r["seconds"] for r in ex])),
                            peak_gib_max=float(max(r["peak_gib"] for r in ex)))
        summary[m] = res
        data[m] = (info, names, shots, per_shot, sums, rows, depth)
        print(m, json.dumps(res, indent=1))
    (folder / "summary.json").write_text(json.dumps(summary, indent=1))

    # ---- summary figure: depth profiles of the summed-gradient error + per-shot errors ----------------------
    fig = plt.figure(figsize=(16, 9.5))
    g = fig.add_gridspec(2, len(models), height_ratios=[1.25, 1], hspace=0.42, wspace=0.22, left=0.06, right=0.98,
                         top=0.83, bottom=0.07)
    for k, m in enumerate(models):
        info, names, shots, per_shot, sums, rows, depth = data[m]
        ax = fig.add_subplot(g[0, k])
        ref = sums["exact"]
        refrow = np.sqrt((ref ** 2).sum(axis=(0, 2)))
        for n in names[1:]:
            err = np.sqrt(((sums[n] - ref) ** 2).sum(axis=(0, 2))) / np.maximum(refrow, 1e-300)
            ax.plot(err * 100, depth, color=COLORS.get(n, "k"), lw=1.6, label=label(n))
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_xlabel("relative error of the 16-shot gradient in each depth row (%)")
        if k == 0:
            ax.set_ylabel("depth below the surface (m)")
        ax.set_title(f"{TITLES.get(m, m)}\nsummed over {len(shots)} shots", fontsize=10, pad=22)
        ax.grid(alpha=0.3, which="both")
        ax.plot([], [], color="#999999", lw=1, ls="--", label="exact gradient norm per row (top axis, max = 1)")
        if k == 0:
            fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=5,
                       frameon=False, fontsize=9)
        ax2 = ax.twiny()
        ax2.plot(refrow / refrow.max(), depth, color="#999999", lw=1, ls="--")
        ax2.set_xlim(0, 1.05)
        ax2.tick_params(labelsize=7, colors="#777777")

        ax = fig.add_subplot(g[1, k])
        for j, n in enumerate(names[1:]):
            vals = np.array([r["rel"] for r in per_shot[n]]) * 100
            x = j + np.linspace(-0.25, 0.25, len(vals))
            ax.scatter(x, vals, s=16, color=COLORS.get(n, "k"), zorder=3)
            ax.plot([j - 0.32, j + 0.32], [np.median(vals)] * 2, color="k", lw=1.2)
            v = summary[m]["methods"][n]["summed_rel"] * 100
            ax.plot([j], [v], marker="*", ms=13, color="k", zorder=4)
            ax.annotate(f"{v:.2f} %", (j + 0.34, v), fontsize=8, va="center")
        ax.set_xticks(range(len(names) - 1))
        ax.set_xticklabels([label(n).replace(" (FWI setting)", "").replace("block Tucker ", "block\n").replace("global Tucker ", "global\n")
                            for n in names[1:]], fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel("gradient relative error (%)" if k == 0 else "")
        ax.set_title("per shot (dots, line = median), star = 16-shot sum", fontsize=9)
        ax.grid(alpha=0.3, axis="y", which="both")
    fig.suptitle("Overthrust 4 cm (19.8 M cells, 250 MHz, 1402 steps), MS-SSIM eps gradient: compressed wavefield history vs "
                 "exact (two-level checkpointing)\nblock Tucker = 32^3 blocks, rank per block from the tolerance, adjoint "
                 "shadow from the surface receivers, rhs from E; error = ||g - g_exact|| / ||g_exact|| over the ground",
                 fontsize=10.5, y=0.985)
    path = folder / "gradcheck_summary.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(path)

    # ---- gradient sections per model -------------------------------------------------------------------------
    for m in models:
        info, names, shots, per_shot, sums, rows, depth = data[m]
        ref = sums["exact"]
        nx, ny, nz = ref.shape
        dx = info["dx"]
        show = [n for n in ("block0.03", "block0.01", "r16") if n in sums]
        iz = nz // 2
        iy = int(np.argmin(np.abs(depth - 1.5)))
        c = float(np.percentile(np.abs(ref), 99.5))
        fig, axes = plt.subplots(2, 1 + 2 * len(show), figsize=(4.2 * (1 + 2 * len(show)), 8.4),
                                 gridspec_kw=dict(height_ratios=[ny * 2.2, nz]), constrained_layout=True)
        panels = [("exact", ref, 1)]
        for n in show:
            panels.append((label(n), sums[n], 1))
            amp = max(1, int(round(0.3 / max(summary[m]["methods"][n]["summed_rel"], 1e-6))))
            amp = 1 if n.startswith("r") else amp
            panels.append((f"{label(n).split(' (')[0]} - exact" + (f" (x{amp})" if amp > 1 else ""), (sums[n] - ref) * amp, amp))
        for j, (title, arr, amp) in enumerate(panels):
            ax = axes[0, j]
            ax.imshow(arr[:, :, iz].T, extent=(0, nx * dx, ny * dx, 0), aspect="auto", cmap="RdBu_r", vmin=-c, vmax=c,
                      interpolation="nearest")
            ax.set_title(f"{title}\nx-depth section, z = {iz * dx:.2f} m", fontsize=8.5)
            ax.axhline(depth[iy], color="#555555", lw=0.6, ls=":")
            ax.set_xlabel("x (m)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax = axes[1, j]
            ax.imshow(arr[:, iy, :].T, extent=(0, nx * dx, nz * dx, 0), aspect="equal", cmap="RdBu_r", vmin=-c, vmax=c,
                      interpolation="nearest")
            ax.set_title(f"depth slice {depth[iy]:.2f} m", fontsize=8.5)
            ax.set_xlabel("x (m)", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0, 0].set_ylabel("depth (m)", fontsize=8)
        axes[1, 0].set_ylabel("z (m)", fontsize=8)
        fig.suptitle(f"16-shot MS-SSIM eps gradient on the {TITLES.get(m, m)}; same colour limit in every panel "
                     f"(99.5th percentile of |exact|), differences amplified where marked", fontsize=10)
        path = folder / f"gradcheck_sections_{m}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
