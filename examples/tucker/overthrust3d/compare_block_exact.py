"""Block-Tucker FWI vs exact-gradient FWI on the same full-size Overthrust run (same data, start, optimiser).

Reads both run dirs (slices/, history.json, and when present snapshots/eps_itNNNN.npy and
gradients/g_itNNNN.npy) and writes a figure + json: model updates of both runs and their difference,
RMSE / gcorr / ucorr per iteration, time per iteration and stored history per shot.

    python compare_block_exact.py <exact run> <block run> <iteration index> <out dir> [block label] [file tag]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXACT, BLOCK = "#333333", "#8a4fd3"


def load_history(run: Path) -> dict:
    out = {}
    for h in json.loads((run / "history.json").read_text()):
        out[h["iteration"]] = h  # resumed runs may repeat an iteration: keep the last record
    return out


def main() -> None:
    exact_dir, block_dir, it, out = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    label = sys.argv[5] if len(sys.argv) > 5 else "block Tucker tol 3 % + source region"
    tag = sys.argv[6] if len(sys.argv) > 6 else ""
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((block_dir / "meta.json").read_text())
    nx, depth, nz = meta["ground_shape"]
    air = meta["shape"][1] - depth
    lx, ld, lz = meta["size_m"]
    he, hb = load_history(exact_dir), load_history(block_dir)
    res: dict = {"iteration_index": it}

    # ---- full-volume comparison when the snapshot / gradient files exist ----
    snap = f"eps_it{it + 1:04d}.npy"
    if (exact_dir / "snapshots" / snap).exists() and (block_dir / "snapshots" / snap).exists():
        me = np.load(exact_dir / "snapshots" / snap).astype(np.float64)
        mb = np.load(block_dir / "snapshots" / snap).astype(np.float64)
        m0 = np.load(block_dir / "eps_init.npy")[:, air:].astype(np.float64)
        mt = np.load(block_dir / "eps_true.npy")[:, air:].astype(np.float64)
        ue, ub = me - m0, mb - m0
        res["model"] = dict(
            rms_block_minus_exact=float(np.sqrt(np.mean((mb - me) ** 2))),
            rel_update_diff=float(np.linalg.norm(ub - ue) / np.linalg.norm(ue)),
            corr_updates=float((ue * ub).sum() / np.sqrt((ue * ue).sum() * (ub * ub).sum())),
            rmse_exact=float(np.sqrt(np.mean((me - mt) ** 2))), rmse_block=float(np.sqrt(np.mean((mb - mt) ** 2))))
        del me, mb, m0, mt, ue, ub
    grad = f"g_it{it:04d}.npy"
    if (exact_dir / "gradients" / grad).exists() and (block_dir / "gradients" / grad).exists():
        ge = np.load(exact_dir / "gradients" / grad).astype(np.float64)
        gb = np.load(block_dir / "gradients" / grad).astype(np.float64)
        res["gradient_same_index"] = dict(  # the two runs are at (slightly) different models here
            rel_diff=float(np.linalg.norm(gb - ge) / np.linalg.norm(ge)),
            corr=float((ge * gb).sum() / np.sqrt((ge * ge).sum() * (gb * gb).sum())))
        del ge, gb

    # ---- figure ----
    se = dict(np.load(exact_dir / "slices" / f"it{it:04d}.npz"))
    sb = dict(np.load(block_dir / "slices" / f"it{it:04d}.npz"))
    s0 = dict(np.load(block_dir / "slices" / "it-001.npz"))
    di = int(se["depth_index"][0])
    panels = []
    for name, s in ((f"exact: update after {it + 1} its", se), (f"block Tucker: update after {it + 1} its", sb)):
        panels.append((name, {k: s[k] - s0[k] for k in ("x_slice", "z_slice", "depth_slices")}))
    panels.append(("block - exact", {k: sb[k] - se[k] for k in ("x_slice", "z_slice", "depth_slices")}))
    lim = float(np.ceil(4 * np.sqrt(np.mean(panels[0][1]["x_slice"][air:] ** 2)) * 4) / 4)
    lim = max(lim, 0.25)
    dlim = float(4 * np.sqrt(np.mean(panels[2][1]["x_slice"][air:] ** 2)))  # the difference gets its own scale
    dlim = float(f"{max(dlim, 1e-3):.1g}")
    fig = plt.figure(figsize=(16, 15))
    g = fig.add_gridspec(4, 3, width_ratios=[1.25, 1.25, 1], hspace=0.42, wspace=0.22, top=0.92, bottom=0.05,
                         left=0.05, right=0.97)
    for r, (name, d) in enumerate(panels):
        lr = dlim if r == 2 else lim
        kw = dict(cmap="RdBu_r", vmin=-lr, vmax=lr, interpolation="nearest")
        ax = fig.add_subplot(g[r, 0])
        im = ax.imshow(d["x_slice"][air:], extent=(0, lz, ld, 0), aspect="auto", **kw)
        ax.set_title(f"{name}: x = {lx / 2:.1f} m", fontsize=9.5)
        ax.set_ylabel("depth (m)")
        ax.set_xlabel("z (m)", fontsize=8)
        ax = fig.add_subplot(g[r, 1])
        ax.imshow(d["z_slice"][:, air:].T, extent=(0, lx, ld, 0), aspect="auto", **kw)
        ax.set_title(f"{name}: z = {lz / 2:.1f} m", fontsize=9.5)
        ax.set_xlabel("x (m)", fontsize=8)
        ax = fig.add_subplot(g[r, 2])
        ax.imshow(d["depth_slices"][0].T, extent=(0, lx, lz, 0), aspect="equal", **kw)
        ax.set_title(f"{name}: depth {(di - air + 0.5) * ld / depth:.2f} m", fontsize=9.5)
        ax.set_xlabel("x (m)", fontsize=8)
        fig.colorbar(im, ax=fig.axes[-3:], shrink=0.85, pad=0.01,
                     label="block - exact (note scale)" if r == 2 else "change in permittivity")
    common = sorted(set(he) & set(hb))
    ie = np.array(sorted(he)) + 1
    ib = np.array(sorted(hb)) + 1
    ax = fig.add_subplot(g[3, 0])
    ax.plot(ie, [he[i]["rmse"] for i in sorted(he)], color=EXACT, lw=2, label="exact (checkpointing)")
    ax.plot(ib, [hb[i]["rmse"] for i in sorted(hb)], color=BLOCK, lw=2, ls="--", label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("eps RMSE")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(g[3, 1])
    for key, short, ls in (("grad_corr", "gcorr", "-"), ("update_corr", "ucorr", ":")):
        ax.plot(ie, [he[i][key] for i in sorted(he)], color=EXACT, lw=1.8, ls=ls, label=f"exact {short}")
        ax.plot(ib, [hb[i][key] for i in sorted(hb)], color=BLOCK, lw=1.8, ls=ls, label=f"block {short}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("correlation with (true - current)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax = fig.add_subplot(g[3, 2])
    stored = {i: np.array(hb[i]["history_gib"], dtype=float).reshape(-1, 2) for i in hb}  # (stored, spilled) per shot
    ax.plot(ib, [stored[i][:, 0].mean() for i in sorted(hb)], color=BLOCK, lw=2, label="mean over shots")
    ax.plot(ib, [stored[i][:, 0].max() for i in sorted(hb)], color=BLOCK, lw=1, ls=":", label="largest shot")
    ax.plot(ib, [stored[i][:, 1].sum() for i in sorted(hb)], color="#999999", lw=1, ls="--", label="spilled to disk, all shots")
    ax.set_xlabel("iteration")
    ax.set_ylabel("block-Tucker history per shot (GiB)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    raw = hb[sorted(hb)[0]]["raw_history_gib"]
    ax.set_title(f"raw E history {raw:.0f} GiB per shot", fontsize=9)
    res["per_iteration"] = [dict(
        iteration=i, rmse_exact=he[i]["rmse"], rmse_block=hb[i]["rmse"], gcorr_exact=he[i]["grad_corr"],
        gcorr_block=hb[i]["grad_corr"], ucorr_exact=he[i]["update_corr"], ucorr_block=hb[i]["update_corr"],
        wall_exact_s=he[i]["wall_s"], wall_block_s=hb[i]["wall_s"], history_mean_gib=float(stored[i][:, 0].mean()),
        history_max_gib=float(stored[i][:, 0].max()), spilled_gib=float(stored[i][:, 1].sum()),
        host_rss_block_gib=hb[i]["host_maxrss_gib"]) for i in common]
    title = (f"Full-size Overthrust FWI (2 cm, 138.9 M cells, 16 shots, 250 MHz): exact gradient vs {label}, "
             f"after {it + 1} iterations")
    if "model" in res:
        m = res["model"]
        title += (f"\nRMSE exact {m['rmse_exact']:.4f}, block {m['rmse_block']:.4f}; block - exact: rms {m['rms_block_minus_exact']:.4f}, "
                  f"{100 * m['rel_update_diff']:.1f} % of the exact update (corr of updates {m['corr_updates']:.4f})")
    fig.suptitle(title, fontsize=11)
    fig.savefig(out / f"compare_block_exact{tag}_it{it + 1:04d}.png", dpi=100)
    (out / f"compare_block_exact{tag}_it{it + 1:04d}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_iteration"}, indent=1))


if __name__ == "__main__":
    main()
