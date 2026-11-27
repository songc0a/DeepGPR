"""Exact-gradient FWI vs two block-Tucker FWIs of the full-size Overthrust run at the same iteration.

Model updates (x / z / depth slices) of the three runs, the two differences to the exact run on one shared colour
scale, and per-iteration curves (RMSE, RMSE difference to exact, gradient correlation, stored history per shot).
When eps_itNNNN.npy / g_itNNNN.npy exist for all three runs, the full-volume differences are added to the title/json.

    python compare_three.py <slice iteration index> <out dir> <exact dir> <block dir 1> <block dir 2> [eps_init.npy]
                            [label 1] [label 2] [file tag]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUNS = (("exact (checkpointing)", "#333333", "-"),
        ("block Tucker 3 %, fp32", "#8a4fd3", "--"),
        ("block Tucker 1.5 %, fp16", "#e07b1a", "-."))
KEYS = ("x_slice", "z_slice", "depth_slices")


def load_history(run: Path) -> dict:
    out = {}
    for h in json.loads((run / "history.json").read_text()):
        out[h["iteration"]] = h  # a resumed run may repeat an iteration: keep the last record
    return out


def full_volume(dirs: list[Path], it: int, air: int, init: Path) -> dict:
    """rms / relative differences of the models (after it+1 its) and gradients (at it) to the exact run."""
    res = {}
    snap, grad = f"eps_it{it + 1:04d}.npy", f"g_it{it:04d}.npy"
    if all((d / "snapshots" / snap).exists() for d in dirs):
        m0 = np.load(init)[:, air:].astype(np.float64)
        me = np.load(dirs[0] / "snapshots" / snap).astype(np.float64)
        ue = me - m0
        for d, (name, _, _) in zip(dirs[1:], RUNS[1:]):
            mb = np.load(d / "snapshots" / snap).astype(np.float64)
            ub = mb - m0
            res[name] = dict(model_rms_diff=float(np.sqrt(np.mean((mb - me) ** 2))),
                             model_rel_update_diff=float(np.linalg.norm(ub - ue) / np.linalg.norm(ue)),
                             model_corr_updates=float((ue * ub).sum() / np.sqrt((ue * ue).sum() * (ub * ub).sum())))
            del mb, ub
        del me, ue, m0
    if all((d / "gradients" / grad).exists() for d in dirs):
        ge = np.load(dirs[0] / "gradients" / grad).astype(np.float64)
        for d, (name, _, _) in zip(dirs[1:], RUNS[1:]):
            gb = np.load(d / "gradients" / grad).astype(np.float64)
            res.setdefault(name, {}).update(
                grad_rel_diff=float(np.linalg.norm(gb - ge) / np.linalg.norm(ge)),
                grad_corr=float((ge * gb).sum() / np.sqrt((ge * ge).sum() * (gb * gb).sum())))
            del gb
    return res


def main() -> None:
    global RUNS
    it, out = int(sys.argv[1]), Path(sys.argv[2])
    if len(sys.argv) > 8:
        RUNS = (RUNS[0], (sys.argv[7], RUNS[1][1], RUNS[1][2]), (sys.argv[8], RUNS[2][1], RUNS[2][2]))
    tag = sys.argv[9] if len(sys.argv) > 9 else ""
    dirs = [Path(p) for p in sys.argv[3:6]]
    init = Path(sys.argv[6]) if len(sys.argv) > 6 else dirs[0] / "eps_init.npy"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((dirs[0] / "meta.json").read_text())
    nx, depth, nz = meta["ground_shape"]
    air = meta["shape"][1] - depth
    lx, ld, lz = meta["size_m"]
    hist = [load_history(d) for d in dirs]
    s0 = dict(np.load(dirs[0] / "slices" / "it-001.npz"))
    sl = [dict(np.load(d / "slices" / f"it{it:04d}.npz")) for d in dirs]
    di = int(sl[0]["depth_index"][0])
    upd = [{k: s[k] - s0[k] for k in KEYS} for s in sl]
    diffs = [{k: u[k] - upd[0][k] for k in KEYS} for u in upd[1:]]
    lim = max(float(np.ceil(4 * np.sqrt(np.mean(upd[0]["x_slice"][air:] ** 2)) * 4) / 4), 0.25)
    dlim = float(f"{max(4 * np.sqrt(np.mean(diffs[0]['x_slice'][air:] ** 2)), 1e-3):.1g}")
    rows = [(f"{RUNS[r][0]}: update", upd[r], lim) for r in range(3)]
    rows += [(f"{RUNS[r + 1][0]} - exact", diffs[r], dlim) for r in range(2)]

    fig = plt.figure(figsize=(16, 24))
    g = fig.add_gridspec(7, 3, width_ratios=[1.25, 1.25, 1], height_ratios=[1, 1, 1, 1, 1, 0.95, 0.95],
                         hspace=0.5, wspace=0.22, top=0.94, bottom=0.035, left=0.05, right=0.97)
    for r, (name, d, lr) in enumerate(rows):
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
                     label="difference to exact (note scale)" if r >= 3 else "change in permittivity")

    def curve(ax, key, runs=(0, 1, 2), fn=None):
        for r in runs:
            its = sorted(hist[r])
            y = [fn(hist[r][i], i) if fn else hist[r][i][key] for i in its]
            ax.plot(np.array(its) + 1, y, color=RUNS[r][1], ls=RUNS[r][2], lw=2, label=RUNS[r][0])
        ax.axvline(it + 1, color="#999999", lw=0.8, ls=":")
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)

    ax = fig.add_subplot(g[5, 0])
    curve(ax, "rmse")
    ax.set_ylabel("eps RMSE to the true model")
    ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(g[5, 1])
    curve(ax, None, (1, 2), lambda h, i: 1e4 * (h["rmse"] - hist[0][i]["rmse"]) if i in hist[0] else np.nan)
    ax.axhline(0, color=RUNS[0][1], lw=1)
    ax.set_ylabel("RMSE(block) - RMSE(exact)  [1e-4]")
    ax.set_title("same iteration, same start / data / optimiser", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(g[5, 2])
    curve(ax, "grad_corr")
    ax.set_ylabel("gcorr: corr(-gradient, true - current)")
    ax = fig.add_subplot(g[6, 0])
    curve(ax, None, (1, 2), lambda h, i: float(np.array(h["history_gib"], dtype=float).reshape(-1, 2)[:, 0].mean()))
    ax.set_ylabel("stored E history per shot, mean (GiB)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"raw E history {hist[1][0]['raw_history_gib']:.0f} GiB per shot", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(g[6, 1])
    curve(ax, "update_corr")
    ax.set_ylabel("ucorr: corr(update, true - start)")
    ax = fig.add_subplot(g[6, 2])
    curve(ax, None, fn=lambda h, i: h.get("host_maxrss_gib", np.nan))
    ax.set_ylabel("host max RSS per process (GiB)")
    ax.set_ylim(bottom=0)

    common = sorted(set(hist[0]) & set(hist[1]) & set(hist[2]))
    res = dict(slice_iteration_index=it, latest_common_iteration_index=common[-1],
               slice_rms_diff_to_exact={RUNS[r + 1][0]: float(np.sqrt(np.mean(diffs[r]["x_slice"][air:] ** 2)))
                                        for r in range(2)},
               full_volume=full_volume(dirs, it, air, init),
               per_iteration=[dict(iteration=i, **{f"{k}_{n}": hist[r][i][k] for r, n in enumerate(("exact", "b3", "b15fp16"))
                                                   for k in ("rmse", "grad_corr", "update_corr", "wall_s")}) for i in common])
    title = (f"Full-size Overthrust FWI (2 cm, 138.9 M cells, 16 shots, 250 MHz): exact vs block-Tucker histories, "
             f"model updates after {it + 1} iterations")
    r_e = hist[0][it]["rmse"]
    title += f"\nRMSE after {it + 1} its: exact {r_e:.4f}, " + ", ".join(
        f"{RUNS[r][0]} {hist[r][it]['rmse']:.4f}" for r in (1, 2))
    fv = res["full_volume"]
    if all("model_rel_update_diff" in fv.get(RUNS[r][0], {}) for r in (1, 2)):
        title += "\nmodel - exact (full volume): " + ", ".join(
            f"{RUNS[r][0]} {100 * fv[RUNS[r][0]]['model_rel_update_diff']:.1f} % of the exact update" for r in (1, 2))
    fig.suptitle(title, fontsize=11)
    stem = f"compare_three{tag}_it{it + 1:04d}"
    fig.savefig(out / f"{stem}.png", dpi=100)
    (out / f"{stem}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_iteration"}, indent=1))


if __name__ == "__main__":
    main()
