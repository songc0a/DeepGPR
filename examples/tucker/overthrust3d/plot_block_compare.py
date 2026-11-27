"""Figures and GIFs of block_wavefield_compare.py: exact vs block-Tucker-recovered Ez (+ global Tucker reference).

    python plot_block_compare.py <run dir> [<other run dirs> ...] [--times ...]

The first directory is drawn (snapshots + GIF); the error / compression curves include every directory.
Cells whose forward values never enter the eps gradient at that frame (fixed air, adjoint shadow) are grey.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

GLOBAL_COLORS = {16: "#d9822b", 32: "#c0392b", 64: "#8a4fd3"}
TOL_COLORS = {0.003: "#123f7a", 0.01: "#2a6fdb", 0.03: "#5b9be8", 0.1: "#9cc0f2"}


def load(d: Path) -> dict:
    info = json.loads((d / "info.json").read_text())
    e = np.load(d / "errors.npz")
    return dict(dir=d, info=info, e=e)


def curves(run: dict, method: str) -> dict:
    a = run["e"][f"acc_{method}"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = a[:, 1] > 1e-8 * a[:, 1].max()
        err = np.where(ok, np.sqrt(a[:, 0] / np.where(ok, a[:, 1], 1)), np.nan)
        okd = a[:, 3] > 1e-8 * np.nanmax(a[:, 3])
        derr = np.where(okd, np.sqrt(a[:, 2] / np.where(okd, a[:, 3], 1)), np.nan)
    return dict(err=err, derr=derr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--times", type=float, nargs="+", default=[10, 20, 30, 40, 55, 75])
    p.add_argument("--gif-every", type=int, default=2)
    p.add_argument("--clip-pct", type=float, default=99.5)
    p.add_argument("--reference", type=int, default=64, help="global Tucker rank shown in the snapshot figure")
    p.add_argument("--amplify", type=float, default=None, help="factor for the amplified difference (default 0.3 / tol)")
    p.add_argument("--global-from", default=None, help="run dir with the global Tucker reference (same shot and frames)")
    a = p.parse_args()
    runs = [load(Path(d)) for d in a.dirs]
    main_run = runs[0]
    info = main_run["info"]
    out = main_run["dir"]
    v = np.load(out / "vertical_Ez.npz")
    dt_ns, dx, air, pml = info["dt"] * 1e9, info["dx"], info["air"], info["pml"]
    nx, ny = info["shape"][0], info["shape"][1]
    keep_rows = np.array(info["keep_until_rows"])[pml:pml + ny]  # physical rows
    extent = (0, nx * dx, (ny - air) * dx, -air * dx)
    xsrc = info["source_cell"][0] * dx
    steps = v["steps"]
    exact = v["exact"]
    tol = info["tolerance"]
    ref = f"r{a.reference}"
    raw_frame = info["storage"]["raw_bytes"] / (info["nt"] + 1)
    frames_all = np.arange(info["nt"] + 1)
    t_all = frames_all * dt_ns

    def masked(n: int, data: np.ndarray) -> np.ndarray:
        rows_off = keep_rows < n  # (ny,) rows not needed at frame n
        return np.where(rows_off[None, :], np.nan, data)

    def clip_of(i: int) -> float:
        c = float(np.percentile(np.abs(exact[i][:, air:]), a.clip_pct))
        return c if c > 0 else 1.0

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#b8b8b8")

    def section(ax, data, c, title):
        ax.imshow(data.T, extent=extent, aspect="auto", cmap=cmap, vmin=-c, vmax=c, interpolation="nearest")
        ax.axhline(0, color="#555555", lw=0.5)
        ax.plot([xsrc], [0], marker="v", color="k", ms=5, clip_on=False)
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(labelsize=7)

    # Global Tucker reference: from the drawn run, else from the first listed run that computed it (same frames).
    pool = runs + ([load(Path(a.global_from))] if a.global_from else [])
    ref_run = next(r for r in pool if a.reference in r["info"]["global_ranks"])
    assert ref_run["info"]["shot"] == info["shot"] and ref_run["info"]["nt"] == info["nt"]
    vref = v if ref_run is main_run else np.load(ref_run["dir"] / "vertical_Ez.npz")
    assert np.array_equal(vref["steps"], steps)
    gl_run = ref_run
    cb = curves(main_run, "block")
    cr = curves(ref_run, ref)
    frame_bytes = main_run["e"]["record_bytes"].sum(1)
    ratio_frame = np.where(frame_bytes > 0, raw_frame / np.maximum(frame_bytes, 1), np.nan)

    def region_rel(i: int, rec: np.ndarray) -> float:
        n = steps[i]
        m = (keep_rows >= n)[None, :] & (np.arange(ny) >= air)[None, :]
        num = np.linalg.norm((rec - exact[i])[np.broadcast_to(m, rec.shape)])
        den = np.linalg.norm(exact[i][np.broadcast_to(m, rec.shape)])
        return float(num / den) if den > 0 else float("nan")

    sub = (f"Overthrust 2 cm, true model, shot {info['shot']} (x {xsrc:.2f} m, z {info['source_cell'][2] * dx:.2f} m), "
           f"{info['freq'] / 1e6:.0f} MHz z-dipole; Ez on the x-depth section through the source")
    st = info["storage"]
    head = (f"block Tucker {info['block']}^3, tolerance {tol:g} per block, floor {info['floor']:g}, adjoint shadow margin "
            f"{info['margin_ns']:g} ns: history {st['block_stored_bytes'] / 1024**3:.1f} GiB of {st['raw_bytes'] / 1024**3:.0f} GiB "
            f"(compression {st['block_ratio']:.0f}x; global r{a.reference} {gl_run['info']['storage'][f'r{a.reference}_ratio']:.0f}x)")
    picks = [int(np.argmin(np.abs(steps * dt_ns - t))) for t in a.times]
    k = a.amplify or max(1.0, round(0.3 / tol))
    fig = plt.figure(figsize=(25, 2.0 * len(picks) + 3.6))
    g = fig.add_gridspec(len(picks) + 1, 5, height_ratios=[1] * len(picks) + [1.5], hspace=0.55, wspace=0.12,
                         left=0.035, right=0.995, top=0.93, bottom=0.05)
    for row, i in enumerate(picks):
        n, c, t = steps[i], clip_of(i), steps[i] * dt_ns
        rec, rr = v["block"][i], vref[ref][i]
        section(fig.add_subplot(g[row, 0]), exact[i], c, f"exact, t = {t:.1f} ns")
        section(fig.add_subplot(g[row, 1]), masked(n, rec), c,
                f"block Tucker (tol {tol:g}) recovered, frame compression {ratio_frame[n]:.0f}x")
        section(fig.add_subplot(g[row, 2]), masked(n, rec - exact[i]), c,
                f"block - exact: section {region_rel(i, rec) * 100:.2f} %, 3D {cb['err'][n] * 100:.2f} %")
        section(fig.add_subplot(g[row, 3]), masked(n, k * (rec - exact[i])), c, f"(block - exact) x {k:g}")
        section(fig.add_subplot(g[row, 4]), masked(n, rr - exact[i]), c,
                f"global r{a.reference} - exact: section {region_rel(i, rr) * 100:.1f} %, 3D {cr['err'][n] * 100:.1f} %")
        fig.axes[-5].set_ylabel("depth (m)", fontsize=8)
        if row == len(picks) - 1:
            for ax in fig.axes[-5:]:
                ax.set_xlabel("x (m)", fontsize=8)
    bottom = g[-1, :].subgridspec(1, 3, wspace=0.22)
    panels = ((0, "relative error of E^n (gradient region)", "err"), (1, "relative error of E^(n+1) - E^n (R^n proxy)", "derr"))
    for col, title, key in panels:
        ax = fig.add_subplot(bottom[0, col])
        for run in runs:
            ci = curves(run, "block")
            tl = run["info"]["tolerance"]
            y, x = ci[key], t_all
            if key == "derr":
                sel = ~np.isnan(y)
                y, x = y[sel], x[sel]
            ax.semilogy(x, y * 100, color=TOL_COLORS.get(tl, "k"), lw=1.8 if run is main_run else 1.2,
                        label=f"block tol {tl:g}")
        for r in gl_run["info"]["global_ranks"]:
            ci = curves(gl_run, f"r{r}")
            y, x = ci[key], t_all
            if key == "derr":
                sel = ~np.isnan(y)
                y, x = y[sel], x[sel]
            ax.semilogy(x, y * 100, color=GLOBAL_COLORS.get(r, "k"), lw=1.2, ls="--", label=f"global r{r}")
        for i in picks:
            ax.axvline(steps[i] * dt_ns, color="#cccccc", lw=0.8, zorder=0)
        ax.set_xlabel("time (ns)")
        ax.set_ylabel("relative error (%)")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.set_xlim(0, t_all[-1])
        ax.legend(frameon=False, fontsize=7, ncol=2)
    ax = fig.add_subplot(bottom[0, 2])
    for run in runs:
        fb = run["e"]["record_bytes"].sum(1)
        rf = np.where(fb > 0, raw_frame / np.maximum(fb, 1), np.nan)
        tl = run["info"]["tolerance"]
        ax.semilogy(t_all, rf, color=TOL_COLORS.get(tl, "k"), lw=1.8 if run is main_run else 1.2,
                    label=f"block tol {tl:g} (whole history {run['info']['storage']['block_ratio']:.0f}x)")
    for r in gl_run["info"]["global_ranks"]:
        ax.axhline(gl_run["info"]["storage"][f"r{r}_ratio"], color=GLOBAL_COLORS.get(r, "k"), lw=1.2, ls="--", label=f"global r{r}")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("raw / stored bytes")
    ax.set_title("compression ratio of each frame (Ex, Ey, Ez)", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.set_xlim(0, t_all[-1])
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"{sub}\n{head}; colour limit per row = {a.clip_pct} th percentile of |exact| in the ground; grey = "
                 f"not needed by the eps gradient (fixed air, adjoint shadow)", fontsize=10.5)
    path = out / f"snapshots_block_tol{tol:g}.png"
    fig.savefig(path, dpi=105)
    plt.close(fig)
    print(path)

    frames = []
    for i in range(0, len(steps), a.gif_every):
        n, c, t = steps[i], clip_of(i), steps[i] * dt_ns
        rec = v["block"][i]
        fig, axes = plt.subplots(3, 1, figsize=(9.6, 8.0), constrained_layout=True)
        section(axes[0], exact[i], c, "exact (direct FDTD propagation)")
        section(axes[1], masked(n, rec), c,
                f"block Tucker (32^3, tol {tol:g}): compressed, then recovered; frame compression "
                + (f"{ratio_frame[n]:.0f}x" if np.isfinite(ratio_frame[n]) else "(nothing stored)"))
        e3 = cb["err"][n]
        section(axes[2], masked(n, k * (rec - exact[i])), c,
                f"(recovered - exact) x {k:g} (otherwise same colour limit)"
                + (f": 3D gradient-region error {e3 * 100:.2f} %" if np.isfinite(e3) else ""))
        for ax in axes:
            ax.set_ylabel("depth (m)", fontsize=8)
        axes[2].set_xlabel("x (m)", fontsize=8)
        fig.suptitle(f"Overthrust 2 cm, shot {info['shot']}, Ez section through the source, t = {t:5.1f} ns "
                     f"(grey: not needed by the gradient)", fontsize=10)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    palette = frames[len(frames) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    gif = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    path = out / f"wavefield_block_tol{tol:g}.gif"
    gif[0].save(path, save_all=True, append_images=gif[1:], duration=70, loop=0)
    print(path, len(gif), "frames")


if __name__ == "__main__":
    main()
