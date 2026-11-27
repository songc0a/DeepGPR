"""Figures and GIFs of wavefield_compare.py output: exact vs Tucker-recovered Ez and their difference.

    python plot_wavefield_compare.py <out dir> [--times 10 20 30 40 55 75] [--gif-every 2]
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

RANK_COLORS = {16: "#d9822b", 24: "#c0392b", 32: "#2a6fdb", 36: "#17a2b8", 48: "#2e9e6a", 64: "#8a4fd3"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--times", type=float, nargs="+", default=[10, 20, 30, 40, 55, 75])
    p.add_argument("--gif-every", type=int, default=2, help="use every n-th saved section in the GIF")
    p.add_argument("--clip-pct", type=float, default=99.5)
    p.add_argument("--ranks", type=int, nargs="+", default=None, help="ranks to draw (default: all in the run)")
    p.add_argument("--extra", nargs="*", default=[], help="other output dirs whose error curves are overlaid")
    a = p.parse_args()
    out = Path(a.out)
    info = json.loads((out / "info.json").read_text())
    v = np.load(out / "vertical_Ez.npz")
    e = np.load(out / "errors.npz")
    dt_ns, dx, air = info["dt"] * 1e9, info["dx"], info["air"]
    nx, ny = info["shape"][0], info["shape"][1]
    extent = (0, nx * dx, (ny - air) * dx, -air * dx)  # x (m), depth (m) below the surface
    xsrc = info["source_cell"][0] * dx
    steps = v["steps"]
    ranks = a.ranks or info["ranks"]

    def curves(ee, rr):
        rf = ee["ref"].sum(1)
        good = rf > 1e-8 * rf.max()
        return {r: np.where(good, np.sqrt(ee[f"err_r{r}"].sum(1) / np.where(good, rf, 1)), np.nan) for r in rr}

    def ratio(inf, r):
        if "storage" in inf and f"r{r}" in inf["storage"]:
            return inf["storage"][f"r{r}"]["ratio"], inf["storage"][f"r{r}"]["stored_bytes"]
        q = inf["passes"][f"r{r}"]
        return q["raw_bytes"] / q["stored_bytes"], q["stored_bytes"]

    frame_err = curves(e, info["ranks"])
    comp = {r: ratio(info, r) for r in info["ranks"]}
    for d in a.extra:
        ie = json.loads((Path(d) / "info.json").read_text())
        new = [r for r in ie["ranks"] if r not in frame_err]
        frame_err.update(curves(np.load(Path(d) / "errors.npz"), new))
        comp.update({r: ratio(ie, r) for r in new})
    t_all = e["steps"] * dt_ns
    exact = v["exact"]

    def clip_of(i: int) -> float:
        c = float(np.percentile(np.abs(exact[i]), a.clip_pct))
        return c if c > 0 else 1.0

    def section(ax, data, c, title):
        im = ax.imshow(data.T, extent=extent, aspect="auto", cmap="RdBu_r", vmin=-c, vmax=c, interpolation="nearest")
        ax.axhline(0, color="#555555", lw=0.5)
        ax.plot([xsrc], [0], marker="v", color="k", ms=5, clip_on=False)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        return im

    def slice_rel(i: int, r: int) -> float:
        return float(np.linalg.norm(v[f"r{r}"][i] - exact[i]) / max(np.linalg.norm(exact[i]), 1e-30))

    sub = (f"Overthrust 2 cm, true model, shot {info['shot']} (x {xsrc:.2f} m, z {info['source_cell'][2] * dx:.2f} m), "
           f"{info['freq'] / 1e6:.0f} MHz, z-dipole; Ez on the x-depth section through the source")
    picks = [int(np.argmin(np.abs(steps * dt_ns - t))) for t in a.times]
    for r in ranks:
        rec = v[f"r{r}"]
        fig = plt.figure(figsize=(16, 2.0 * len(picks) + 3.2))
        g = fig.add_gridspec(len(picks) + 1, 3, height_ratios=[1] * len(picks) + [1.35], hspace=0.55, wspace=0.12,
                             left=0.05, right=0.98, top=0.93, bottom=0.05)
        for row, i in enumerate(picks):
            c = clip_of(i)
            t = steps[i] * dt_ns
            section(fig.add_subplot(g[row, 0]), exact[i], c, f"exact, t = {t:.1f} ns")
            section(fig.add_subplot(g[row, 1]), rec[i], c, f"Tucker r{r} compressed -> recovered, t = {t:.1f} ns")
            section(fig.add_subplot(g[row, 2]), rec[i] - exact[i], c,
                    f"recovered - exact (same colour limit): section {slice_rel(i, r) * 100:.1f} %, "
                    f"3D frame {frame_err[r][steps[i]] * 100:.1f} %")
            fig.axes[-3].set_ylabel("depth (m)", fontsize=8)
            if row == len(picks) - 1:
                for ax in fig.axes[-3:]:
                    ax.set_xlabel("x (m)", fontsize=8)
        ax = fig.add_subplot(g[-1, :])
        for rr in sorted(frame_err):
            ax.plot(t_all, frame_err[rr] * 100, color=RANK_COLORS.get(rr, "k"), lw=2 if rr == r else 1.2,
                    label=f"Tucker r{rr} ({comp[rr][0]:.0f}x)" + (" (this figure)" if rr == r else ""))
        for i in picks:
            ax.axvline(steps[i] * dt_ns, color="#bbbbbb", lw=0.8, zorder=0)
        ax.set_xlabel("time (ns)")
        ax.set_ylabel("relative error (%)")
        ax.set_title("relative L2 error of every recovered E frame (Ex, Ey, Ez; 3D physical domain incl. air, PML excluded)",
                     fontsize=9)
        ax.set_xlim(0, t_all[-1])
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        s = info["summary"][f"r{r}"]
        fig.suptitle(f"{sub}\nTucker r{r} (oversampling 8, per-frame 3D Tucker, FWI settings; history {comp[r][1] / 1024**3:.2f} GiB, "
                     f"compression {comp[r][0]:.0f}x): whole-history relative error "
                     f"{s['relative_l2_all'] * 100:.1f} % (Ex {s['relative_l2_component']['Ex'] * 100:.1f}, "
                     f"Ey {s['relative_l2_component']['Ey'] * 100:.1f}, Ez {s['relative_l2_component']['Ez'] * 100:.1f} %); "
                     f"colour limit per row = {a.clip_pct} th percentile of |exact|", fontsize=10)
        path = out / f"snapshots_r{r}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(path)

        # GIF: three panels, every gif_every-th saved section.
        frames = []
        for i in range(0, len(steps), a.gif_every):
            c = clip_of(i)
            t = steps[i] * dt_ns
            fig, axes = plt.subplots(3, 1, figsize=(9.6, 8.0), constrained_layout=True)
            section(axes[0], exact[i], c, "exact (direct FDTD propagation)")
            section(axes[1], rec[i], c, f"Tucker r{r}: compressed, then recovered")
            fe = frame_err[r][steps[i]]
            section(axes[2], rec[i] - exact[i], c,
                    f"recovered - exact (same colour limit): section {slice_rel(i, r) * 100:.1f} %, "
                    f"3D frame {fe * 100:.1f} %" if np.isfinite(fe) else "recovered - exact")
            for ax in axes:
                ax.set_ylabel("depth (m)", fontsize=8)
            axes[2].set_xlabel("x (m)", fontsize=8)
            axes[1].set_title(f"Tucker r{r} (compression {comp[r][0]:.0f}x): compressed, then recovered", fontsize=9)
            fig.suptitle(f"Overthrust 2 cm, shot {info['shot']}, Ez section through the source, t = {t:5.1f} ns", fontsize=10)
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
            plt.close(fig)
        palette = frames[len(frames) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        gif = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
        path = out / f"wavefield_r{r}.gif"
        gif[0].save(path, save_all=True, append_images=gif[1:], duration=70, loop=0, optimize=False)
        print(path, len(gif), "frames")


if __name__ == "__main__":
    main()
