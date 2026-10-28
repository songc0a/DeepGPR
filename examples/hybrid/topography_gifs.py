"""Same undulating ground, two discretisations: staircase Yee-FDTD vs hybrid DGTD-FDTD.

For each model of ``models.CASES`` the script runs, on the *same* topography
and subsurface:

* native DeepGPR Yee-FDTD (``DeepGPR.compute``) at ``dx`` - the surface can
  only be represented node by node (a staircase), time step at its own CFL
  limit;
* the hybrid solver (``DeepGPR.hybrid.HybridTM2D``) on the same 2 cm model -
  a DGTD band of 2 cm triangles whose edges lie on the surface, Yee part at
  ``dx/3`` (``fdtd_refinement=3``, no extra time steps to speak of);
* an independent reference: native Yee-FDTD at ``dx / 8`` with cell-averaged
  eps and sigma at the surface (second order for Ez).

All time steps divide the frame interval exactly (see ``models.py`` for the
source/receiver time convention), so runs with different ``dt`` compare at the
same times. Outputs per model (``examples/hybrid/results/``): a GIF with the
full wavefields, zooms that show the Yee cells / triangle mesh at the surface
and the error of each solver against the reference; a mesh figure; a trace
figure. The equal-accuracy tables are produced by ``phase2_benchmark.py``.

Run from the repository root (CUDA recommended)::

    python examples/hybrid/topography_gifs.py --cases B_sine slope C_irregular
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import DeepGPR  # noqa: E402
from DeepGPR.hybrid import FieldSampler  # noqa: E402
from models import (  # noqa: E402
    CASES, DEPTH, DX, FRAME_DT, ROOT, WIDTH, aligned_dt, antenna_positions, build_dg, nt_of,
    run_dg, run_native, yee_model,
)

REF = 8
CMAP = LinearSegmentedColormap.from_list(
    "bluered", ["#0d366b", "#2a78d6", "#86b6ef", "#f0efec", "#f0a5a4", "#e34948", "#8a1c1c"])
INK, INK2, GRID = "#0b0b0b", "#52514e", "#8c8b86"


# ---------------------------------------------------------------- drawing helpers
def staircase_outline(air, dx):
    """Step polyline (y, x) of the air/ground boundary seen by the Yee grid."""
    first = np.argmax(~air, axis=0)                 # first ground node per column
    xb = (first - 0.5) * dx
    ys = np.arange(air.shape[1]) * dx
    pts = []
    for j in range(air.shape[1]):
        pts += [(ys[j] - 0.5 * dx, xb[j]), (ys[j] + 0.5 * dx, xb[j])]
    return np.array(pts)


def mesh_segments(sol, window):
    x0, x1, y0, y1 = window
    v = sol.mesh.vertices - np.array([sol.ox, sol.oy])
    tri = sol.mesh.triangles
    p = v[tri]
    keep = ((p[:, :, 0].max(1) >= x0) & (p[:, :, 0].min(1) <= x1)
            & (p[:, :, 1].max(1) >= y0) & (p[:, :, 1].min(1) <= y1))
    edges = np.concatenate([tri[keep][:, [0, 1]], tri[keep][:, [1, 2]], tri[keep][:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return [[(v[a, 1], v[a, 0]), (v[b, 1], v[b, 0])] for a, b in edges]


def yee_segments(window, dx):
    """Boundaries of the material cells centred on the Ez nodes."""
    x0, x1, y0, y1 = window
    xs = (np.arange(math.floor(x0 / dx), math.ceil(x1 / dx) + 1) - 0.5) * dx
    ys = (np.arange(math.floor(y0 / dx), math.ceil(y1 / dx) + 1) - 0.5) * dx
    return ([[(y0, x), (y1, x)] for x in xs if x0 <= x <= x1]
            + [[(y, x0), (y, x1)] for y in ys if y0 <= y <= y1])


def style(ax, title, xlabel=True, ylabel=True):
    ax.set_title(title, fontsize=9.5, color=INK)
    if xlabel:
        ax.set_xlabel("y (m)", fontsize=8.5, color=INK2)
    if ylabel:
        ax.set_ylabel("depth x (m)", fontsize=8.5, color=INK2)
    ax.tick_params(labelsize=7.5, colors=INK2)


def make_gif(path, case, sol, fd_frames, hy, ref, src_fd, src_hy, air):
    """Rows: full wavefields, surface zooms with cells/mesh, errors vs reference."""
    x0, x1, y0, y1 = case["zoom"]
    nframes = min(len(fd_frames), len(hy["full"]), len(ref))
    vmax = 0.2 * np.percentile(np.abs(np.concatenate([fd_frames[8:], hy["full"][8:]])), 99.7)
    emax = vmax / 4
    ys = np.linspace(0, WIDTH, 1200)
    curve = case["surface"](ys)
    stair = staircase_outline(air, DX)
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 10.4), constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[1.0, 0.95, 1.0]))
    ext_cells = [-0.5 * DX, WIDTH - 0.5 * DX, DEPTH - 0.5 * DX, -0.5 * DX]
    imgs = {}
    # Row 1: full wavefields (both on the 2 cm node grid).
    imgs["fd"] = axes[0, 0].imshow(fd_frames[0], cmap=CMAP, vmin=-vmax, vmax=vmax,
                                   extent=ext_cells, interpolation="bilinear")
    imgs["hy"] = axes[0, 1].imshow(hy["full"][0], cmap=CMAP, vmin=-vmax, vmax=vmax,
                                   extent=ext_cells, interpolation="bilinear")
    style(axes[0, 0], f"DeepGPR Yee-FDTD, dx = {DX * 100:.0f} cm (staircase surface)",
          xlabel=False)
    style(axes[0, 1], f"Hybrid: DGTD h = {DX * 100:.0f} cm (body-fitted), "
                      f"Yee {sol.dx * 1000:.1f} mm", xlabel=False, ylabel=False)
    axes[0, 0].plot(stair[:, 0], stair[:, 1], color=INK, lw=0.9)
    axes[0, 1].plot(ys, curve, color=INK, lw=0.9)
    for xb in case["band"]:
        axes[0, 1].axhline(xb, color=INK2, lw=0.7, ls="--")
    axes[0, 1].text(0.08, case["band"][0] - 0.05, "DGTD band (dashed)", fontsize=7.5,
                    color=INK2)
    for ax, sxy in ((axes[0, 0], src_fd), (axes[0, 1], src_hy)):
        ax.add_patch(Rectangle((y0, x0), y1 - y0, x1 - x0, fill=False, ec=INK, lw=0.8))
        ax.plot(sxy[1], sxy[0], marker="v", ms=6, color=INK, ls="none")
    # Row 2: zooms with the Yee cells and the triangle mesh.
    zx = (int(round(x0 / DX)), int(round(x1 / DX)) + 1)
    zy = (int(round(y0 / DX)), int(round(y1 / DX)) + 1)
    imgs["fdz"] = axes[1, 0].imshow(fd_frames[0][zx[0]:zx[1], zy[0]:zy[1]], cmap=CMAP,
                                    vmin=-vmax, vmax=vmax, interpolation="nearest",
                                    extent=[(zy[0] - 0.5) * DX, (zy[1] - 0.5) * DX,
                                            (zx[1] - 0.5) * DX, (zx[0] - 0.5) * DX])
    imgs["hyz"] = axes[1, 1].imshow(hy["zoom"][0], cmap=CMAP, vmin=-vmax, vmax=vmax,
                                    extent=[y0, y1, x1, x0], interpolation="bilinear")
    axes[1, 0].add_collection(LineCollection(yee_segments(case["zoom"], DX), colors=GRID,
                                             linewidths=0.35))
    axes[1, 1].add_collection(LineCollection(mesh_segments(sol, case["zoom"]), colors=GRID,
                                             linewidths=0.35))
    axes[1, 0].plot(stair[:, 0], stair[:, 1], color=INK, lw=1.4, label="surface seen by FDTD")
    axes[1, 0].plot(ys, curve, color=INK, lw=0.8, ls=":", label="true surface")
    axes[1, 1].plot(ys, curve, color=INK, lw=1.4, label="true surface = mesh edges")
    style(axes[1, 0], "zoom: Yee cells (2 cm) and staircase surface", xlabel=False)
    style(axes[1, 1], "zoom: DGTD triangle mesh (P1, h = 2 cm)", xlabel=False, ylabel=False)
    for ax, sxy in ((axes[1, 0], src_fd), (axes[1, 1], src_hy)):
        ax.plot(sxy[1], sxy[0], marker="v", ms=8, color=INK, ls="none")
        ax.set_xlim(y0, y1)
        ax.set_ylim(x1, x0)
        ax.legend(loc="lower left", fontsize=7, framealpha=0.85)
    # Row 3: errors against the dx/8 reference.
    imgs["fde"] = axes[2, 0].imshow(fd_frames[0] - ref[0], cmap=CMAP, vmin=-emax, vmax=emax,
                                    extent=ext_cells, interpolation="bilinear")
    imgs["hye"] = axes[2, 1].imshow(hy["full"][0] - ref[0], cmap=CMAP, vmin=-emax, vmax=emax,
                                    extent=ext_cells, interpolation="bilinear")
    style(axes[2, 0], f"error of Yee-FDTD vs reference (averaged FDTD dx/{REF}), colour range / 4")
    style(axes[2, 1], f"error of hybrid vs reference (averaged FDTD dx/{REF}), colour range / 4",
          ylabel=False)
    axes[2, 0].plot(ys, curve, color=INK, lw=0.6)
    axes[2, 1].plot(ys, curve, color=INK, lw=0.6)
    for ax in (axes[0, 0], axes[0, 1], axes[2, 0], axes[2, 1]):
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(DEPTH, 0)
    cb = fig.colorbar(imgs["hy"], ax=axes[:2, :].ravel().tolist(), shrink=0.6, pad=0.01)
    cb.set_label("Ez (V/m), clipped", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb2 = fig.colorbar(imgs["hye"], ax=axes[2, :].ravel().tolist(), shrink=0.9, pad=0.01)
    cb2.set_label("Ez error (V/m)", fontsize=8, color=INK2)
    cb2.ax.tick_params(labelsize=7, colors=INK2)
    stamp = fig.suptitle("", fontsize=10.5, color=INK)

    def update(k):
        imgs["fd"].set_data(fd_frames[k])
        imgs["hy"].set_data(hy["full"][k])
        imgs["fdz"].set_data(fd_frames[k][zx[0]:zx[1], zy[0]:zy[1]])
        imgs["hyz"].set_data(hy["zoom"][k])
        imgs["fde"].set_data(fd_frames[k] - ref[k])
        imgs["hye"].set_data(hy["full"][k] - ref[k])
        stamp.set_text(f"{case['label']}: same surface, same 2 cm model    "
                       f"t = {(k + 1) * FRAME_DT * 1e9:4.1f} ns")
        return list(imgs.values()) + [stamp]

    FuncAnimation(fig, update, frames=nframes).save(path, writer=PillowWriter(fps=8), dpi=72)
    plt.close(fig)


def mesh_figure(path, case, sol, air):
    stair = staircase_outline(air, DX)
    ys = np.linspace(0, WIDTH, 2400)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[0.9, 1.3]))
    band = case["band"]
    window = (band[0], band[1], 0.0, WIDTH)
    axes[0].add_collection(LineCollection(mesh_segments(sol, window), colors=INK2,
                                          linewidths=0.15))
    axes[0].plot(ys, case["surface"](ys), color="#e34948", lw=0.9, label="ground surface")
    axes[0].add_patch(Rectangle((case["zoom"][2], case["zoom"][0]),
                                case["zoom"][3] - case["zoom"][2],
                                case["zoom"][1] - case["zoom"][0], fill=False, ec=INK, lw=1.0))
    axes[0].set_xlim(0, WIDTH)
    axes[0].set_ylim(band[1], band[0])
    axes[0].set_aspect("equal")
    axes[0].legend(loc="lower right", fontsize=8)
    q = sol.mesh.quality()
    style(axes[0], f"DGTD band mesh (Delaunay, {q['elements']} triangles incl. side CPML "
                   f"columns, minimum angle {q['min_angle_deg']:.1f} deg, edges "
                   f"{q['min_edge'] * 100:.1f}-{q['max_edge'] * 100:.1f} cm); box = zoom")
    x0, x1, y0, y1 = case["zoom"]
    ax = axes[1]
    ax.add_collection(LineCollection(yee_segments(case["zoom"], DX), colors="#86b6ef",
                                     linewidths=0.6, label="Yee material cells (FDTD)"))
    ax.add_collection(LineCollection(mesh_segments(sol, case["zoom"]), colors=INK2,
                                     linewidths=0.5, label="DGTD triangles"))
    ax.plot(stair[:, 0], stair[:, 1], color="#2a78d6", lw=1.8, label="surface seen by FDTD")
    ax.plot(ys, case["surface"](ys), color="#e34948", lw=1.8, label="true surface (DGTD edges)")
    ax.set_xlim(y0, y1)
    ax.set_ylim(x1, x0)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    style(ax, "zoom: what each solver sees at the surface")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def trace_figure(path, case, t, traces, rec_xy, picks, errors):
    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 6.8), sharex=True,
                             constrained_layout=True)
    styles = {f"reference FDTD dx/{REF}": dict(color=INK, lw=2.0),
              "Yee-FDTD dx (staircase)": dict(color="#eb6834", lw=1.2),
              "hybrid DGTD-FDTD": dict(color="#2a78d6", lw=1.2, ls="--")}
    for ax, k in zip(axes, picks):
        for name, kw in styles.items():
            ax.plot(t * 1e9, traces[name][:, k], label=name, **kw)
        ax.set_ylabel(f"Ez, y = {rec_xy[k][1]:.1f} m", fontsize=8, color=INK2)
        ax.tick_params(labelsize=8, colors=INK2)
        ax.grid(color="#e6e5e1", lw=0.5)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time (ns)", fontsize=9, color=INK2)
    fig.suptitle(f"{case['label']}: receivers 0.1 m above the ground\nrelative L2 vs reference "
                 f"(offsets >= 0.3 m): Yee-FDTD {errors[0]:.3f}, hybrid {errors[1]:.3f}",
                 fontsize=10, color=INK)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main
def run_case(name, case, device, outdir):
    rec_y = np.arange(0.2, WIDTH - 0.2 + 1e-9, 0.1)
    src = antenna_positions(case, [case["source_y"]])          # (1, 2), on 2 cm nodes
    rec = antenna_positions(case, rec_y)[None]                  # (1, nrx, 2)
    nx, ny = int(round(DEPTH / DX)), int(round(WIDTH / DX))
    x0, x1, y0, y1 = case["zoom"]
    probe = build_dg(case, device)
    dt_h, m_h = aligned_dt(probe.dt)
    del probe
    sol = build_dg(case, device, dt=dt_h)
    samplers = {"full": FieldSampler(sol, np.arange(nx) * DX, np.arange(ny) * DX),
                "zoom": FieldSampler(sol, np.arange(x0, x1 + 1e-9, 0.004),
                                     np.arange(y0, y1 + 1e-9, 0.004))}
    hy_data, hy_frames, _, _ = run_dg(sol, nt_of(dt_h), src, rec, frame_every=m_h,
                                      samplers=samplers)
    runs = {}
    for k in (1, REF):
        dx = DX / k
        # Staircase Yee at dx; the dx/8 reference uses cell-averaged materials.
        eps, sig, air = yee_model(dx, case, averaged=(k == REF))
        dt, m = aligned_dt(0.95 * DeepGPR.max_stable_time_step(dx, 10, 10, 1))
        data, fr, _, _ = run_native(eps, sig, dx, dt, nt_of(dt), src, rec, device,
                                    frame_every=m, decimate=k)
        runs[k] = dict(dt=dt, data=data[0], frames=fr, air=air)
    t_fd = (np.arange(nt_of(runs[1]["dt"])) + 1) * runs[1]["dt"]

    def interp(dt, d):
        t = (np.arange(d.shape[0]) + 1) * dt
        return np.stack([np.interp(t_fd, t, d[:, j]) for j in range(d.shape[1])], 1)

    ref = interp(runs[REF]["dt"], runs[REF]["data"])
    fd_i, hy_i = interp(runs[1]["dt"], runs[1]["data"]), interp(dt_h, hy_data[0])
    keep = np.abs(rec[0, :, 1] - src[0, 1]) >= 0.3
    err = lambda d: float(np.linalg.norm((d - ref)[:, keep]) / np.linalg.norm(ref[:, keep]))  # noqa: E731
    make_gif(outdir / f"{name}_staircase_fdtd_vs_hybrid_dgtd.gif", case, sol,
             runs[1]["frames"], hy_frames, runs[REF]["frames"], src[0], src[0], runs[1]["air"])
    mesh_figure(outdir / f"{name}_mesh.png", case, sol, runs[1]["air"])
    picks = [int(np.argmin(np.abs(rec[0, :, 1] - (src[0, 1] + o)))) for o in (0.6, 1.5, 2.5)]
    trace_figure(outdir / f"{name}_traces.png", case, t_fd,
                 {f"reference FDTD dx/{REF}": ref, "Yee-FDTD dx (staircase)": fd_i,
                  "hybrid DGTD-FDTD": hy_i}, rec[0], picks, (err(fd_i), err(hy_i)))
    print(f"{name}: relative L2 vs reference - Yee {err(fd_i):.4f}, hybrid {err(hy_i):.4f}",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["B_sine", "slope", "C_irregular"])
    parser.add_argument("--outdir", default=str(ROOT / "examples" / "hybrid" / "results"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name in args.cases:
        run_case(name, CASES[name], torch.device(args.device), outdir)


if __name__ == "__main__":
    main()
