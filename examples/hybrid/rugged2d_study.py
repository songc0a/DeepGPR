"""2D study on rugged topography: staircase Yee-FDTD vs the hybrid DGTD-FDTD.

Model (DeepGPR layout, axis 0 = depth x, axis 1 = horizontal y; 2.5 cm model grid,
1.6 m x 3.2 m, 0.3 m CPML):

* surface: datum 0.55 m with a 0.24 m ridge (Gaussian, y = 1.05 m), a 0.17 m
  V-shaped gully (y = 2.15 m) and 2 cm roughness of 0.16 m wavelength: 0.43 m
  of relief, slopes up to 61 deg, steeper than 30 deg on half of the profile;
* ground eps_r 9, sigma 5 mS/m (wet soil: strong surface contrast), a known
  deep reflector (eps_r 16 below 1.3 m) and three near-surface eps_r anomalies
  (14, 5, 13; radius 9-10 cm, centred 0.2 m below the local surface under the
  ridge flanks and the gully flank), where the surface representation matters;
* 14 Ez sources every 0.2 m and 29 receivers every 0.1 m, 5 cm above the local
  surface (ground coupled), all on 2.5 cm nodes; 400 MHz Ricker; 24 ns.

Solvers on the same model and the same antenna positions:

* ``fdtd``       native DeepGPR on the 2.5 cm grid, staircase surface (node above
                 the surface = air) - what FDTD-FWI uses on the model grid;
* ``fdtd_r3``    native DeepGPR on 2.5/3 cm (the Yee spacing of the hybrid), staircase;
* ``fdtd_r3avg`` native DeepGPR on 2.5/3 cm with cell-averaged surface materials
                 (forward part only; a sub-cell FDTD treatment for comparison);
* ``fdtd_r5``    native DeepGPR on 2.5/5 cm, staircase (forward part only: the
                 FDTD grid that matches the hybrid's accuracy);
* ``hybrid``     DGTD band 0.15-0.95 m (Delaunay P1 triangles, h = 1.25 cm, the
                 surface is a chain of triangle edges) + Yee at 2.5/3 cm. With
                 h = 2.5 cm the band is the accuracy bottleneck here (400 MHz in
                 eps_r 9: 10 elements per wavelength, and DG elements take the
                 material of the 2.5 cm pixel at their centroid): 6.3 % instead of 2.5 %;
* reference: native DeepGPR with cell-averaged materials on 2.5/9 cm (and on
  2.5/15 cm to bound its own error); the 2.5 cm model upsampled pixel by pixel.
  Odd factors keep pixel boundaries between nodes (with even factors the
  boundary nodes of the reference coincide with those of even FDTD grids and
  flatter them).

Part ``forward``: gather errors against the reference (all shots, offsets >=
0.2 m), time and peak GPU memory of the 14-shot runs, a GIF of the wavefields
(full view, zoom with the Yee cells / triangles, error against the reference),
a mesh figure and a trace figure.

Part ``fwi``: eps_r-only FWI (sigma known) with ``fdtd``, ``fdtd_r3`` and
``hybrid``. Observed data: the 2.5/9 cm reference (independent solver, grid and
surface treatment). L-BFGS with strong Wolfe line search; stage 1 low-passed to
250 MHz, stage 2 full band; each stage's misfit normalised by its start value.
GIF of the models per iteration with misfit and model-error curves; time and
memory per iteration.

    python examples/hybrid/rugged2d_study.py --part forward
    python examples/hybrid/rugged2d_study.py --part fwi
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import DeepGPR  # noqa: E402
from DeepGPR import apply_filter  # noqa: E402
from DeepGPR.hybrid import FieldSampler, HybridTM2D, make_propagator, resample  # noqa: E402
from DeepGPR.hybrid.staircase import StaircaseHybridTM2D  # noqa: E402
from DeepGPR.hybrid.topography import CallableSurface  # noqa: E402
from topography_gifs import CMAP, mesh_segments, staircase_outline, yee_segments  # noqa: E402

OUT = HERE / "results" / "rugged2d"
DX = 0.025
DEPTH, WIDTH = 1.6, 3.2
PML_M = 0.3
FREQ, T_MAX, FRAME_DT = 4.0e8, 24e-9, 0.25e-9
T0 = 1.2 / FREQ
EPS_BG, SIG_BG = 9.0, 0.005
REFLECTOR_DEPTH, EPS_REFLECTOR = 1.30, 16.0
BAND = (0.15, 0.95)
HEIGHT = 0.05
MUTE = 0.2
SRC_Y = np.round(np.arange(0.3, 2.91, 0.2), 6)
REC_Y = np.round(np.arange(0.2, 3.01, 0.1), 6)
GIF_SHOT = 5                       # source at y = 1.3 m, on the ridge flank
ZOOM = (0.20, 0.62, 0.80, 1.55)    # (x0, x1, y0, y1): ridge
REFS = (9, 15)
REF = REFS[0]
DG_H = DX / 2
INK, INK2 = "#0b0b0b", "#52514e"
COLORS = {"fdtd": "#eb6834", "fdtd_r3": "#0f766e", "fdtd_r3avg": "#a855f7", "fdtd_r5": "#9c8a1e",
          "hybrid": "#2a78d6", "stair": "#be185d", "ref": INK}
LABELS = {"fdtd": "FDTD staircase, 2.5 cm", "fdtd_r3": "FDTD staircase, 0.83 cm",
          "fdtd_r3avg": "FDTD cell-averaged, 0.83 cm", "hybrid": "hybrid DGTD + Yee 0.83 cm",
          "fdtd_r5": "FDTD staircase, 0.5 cm", "stair": "hybrid, staircase interface (DG layer)",
          "ref": "reference (averaged FDTD, 2.5/9 cm)"}
LAYER = (0.10, 0.075)          # staircase hybrid: DG at least 10 cm above / 7.5 cm below the surface


def surface_depth(y):
    y = np.asarray(y, dtype=float)
    ridge = -0.24 * np.exp(-((y - 1.05) / 0.17) ** 2)
    gully = 0.17 * np.clip(1.0 - np.abs(y - 2.15) / 0.22, 0.0, None)
    rough = 0.02 * np.sin(2 * np.pi * y / 0.16)
    return 0.55 + ridge + gully + rough


SURFACE = CallableSurface(surface_depth)
ANOMALIES = [  # (y, depth below the local surface, radius, eps_r)
    (0.70, 0.20, 0.10, 13.0),
    (1.30, 0.20, 0.10, 14.0),
    (2.40, 0.20, 0.09, 5.0),
]


# ---------------------------------------------------------------- models
def background(dx=DX):
    nx, ny = int(round(DEPTH / dx)), int(round(WIDTH / dx))
    x = torch.arange(nx, dtype=torch.float64)[:, None] * dx
    eps = torch.full((nx, ny), EPS_BG, dtype=torch.float64)
    eps = torch.where(x + 0.5 * dx > REFLECTOR_DEPTH,
                      torch.clamp((x + 0.5 * dx - REFLECTOR_DEPTH) / dx, 0, 1)
                      * (EPS_REFLECTOR - EPS_BG) + EPS_BG, eps).expand(nx, ny).clone()
    return eps, torch.full((nx, ny), SIG_BG, dtype=torch.float64)


def true_model():
    """The 2.5 cm model grid (what the user supplies)."""
    eps, sig = background()
    nx, ny = eps.shape
    x = torch.arange(nx, dtype=torch.float64)[:, None] * DX
    y = torch.arange(ny, dtype=torch.float64)[None, :] * DX
    for yc, below, r, e in ANOMALIES:
        xc = float(surface_depth(yc)) + below
        eps[(x - xc) ** 2 + (y - yc) ** 2 <= r * r] = e
    return eps, sig


def on_grid(model, dx):
    """2.5 cm pixels resampled to a finer grid (nearest pixel)."""
    eps_c, sig_c = model
    nx, ny = int(round(DEPTH / dx)), int(round(WIDTH / dx))
    i = np.clip(np.floor(np.arange(nx) * dx / DX + 0.5 + 1e-9).astype(int), 0, eps_c.shape[0] - 1)
    j = np.clip(np.floor(np.arange(ny) * dx / DX + 0.5 + 1e-9).astype(int), 0, eps_c.shape[1] - 1)
    return eps_c[i][:, j].clone(), sig_c[i][:, j].clone()


def yee_model(model, dx, averaged):
    """Yee model on ``dx``: staircase (node above the surface = air) or the
    second-order cell average of the ground fraction (Ez tangential in 2D TM)."""
    eps_g, sig_g = on_grid(model, dx)
    nx, ny = eps_g.shape
    x = np.arange(nx)[:, None] * dx
    if averaged:
        frac = np.zeros((nx, ny))
        for k in range(8):
            y = (np.arange(ny) + (k + 0.5) / 8 - 0.5) * dx
            frac += np.clip((x + 0.5 * dx - surface_depth(y)[None]) / dx, 0.0, 1.0)
        frac /= 8
    else:
        frac = (x >= surface_depth(np.arange(ny) * dx)[None]).astype(float)
    f = torch.as_tensor(frac)
    return f * eps_g + (1 - f), f * sig_g, frac < 0.5


def antennas():
    """``(src (S, 2), rec (S, nrx, 2))`` 5 cm above the surface, on 2.5 cm nodes
    in the air (moved up one node if the rounding would put them below)."""
    def place(ys):
        x = np.round((surface_depth(ys) - HEIGHT) / DX) * DX
        x = np.where(x > surface_depth(ys) - 0.01, x - DX, x)
        return np.stack([x, ys], 1)
    src = place(SRC_Y)
    rec = np.broadcast_to(place(REC_Y)[None], (len(SRC_Y), len(REC_Y), 2)).copy()
    return src, rec


def mute_mask(src, rec):
    return np.abs(rec[:, :, 1] - src[:, None, 1]) >= MUTE


def wavelet(nt, dt):
    t = (np.arange(nt) + 0.5) * dt - T0
    a = (np.pi * FREQ * t) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a))


def aligned_dt(dt_max):
    m = math.ceil(FRAME_DT / dt_max * (1 + 1e-9))
    return FRAME_DT / m, m


def nt_of(dt):
    return int(round(T_MAX / dt))


def sync():
    torch.cuda.synchronize()


# ---------------------------------------------------------------- runs
def run_native(eps, sig, dx, dt, src, rec, frame_every=None, decimate=1):
    """Native DeepGPR; returns ``(data (S, nt, nrx), frames of shot 0, seconds, peak GB)``."""
    nt = nt_of(dt)
    pml = int(round(PML_M / dx))
    node = lambda p: np.rint(np.asarray(p) / dx).astype(int)  # noqa: E731
    s, r = node(src), node(rec)
    src_t = torch.tensor([[[i, j, 0]] for i, j in s], dtype=torch.int, device="cuda")
    rec_t = torch.tensor([[[i, j, 0] for i, j in shot] for shot in r], dtype=torch.int,
                         device="cuda")
    w = wavelet(nt, dt).float().cuda()
    nx, ny = eps.shape
    seg = frame_every or nt
    E = H = P = None
    frames, data = [], []
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    sync()
    t0 = time.perf_counter()
    for n0 in range(0, nt, seg):
        n1 = min(nt, n0 + seg)
        _, E, H, P, d = DeepGPR.compute(
            device="cuda", dx=dx, dt=dt, source_amplitudes=w[n0:n1].reshape(1, -1, 1),
            source_location=src_t, receiver_location=rec_t, eps_r=eps.float(),
            sigma=sig.float(), pmlthick=pml, E=E, H=H, PML=P, save_wavefield_history=False)
        data.append(d)
        if frame_every:
            frames.append(E[2][0, pml:pml + nx:decimate, pml:pml + ny:decimate, 0].cpu().numpy())
    sync()
    seconds = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
    return (torch.cat(data, 1).cpu().numpy(), np.stack(frames) if frames else None, seconds,
            peak)


def build_hybrid(model, dt=None, nshot=1, dtype=torch.float32):
    eps, sig = model
    return HybridTM2D(eps, sig, DX, surface=SURFACE, dg_band=BAND,
                      pmlthick=int(round(PML_M / DX)), dg_order=1, dg_h=DG_H, mesh="delaunay",
                      fdtd_refinement=3, dt=dt, nshot=nshot, device="cuda", dtype=dtype)


def build_stair(model, dt=None, nshot=1, dtype=torch.float32):
    eps, sig = model
    return StaircaseHybridTM2D(eps, sig, DX, surface=SURFACE, layer=LAYER,
                               pmlthick=int(round(PML_M / DX)), dg_order=1, dg_h=DG_H,
                               fdtd_refinement=3, dt=dt, nshot=nshot, device="cuda", dtype=dtype)


def run_hybrid(sol, src, rec, samplers=None, frame_every=None):
    frames = {k: [] for k in (samplers or {})}

    def grab(n, s):
        if (n + 1) % frame_every == 0:
            for k, smp in samplers.items():
                frames[k].append(smp())

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    nt = nt_of(sol.dt)
    res = sol.run(wavelet(nt, sol.dt), np.asarray(src)[:, None, :], np.asarray(rec),
                  callback=grab if samplers else None, callback_every=1 if samplers else 0)
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
    return (res["receivers"].cpu().numpy(), {k: np.stack(v) for k, v in frames.items()},
            res["seconds"], peak)


def to_axis(data, dt, t_out):
    t = (np.arange(data.shape[1]) + 1) * dt
    out = np.empty((data.shape[0], len(t_out), data.shape[2]))
    for s in range(data.shape[0]):
        for j in range(data.shape[2]):
            out[s, :, j] = np.interp(t_out, t, data[s, :, j])
    return out


# ---------------------------------------------------------------- forward part
def forward():
    OUT.mkdir(parents=True, exist_ok=True)
    model = true_model()
    src, rec = antennas()
    keep = mute_mask(src, rec)
    runs = {}
    # native FDTD: timing runs (all shots) and frame runs (GIF shot only)
    specs = {"fdtd": (1, False), "fdtd_r3": (3, False), "fdtd_r3avg": (3, True),
             "fdtd_r5": (5, False)}
    specs.update({f"ref{k}": (k, True) for k in REFS})
    for name, (k, averaged) in specs.items():
        dx = DX / k
        eps, sig, air = yee_model(model, dx, averaged)
        dt, m = aligned_dt(0.95 * DeepGPR.max_stable_time_step(dx, 10, 10, 1))
        data, _, sec, peak = run_native(eps, sig, dx, dt, src, rec)
        frames = fine = None
        if name in ("fdtd", "fdtd_r3", f"ref{REF}"):
            # full-resolution frames of the FDTD grids (zoom), decimated to 2.5 cm for the
            # full view and the errors; the reference only on the 2.5 cm nodes
            dec = k if name.startswith("ref") else 1
            _, fr, _, _ = run_native(eps, sig, dx, dt, src[GIF_SHOT:GIF_SHOT + 1],
                                     rec[GIF_SHOT:GIF_SHOT + 1], frame_every=m, decimate=dec)
            is_ref = name.startswith("ref")
            frames = fr if is_ref else fr[:, ::k, ::k]
            fine = None if is_ref else fr
        runs[name] = dict(dt=dt, data=data, frames=frames, fine=fine, seconds=sec, peak=peak,
                          air=air, nt=nt_of(dt), cells=int(np.prod(eps.shape)))
        print(f"{name:11s} dx {dx * 1000:6.2f} mm  nt {nt_of(dt):5d}  {sec:6.2f} s  "
              f"{peak:.2f} GB", flush=True)
    # hybrids: planar DGTD band and staircase DGTD layer
    nx, ny = int(round(DEPTH / DX)), int(round(WIDTH / DX))
    x0, x1, y0, y1 = ZOOM
    for name, build in (("hybrid", build_hybrid), ("stair", build_stair)):
        probe = build(model)
        dt_h, m_h = aligned_dt(probe.dt)
        del probe
        sol = build(model, dt=dt_h, nshot=len(SRC_Y))
        data, _, sec, peak = run_hybrid(sol, src, rec)
        one = build(model, dt=dt_h, nshot=1)
        samplers = {"full": FieldSampler(one, np.arange(nx) * DX, np.arange(ny) * DX),
                    "zoom": FieldSampler(one, np.arange(x0, x1 + 1e-9, 0.004),
                                         np.arange(y0, y1 + 1e-9, 0.004))}
        _, frames, _, _ = run_hybrid(one, src[GIF_SHOT:GIF_SHOT + 1],
                                     rec[GIF_SHOT:GIF_SHOT + 1], samplers, m_h)
        runs[name] = dict(dt=dt_h, data=data, frames=frames, seconds=sec, peak=peak,
                          nt=nt_of(dt_h), mesh=sol.mesh.quality(), K=sol.dg.K, sol=one,
                          summary=sol.summary())
        print(f"{name:11s} dt {dt_h:.3g}  nt {nt_of(dt_h):5d}  {sec:6.2f} s  {peak:.2f} GB  "
              f"{sol.dg.K} triangles", flush=True)
    # errors on the coarse common axis
    t_out = (np.arange(nt_of(runs["fdtd"]["dt"])) + 1) * runs["fdtd"]["dt"]
    gathers = {k: to_axis(v["data"], v["dt"], t_out) for k, v in runs.items()}
    ref = gathers[f"ref{REF}"]
    sel = np.broadcast_to(keep[:, None, :], ref.shape)          # offsets >= MUTE

    def rel(d):
        return float(np.linalg.norm((d - ref)[sel]) / np.linalg.norm(ref[sel]))
    table = {}
    for k in ("fdtd", "fdtd_r3", "fdtd_r3avg", "fdtd_r5", "hybrid", "stair", f"ref{REFS[1]}"):
        table[k] = dict(error=rel(gathers[k]), seconds=runs[k]["seconds"],
                        peak_GB=runs[k]["peak"], nt=runs[k]["nt"], dt=runs[k]["dt"])
        print(f"{k:11s} error vs ref {table[k]['error'] * 100:6.2f} %", flush=True)
    for k in ("hybrid", "stair"):
        table[k]["mesh"] = runs[k]["mesh"]
        table[k]["triangles"] = runs[k]["K"]
    table["stair"]["segments"] = runs["stair"]["summary"]["segments"]
    table["stair"]["concave_corners"] = runs["stair"]["summary"]["concave_corners"]
    table["stair"]["layer"] = LAYER
    (OUT / "forward.json").write_text(json.dumps(table, indent=1, default=float))
    np.savez_compressed(OUT / "forward_gathers.npz", t=t_out, src=src, rec=rec, keep=keep,
                        ref_raw=runs[f"ref{REF}"]["data"], ref_dt=runs[f"ref{REF}"]["dt"],
                        **{k: gathers[k] for k in gathers})
    mesh_figure(OUT / "mesh.png", runs["hybrid"]["sol"], runs["fdtd"]["air"],
                runs["fdtd_r3"]["air"])
    trace_figure(OUT / "traces.png", t_out, gathers, src, rec, table)
    forward_gif(OUT / "forward.gif", runs, src)
    summary_figure(OUT / "forward_summary.png", table)
    staircase_figure(OUT / "interfaces_staircase.png", runs["stair"]["sol"])


# ---------------------------------------------------------------- figures (forward)
def style(ax, title, xlabel=True, ylabel=True):
    ax.set_title(title, fontsize=9, color=INK)
    if xlabel:
        ax.set_xlabel("y (m)", fontsize=8, color=INK2)
    if ylabel:
        ax.set_ylabel("depth x (m)", fontsize=8, color=INK2)
    ax.tick_params(labelsize=7, colors=INK2)


def forward_gif(path, runs, src):
    x0, x1, y0, y1 = ZOOM
    fd, fd3, ref = runs["fdtd"]["frames"], runs["fdtd_r3"]["frames"], runs[f"ref{REF}"]["frames"]
    dgf = {k: runs[k]["frames"] for k in ("hybrid", "stair")}
    n = min(len(fd), len(fd3), len(ref), *(len(v["full"]) for v in dgf.values()))
    vmax = 0.25 * np.percentile(np.abs(ref[8:n]), 99.7)
    emax = vmax / 4
    ys = np.linspace(0, WIDTH, 1600)
    curve = surface_depth(ys)
    stair1 = staircase_outline(runs["fdtd"]["air"], DX)
    stair3 = staircase_outline(runs["fdtd_r3"]["air"], DX / 3)
    ext = [-0.5 * DX, WIDTH - 0.5 * DX, DEPTH - 0.5 * DX, -0.5 * DX]
    fig, axes = plt.subplots(3, 4, figsize=(20, 10.4), constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[0.9, 1.0, 0.9]))
    cols = [("fdtd", fd, stair1, DX), ("fdtd_r3", fd3, stair3, DX / 3), ("hybrid", None, None, None),
            ("stair", None, None, None)]
    fine = {"fdtd": runs["fdtd"]["fine"], "fdtd_r3": runs["fdtd_r3"]["fine"]}

    def zoom_box(h):
        k = int(round(DX / h))
        zx = (int(round(x0 / h)), int(round(x1 / h)) + 1)
        zy = (int(round(y0 / h)), int(round(y1 / h)) + 1)
        return k, zx, zy, [(zy[0] - 0.5) * h, (zy[1] - 0.5) * h, (zx[1] - 0.5) * h, (zx[0] - 0.5) * h]
    imgs = {}
    for c, (name, frames, stair, h) in enumerate(cols):
        hy = dgf.get(name)
        f0 = hy["full"][0] if hy is not None else frames[0]
        imgs[f"f{c}"] = axes[0, c].imshow(f0, cmap=CMAP, vmin=-vmax, vmax=vmax, extent=ext,
                                          interpolation="bilinear")
        axes[0, c].plot(ys, curve, color=INK, lw=0.8)
        if stair is not None:
            axes[0, c].plot(stair[:, 0], stair[:, 1], color=INK, lw=0.6)
        elif name == "hybrid":
            for xb in BAND:
                axes[0, c].axhline(xb, color=INK2, lw=0.7, ls="--")
        else:
            sl = runs[name]["sol"]
            poly = sl.mesh.info["polygon"] - np.array([sl.ox, sl.oy])
            pc = np.concatenate([poly, poly[:1]])
            axes[0, c].plot(pc[:, 1], pc[:, 0], color=INK2, lw=0.7, ls="--")
        axes[0, c].add_patch(Rectangle((y0, x0), y1 - y0, x1 - x0, fill=False, ec=INK, lw=0.8))
        axes[0, c].plot(src[GIF_SHOT, 1], src[GIF_SHOT, 0], "v", ms=6, color=INK)
        style(axes[0, c], LABELS[name] + ("  (DGTD dashed)" if hy is not None else ""),
              xlabel=False, ylabel=(c == 0))
        # zoom with the grid each solver uses
        if hy is not None:
            sl = runs[name]["sol"]
            imgs[f"z{c}"] = axes[1, c].imshow(hy["zoom"][0], cmap=CMAP, vmin=-vmax, vmax=vmax,
                                              extent=[y0, y1, x1, x0], interpolation="bilinear")
            axes[1, c].add_collection(LineCollection(mesh_segments(sl, ZOOM), colors="#8c8b86",
                                                     linewidths=0.35))
            axes[1, c].plot(ys, curve, color=INK, lw=1.4, label="surface = triangle edges")
            if name == "stair":
                poly = sl.mesh.info["polygon"] - np.array([sl.ox, sl.oy])
                pc = np.concatenate([poly, poly[:1]])
                axes[1, c].plot(pc[:, 1], pc[:, 0], color="#e34948", lw=1.6,
                                label="DGTD-FDTD interface")
            title = f"zoom: DGTD triangles (P1, h = {DG_H * 100:.2f} cm)"
        else:
            k, zx, zy, zext = zoom_box(h)
            imgs[f"z{c}"] = axes[1, c].imshow(fine[name][0][zx[0]:zx[1], zy[0]:zy[1]], cmap=CMAP,
                                              vmin=-vmax, vmax=vmax, interpolation="nearest",
                                              extent=zext)
            axes[1, c].add_collection(LineCollection(yee_segments(ZOOM, h), colors="#8c8b86",
                                                     linewidths=0.3 if k > 1 else 0.45))
            axes[1, c].plot(stair[:, 0], stair[:, 1], color=INK, lw=1.4,
                            label="surface seen by FDTD")
            axes[1, c].plot(ys, curve, color=INK, lw=0.8, ls=":", label="true surface")
            title = f"zoom: Yee cells ({h * 100:.2f} cm) and staircase"
        axes[1, c].plot(src[GIF_SHOT, 1], src[GIF_SHOT, 0], "v", ms=8, color=INK)
        axes[1, c].set_xlim(y0, y1)
        axes[1, c].set_ylim(x1, x0)
        axes[1, c].legend(loc="lower left", fontsize=7, framealpha=0.85)
        style(axes[1, c], title, xlabel=False, ylabel=(c == 0))
        e0 = (hy["full"][0] if hy is not None else frames[0]) - ref[0]
        imgs[f"e{c}"] = axes[2, c].imshow(e0, cmap=CMAP, vmin=-emax, vmax=emax, extent=ext,
                                          interpolation="bilinear")
        axes[2, c].plot(ys, curve, color=INK, lw=0.6)
        style(axes[2, c], f"error vs reference ({LABELS[name]}), colour range / 4",
              ylabel=(c == 0))
    for ax in list(axes[0]) + list(axes[2]):
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(DEPTH, 0)
    cb = fig.colorbar(imgs["f3"], ax=axes[:2, :].ravel().tolist(), shrink=0.6, pad=0.01)
    cb.set_label("Ez (clipped)", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    stamp = fig.suptitle("", fontsize=11, color=INK)

    def update(k):
        for c, (name, frames, _, _) in enumerate(cols):
            if name in dgf:
                hy = dgf[name]
                imgs[f"f{c}"].set_data(hy["full"][k])
                imgs[f"z{c}"].set_data(hy["zoom"][k])
                imgs[f"e{c}"].set_data(hy["full"][k] - ref[k])
            else:
                _, zx, zy, _ = zoom_box(cols[c][3])
                imgs[f"f{c}"].set_data(frames[k])
                imgs[f"z{c}"].set_data(fine[name][k][zx[0]:zx[1], zy[0]:zy[1]])
                imgs[f"e{c}"].set_data(frames[k] - ref[k])
        stamp.set_text("Rugged topography (relief 0.43 m, slopes up to 61 deg), eps_r 9 ground, "
                       f"same 2.5 cm model    t = {(k + 1) * FRAME_DT * 1e9:5.2f} ns")
        return list(imgs.values()) + [stamp]

    # every second frame (0.5 ns) keeps the GIF small
    FuncAnimation(fig, update, frames=range(1, n, 2)).save(path, writer=PillowWriter(fps=6),
                                                            dpi=70)
    plt.close(fig)


def mesh_figure(path, sol, air1, air3):
    ys = np.linspace(0, WIDTH, 3200)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.6), constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[0.8, 1.3]))
    window = (BAND[0], BAND[1], 0.0, WIDTH)
    axes[0].add_collection(LineCollection(mesh_segments(sol, window), colors=INK2,
                                          linewidths=0.15))
    axes[0].plot(ys, surface_depth(ys), color="#e34948", lw=0.9, label="ground surface")
    x0, x1, y0, y1 = ZOOM
    axes[0].add_patch(Rectangle((y0, x0), y1 - y0, x1 - x0, fill=False, ec=INK, lw=1.0))
    axes[0].set_xlim(0, WIDTH)
    axes[0].set_ylim(BAND[1], BAND[0])
    axes[0].set_aspect("equal")
    axes[0].legend(loc="lower right", fontsize=8)
    q = sol.mesh.quality()
    style(axes[0], f"DGTD band (Delaunay, {q['elements']} triangles incl. CPML columns, "
                   f"minimum angle {q['min_angle_deg']:.1f} deg); box = zoom")
    ax = axes[1]
    ax.add_collection(LineCollection(yee_segments(ZOOM, DX), colors="#86b6ef", linewidths=0.7))
    ax.add_collection(LineCollection(mesh_segments(sol, ZOOM), colors=INK2, linewidths=0.5))
    s1 = staircase_outline(air1, DX)
    s3 = staircase_outline(air3, DX / 3)
    ax.plot(s1[:, 0], s1[:, 1], color="#eb6834", lw=1.8, label="surface seen by FDTD 2.5 cm")
    ax.plot(s3[:, 0], s3[:, 1], color="#0f766e", lw=1.2, label="surface seen by FDTD 0.83 cm")
    ax.plot(ys, surface_depth(ys), color="#e34948", lw=1.8, label="true surface (DGTD edges)")
    ax.plot([], [], color="#86b6ef", lw=0.7, label="Yee material cells 2.5 cm")
    ax.plot([], [], color=INK2, lw=0.5, label="DGTD triangles")
    ax.set_xlim(y0, y1)
    ax.set_ylim(x1, x0)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    style(ax, "zoom (ridge): what each solver sees at the surface")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def trace_figure(path, t, gathers, src, rec, table):
    s = GIF_SHOT
    picks = [int(np.argmin(np.abs(rec[s, :, 1] - (src[s, 1] + o)))) for o in (-0.5, 0.3, 0.9)]
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 7.2), sharex=True,
                             constrained_layout=True)
    for ax, j in zip(axes, picks):
        ax.plot(t * 1e9, gathers[f"ref{REF}"][s, :, j], color=INK, lw=2.0, label=LABELS["ref"])
        for k in ("fdtd", "fdtd_r3", "hybrid", "stair"):
            ax.plot(t * 1e9, gathers[k][s, :, j], color=COLORS[k], lw=1.1,
                    ls="--" if k in ("hybrid", "stair") else "-",
                    label=f"{LABELS[k]} ({table[k]['error'] * 100:.1f} %)")
        ax.set_ylabel(f"Ez, y = {rec[s, j, 1]:.1f} m", fontsize=8, color=INK2)
        ax.tick_params(labelsize=8, colors=INK2)
        ax.grid(color="#e6e5e1", lw=0.5)
    axes[0].legend(fontsize=7.5, loc="upper right")
    axes[-1].set_xlabel("time (ns)", fontsize=9, color=INK2)
    fig.suptitle(f"source y = {src[s, 1]:.1f} m (ridge flank); legend: relative L2 error of "
                 "all gathers", fontsize=10, color=INK)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary_figure(path, table):
    names = ["fdtd", "fdtd_r3", "fdtd_r3avg", "fdtd_r5", "hybrid", "stair"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, key, label, scale in ((axes[0], "error", "gather error vs reference (%)", 100),
                                  (axes[1], "seconds", "time, 14 shots (s)", 1),
                                  (axes[2], "peak_GB", "peak GPU memory (GB)", 1)):
        vals = [table[n][key] * scale for n in names]
        bars = ax.barh(range(len(names)), vals, color=[COLORS[n] for n in names], height=0.6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([LABELS[n] for n in names], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(label, fontsize=9.5, color=INK)
        ax.tick_params(labelsize=8, colors=INK2)
        for b, v in zip(bars, vals):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:.3g}", va="center",
                    fontsize=8, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- FWI part
def fwi(solvers, iters, merge=False):
    OUT.mkdir(parents=True, exist_ok=True)
    fw = np.load(OUT / "forward_gathers.npz")
    src, rec = fw["src"], fw["rec"]
    keep = torch.as_tensor(fw["keep"], device="cuda")
    ref_data = torch.as_tensor(fw["ref_raw"])
    t_ref = (torch.arange(ref_data.shape[1], dtype=torch.float64) + 1) * float(fw["ref_dt"])
    eps0, sig0 = background()
    true_e, _ = true_model()
    ground = torch.as_tensor(np.arange(eps0.shape[0])[:, None] * DX
                             >= surface_depth(np.arange(eps0.shape[1]) * DX)[None])
    err = lambda m: float((m - true_e)[ground].norm() / (true_e - eps0)[ground].norm())  # noqa: E731
    results = {}
    for name in solvers:
        r = 3 if name in ("fdtd_r3", "hybrid", "stair") else 1
        if name == "stair":
            from DeepGPR.hybrid import HybridPropagator
            prop = HybridPropagator(build_stair(background(), nshot=len(SRC_Y)), src[:, None, :],
                                    rec)
        elif name == "hybrid":
            prop = make_propagator("hybrid_dgtd", eps0, sig0, DX, src[:, None, :], rec,
                                   surface=SURFACE, pmlthick=int(round(PML_M / DX)),
                                   dg_band=BAND, dg_order=1, dg_h=DG_H, mesh="delaunay",
                                   fdtd_refinement=3, device="cuda", dtype=torch.float32)
        else:
            dt = 0.95 * DeepGPR.max_stable_time_step(DX / r, 10, 10, 1)
            extra = {"model_gradient_sampling_interval": "auto"} if r > 1 else {}
            prop = make_propagator("fdtd", eps0, sig0, DX, src[:, None, :], rec, surface=SURFACE,
                                   pmlthick=int(round(PML_M / DX)), dt=dt, device="cuda",
                                   fdtd_refinement=r, **extra)
        nt = nt_of(prop.dt)
        dobs = resample(ref_data[None] if ref_data.ndim == 2 else ref_data, t_ref,
                        prop.time_axis(nt)).to("cuda", torch.float32)
        mute = keep[:, None, :].to(torch.float32)
        dobs = dobs * mute
        w = wavelet(nt, prop.dt).to("cuda", torch.float32)
        fs = 1.0 / prop.dt
        E0, S0 = eps0.to("cuda", torch.float32), sig0.to("cuda", torch.float32)
        u = torch.zeros_like(E0, requires_grad=True)
        def misfit_full():
            """Full-band normalised misfit (recorded every iteration, not timed)."""
            with torch.no_grad():
                d = prop((E0 + u).clamp(min=1.0), S0, w[None]).float() * mute
                return float(((d - dobs) ** 2).sum() / (dobs ** 2).sum())

        hist = dict(misfit=[misfit_full()], eps_err=[err(eps0)], seconds=[], evals=[],
                    eps=[eps0.numpy()])
        peak = 0.0
        evals = [0]
        for cutoff, n_it in ((2.5e8, max(1, iters // 3)), (None, iters - max(1, iters // 3))):
            norm = [None]

            def objective():
                eps = (E0 + u).clamp(min=1.0)
                d = prop(eps, S0, w[None]).float() * mute
                obs = dobs
                if cutoff is not None:
                    d, obs = apply_filter(d, fs, cutoff), apply_filter(dobs, fs, cutoff)
                loss = ((d - obs) ** 2).sum() / (obs ** 2).sum()
                evals[0] += 1
                return loss

            opt = torch.optim.LBFGS([u], lr=1.0, history_size=10, max_iter=1,
                                    line_search_fn="strong_wolfe")
            for _ in range(n_it):
                torch.cuda.reset_peak_memory_stats()
                sync()
                t0 = time.perf_counter()
                n0 = evals[0]

                def closure():
                    opt.zero_grad()
                    raw = objective()
                    if norm[0] is None:
                        norm[0] = float(raw.detach())
                    loss = raw / norm[0]
                    loss.backward()
                    return loss

                opt.step(closure)
                sync()
                hist["seconds"].append(time.perf_counter() - t0)
                peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 30)
                eps_now = (E0 + u).clamp(min=1.0).detach().cpu().double()
                hist["misfit"].append(misfit_full())
                hist["eps_err"].append(err(eps_now))
                hist["evals"].append(evals[0] - n0)
                hist["eps"].append(eps_now.numpy())
                print(f"{name:8s} it {len(hist['seconds']):3d} misfit {hist['misfit'][-1]:.4e} "
                      f"eps err {hist['eps_err'][-1]:.3f} ({hist['seconds'][-1]:.1f} s, "
                      f"{hist['evals'][-1]} evaluations)", flush=True)
        results[name] = dict(misfit=hist["misfit"], eps_err=hist["eps_err"],
                             seconds=hist["seconds"], evals=hist["evals"], peak_GB=peak,
                             nt=nt, dt=prop.dt, eps=np.stack(hist["eps"]))
        prop = None   # free the propagator before the next solver
        torch.cuda.empty_cache()
    if merge and (OUT / "fwi.json").exists():
        # keep earlier runs of the solvers not rerun now (same schedule assumed)
        old = json.loads((OUT / "fwi.json").read_text())
        models = np.load(OUT / "fwi_models.npz")
        for k, v in old.items():
            if k not in results and k in models.files:
                results[k] = dict(v, eps=models[k])
        results = {k: results[k] for k in ("fdtd", "fdtd_r3", "hybrid", "stair") if k in results}
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "eps"} for k, v in results.items()}
    for k, v in summary.items():
        v["final_eps_err"] = v["eps_err"][-1]
        v["total_seconds"] = float(np.sum(v["seconds"]))
        v["seconds_per_evaluation"] = float(np.sum(v["seconds"]) / np.sum(v["evals"]))
    (OUT / "fwi.json").write_text(json.dumps(summary, indent=1, default=float))
    fwi_summary_figure(OUT / "fwi_summary.png", summary)
    np.savez_compressed(OUT / "fwi_models.npz", true=true_e.numpy(), init=eps0.numpy(),
                        ground=ground.numpy(), **{k: v["eps"] for k, v in results.items()})
    fwi_gif(OUT / "fwi.gif", results, true_e.numpy(), ground.numpy())


def fwi_summary_figure(path, summary):
    names = [k for k in ("fdtd", "fdtd_r3", "hybrid", "stair") if k in summary]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), constrained_layout=True)
    items = ((axes[0], lambda v: v["final_eps_err"], "final eps_r model error"),
             (axes[1], lambda v: min(v["eps_err"]), "best eps_r model error (any iteration)"),
             (axes[2], lambda v: v["seconds_per_evaluation"], "time per misfit + gradient (s)"),
             (axes[3], lambda v: v["peak_GB"], "peak GPU memory (GB)"))
    for ax, f, label in items:
        vals = [f(summary[n]) for n in names]
        bars = ax.barh(range(len(names)), vals, color=[COLORS[n] for n in names], height=0.6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([LABELS[n] for n in names], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(label, fontsize=9.5, color=INK)
        ax.tick_params(labelsize=8, colors=INK2)
        for b, v in zip(bars, vals):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:.3g}", va="center",
                    fontsize=8, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fwi_gif(path, results, true_e, ground):
    names = list(results)
    n = max(len(r["eps"]) for r in results.values())
    ys = np.linspace(0, WIDTH, 1600)
    ext = [-0.5 * DX, WIDTH - 0.5 * DX, DEPTH - 0.5 * DX, -0.5 * DX]
    show = lambda m: np.where(ground, m, np.nan)  # noqa: E731
    fig = plt.figure(figsize=(15, 3.1 * (len(names) + 1) / 2 + 3.2), constrained_layout=True)
    gs = fig.add_gridspec((len(names) + 2) // 2 + 1, 2)
    panels = [("true", true_e)] + [(k, results[k]["eps"][0]) for k in names]
    imgs = []
    for i, (k, m) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        im = ax.imshow(show(m), cmap="viridis", vmin=4.0, vmax=16.0, extent=ext,
                       interpolation="nearest")
        ax.plot(ys, surface_depth(ys), color=INK, lw=0.8)
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(DEPTH, 0)
        style(ax, "true eps_r" if k == "true" else LABELS[k], xlabel=False)
        imgs.append((k, im, ax))
    fig.colorbar(imgs[0][1], ax=[a for _, _, a in imgs], shrink=0.8, label="eps_r")
    axm = fig.add_subplot(gs[-1, 0])
    axe = fig.add_subplot(gs[-1, 1])
    lines = []
    for k in names:
        r = results[k]
        it = np.arange(len(r["misfit"]))
        axm.semilogy(it, np.asarray(r["misfit"]) / r["misfit"][0], color=COLORS[k], lw=1.5,
                     label=LABELS[k])
        axe.plot(np.arange(len(r["eps_err"])), r["eps_err"], color=COLORS[k], lw=1.5,
                 label=LABELS[k])
    for ax, title in ((axm, "full-band data misfit / initial"),
                      (axe, "eps_r model error ||m - m_true|| / ||m_true - m_0||")):
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("L-BFGS iteration", fontsize=8, color=INK2)
        ax.tick_params(labelsize=7, colors=INK2)
        ax.grid(color="#e6e5e1", lw=0.5)
        ax.legend(fontsize=7.5)
        lines.append(ax.axvline(0, color=INK2, lw=0.8))
    stamp = fig.suptitle("", fontsize=11, color=INK)

    def update(i):
        for k, im, ax in imgs[1:]:
            eps = results[k]["eps"]
            im.set_data(show(eps[min(i, len(eps) - 1)]))
            ax.set_title(f"{LABELS[k]}: eps error {results[k]['eps_err'][min(i, len(eps) - 1)]:.3f}",
                         fontsize=9, color=INK)
        for ln in lines:
            ln.set_xdata([i, i])
        stamp.set_text(f"eps_r-only FWI on rugged topography (observed: averaged FDTD 2.5/{REF} cm)"
                       f"    iteration {i}")
        return [im for _, im, _ in imgs] + lines + [stamp]

    FuncAnimation(fig, update, frames=n).save(path, writer=PillowWriter(fps=2), dpi=72)
    plt.close(fig)


def staircase_figure(path, sol):
    """Where the staircase hybrid's DGTD-FDTD interface is: full section and zooms."""
    import matplotlib as mpl
    mpl.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    poly = sol.mesh.info["polygon"] - np.array([sol.ox, sol.oy])     # physical frame
    pc = np.concatenate([poly, poly[:1]])
    h = sol.dx
    eps, _ = true_model()
    eps_air = eps.numpy().copy()
    x = np.arange(eps.shape[0])[:, None] * DX
    eps_air[x < surface_depth(np.arange(eps.shape[1]) * DX)[None]] = np.nan
    src, rec = antennas()
    fig = plt.figure(figsize=(14, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(eps_air, cmap="viridis", vmin=1, vmax=16, extent=[-DX / 2, WIDTH - DX / 2,
                   DEPTH - DX / 2, -DX / 2], interpolation="nearest")
    from matplotlib.patches import Polygon as MplPolygon
    ax.add_patch(MplPolygon(np.stack([poly[:, 1], poly[:, 0]], 1), closed=True, fc="#2a78d6",
                            ec="none", alpha=0.25, zorder=2))
    ax.plot(pc[:, 1], pc[:, 0], color="#e34948", lw=1.8, zorder=4, label="DGTD-FDTD 交界面（阶梯折线）")
    ys = np.linspace(0, WIDTH, 3000)
    ax.plot(ys, surface_depth(ys), color=INK, lw=1.0, zorder=3, label="地表（DG 三角形的边）")
    for xb in BAND:
        ax.axhline(xb, color=INK2, lw=0.9, ls="--", zorder=3)
    ax.text(0.05, BAND[0] - 0.015, "原平面带的交界面（对比）", fontsize=8.5, color=INK2)
    ax.plot(rec[0, :, 1], rec[0, :, 0], ".", ms=3, color=INK, zorder=5, label="接收点")
    ax.plot(src[:, 1], src[:, 0], "v", ms=6, color="#eb6834", zorder=5, label="源")
    za = (0.18, 0.48, 0.80, 1.10)
    zb = (0.60, 0.92, 2.00, 2.32)
    for z, lab in ((za, "放大 A"), (zb, "放大 B")):
        ax.add_patch(Rectangle((z[2], z[0]), z[3] - z[2], z[1] - z[0], fill=False, ec=INK, lw=1.1,
                               zorder=6))
        ax.text(z[3] + 0.02, z[0] + 0.03, lab, fontsize=9, color=INK)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(DEPTH, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("深度 x (m)")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01, label="εr（空气为白色）")
    q = sol.mesh.quality()
    ax.set_title(f"阶梯交界面：DGTD 只占地表附近一层（地表以上 ≥ {LAYER[0] * 100:.0f} cm、以下 ≥ "
                 f"{LAYER[1] * 100:.1f} cm，{q['elements']} 个三角形），其余全部是 Yee-FDTD",
                 fontsize=11.5, color=INK)

    def zoom(a, z, title):
        x0, x1, y0, y1 = z
        rows = np.arange(np.ceil(x0 / h), np.floor(x1 / h) + 1) * h
        cols = np.arange(np.ceil(y0 / h), np.floor(y1 / h) + 1) * h
        XX, YY = np.meshgrid(rows, cols, indexing="ij")
        yee = ~sol.in_band(XX + sol.ox, YY + sol.oy) | np.isclose(
            _dist_poly(np.stack([XX.ravel(), YY.ravel()], 1), pc).reshape(XX.shape), 0, atol=1e-9)
        segs = []
        for i, r_ in enumerate(rows):
            for j in range(len(cols) - 1):
                if yee[i, j] and yee[i, j + 1]:
                    segs.append([(cols[j], r_), (cols[j + 1], r_)])
        for j, c_ in enumerate(cols):
            for i in range(len(rows) - 1):
                if yee[i, j] and yee[i + 1, j]:
                    segs.append([(c_, rows[i]), (c_, rows[i + 1])])
        a.add_collection(LineCollection(segs, colors="#86b6ef", linewidths=0.7))
        a.plot(YY[yee], XX[yee], ".", ms=2.5, color="#2a78d6")
        a.add_collection(LineCollection(mesh_segments(sol, z), colors=INK2, linewidths=0.6))
        a.plot(pc[:, 1], pc[:, 0], color="#e34948", lw=2.2)
        a.plot(ys, surface_depth(ys), color=INK, lw=1.4)
        a.plot(src[:, 1], src[:, 0], "v", ms=8, color="#eb6834")
        a.set_xlim(y0, y1)
        a.set_ylim(x1, x0)
        a.set_aspect("equal")
        a.set_xlabel("y (m)")
        a.set_ylabel("深度 x (m)")
        a.set_title(title, fontsize=10.5, color=INK)

    zoom(fig.add_subplot(gs[1, 0]), za, "放大 A：陡脊左翼（红 = 交界面，蓝 = Yee 0.83 cm，灰 = 三角形）")
    zoom(fig.add_subplot(gs[1, 1]), zb, "放大 B：V 形沟（台阶的竖直段与凹/凸角）")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _dist_poly(pts, poly):
    from DeepGPR.hybrid.mesh import _distance_to_polyline
    return _distance_to_polyline(pts, poly)


def interface_figure(path):
    """Where the hybrid's two DGTD-FDTD interfaces are: full section and zooms."""
    import matplotlib as mpl
    mpl.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    sol = build_hybrid(true_model())
    xt, xb = sol.x_top - sol.ox, sol.x_bottom - sol.ox          # physical depths of the rows
    h = sol.dx
    pml = PML_M
    eps, _ = true_model()
    eps_air = eps.numpy().copy()
    x = np.arange(eps.shape[0])[:, None] * DX
    eps_air[x < surface_depth(np.arange(eps.shape[1]) * DX)[None]] = np.nan
    src, rec = antennas()
    fig = plt.figure(figsize=(14, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])
    ax = fig.add_subplot(gs[0, :])
    ax.add_patch(Rectangle((-pml, -pml), WIDTH + 2 * pml, DEPTH + 2 * pml, fc="#e6e5e1",
                           ec="none", hatch="///", alpha=0.6, zorder=0))
    ax.add_patch(Rectangle((0, 0), WIDTH, DEPTH, fc="white", ec="none", zorder=0.5))
    im = ax.imshow(eps_air, cmap="viridis", vmin=1, vmax=16, extent=[-DX / 2, WIDTH - DX / 2,
                   DEPTH - DX / 2, -DX / 2], interpolation="nearest", zorder=1)
    ax.add_patch(Rectangle((-pml, xt), WIDTH + 2 * pml, xb - xt, fc="#2a78d6", ec="none",
                           alpha=0.18, zorder=2))
    ys = np.linspace(-pml, WIDTH + pml, 2000)
    ax.plot(ys, surface_depth(np.clip(ys, 0, WIDTH)), color=INK, lw=1.0, zorder=3)
    for xi, name, above, below in ((xt, "上交界面", "Yee-FDTD（空气，网格 0.83 cm）",
                                    "DGTD 三角网格（P1，h = 1.25 cm）"),
                                   (xb, "下交界面", "DGTD 三角网格",
                                    "Yee-FDTD（地下，网格 0.83 cm）")):
        ax.axhline(xi, color="#e34948", lw=2.4, zorder=4)
        ax.text(WIDTH + pml + 0.03, xi, f"{name}\nx = {xi:.3f} m", va="center", fontsize=10.5,
                color="#e34948", fontweight="bold")
        ax.annotate(f"上方：{above}\n下方：{below}", xy=(3.02, xi), xytext=(2.55, xi + (
            -0.12 if name == "上交界面" else 0.22)), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8), zorder=5,
            bbox=dict(fc="white", ec=INK2, lw=0.5, alpha=0.9))
    ax.plot(rec[0, :, 1], rec[0, :, 0], ".", ms=3, color=INK, zorder=5, label="接收点（29 个）")
    ax.plot(src[:, 1], src[:, 0], "v", ms=6, color="#eb6834", zorder=5, label="源（14 个）")
    ax.text(1.6, 0.06, "Yee-FDTD 区（空气）", ha="center", fontsize=10, color=INK)
    ax.text(2.75, 0.33, "DGTD 带（浅蓝）\n地表（黑线）是三角形的边", ha="center", va="center",
            fontsize=9.5, color="#0d366b", bbox=dict(fc="white", ec="none", alpha=0.75))
    ax.text(1.6, 1.12, "Yee-FDTD 区（地下，含异常体下方与深部反射层）", ha="center", fontsize=10,
            color="white")
    ax.text(-pml / 2, DEPTH / 2, "CPML", rotation=90, ha="center", va="center", fontsize=9,
            color=INK2)
    ax.text(WIDTH + pml / 2, 1.45, "CPML", rotation=90, ha="center", va="center",
            fontsize=9, color=INK2)
    ax.text(1.6, -pml / 2, "CPML", ha="center", va="center", fontsize=9, color=INK2)
    ax.text(1.6, DEPTH + pml / 2, "CPML", ha="center", va="center", fontsize=9, color=INK2)
    zt = (xt - 0.05, xt + 0.06, 1.00, 1.22)
    zb = (xb - 0.06, xb + 0.05, 2.00, 2.22)
    for z in (zt, zb):
        ax.add_patch(Rectangle((z[2], z[0]), z[3] - z[2], z[1] - z[0], fill=False, ec=INK, lw=1.2,
                               zorder=6))
    ax.text(zt[3] + 0.02, zt[0], "放大 A", fontsize=9, color=INK, va="top")
    ax.text(zb[3] + 0.02, zb[1], "放大 B", fontsize=9, color=INK, va="bottom")
    ax.set_xlim(-pml, WIDTH + pml + 0.55)
    ax.set_ylim(DEPTH + pml, -pml)
    ax.set_aspect("equal")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("深度 x (m)")
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01, label="εr（空气为白色）")
    ax.set_title("混合网格的两个交界面：DGTD 带只覆盖地表附近（0.15–0.95 m），其余全部是 Yee-FDTD",
                 fontsize=12, color=INK)

    def zoom(a, z, title, band_below):
        x0, x1, y0, y1 = z
        xi = xt if band_below else xb
        # Yee Ez lattice on the FDTD side only (the nodes inside the band are not updated)
        rows = np.arange(np.ceil(x0 / h), np.floor(x1 / h) + 1) * h
        cols = np.arange(np.ceil(y0 / h), np.floor(y1 / h) + 1) * h
        yee_rows = rows[rows <= xi + 1e-9] if band_below else rows[rows >= xi - 1e-9]
        segs = [[(y0, r), (y1, r)] for r in yee_rows]
        lo, hi = (x0, xi) if band_below else (xi, x1)
        segs += [[(c, lo), (c, hi)] for c in cols]
        a.add_collection(LineCollection(segs, colors="#86b6ef", linewidths=0.8))
        YY, XX = np.meshgrid(cols, yee_rows)
        a.plot(YY.ravel(), XX.ravel(), ".", ms=3, color="#2a78d6")
        a.add_collection(LineCollection(mesh_segments(sol, z), colors=INK2, linewidths=0.7))
        a.axhline(xi, color="#e34948", lw=2.2)
        a.set_xlim(y0, y1)
        a.set_ylim(x1, x0)
        a.set_aspect("equal")
        a.set_xlabel("y (m)")
        a.set_ylabel("深度 x (m)")
        yee_side = "上方" if band_below else "下方"
        dg_side = "下方" if band_below else "上方"
        a.set_title(title, fontsize=10.5, color=INK)
        a.text(y0 + 0.004, xi + (-0.004 if band_below else 0.004),
               f"交界面 = Yee 的一行 Ez 节点（x = {xi:.3f} m）",
               va="bottom" if band_below else "top", fontsize=8.5, color="#e34948",
               bbox=dict(fc="white", ec="none", alpha=0.8))
        a.text(y1 - 0.004, x0 + 0.004 if band_below else x1 - 0.004,
               f"{yee_side}：Yee 网格 {h * 100:.2f} cm（蓝点 = Ez 节点）",
               ha="right", va="top" if band_below else "bottom", fontsize=8.5, color="#2a78d6",
               bbox=dict(fc="white", ec="none", alpha=0.8))
        a.text(y1 - 0.004, x1 - 0.004 if band_below else x0 + 0.004,
               f"{dg_side}：DGTD 三角形（节点与 Yee 不对齐，用 mortar 投影耦合）",
               ha="right", va="bottom" if band_below else "top", fontsize=8.5, color=INK,
               bbox=dict(fc="white", ec="none", alpha=0.8))

    zoom(fig.add_subplot(gs[1, 0]), zt, "放大 A：上交界面（空气中）", True)
    zoom(fig.add_subplot(gs[1, 1]), zb, "放大 B：下交界面（地下）", False)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return xt, xb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=("forward", "fwi", "all", "interfaces"), default="all")
    ap.add_argument("--iters", type=int, default=24)
    ap.add_argument("--solvers", nargs="+", default=["fdtd", "fdtd_r3", "hybrid", "stair"])
    ap.add_argument("--merge", action="store_true",
                    help="add the solvers run now to the existing fwi.json / fwi_models.npz")
    args = ap.parse_args()
    if args.part in ("forward", "all"):
        forward()
    if args.part in ("fwi", "all"):
        fwi(args.solvers, args.iters, args.merge)
    if args.part in ("interfaces", "all"):
        print("interfaces at x =", interface_figure(OUT / "interfaces.png"))


if __name__ == "__main__":
    main()
