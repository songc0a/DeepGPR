"""Phase 2 benchmark: Yee-FDTD (staircase), full DGTD and hybrid DGTD-FDTD on topography.

For every model in ``models.CASES`` the script runs

* the reference: native DeepGPR FDTD at ``dx/8`` (2.5 mm) with cell-averaged
  eps and sigma at the surface (second order for Ez in 2D TM; it agrees with
  ``dx/16`` to 0.02 % / 0.01 % / 0.27 % for models A / B / C);
* native FDTD with the staircase surface (DeepGPR default) at ``dx``, ``dx/2``,
  ``dx/4``, and with cell-averaged materials at ``dx``, ``dx/2``;
* full DGTD (``dg_band=(None, None)``) with P1 (``h = dx``) and P2 (``h = 2 dx``);
* the hybrid with P1 (``h = dx``) and P2 (``h = 2 dx``), Yee part at ``dx/3``
  (``fdtd_refinement=3``; no extra time steps), plus P1 with the Yee part at
  ``dx`` for comparison;

for two acquisitions - a common-shot gather (57 receivers) and a
common-offset B-scan (26 positions, Tx-Rx offset 0.4 m) - and records the
relative L2 / median per-trace errors against the reference, wall time,
peak GPU memory, unknowns and time steps. Figures: wavefield snapshots,
snapshot errors, B-scans and B-scan errors. Tables:
``results/phase2_summary.json`` and ``results/phase2_tables.md``.

Run from the repository root (CUDA)::

    python examples/hybrid/phase2_benchmark.py [--cases A_flat B_sine C_irregular slope]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import DeepGPR  # noqa: E402
from DeepGPR.hybrid import FieldSampler  # noqa: E402
from models import (  # noqa: E402
    CASES, DEPTH, DX, EPS_SOIL, FRAME_DT, FREQ, PML_M, ROOT, WIDTH, aligned_dt, antenna_positions,
    build_dg, nt_of, run_dg, run_native, yee_model,
)

REF = 8
BSCAN_Y = np.arange(0.5, 5.5 + 1e-9, 0.2)
BSCAN_OFFSET = 0.4
GATHER_Y = np.arange(0.2, WIDTH - 0.2 + 1e-9, 0.1)
SNAP_TIMES = (8e-9, 16e-9, 24e-9)
MAX_NATIVE_CELLS = 6e7      # cells x shots per native call
CMAP = LinearSegmentedColormap.from_list(
    "bluered", ["#0d366b", "#2a78d6", "#86b6ef", "#f0efec", "#f0a5a4", "#e34948", "#8a1c1c"])
INK, INK2 = "#0b0b0b", "#52514e"
C0 = 299792458.0


def geometry(case):
    src_g = antenna_positions(case, [case["source_y"]])           # (1, 2)
    rec_g = antenna_positions(case, GATHER_Y)[None]               # (1, 57, 2)
    tx = antenna_positions(case, BSCAN_Y - BSCAN_OFFSET / 2)      # (26, 2)
    rx = antenna_positions(case, BSCAN_Y + BSCAN_OFFSET / 2)[:, None]
    return src_g, rec_g, tx, rx


def native_bscan(eps, sig, dx, dt, nt, tx, rx, device):
    per_call = max(1, int(MAX_NATIVE_CELLS // eps.numel()))
    data, seconds, peak = [], 0.0, 0
    for k in range(0, len(tx), per_call):
        d, _, s, p = run_native(eps, sig, dx, dt, nt, tx[k:k + per_call], rx[k:k + per_call],
                                device)
        data.append(d)
        seconds += s
        peak = max(peak, p)
    return np.concatenate(data, 0), seconds, peak


def to_axis(t_target, dt, d):
    """Resample ``d (nshot, nt, nrx)`` sampled at ``(n + 1) dt`` onto ``t_target``."""
    t = (np.arange(d.shape[1]) + 1) * dt
    out = np.empty((d.shape[0], t_target.size, d.shape[2]))
    for s in range(d.shape[0]):
        for j in range(d.shape[2]):
            out[s, :, j] = np.interp(t_target, t, d[s, :, j])
    return out


def errors(d, ref, keep=None):
    d = d.reshape(-1, d.shape[1], d.shape[2]).transpose(1, 0, 2).reshape(d.shape[1], -1)
    r = ref.reshape(-1, ref.shape[1], ref.shape[2]).transpose(1, 0, 2).reshape(ref.shape[1], -1)
    if keep is not None:
        d, r = d[:, keep], r[:, keep]
    per = np.linalg.norm(d - r, axis=0) / np.linalg.norm(r, axis=0)
    return dict(rel_l2=float(np.linalg.norm(d - r) / np.linalg.norm(r)),
                median_trace=float(np.median(per)), max_trace=float(per.max()))


def run_case(name, case, device, outdir):
    src_g, rec_g, tx, rx = geometry(case)
    nx, ny = int(round(DEPTH / DX)), int(round(WIDTH / DX))
    node_grid = (np.arange(nx) * DX, np.arange(ny) * DX)
    methods, frames = {}, {}

    # ---------------- native Yee-FDTD: staircase (DeepGPR default) and
    # cell-averaged eps/sigma; the averaged dx/8 run is the reference.
    for k, averaged in ((1, False), (2, False), (4, False), (1, True), (2, True), (REF, True)):
        dx = DX / k
        eps, sig, air = yee_model(dx, case, averaged)
        dt, m = aligned_dt(0.95 * DeepGPR.max_stable_time_step(dx, 10, 10, 1))
        nt = nt_of(dt)
        pml = int(round(PML_M / dx))
        ext_cells = (eps.shape[0] + 2 * pml) * (eps.shape[1] + 2 * pml)  # incl. CPML
        if (k, averaged) in ((1, False), (REF, True)):
            _, fr, _, _ = run_native(eps, sig, dx, dt, nt, src_g, rec_g, device,
                                     frame_every=m, decimate=k)
            frames[k] = fr
        g, secs, peak = native_bscan(eps, sig, dx, dt, nt, src_g, rec_g, device)
        b, bsecs, bpeak = native_bscan(eps, sig, dx, dt, nt, tx, rx, device)
        label = ("reference" if k == REF else
                 f"Yee dx/{k} averaged" if averaged else f"Yee dx/{k}")
        methods[label] = dict(dt=dt, gather=g, bscan=b, seconds=secs, bscan_seconds=bsecs,
                              peak_mb=peak / 2**20, bscan_peak_mb=bpeak / 2**20,
                              unknowns=3 * ext_cells, steps=nt, cells=ext_cells)
        print(f"  {name}: {label} done ({secs:.2f} s, B-scan {bsecs:.1f} s)", flush=True)

    # ---------------- DGTD variants
    variants = [("full DGTD P1", "dgtd", 1, None, None), ("full DGTD P2", "dgtd", 2, None, None),
                ("hybrid P1, Yee dx", "hybrid", 1, None, 1),
                ("hybrid P1", "hybrid", 1, None, None), ("hybrid P2", "hybrid", 2, None, None)]
    for band in case.get("extra_bands", ()):
        variants.append((f"hybrid P1, band {band[0]}-{band[1]} m", "hybrid", 1, band, None))
    for label, mode, order, band, refinement in variants:
        t0 = time.perf_counter()
        probe = build_dg(case, device, mode=mode, order=order, band=band, refinement=refinement)
        dt, m = aligned_dt(probe.dt)
        del probe
        sol = build_dg(case, device, mode=mode, order=order, dt=dt, band=band,
                       refinement=refinement)
        setup = time.perf_counter() - t0
        nt = nt_of(dt)
        samplers = ({"full": FieldSampler(sol, *node_grid)}
                    if order == 1 and band is None and refinement is None else None)
        g, fr, secs, peak = run_dg(sol, nt, src_g, rec_g, frame_every=m, samplers=samplers)
        if samplers:
            frames[label] = fr["full"]
        info = sol.summary()
        del sol
        torch.cuda.empty_cache()
        solb = build_dg(case, device, mode=mode, order=order, dt=dt, nshot=len(tx), band=band,
                        refinement=refinement)
        b, _, bsecs, bpeak = run_dg(solb, nt, tx, rx)
        del solb
        torch.cuda.empty_cache()
        unknowns = info.get("dg_dofs", 0) + info.get("fdtd_dofs", 0)
        methods[label] = dict(dt=dt, gather=g, bscan=b, seconds=secs, bscan_seconds=bsecs,
                              peak_mb=peak / 2**20, bscan_peak_mb=bpeak / 2**20,
                              unknowns=int(unknowns), steps=nt, setup_seconds=setup,
                              dg_elements=info["dg_elements"],
                              yee_nodes=info.get("fdtd_active_ez_nodes", 0),
                              min_angle_deg=info["min_angle_deg"])
        print(f"  {name}: {label} done ({secs:.2f} s, B-scan {bsecs:.1f} s)", flush=True)

    # ---------------- errors on the coarse Yee time axis
    t_axis = (np.arange(nt_of(methods["Yee dx/1"]["dt"])) + 1) * methods["Yee dx/1"]["dt"]
    ref_g = to_axis(t_axis, methods["reference"]["dt"], methods["reference"]["gather"])
    ref_b = to_axis(t_axis, methods["reference"]["dt"], methods["reference"]["bscan"])
    keep = np.abs(rec_g[0, :, 1] - src_g[0, 1]) >= 0.3
    table = {}
    resampled = {}
    for label, mth in methods.items():
        g = to_axis(t_axis, mth["dt"], mth["gather"])
        b = to_axis(t_axis, mth["dt"], mth["bscan"])
        resampled[label] = (g, b)
        row = {k: v for k, v in mth.items() if k not in ("gather", "bscan")}
        if label != "reference":
            row["gather_error"] = errors(g, ref_g, keep)
            row["bscan_error"] = errors(b, ref_b)
        table[label] = row

    # ---------------- figures
    snapshot_figures(outdir, name, case, frames)
    bscan_figure(outdir / f"phase2_{name}_bscan.png", case, t_axis, ref_b, resampled)
    np.savez_compressed(outdir / f"phase2_{name}_data.npz", t=t_axis,
                        **{f"gather_{k.replace(' ', '_').replace('/', '')}": v[0]
                           for k, v in resampled.items()},
                        **{f"bscan_{k.replace(' ', '_').replace('/', '')}": v[1]
                           for k, v in resampled.items()})
    return table


def snapshot_figures(outdir, name, case, frames):
    rows = [("reference", REF, f"reference: Yee-FDTD dx/{REF} = 2.5 mm, cell-averaged"),
            ("Yee", 1, "Yee-FDTD dx = 2 cm (staircase)"),
            ("full", "full DGTD P1", "full DGTD P1, h = 2 cm"),
            ("hybrid", "hybrid P1", "hybrid DGTD-FDTD P1, h = 2 cm, Yee 6.7 mm")]
    idx = [int(round(t / FRAME_DT)) - 1 for t in SNAP_TIMES]
    ref = frames[REF]
    vmax = 0.2 * np.percentile(np.abs(ref[min(8, len(ref) // 2):]), 99.7)
    ext = [-0.5 * DX, WIDTH - 0.5 * DX, DEPTH - 0.5 * DX, -0.5 * DX]
    ys = np.linspace(0, WIDTH, 1200)
    for kind in ("field", "error"):
        sub = rows if kind == "field" else rows[1:]
        fig, axes = plt.subplots(len(sub), len(idx), figsize=(12, 2.35 * len(sub) + 0.6),
                                 constrained_layout=True, squeeze=False)
        for r, (_, key, title) in enumerate(sub):
            for c, k in enumerate(idx):
                data = frames[key][k] if kind == "field" else frames[key][k] - ref[k]
                scale = vmax if kind == "field" else vmax / 4
                im = axes[r, c].imshow(data, cmap=CMAP, vmin=-scale, vmax=scale, extent=ext,
                                       interpolation="bilinear")
                axes[r, c].plot(ys, case["surface"](ys), color=INK, lw=0.6)
                if key in ("hybrid P1",):
                    for xb in case["band"]:
                        axes[r, c].axhline(xb, color=INK2, lw=0.6, ls="--")
                axes[r, c].set_title(f"{title}\nt = {SNAP_TIMES[c] * 1e9:.0f} ns", fontsize=8,
                                     color=INK)
                axes[r, c].tick_params(labelsize=7, colors=INK2)
        cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
        cb.set_label("Ez (V/m)" if kind == "field" else "Ez error vs reference (V/m), range / 4",
                     fontsize=8, color=INK2)
        fig.suptitle(case["label"] + (" - wavefield snapshots" if kind == "field"
                                      else " - snapshot errors"), fontsize=10, color=INK)
        fig.savefig(outdir / f"phase2_{name}_snapshots_{kind}.png", dpi=110)
        plt.close(fig)


def bscan_figure(path, case, t, ref_b, resampled):
    show = [("reference", ref_b), ("Yee dx/1", resampled["Yee dx/1"][1]),
            ("full DGTD P1", resampled["full DGTD P1"][1]),
            ("hybrid P1", resampled["hybrid P1"][1])]
    gain = (t[:, None] * 1e9 + 1.0) ** 1.5
    img = lambda b: (b[:, :, 0].T * gain)  # noqa: E731
    vmax = np.percentile(np.abs(img(ref_b)), 99)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.4), sharey=True, constrained_layout=True)
    ext = [BSCAN_Y[0], BSCAN_Y[-1], t[-1] * 1e9, t[0] * 1e9]
    for c, (label, b) in enumerate(show):
        axes[0, c].imshow(img(b), cmap=CMAP, vmin=-vmax, vmax=vmax, aspect="auto", extent=ext,
                          interpolation="bilinear")
        axes[0, c].set_title(label if label != "Yee dx/1" else "Yee dx (staircase)", fontsize=9,
                             color=INK)
        if c:
            e = errors(b, ref_b)
            axes[1, c].imshow(img(b - ref_b), cmap=CMAP, vmin=-vmax / 10, vmax=vmax / 10,
                              aspect="auto", extent=ext, interpolation="bilinear")
            axes[1, c].set_title(f"error x10 (rel. L2 {e['rel_l2']:.3f})", fontsize=9, color=INK)
        else:
            axes[1, c].axis("off")
        for r in range(2):
            axes[r, c].tick_params(labelsize=7, colors=INK2)
    axes[0, 0].set_ylabel("time (ns), gain t^1.5", fontsize=8, color=INK2)
    axes[1, 1].set_ylabel("time (ns)", fontsize=8, color=INK2)
    for c in range(4):
        axes[1, c].set_xlabel("antenna midpoint y (m)", fontsize=8, color=INK2)
    fig.suptitle(f"{case['label']} - common-offset B-scans (offset {BSCAN_OFFSET} m, antennas "
                 f"~0.1 m above ground)", fontsize=10, color=INK)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def markdown(summary):
    lines = []
    lam = C0 / (FREQ * EPS_SOIL ** 0.5)
    for name, table in summary.items():
        lines.append(f"### {CASES[name]['label']}\n")
        lines.append("| method | unknowns | steps | dt (ps) | gather L2 / median | "
                     "B-scan L2 / median | gather time (s) | B-scan time (s) | "
                     "peak GPU (MB, gather / B-scan) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for label, r in table.items():
            if label == "reference":
                continue
            g, b = r["gather_error"], r["bscan_error"]
            lines.append(
                f"| {label} | {r['unknowns']:,} | {r['steps']} | {r['dt'] * 1e12:.1f} | "
                f"{g['rel_l2'] * 100:.2f} % / {g['median_trace'] * 100:.2f} % | "
                f"{b['rel_l2'] * 100:.2f} % / {b['median_trace'] * 100:.2f} % | "
                f"{r['seconds']:.2f} | {r['bscan_seconds']:.1f} | "
                f"{r['peak_mb']:.0f} / {r['bscan_peak_mb']:.0f} |")
        ref = table["reference"]
        lines.append(f"\nReference: {ref['unknowns']:,} unknowns, {ref['steps']} steps, "
                     f"{ref['seconds']:.1f} s (gather), {ref['bscan_seconds']:.1f} s (B-scan).\n")
    lines.append(f"Centre wavelength in the ground {lam:.3f} m = {lam / DX:.1f} cells of 2 cm.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--outdir", default=str(ROOT / "examples" / "hybrid" / "results"))
    args = parser.parse_args()
    device = torch.device("cuda")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.cases:
        summary[name] = run_case(name, CASES[name], device, outdir)
        with open(outdir / "phase2_summary.json", "w") as fh:
            json.dump(summary, fh, default=float, indent=1)
        with open(outdir / "phase2_tables.md", "w") as fh:
            fh.write(markdown(summary))
        print(markdown({name: summary[name]}), flush=True)


if __name__ == "__main__":
    main()
