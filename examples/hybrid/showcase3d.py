"""Where does the hybrid DGTD-FDTD beat a conventional staircase FDTD?  Forward accuracy
against cost on a family of rugged-terrain models (design of the paper's showcase model).

Model family (3D, 2 cm model grid, 0.8 m x 0.8 m x 0.8 m, 8-cell absorbers): a rugged
surface (``moderate``: the pattern of fwi3d_lts.py / amr_terrain3d.py, slopes up to about
45 deg; ``steep``: shorter and higher undulations plus denser roughness, slopes up to about
60 deg), a homogeneous ground (``dry``: eps_r 5, 3 mS/m; ``wet``: eps_r 15, 10 mS/m), a sphere
(1.5 x the ground eps_r, radius 8 cm, 0.48 m deep) and a layer (1.6 x the ground eps_r)
below 0.62 m; 3 sources and 7 x 7 receivers 4 cm below the top (in the air), 400 MHz Ricker.

Solvers (antennas on Ez nodes shared by all grids: odd refinements of the 2 cm grid only):

* reference: native DeepGPR on 2.857 mm (refinement 7) with cell-averaged surface materials;
* conventional FDTD: native DeepGPR with cell-averaged surface materials on 2 cm, 6.67 mm
  and 4 mm (refinements 1, 3, 5);
* hybrid: :class:`HybridMaxwell3D`, P1 (or P2) band 0.08-0.30/0.34 m, flux 0.1, Yee
  refinement 1 or 3, global stepping (or LTS).

Metrics: relative L2 data error against the reference over all shots and receivers
(offsets below 12 cm muted), plain and with the (t - 1 ns)^2 gain of the FWI studies;
forward cost in milliseconds per shot per simulated nanosecond (RTX 4090, float32).

    python examples/hybrid/showcase3d.py run CASE [--hybrid r:order:lts ...]
    python examples/hybrid/showcase3d.py report

CASE: dry_moderate, dry_steep, wet_moderate, wet_steep.  Results in
``examples/hybrid/results/showcase3d/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import fwi3d_lts as F  # noqa: E402
from DeepGPR.hybrid import HybridMaxwell3D, Surface3D  # noqa: E402

OUT = HERE / "results" / "showcase3d"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
REF_R = 7
NATIVE_R = (1, 3, 5)
CASES = {
    "dry_moderate": dict(eps=5.0, sig=0.003, relief="moderate", t_max=12e-9),
    "dry_steep": dict(eps=5.0, sig=0.003, relief="steep", t_max=12e-9),
    "wet_moderate": dict(eps=15.0, sig=0.010, relief="moderate", t_max=15e-9),
    "wet_steep": dict(eps=15.0, sig=0.010, relief="steep", t_max=15e-9),
}


def surface(relief):
    """Elevation samples on the 0.78 m square, PCHIP between them; returns (surface, band)."""
    rng = np.random.default_rng(3)
    if relief == "moderate":
        ey = np.linspace(0.0, 0.78, 11)                      # 7.8 cm spacing
        Y, Z = np.meshgrid(ey, ey, indexing="ij")
        elev = (0.05 * np.sin(2 * np.pi * Y / 0.45) * np.cos(2 * np.pi * Z / 0.55)
                + 0.025 * rng.uniform(-1.0, 1.0, Y.shape))
        return Surface3D.from_grid(ey, ey, 0.19 - elev), (0.08, 0.30)
    ey = np.linspace(0.0, 0.78, 21)                          # 3.9 cm spacing
    Y, Z = np.meshgrid(ey, ey, indexing="ij")
    elev = (0.07 * np.sin(2 * np.pi * Y / 0.32) * np.cos(2 * np.pi * Z / 0.38)
            + 0.03 * rng.uniform(-1.0, 1.0, Y.shape))
    return Surface3D.from_grid(ey, ey, 0.20 - elev), (0.08, 0.34)


def configure(case):
    c = CASES[case]
    F.SHAPE = (40, 40, 40)
    F.T_MAX = c["t_max"]
    F.EPS_BG, F.SIG_BG = c["eps"], c["sig"]
    F.LAYER_X, F.EPS_LAYER = 0.62, 1.6 * c["eps"]
    F.SPHERES = [(0.48, 0.40, 0.39, 0.08, 1.5 * c["eps"], None)]
    F.SURFACE, F.BAND = surface(c["relief"])
    F.SRC_YZ = [(0.22, 0.39), (0.40, 0.39), (0.58, 0.39)]
    F.REC_YZ = [(round(y, 2), round(z, 2)) for y in np.arange(0.10, 0.71, 0.10)
                for z in np.arange(0.09, 0.70, 0.10)]
    F.ANT_X = 0.04
    yy, zz = np.meshgrid(np.arange(F.SHAPE[1]) * F.DX, np.arange(F.SHAPE[2]) * F.DX, indexing="ij")
    s = F.SURFACE(yy, zz)
    assert F.BAND[0] < s.min() and s.max() < F.BAND[1], (s.min(), s.max(), F.BAND)
    slope = np.hypot(*np.gradient(F.SURFACE(*np.meshgrid(np.arange(0, 0.78, 0.002),
                                                           np.arange(0, 0.78, 0.002),
                                                           indexing="ij")), 0.002))
    return dict(surface_min=float(s.min()), surface_max=float(s.max()),
                max_slope_deg=float(np.degrees(np.arctan(slope.max()))),
                p95_slope_deg=float(np.degrees(np.arctan(np.percentile(slope, 95)))))


def native(case, r):
    """Native DeepGPR with cell-averaged surface on 2 cm / r (cached)."""
    F.OUT = OUT / case / f"native_r{r}"
    F.OBS_REFINE = r
    d = F.observed_data(DEV)
    z = np.load(F.OUT / "observed.npz")
    return d.double(), float(z["seconds"])


def hybrid(case, r, order, lts):
    e0, s0 = F.true_ground()
    kw = dict(lts="auto") if lts else {}
    t0 = time.perf_counter()
    sol = HybridMaxwell3D(e0, s0, F.DX, surface=F.SURFACE, dg_band=F.BAND, pmlthick=F.PML,
                          flux=0.1, fdtd_refinement=r, dg_order=order, nshot=1, device=DEV,
                          dtype=torch.float32, **kw)
    fw = F.Forward(sol, DEV)
    setup = time.perf_counter() - t0
    e, s = (torch.tensor(a, dtype=torch.float32, device=DEV) for a in (e0, s0))
    with torch.no_grad():
        fw(e, s, 0)                                           # warm-up (graph capture)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        d = torch.cat([fw(e, s, i) for i in range(len(F.SRC_YZ))])
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    info = sol.summary()
    meta = dict(p=sol.lts, dt=sol.dt, setup_seconds=setup, yee_dofs=info["fdtd_dofs"],
                dg_dofs=info["dg_dofs"])
    del fw, sol
    torch.cuda.empty_cache()
    return d.double().cpu(), sec, meta


def errors(d, ref):
    mute = F.mute_mask()[:, None, :].double()
    g = F.gain().double()[None, :, None]
    out = {}
    for name, w in (("plain", mute), ("gained", mute * g)):
        out[name] = float(((d - ref) * w).norm() / (ref * w).norm())
    return out


def run(case, hybrids):
    geo = configure(case)
    (OUT / case).mkdir(parents=True, exist_ok=True)
    path = OUT / case / "results.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res["geometry"] = geo
    ns = F.T_MAX * 1e9
    nshot = len(F.SRC_YZ)
    ref, sec = native(case, REF_R)
    res[f"native_r{REF_R}"] = dict(ms_per_shot_ns=1e3 * sec / nshot / ns, h_mm=20.0 / REF_R)
    print(f"[{case}] {geo}; reference r={REF_R}: {1e3 * sec / nshot / ns:.0f} ms/shot/ns", flush=True)
    for r in NATIVE_R:
        d, sec = native(case, r)
        res[f"native_r{r}"] = dict(errors(d, ref), ms_per_shot_ns=1e3 * sec / nshot / ns,
                                   h_mm=20.0 / r)
        print(f"  native {20 / r:5.2f} mm: {res[f'native_r{r}']}", flush=True)
        path.write_text(json.dumps(res, indent=1))
    for spec in hybrids:
        r, order, lts = spec.split(":")
        r, order, lts = int(r), int(order), lts == "lts"
        key = f"hybrid_r{r}_P{order}_{'lts' if lts else 'global'}"
        d, sec, meta = hybrid(case, r, order, lts)
        res[key] = dict(errors(d, ref), ms_per_shot_ns=1e3 * sec / nshot / ns, h_mm=20.0 / r,
                        **meta)
        print(f"  {key}: {res[key]}", flush=True)
        path.write_text(json.dumps(res, indent=1))
    path.write_text(json.dumps(res, indent=1))


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = [c for c in CASES if (OUT / c / "results.json").exists()]
    fig, axes = plt.subplots(1, len(cases), figsize=(4.6 * len(cases), 4.2), squeeze=False)
    lines = ["# Hybrid vs conventional FDTD: forward accuracy and cost", "",
             "Data error against native DeepGPR on 2.857 mm (cell-averaged surface), offsets "
             "below 12 cm muted; plain / with the (t - 1 ns)^2 gain. Cost: ms per shot per "
             "simulated ns (RTX 4090, float32).", ""]
    for ax, case in zip(axes[0], cases):
        res = json.loads((OUT / case / "results.json").read_text())
        g = res["geometry"]
        lines += [f"## {case} (surface {g['surface_min']:.2f}-{g['surface_max']:.2f} m, "
                  f"max slope {g['max_slope_deg']:.0f} deg, 95th pct {g['p95_slope_deg']:.0f} deg)", "",
                  "| solver | error plain | error gained | ms/shot/ns |", "|---|---|---|---|"]
        for kind, color, mark in (("native", "#c2410c", "s"), ("hybrid", "#1d4ed8", "o")):
            pts = sorted((v["ms_per_shot_ns"], 100 * v["gained"], k) for k, v in res.items()
                         if k.startswith(kind) and "gained" in v)
            if pts:
                ax.loglog([p[0] for p in pts], [p[1] for p in pts], mark + "-", color=color,
                          label="conventional FDTD" if kind == "native" else "hybrid")
                for x, y, k in pts:
                    ax.annotate(k.replace("native_", "").replace("hybrid_", ""), (x, y), fontsize=7)
        for k, v in res.items():
            if "gained" in v:
                lines.append(f"| {k} | {100 * v['plain']:.2f} % | {100 * v['gained']:.2f} % | "
                             f"{v['ms_per_shot_ns']:.0f} |")
        lines.append("")
        ax.set_title(case)
        ax.set_xlabel("ms per shot per simulated ns")
        ax.set_ylabel("data error with gain [%]")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pareto.png", dpi=120)
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "report"))
    ap.add_argument("case", nargs="?", choices=tuple(CASES))
    ap.add_argument("--hybrid", nargs="*", default=["1:1:global", "3:1:global"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.what == "run":
        run(a.case, a.hybrid)
    else:
        report()


if __name__ == "__main__":
    main()
