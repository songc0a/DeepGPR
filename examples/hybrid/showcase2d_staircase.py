"""How fine must a conventional (staircase) FDTD grid be on rugged terrain?  A 2D test of
the premise of the showcase model: the hybrid can only win where a staircase FDTD needs
cells much finer than about lambda/15 (2/3 cm at 400 MHz).  Native DeepGPR (TM, Ez) only,
on the CPU.

Profile 1.2 m long, 0.6 m deep (air above a steep rugged surface 0.12-0.30 m deep: sine
plus PCHIP roughness, local slopes up to about 60 deg); ground ``dry`` (eps_r 5, 3 mS/m) or
``wet`` (a 2.5 cm wet layer, eps_r 18, 20 mS/m, along the surface over the dry ground); a
reflector (eps_r 9) below 0.45 m.  8 common-offset positions (offset 12 cm), antennas
``fixed`` (a horizontal line 0.10 m deep, on nodes of every grid: 2-20 cm above the ground)
or ``coupled`` (2 cm above the local surface, snapped to the nearest node of each grid).
400 MHz Ricker, 12 ns.  Surface materials ``avg`` (area fractions of air / wet / dry from
10 x 10 sub-samples per cell) or ``bin`` (cell centre).  Reference: 1.25 mm, ``avg``.

    python examples/hybrid/showcase2d_staircase.py [--threads 14]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import DeepGPR  # noqa: E402

OUT = ROOT / "examples" / "hybrid" / "results" / "showcase2d"
LY, LX = 1.2, 0.6
FREQ, T0, T_MAX = 4e8, 3e-9, 12e-9
PML_M = 0.16
DT_OUT = 2e-11
MAT = {"air": (1.0, 0.0), "wet": (18.0, 0.020), "dry": (5.0, 0.003), "deep": (9.0, 0.003)}
WET_D, DEEP_X = 0.025, 0.45
TX_Y = 0.30 + 0.08 * np.arange(8)
OFFSET = 0.12
ANT_X_FIXED, ANT_H_COUPLED = 0.10, 0.02
GRIDS_MM = (20.0, 10.0, 20.0 / 3, 5.0, 4.0, 20.0 / 7, 2.0)
REF_MM, CHECK_MM = 1.25, 1.0


def surface():
    rng = np.random.default_rng(7)
    yk = np.arange(0.0, LY + 1e-9, 0.04)
    rough = PchipInterpolator(yk, 0.02 * rng.uniform(-1.0, 1.0, yk.size))
    return lambda y: np.clip(0.21 + 0.05 * np.sin(2 * np.pi * y / 0.40 + 0.3) + rough(y), 0.12, 0.30)


S = surface()


def medium(x, y, ground):
    """Material index at points: 0 air, 1 wet, 2 dry, 3 deep."""
    s = S(y)
    m = np.where(x < s, 0, 2)
    if ground == "wet":
        m = np.where((x >= s) & (x < s + WET_D), 1, m)
    return np.where(x >= DEEP_X, 3, m)


def materials(h, ground, mode):
    nx, ny = int(round(LX / h)) + 1, int(round(LY / h)) + 1
    xs, ys = np.arange(nx) * h, np.arange(ny) * h
    vals = np.array([MAT[k] for k in ("air", "wet", "dry", "deep")])
    if mode == "bin":
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        idx = medium(X, Y, ground)
        return vals[idx, 0], vals[idx, 1]
    sub = 10
    off = (np.arange(sub) + 0.5) / sub - 0.5
    eps = np.zeros((nx, ny))
    sig = np.zeros((nx, ny))
    for ox in off:
        X, Y = np.meshgrid(xs + ox * h, ys, indexing="ij")
        for oy in off:
            idx = medium(X, Y + oy * h, ground)
            eps += vals[idx, 0]
            sig += vals[idx, 1]
    return eps / sub ** 2, sig / sub ** 2


def antennas(h, ants):
    """(src, rec) node indices (nshot, 1, 3) for grid spacing h."""
    def node(x, y):
        return [int(round(x / h)), int(round(y / h)), 0]
    src, rec = [], []
    for y in TX_Y:
        yr = y + OFFSET
        if ants == "fixed":
            xs, xr = ANT_X_FIXED, ANT_X_FIXED
        else:
            xs, xr = S(y) - ANT_H_COUPLED, S(yr) - ANT_H_COUPLED
        src.append([node(xs, y)])
        rec.append([node(xr, yr)])
    return torch.tensor(src, dtype=torch.int32), torch.tensor(rec, dtype=torch.int32)


def ricker(t):
    a = (np.pi * FREQ * (t - T0)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def simulate(h, ground, ants, mode, amp_scale):
    eps, sig = materials(h, ground, mode)
    dt = 0.95 * h / (299792458.0 * np.sqrt(2.0))
    nt = int(np.ceil(T_MAX / dt)) + 1
    w = torch.tensor(ricker((np.arange(nt) + 0.5) * dt) * amp_scale(h), dtype=torch.float32)
    src, rec = antennas(h, ants)
    pml = int(round(PML_M / h))
    t0 = time.time()
    out = DeepGPR.compute(device="cpu", dx=h, dt=dt, source_amplitudes=w.reshape(1, -1, 1),
                          source_location=src, receiver_location=rec,
                          eps_r=torch.tensor(eps, dtype=torch.float32)[..., None],
                          sigma=torch.tensor(sig, dtype=torch.float32)[..., None],
                          fdtd_order=2, pmlthick=pml, save_wavefield_history=False)
    sec = time.time() - t0
    d = out[-1][:, :, 0].double().numpy()                  # (nshot, nt), sample n at (n + 1) dt
    t_sim = (np.arange(nt) + 1) * dt
    t_out = np.arange(1, int(round(T_MAX / DT_OUT)) + 1) * DT_OUT
    return np.stack([np.interp(t_out, t_sim, tr) for tr in d]), sec


def amplitude_scaling(threads):
    """Exponent k of the source scaling h^-k that makes a homogeneous-air run grid-independent."""
    res = {}
    for k in (0, 1, 2):
        traces = []
        for h in (4e-3, 2e-3):
            eps = np.ones((int(round(LX / h)) + 1, int(round(LY / h)) + 1))
            dt = 0.95 * h / (299792458.0 * np.sqrt(2.0))
            nt = int(np.ceil(6e-9 / dt)) + 1
            w = torch.tensor(ricker((np.arange(nt) + 0.5) * dt) * h ** -k, dtype=torch.float32)
            s = torch.tensor([[[int(round(0.3 / h)), int(round(0.4 / h)), 0]]], dtype=torch.int32)
            r = torch.tensor([[[int(round(0.3 / h)), int(round(0.6 / h)), 0]]], dtype=torch.int32)
            out = DeepGPR.compute(device="cpu", dx=h, dt=dt, source_amplitudes=w.reshape(1, -1, 1),
                                  source_location=s, receiver_location=r,
                                  eps_r=torch.tensor(eps, dtype=torch.float32)[..., None],
                                  sigma=torch.zeros(eps.shape + (1,)), fdtd_order=2,
                                  pmlthick=int(round(PML_M / h)), save_wavefield_history=False)
            d = out[-1][0, :, 0].double().numpy()
            t_out = np.arange(1, 300) * DT_OUT
            traces.append(np.interp(t_out, (np.arange(nt) + 1) * dt, d))
        res[k] = float(np.abs(traces[0]).max() / np.abs(traces[1]).max())
    k = min(res, key=lambda q: abs(np.log(res[q])))
    print(f"source scaling: peak ratio 4 mm / 2 mm for h^-k: {res} -> k = {k}", flush=True)
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=14)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    k = res.get("source_k")
    if k is None:
        k = amplitude_scaling(a.threads)
        res["source_k"] = k
    scale = lambda h: h ** -k  # noqa: E731
    ys = np.arange(0, LY, 0.001)
    slope = np.degrees(np.arctan(np.abs(np.gradient(S(ys), 0.001))))
    res["surface"] = dict(min=float(S(ys).min()), max=float(S(ys).max()),
                          max_slope_deg=float(slope.max()), p90_slope_deg=float(np.percentile(slope, 90)))
    print(res["surface"], flush=True)
    t_out = np.arange(1, int(round(T_MAX / DT_OUT)) + 1) * DT_OUT
    gain = np.clip((t_out - 1e-9) / 1e-9, 0, None) ** 2
    for ground in ("dry", "wet"):
        for ants in ("fixed", "coupled"):
            key = f"{ground}_{ants}"
            ref, sec = simulate(REF_MM * 1e-3, ground, ants, "avg", scale)
            print(f"[{key}] reference {REF_MM} mm: {sec:.0f} s", flush=True)
            rows = res.setdefault(key, {})
            rows["ref_seconds"] = sec

            def err(d):
                return dict(plain=float(np.linalg.norm(d - ref) / np.linalg.norm(ref)),
                            gained=float(np.linalg.norm((d - ref) * gain) / np.linalg.norm(ref * gain)))
            if key == "wet_coupled":
                d, sec = simulate(CHECK_MM * 1e-3, ground, ants, "avg", scale)
                rows["check_1mm"] = dict(err(d), seconds=sec)
                print(f"  1 mm vs reference: {rows['check_1mm']}", flush=True)
            for hmm in GRIDS_MM:
                for mode in ("avg", "bin"):
                    d, sec = simulate(hmm * 1e-3, ground, ants, mode, scale)
                    rows[f"{mode}_{hmm:.2f}mm"] = dict(err(d), seconds=sec, h_mm=hmm)
                    print(f"  {mode} {hmm:5.2f} mm: plain {100 * rows[f'{mode}_{hmm:.2f}mm']['plain']:6.2f} %  "
                          f"gained {100 * rows[f'{mode}_{hmm:.2f}mm']['gained']:6.2f} %  ({sec:.1f} s)", flush=True)
            path.write_text(json.dumps(res, indent=1))
    path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
