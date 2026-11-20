"""Ground-coupled antennas on steep terrain (2D, CPU): does exact antenna placement fix the
staircase error, and does the hybrid with antennas inside the DG band beat it?
Same profile, materials and antennas as showcase2d_staircase.py (antennas 2 cm above the
local surface).  Arms (all through HybridTM2D, TM, float32, CPU):

* ``yee``: pure Yee with cell-averaged materials and bilinear off-grid point sources /
  receivers at the exact antenna positions (no snapping), spacing h;
* ``hyb``: hybrid, Yee 6.67 mm outside a P1 DG band 0.10-0.32 m (Delaunay, 1 cm elements,
  4 mm within 3 cm of the surface), flux 0.1, antennas inside the band at their exact
  positions; the ground model on the 6.67 mm grid (a wet layer is then only sampled per
  element, not followed by the mesh).

Reference: ``yee`` at 1.25 mm.  Errors also against the snapped native reference.

    python examples/hybrid/showcase2d_hybrid.py [--threads 14]
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
import showcase2d_staircase as S2  # noqa: E402
import DeepGPR  # noqa: E402
from DeepGPR.hybrid import HybridPropagator, HybridTM2D  # noqa: E402
from DeepGPR.hybrid.topography import CallableSurface  # noqa: E402

OUT = S2.OUT
T_OUT = np.arange(1, int(round(S2.T_MAX / S2.DT_OUT)) + 1) * S2.DT_OUT


def positions(ants="coupled"):
    src, rec = [], []
    for y in S2.TX_Y:
        yr = y + S2.OFFSET
        if ants == "fixed":
            xs = xr = S2.ANT_X_FIXED
        else:
            xs, xr = S2.S(y) - S2.ANT_H_COUPLED, S2.S(yr) - S2.ANT_H_COUPLED
        src.append([[xs, y]])
        rec.append([[xr, yr]])
    return np.array(src), np.array(rec)


def ground_model(h, ground):
    """Cell-averaged ground materials with the ground continued over the surface (the
    hybrid imposes air from the surface itself)."""
    nx, ny = int(round(S2.LX / h)) + 1, int(round(S2.LY / h)) + 1
    xs, ys = np.arange(nx) * h, np.arange(ny) * h
    sub = 10
    off = (np.arange(sub) + 0.5) / sub - 0.5
    eps = np.zeros((nx, ny))
    sig = np.zeros((nx, ny))
    for ox in off:
        X, Y = np.meshgrid(xs + ox * h, ys, indexing="ij")
        for oy in off:
            s = S2.S(Y + oy * h)
            m = np.full(X.shape, 2)
            if ground == "wet":
                m = np.where(X < s + S2.WET_D, 1, m)          # layer continued upwards
            m = np.where(X >= S2.DEEP_X, 3, m)
            vals = np.array([S2.MAT[k] for k in ("air", "wet", "dry", "deep")])
            eps += vals[m, 0]
            sig += vals[m, 1]
    return eps / sub ** 2, sig / sub ** 2


def run(sol, eps, sig, ants="coupled", factor=1.0):
    src, rec = positions(ants)
    prop = HybridPropagator(sol, src, rec)
    nt = int(np.ceil(S2.T_MAX / sol.dt)) + 1
    w = torch.tensor(S2.ricker((np.arange(nt) + 0.5) * sol.dt) * factor, dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        d = prop(torch.tensor(eps, dtype=torch.float32), torch.tensor(sig, dtype=torch.float32), w[None])
    sec = time.time() - t0
    t = prop.time_axis(nt).double().numpy()
    d = d[:, :, 0].double().numpy()
    return np.stack([np.interp(T_OUT, t, tr) for tr in d]), sec


def yee(h, ground, factor):
    eps, sig = S2.materials(h, ground, "avg")
    sol = HybridTM2D(eps, sig, h, pmlthick=int(round(S2.PML_M / h)), nshot=len(S2.TX_Y),
                     device="cpu", dtype=torch.float32)
    return run(sol, eps, sig, factor=factor)


def hyb(ground, factor, h=0.02 / 3):
    eps, sig = ground_model(h, ground)
    t0 = time.time()
    sol = HybridTM2D(eps, sig, h, surface=CallableSurface(S2.S), dg_band=(0.10, 0.32),
                     pmlthick=int(round(S2.PML_M / h)), mesh="delaunay", dg_h=0.01,
                     mesh_options=dict(h_surface=0.004, refine_width=0.03), flux=0.1,
                     nshot=len(S2.TX_Y), device="cpu", dtype=torch.float32)
    setup = time.time() - t0
    d, sec = run(sol, eps, sig, factor=factor)
    info = dict(dt=sol.dt, setup_seconds=setup, dg_dofs=int(sol.summary().get("dg_dofs", -1)))
    return d, sec, info


def calibrate():
    """Amplitude factor native / HybridTM2D (pure Yee), homogeneous air, nodes, 2 mm."""
    h = 0.002
    nx, ny = int(round(S2.LX / h)) + 1, int(round(S2.LY / h)) + 1
    eps, sig = np.ones((nx, ny)), np.zeros((nx, ny))
    xs, ys, yr = 0.30, 0.40, 0.60
    dt = 0.95 * h / (299792458.0 * np.sqrt(2.0))
    nt = int(np.ceil(6e-9 / dt)) + 1
    w = torch.tensor(S2.ricker((np.arange(nt) + 0.5) * dt), dtype=torch.float32)
    out = DeepGPR.compute(device="cpu", dx=h, dt=dt, source_amplitudes=w.reshape(1, -1, 1),
                          source_location=torch.tensor([[[int(round(xs / h)), int(round(ys / h)), 0]]], dtype=torch.int32),
                          receiver_location=torch.tensor([[[int(round(xs / h)), int(round(yr / h)), 0]]], dtype=torch.int32),
                          eps_r=torch.tensor(eps, dtype=torch.float32)[..., None],
                          sigma=torch.tensor(sig, dtype=torch.float32)[..., None], fdtd_order=2,
                          pmlthick=int(round(S2.PML_M / h)), save_wavefield_history=False)
    t_n = (np.arange(nt) + 1) * dt
    dn = np.interp(T_OUT[:250], t_n, out[-1][0, :, 0].double().numpy())
    sol = HybridTM2D(eps, sig, h, pmlthick=int(round(S2.PML_M / h)), nshot=1, device="cpu",
                     dtype=torch.float32, dt=dt)
    prop = HybridPropagator(sol, np.array([[[xs, ys]]]), np.array([[[xs, yr]]]))
    with torch.no_grad():
        dh = prop(torch.tensor(eps, dtype=torch.float32), torch.tensor(sig, dtype=torch.float32), w[None])
    dh = np.interp(T_OUT[:250], prop.time_axis(nt).double().numpy(), dh[0, :, 0].double().numpy())
    f = float(np.dot(dn, dh) / np.dot(dh, dh))
    mis = float(np.linalg.norm(dn - f * dh) / np.linalg.norm(dn))
    print(f"calibration native / hybrid-Yee: factor {f:.6g}, shape mismatch {100 * mis:.3f} %", flush=True)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=14)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    path = OUT / "hybrid_results.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if "factor" not in res:
        res["factor"] = calibrate()
        path.write_text(json.dumps(res, indent=1))
    f = res["factor"]
    gain = np.clip((T_OUT - 1e-9) / 1e-9, 0, None) ** 2

    def err(d, ref):
        return dict(plain=float(np.linalg.norm(d - ref) / np.linalg.norm(ref)),
                    gained=float(np.linalg.norm((d - ref) * gain) / np.linalg.norm(ref * gain)))
    native_ref = {}
    for ground in ("dry", "wet"):
        native_ref[ground], _ = S2.simulate(S2.REF_MM * 1e-3, ground, "coupled", "avg", lambda h: 1.0)
    for ground in ("dry", "wet"):
        rows = res.setdefault(ground, {})
        ref, sec = yee(S2.REF_MM * 1e-3, ground, f)
        rows["yee_ref_seconds"] = sec
        rows["native_1.25mm_vs_yee_1.25mm"] = err(native_ref[ground], ref)
        print(f"[{ground}] yee {S2.REF_MM} mm reference {sec:.0f} s; native (snapped) 1.25 mm vs it: "
              f"{rows['native_1.25mm_vs_yee_1.25mm']}", flush=True)
        for hmm in (20.0 / 3, 4.0, 20.0 / 7, 2.0):
            d, sec = yee(hmm * 1e-3, ground, f)
            rows[f"yee_{hmm:.2f}mm"] = dict(err(d, ref), seconds=sec)
            print(f"  yee {hmm:5.2f} mm (off-grid sources): {rows[f'yee_{hmm:.2f}mm']}", flush=True)
            path.write_text(json.dumps(res, indent=1))
        d, sec, info = hyb(ground, f)
        rows["hyb_yee6.67mm"] = dict(err(d, ref), seconds=sec, **info)
        print(f"  hybrid (antennas in the band): {rows['hyb_yee6.67mm']}", flush=True)
        path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
