"""Shared Phase 4 (3D) models and runners for the hybrid examples.

Models (physical frame, axis 0 = depth ``x``, axes 1/2 = horizontal ``y``/``z``),
2 cm model grid, 0.8 m deep and 1.2 m x 1.2 m wide, 10-cell CPML:

* ``A3_flat``      - horizontal ground at 0.30 m;
* ``B3_sine``      - ``d = 0.30 + 0.06 sin(2 pi y / 0.6) cos(2 pi z / 0.8)``;
* ``C3_irregular`` - surface from gridded elevation samples (PCHIP), with a
  high-permittivity and a conductive sphere (defined on the 2 cm grid).

Ground ``eps_r = 6``, ``sigma = 3 mS/m``; 400 MHz Ricker; horizontal ``Ez``
dipole 0.16 m deep (at least 8 cm above the ground), receivers on a line along
``y`` at the same depth and on a buried line at 0.62 m. All antennas are on
coarse ``Ez`` nodes, which are also nodes of the odd refinements used here.

Time conventions as in 2D: source sample ``n`` is ``f((n + 1/2) dt)``,
receiver sample ``n`` is the field at ``(n + 1) dt``. Sources follow DeepGPR's
``J = w dl delta`` with ``dl`` = the 2 cm model spacing for every solver.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import DeepGPR  # noqa: E402
from DeepGPR.hybrid import resample  # noqa: E402
from DeepGPR.hybrid.solver3d import HybridMaxwell3D  # noqa: E402
from DeepGPR.hybrid.topography import Surface3D  # noqa: E402

DX = 0.02
SHAPE = (40, 60, 60)
PML = 10
EPS_SOIL, SIG_SOIL = 6.0, 0.003
FREQ = 4.0e8
T0 = 1.2 / FREQ
T_MAX = 16e-9
BAND = (0.20, 0.44)
ANT_X = 0.16
SRC_YZ = (0.60, 0.61)                 # z on an Ez node: (k + 1/2) DX
REC_Y = np.round(np.arange(0.10, 1.101, 0.04), 6)
BURIED_X = 0.62
OFFSET_MIN = 0.3

_ELEV_Y = np.linspace(0.0, 1.2, 7)
_ELEV_Z = np.linspace(0.0, 1.2, 7)
_ELEV = np.array([[0.00, 0.03, 0.05, 0.02, -0.02, -0.04, -0.01],
                  [0.02, 0.06, 0.04, 0.00, -0.05, -0.03, 0.01],
                  [0.04, 0.05, 0.01, -0.03, -0.04, 0.00, 0.03],
                  [0.01, 0.02, -0.02, -0.05, -0.01, 0.04, 0.05],
                  [-0.02, -0.03, -0.04, 0.00, 0.03, 0.06, 0.03],
                  [-0.04, -0.05, -0.01, 0.03, 0.05, 0.02, 0.00],
                  [-0.01, -0.02, 0.02, 0.04, 0.01, -0.02, -0.03]])
SPHERES = [  # (x, y, z, radius, eps_r, sigma), off the receiver plane z = 0.61
    (0.58, 0.42, 0.47, 0.10, 14.0, SIG_SOIL),
    (0.58, 0.80, 0.75, 0.10, 6.0, 0.05),
]

CASES = {
    "A3_flat": dict(surface=Surface3D.flat(0.30), anomalies=False,
                    label="A3: flat ground"),
    "B3_sine": dict(surface=Surface3D.sine(0.30, 0.06, 0.6, 0.8), anomalies=False,
                    label="B3: sine topography (+-6 cm)"),
    "C3_irregular": dict(surface=Surface3D.from_grid(_ELEV_Y, _ELEV_Z, 0.30 - _ELEV),
                         anomalies=True, label="C3: irregular topography + eps/sigma spheres"),
}


def wavelet(nt, dt):
    t = (np.arange(nt) + 0.5) * dt - T0
    a = (np.pi * FREQ * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def ground_model(case):
    """Ground materials on the 2 cm model grid (anomalies included)."""
    eps = np.full(SHAPE, EPS_SOIL)
    sig = np.full(SHAPE, SIG_SOIL)
    if CASES[case]["anomalies"]:
        x, y, z = np.meshgrid(*(np.arange(n) * DX for n in SHAPE), indexing="ij")
        for (cx, cy, cz, r, e, s) in SPHERES:
            m = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r * r
            eps[m], sig[m] = e, s
    return eps, sig


def staircase_model(case, h=DX):
    """Air above the surface node-wise (what a DeepGPR user gives the native solver)."""
    eps_g, sig_g = ground_model(case)
    r = int(round(DX / h))
    n = tuple(s * r for s in SHAPE)
    idx = [np.clip(np.floor(np.arange(m) / r + 0.5).astype(int), 0, s - 1) for m, s in zip(n, SHAPE)]
    eps = eps_g[np.ix_(*idx)]
    sig = sig_g[np.ix_(*idx)]
    Y, Z = np.meshgrid(np.arange(n[1]) * h, np.arange(n[2]) * h, indexing="ij")
    d = CASES[case]["surface"](Y, Z)
    air = (np.arange(n[0]) * h)[:, None, None] < d[None]
    eps[air], sig[air] = 1.0, 0.0
    return eps, sig


def averaged_model(case, h, sub=3):
    """Reference materials on a grid of spacing ``h``: cell volume fraction of
    ground (exact along x, ``sub x sub`` samples in y/z), arithmetic averages."""
    eps_g, sig_g = ground_model(case)
    r = int(round(DX / h))
    n = tuple(s * r for s in SHAPE)
    idx = [np.clip(np.floor(np.arange(m) / r + 0.5).astype(int), 0, s - 1) for m, s in zip(n, SHAPE)]
    eps_g, sig_g = eps_g[np.ix_(*idx)], sig_g[np.ix_(*idx)]
    off = (np.arange(sub) + 0.5) / sub - 0.5
    frac = np.zeros(n)
    x = np.arange(n[0]) * h
    for oy in off:
        for oz in off:
            Y, Z = np.meshgrid(np.arange(n[1]) * h + oy * h, np.arange(n[2]) * h + oz * h, indexing="ij")
            d = CASES[case]["surface"](Y, Z)
            frac += np.clip((x[:, None, None] + 0.5 * h - d[None]) / h, 0.0, 1.0)
    frac /= sub * sub
    return 1.0 + frac * (eps_g - 1.0), frac * sig_g


def antennas():
    """Source ``(1, 1, 3)`` and receivers ``(1, nrec, 3)`` physical positions [m]."""
    src = np.array([[[ANT_X, SRC_YZ[0], SRC_YZ[1]]]])
    line = [[ANT_X, y, SRC_YZ[1]] for y in REC_Y]
    buried = [[BURIED_X, y, SRC_YZ[1]] for y in REC_Y]
    return src, np.array([line + buried])


def metric_mask():
    """Receivers used in the error metric (offset >= 0.3 m on the surface line)."""
    off = np.abs(REC_Y - SRC_YZ[0])
    return np.concatenate([off >= OFFSET_MIN, np.ones(REC_Y.size, dtype=bool)])


def time_axis(dt_out=2e-11):
    return np.arange(int(T_MAX / dt_out)) * dt_out + dt_out


def run_native(eps, sig, h, *, device="cuda"):
    """Native DeepGPR on spacing ``h`` (moment ``w DX``: amplitude scaled by DX/h)."""
    n = eps.shape
    pml = int(round(PML * DX / h))
    dt = 0.95 * DeepGPR.max_stable_time_step(h, n[0] + 2 * pml, n[1] + 2 * pml, n[2] + 2 * pml)
    nt = int(np.ceil(T_MAX / dt)) + 1
    w = torch.tensor(wavelet(nt, dt) * (DX / h), dtype=torch.float32)
    src, rec = antennas()
    to_idx = lambda p: torch.tensor(np.rint((p - np.array([0, 0, 0.5 * h])) / h).astype(np.int32))  # noqa: E731
    t0 = time.time()
    out = DeepGPR.compute(device=device, dx=h, dt=dt, source_amplitudes=w.reshape(1, -1, 1),
                          source_location=to_idx(src), receiver_location=to_idx(rec),
                          eps_r=torch.tensor(eps, dtype=torch.float32),
                          sigma=torch.tensor(sig, dtype=torch.float32), pmlthick=pml,
                          save_wavefield_history=False)
    if device == "cuda":
        torch.cuda.synchronize()
    sec = time.time() - t0
    data = out[-1][0].double().cpu().numpy()                       # (nt, nrec)
    t = (np.arange(nt) + 1) * dt
    return resample_np(data, t), {"seconds": sec, "dt": dt, "nt": nt, "cells": int(np.prod(n)),
                                  "pml": pml}


def resample_np(data, t_from):
    out = resample(torch.as_tensor(data[None]), torch.as_tensor(t_from), torch.as_tensor(time_axis()))
    return out[0].numpy()


def build_hybrid(case, *, order=1, refinement=1, device="cuda", dtype=torch.float32, **kw):
    eps, sig = ground_model(case)
    return HybridMaxwell3D(eps, sig, DX, surface=CASES[case]["surface"], dg_band=BAND, pmlthick=PML,
                           dg_order=order, fdtd_refinement=refinement, device=device, dtype=dtype,
                           **kw)


def run_hybrid(sol, *, callback=None, callback_every=0):
    nt = int(np.ceil(T_MAX / sol.dt)) + 1
    src, rec = antennas()
    torch.cuda.reset_peak_memory_stats()
    res = sol.run(torch.tensor(wavelet(nt, sol.dt)), src, rec, callback=callback,
                  callback_every=callback_every)
    data = res["receivers"][0].double().cpu().numpy()
    t = (np.arange(nt) + 1) * sol.dt
    info = {"seconds": res["seconds"], "dt": sol.dt, "nt": nt,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
    info.update({k: v for k, v in sol.summary().items() if isinstance(v, (int, float))})
    return resample_np(data, t), info


def rel_error(data, ref, mask=None):
    m = metric_mask() if mask is None else mask
    return float(np.linalg.norm(data[:, m] - ref[:, m]) / np.linalg.norm(ref[:, m]))


def native_snapshots(eps, sig, h, frame_times, z_plane, *, device="cuda"):
    """Ez on the plane ``z = z_plane`` (an Ez node plane) at ``frame_times`` from a
    native DeepGPR run advanced in chunks (states passed back unchanged).
    Returns ``(frames (nf, nx, ny), x, y)`` on the physical grid of spacing ``h``."""
    n = eps.shape
    pml = int(round(PML * DX / h))
    dt = 0.95 * DeepGPR.max_stable_time_step(h, n[0] + 2 * pml, n[1] + 2 * pml, n[2] + 2 * pml)
    steps = [int(round(t / dt)) - 1 for t in frame_times]
    w = torch.tensor(wavelet(max(steps) + 1, dt) * (DX / h), dtype=torch.float32)
    src, rec = antennas()
    to_idx = lambda p: torch.tensor(np.rint((p - np.array([0, 0, 0.5 * h])) / h).astype(np.int32))  # noqa: E731
    kz = int(round(z_plane / h - 0.5)) + pml
    E = H = P = None
    done, frames = 0, []
    eps_t = torch.tensor(eps, dtype=torch.float32)
    sig_t = torch.tensor(sig, dtype=torch.float32)
    for s in steps:
        chunk = s + 1 - done
        if chunk > 0:
            out = DeepGPR.compute(device=device, dx=h, dt=dt,
                                  source_amplitudes=w[done:s + 1].reshape(1, -1, 1),
                                  source_location=to_idx(src), receiver_location=to_idx(rec[:, :1]),
                                  eps_r=eps_t, sigma=sig_t, pmlthick=pml, E=E, H=H, PML=P,
                                  save_wavefield_history=False)
            E, H, P = out[1], out[2], out[3]
            done = s + 1
        frames.append(E[2][0, pml:pml + n[0], pml:pml + n[1], kz].float().cpu().numpy())
    return np.stack(frames), np.arange(n[0]) * h, np.arange(n[1]) * h
