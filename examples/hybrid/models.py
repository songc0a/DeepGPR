"""Shared Phase 2 models and runners for the hybrid examples.

Models (physical frame, axis 0 = depth ``x``, axis 1 = horizontal ``y``):

* ``A_flat``      - horizontal ground at 1.2 m;
* ``B_sine``      - sine surface (amplitude 0.25 m, wavelength 2.4 m);
* ``C_irregular`` - surface interpolated from elevation samples, with a
  high-permittivity and a conductive anomaly (defined on the 2 cm grid, as a
  DeepGPR user supplies a subsurface model);
* ``slope``       - a 16.7 degree ramp (extra case).

Ground ``eps_r = 6``, ``sigma = 3 mS/m``; 300 MHz Ricker; antennas about
0.1 m above the local ground, placed on 2 cm nodes (nodes of every finer grid
used here), so every solver uses identical antenna positions.

Time conventions shared by all runners: the source sample of step ``n`` is
``f((n + 1/2) dt)`` (DeepGPR applies it between ``E^n`` and ``E^(n+1)``) and
receiver sample ``n`` is the field at ``(n + 1) dt``.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import DeepGPR  # noqa: E402
from DeepGPR.hybrid import HybridTM2D, Surface  # noqa: E402

DX = 0.02            # grid spacing of the compared solvers [m]
DEPTH, WIDTH = 3.6, 6.0
PML_M = 0.4          # CPML thickness [m]
EPS_SOIL, SIG_SOIL = 6.0, 0.003
FREQ = 3.0e8
T0 = 1.2 / FREQ
T_MAX = 40e-9
FRAME_DT = 0.5e-9
ANTENNA_HEIGHT = 0.10

ELEV_Y = np.arange(0.0, 6.01, 0.5)
ELEV_Z = np.array([0.0, 0.06, 0.20, 0.30, 0.22, 0.05, -0.10, -0.18, -0.08, 0.10, 0.24, 0.12, 0.0])
ANOMALIES = [  # (x, y, radius, eps_r, sigma)
    (2.40, 2.20, 0.25, 15.0, SIG_SOIL),
    (2.30, 4.20, 0.25, 8.0, 0.05),
]

# ``band``: DGTD band (physical depth) containing the surface. With the Yee part
# refined by 3 (``HYBRID_REFINEMENT``) the interfaces reflect about 9x less and
# a band reaching 0.3-0.6 m below the deepest surface point suffices; model C
# keeps its anomalies in the (refined) Yee region. ``extra_bands`` are
# additional hybrid variants run by the benchmark.
CASES = {
    "A_flat": dict(surface=Surface.flat(1.2), source_y=3.0, anomalies=False,
                   label="Model A: flat ground", zoom=(0.95, 1.50, 2.50, 3.50),
                   band=(0.70, 1.80)),
    "B_sine": dict(surface=Surface.sine(1.2, 0.25, 2.4), source_y=3.0, anomalies=False,
                   label="Model B: sine topography (amplitude 0.25 m, wavelength 2.4 m)",
                   zoom=(1.05, 1.60, 2.45, 3.45), band=(0.70, 1.80)),
    "C_irregular": dict(surface=Surface.from_elevation(ELEV_Y, ELEV_Z, datum=1.2), source_y=2.5,
                        anomalies=True,
                        label="Model C: irregular topography + eps and sigma anomalies",
                        zoom=(0.80, 1.35, 2.00, 3.00), band=(0.70, 1.95),
                        extra_bands=((0.70, 2.90),)),
    "slope": dict(surface=Surface.slope(1.5, -0.3, y0=2.0, y_start=2.0, y_end=4.0), source_y=3.0,
                  anomalies=False, label="slope topography (16.7 deg ramp)",
                  zoom=(0.95, 1.50, 2.50, 3.50), band=(0.70, 1.80)),
}


def wavelet(nt, dt):
    """Ricker sampled at the half steps where DeepGPR applies it."""
    t = (np.arange(nt) + 0.5) * dt - T0
    a = (np.pi * FREQ * t) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a))


def aligned_dt(dt_max):
    """Largest ``FRAME_DT / m`` not exceeding ``dt_max``."""
    m = math.ceil(FRAME_DT / dt_max * (1 + 1e-9))
    return FRAME_DT / m, m


def nt_of(dt):
    return int(round(T_MAX / dt))


def ground_model(dx, anomalies):
    """Ground on a ``dx`` grid: background plus circular anomalies sampled at nodes."""
    nx, ny = int(round(DEPTH / dx)), int(round(WIDTH / dx))
    eps = torch.full((nx, ny), EPS_SOIL, dtype=torch.float64)
    sig = torch.full((nx, ny), SIG_SOIL, dtype=torch.float64)
    if anomalies:
        x = torch.arange(nx, dtype=torch.float64)[:, None] * dx
        y = torch.arange(ny, dtype=torch.float64)[None, :] * dx
        for xc, yc, r, e, s in ANOMALIES:
            inside = (x - xc) ** 2 + (y - yc) ** 2 <= r * r
            eps[inside], sig[inside] = e, s
    return eps, sig


def ground_on(dx, case):
    """The subsurface model the user supplies (a 2 cm grid, ``ground_model(DX)``),
    resampled onto a finer ``dx`` grid by nearest 2 cm pixel."""
    eps_c, sig_c = ground_model(DX, case["anomalies"])
    nx, ny = int(round(DEPTH / dx)), int(round(WIDTH / dx))
    i = np.clip(np.floor(np.arange(nx) * dx / DX + 0.5 + 1e-9).astype(int), 0, eps_c.shape[0] - 1)
    j = np.clip(np.floor(np.arange(ny) * dx / DX + 0.5 + 1e-9).astype(int), 0, eps_c.shape[1] - 1)
    return eps_c[i][:, j].clone(), sig_c[i][:, j].clone()


def ground_fraction(dx, case, averaged, supersample=8):
    """Fraction of every Yee material cell (centred on the Ez node) below the surface.

    ``averaged=False`` is DeepGPR's staircase (node below the surface -> ground);
    ``averaged=True`` integrates exactly along ``x`` and with ``supersample``
    points along ``y``. For Ez (tangential to the surface in 2D TM) the arithmetic
    average of eps and sigma over the cell is the second-order effective medium.
    """
    nx, ny = int(round(DEPTH / dx)), int(round(WIDTH / dx))
    x = np.arange(nx)[:, None] * dx
    if not averaged:
        return (x >= case["surface"](np.arange(ny) * dx)[None]).astype(float)
    frac = np.zeros((nx, ny))
    for k in range(supersample):
        y = (np.arange(ny) + (k + 0.5) / supersample - 0.5) * dx
        frac += np.clip((x + 0.5 * dx - case["surface"](y)[None]) / dx, 0.0, 1.0)
    return frac / supersample


def yee_model(dx, case, averaged=False):
    """Yee model of a case on a ``dx`` grid: ``(eps, sig, air_mask)``."""
    eps_g, sig_g = ground_on(dx, case)
    f = torch.as_tensor(ground_fraction(dx, case, averaged))
    eps = f * eps_g + (1 - f) * 1.0
    sig = f * sig_g
    return eps, sig, (f < 0.5).numpy()


def staircase_model(dx, case):
    """DeepGPR staircase model of the topography (every node above the surface is air)."""
    return yee_model(dx, case, averaged=False)


def antenna_positions(case, ys):
    """Points about ``ANTENNA_HEIGHT`` above the surface, snapped to 2 cm nodes."""
    p = np.stack([case["surface"](np.asarray(ys, dtype=float)) - ANTENNA_HEIGHT,
                  np.asarray(ys, dtype=float)], 1)
    return np.round(p / DX) * DX


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_native(eps, sig, dx, dt, nt, src, rec, device, frame_every=None, decimate=1):
    """Native DeepGPR FDTD.

    Args:
        src: ``(nshot, 2)`` physical positions (rounded to nodes).
        rec: ``(nshot, nrx, 2)``.
        frame_every: Split the run into segments of this many steps (states
            passed back) and keep Ez of shot 0 at the segment ends.

    Returns:
        ``(data (nshot, nt, nrx), frames or None, seconds, peak_bytes)``.
    """
    pml = int(round(PML_M / dx))
    to_node = lambda p: np.rint(np.asarray(p) / dx).astype(int)  # noqa: E731
    s, r = to_node(src), to_node(rec)
    src_t = torch.tensor([[[i, j, 0]] for i, j in s], dtype=torch.int, device=device)
    rec_t = torch.tensor([[[i, j, 0] for i, j in shot] for shot in r], dtype=torch.int,
                         device=device)
    w = wavelet(nt, dt).float().to(device)
    nx, ny = eps.shape
    seg = frame_every or nt
    E = H = P = None
    frames, data = [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    t0 = time.perf_counter()
    for n0 in range(0, nt, seg):
        n1 = min(nt, n0 + seg)
        _, E, H, P, d = DeepGPR.compute(
            device=device, dx=dx, dt=dt, source_amplitudes=w[n0:n1].reshape(1, -1, 1),
            source_location=src_t, receiver_location=rec_t, eps_r=eps.float(),
            sigma=sig.float(), pmlthick=pml, E=E, H=H, PML=P, save_wavefield_history=False)
        data.append(d)
        if frame_every:
            frames.append(E[2][0, pml:pml + nx:decimate, pml:pml + ny:decimate, 0].cpu().numpy())
    sync(device)
    seconds = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    data = torch.cat(data, 1).cpu().numpy()
    return data, (np.stack(frames) if frames else None), seconds, peak


HYBRID_REFINEMENT = 3   # Yee spacing DX / 3 inside the hybrid (no extra time steps)


def build_dg(case, device, *, mode="hybrid", order=1, h=None, dt=None, nshot=1,
             dtype=torch.float32, band=None, refinement=None, **kw):
    """Hybrid (``mode="hybrid"``) or full DGTD (``"dgtd"``) solver for a case.

    The ground model is the 2 cm grid ``ground_model(DX)`` (what a DeepGPR user
    supplies); the surface is the exact curve.
    """
    eps, sig = ground_model(DX, case["anomalies"])
    band = (band or case["band"]) if mode == "hybrid" else (None, None)
    if refinement is None:
        refinement = HYBRID_REFINEMENT if mode == "hybrid" else 1
    return HybridTM2D(eps, sig, DX, surface=case["surface"], dg_band=band,
                      pmlthick=int(round(PML_M / DX)), dg_order=order,
                      dg_h=(DX * order if h is None else h), mesh="delaunay", dt=dt,
                      nshot=nshot, device=device, dtype=dtype, fdtd_refinement=refinement,
                      **kw)


def run_dg(sol, nt, src, rec, frame_every=None, samplers=None):
    """Run a DG/hybrid solver; returns ``(data, frames dict, seconds, peak_bytes)``."""
    frames = {k: [] for k in (samplers or {})}

    def grab(n, s):
        if (n + 1) % frame_every == 0:
            for k, smp in samplers.items():
                frames[k].append(smp())

    device = sol.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    src = np.asarray(src, dtype=float).reshape(sol.nshot, -1, 2)
    rec = np.asarray(rec, dtype=float).reshape(sol.nshot, -1, 2)
    res = sol.run(wavelet(nt, sol.dt), src, rec,
                  callback=grab if samplers else None, callback_every=1 if samplers else 0)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return (res["receivers"].cpu().numpy(), {k: np.stack(v) for k, v in frames.items()},
            res["seconds"], peak)


__all__ = [
    "ANOMALIES", "ANTENNA_HEIGHT", "CASES", "DEPTH", "DX", "FRAME_DT", "PML_M", "T_MAX", "WIDTH",
    "aligned_dt", "antenna_positions", "build_dg", "ground_model", "nt_of", "run_dg",
    "ground_fraction", "ground_on", "run_native", "staircase_model", "sync", "wavelet",
    "yee_model",
]
