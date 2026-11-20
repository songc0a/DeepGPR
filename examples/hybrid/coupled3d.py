"""Ground-coupled antennas over rough, wet ground in 3D: careful staircase FDTD against
the hybrid DGTD-FDTD with the antennas inside the DG band (forward accuracy and cost).

The 2D TM tests (showcase2d_*.py, docs/HYBRID_DGTD.md 12.9) found a careful staircase
(cell-averaged materials, exact antenna positions) accurate everywhere; in 2D TM the field
is tangential to every interface, where the arithmetic cell average is the right
permittivity.  In 3D the field of a ground-coupled antenna also crosses the surface in the
normal direction, where the right average is harmonic, so the staircase may converge
slowly.  This script measures it.

Model (2 cm model grid, 0.58 m x 0.58 m, 8-cell absorbers): air over a homogeneous ground
below a rough surface (wet eps_r 15, 10 mS/m, or dry eps_r 5, 3 mS/m; surface 0.14-0.27 m
deep, slopes up to 65 deg, median 30 deg).  9 shots on a 3 x 3 pattern, each with 8
receivers at offsets 8 and 14 cm along y and z; Ez dipoles (horizontal), 400 MHz Ricker,
14 ns.  Every antenna is on an Ez node shared by all odd refinements of the 2 cm grid
(x, y on 2 cm, z on odd cm), so no solver moves it.

CASE = GROUND_PLACEMENT[_line][_tgt]:

* GROUND ``wet`` | ``dry``;
* PLACEMENT ``coupled`` (antennas 1-3 cm above the local surface, inside the DG band),
  ``air`` (4 cm below the model top, 11-22 cm above the ground, in the Yee grid) or
  ``homog`` (diagnostic: coupled positions in a homogeneous medium, the "air" given the
  ground's material, so a staircase has no geometry error and every hybrid error is the DG
  point source, dispersion or the absorbers);
* ``_line``: finite antennas instead of point dipoles, a 4 cm line current along z with a
  cos^2 taper (same moment), received as the same weighted line average of Ez; each solver
  samples the line at its own resolution (native: the Ez nodes on the line; hybrid: DG
  point sources every 1 mm);
* ``_amr``: as ``_tgt`` in a 0.66 m deep model with the sphere centred 0.48 m deep, so a
  Huygens subgrid box (2.22 mm, ratio 3 on a 6.67 mm main grid, time ratio 1) fits around
  it below the band: the full method (``sub:`` specs, HybridSubgridMaxwell3D: DG band for
  the surface, coarse Yee, refined box for the target) against the uniform staircase;
* ``_tgt``: a deeper model (0.58 m) with a sphere (eps_r 5, 1 mS/m, radius 6 cm, centre
  0.46 m deep under the central shot); every run also simulates the model without the
  sphere, and the report scores the target response (data with minus data without the
  sphere), the signal an image is made of.  The hybrid then takes its Yee materials on its
  own Yee grid (``fdtd_refinement = 1``), averaged exactly like the staircase at that
  spacing, so the sphere is represented identically in both families.

Solvers:

* ``native_avg_k`` / ``native_bin_k``: native DeepGPR on 2 cm / k (k odd), materials
  cell-averaged (arithmetic, volume fractions) or binary (node in air or ground);
* ``hyb_r{r}_P{p}_h{mm}``: HybridMaxwell3D, Yee 2 cm / r outside a P``p`` band 0.06-0.34 m
  with ``dg_h`` element size, flux 0.1, global stepping (``:lts``: local time stepping);
* ``hybsub_r{m}_P{p}_h{mm}`` (spec ``sub:m:p:mm``, ``_amr`` cases): HybridSubgridMaxwell3D,
  main Yee 2 cm / m from a fine model 2 cm / (3 m) (coarsen 3), the P``p`` band, one box of
  ratio 3 around the sphere (m = 3: 6.67 mm main grid, 2.22 mm box; m = 5: 4 mm, 1.33 mm).

    python examples/hybrid/coupled3d.py CASE native --k 3 5 7 9 [--mode avg bin]
    python examples/hybrid/coupled3d.py CASE hybrid --spec 3:1:20 3:2:10 ...
    python examples/hybrid/coupled3d.py CASE report [--ref KEY]

Results in examples/hybrid/results/coupled3d/CASE/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import DeepGPR  # noqa: E402
from DeepGPR.hybrid import HybridMaxwell3D, HybridPropagator, Surface3D, resample  # noqa: E402

OUT = ROOT / "examples" / "hybrid" / "results" / "coupled3d"
DEV = os.environ.get("COUPLED3D_DEVICE", "cuda")
DX = 0.02
SHAPE = (22, 30, 30)                     # 0.42 x 0.58 x 0.58 m
SHAPE_TGT = (30, 30, 30)                 # 0.58 m deep (``_tgt`` cases)
SHAPE_AMR = (33, 30, 30)                 # 0.66 m deep (``_amr`` cases)
PML = 8
FREQ, T0, T_MAX = 4.0e8, 3.0e-9, 14e-9
DT_OUT = 2e-11
BAND = (0.06, 0.34)
GROUND = {"wet": (15.0, 0.010), "dry": (5.0, 0.003)}
SPHERE = (0.46, 0.30, 0.31, 0.06, 5.0, 0.001)        # centre x, y, z, radius, eps_r, sigma
SPHERE_AMR = (0.48, 0.30, 0.31, 0.06, 5.0, 0.001)
SUB_FINE_K = 9                                       # fine model 2 cm / 9 = 2.22 mm
SUB_COARSEN = 3                                      # main Yee 6.67 mm, box ratio 3
SUB_BOX = ((0.4133333, 0.5466667), (0.2333333, 0.3666667), (0.24, 0.38))
SHOTS_YZ = [(y, z) for y in (0.20, 0.30, 0.40) for z in (0.21, 0.31, 0.41)]
OFFSETS = [(0.0, 0.08), (0.0, -0.08), (0.0, 0.14), (0.0, -0.14),
           (0.08, 0.0), (-0.08, 0.0), (0.14, 0.0), (-0.14, 0.0)]
AIR_X = 0.04
LINE_L = 0.04                            # finite antenna length (``_line`` cases)
LINE_DZ_HYB = 0.001                      # quadrature spacing of the line in the hybrid
CASES = ("wet_coupled", "dry_coupled", "wet_air", "wet_homog", "dry_homog",
         "wet_coupled_line", "wet_homog_line", "dry_coupled_line",
         "wet_coupled_line_tgt", "wet_air_tgt", "dry_coupled_line_tgt", "dry_air_tgt",
         "wet_coupled_line_amr")


def ground_of(case):
    return case.split("_")[0]


def is_homog(case):
    return "homog" in case


def is_line(case):
    return "_line" in case


def has_target(case):
    return case.endswith("_tgt") or case.endswith("_amr")


def shape_of(case):
    if case.endswith("_amr"):
        return SHAPE_AMR
    return SHAPE_TGT if has_target(case) else SHAPE


def sphere_of(case):
    return SPHERE_AMR if case.endswith("_amr") else SPHERE


def surface():
    ey = np.linspace(0.0, 0.58, 11)
    Y, Z = np.meshgrid(ey, ey, indexing="ij")
    elev = (0.05 * np.sin(2 * np.pi * Y / 0.40 + 0.5) * np.cos(2 * np.pi * Z / 0.47)
            + 0.025 * np.random.default_rng(11).uniform(-1.0, 1.0, Y.shape))
    return Surface3D.from_grid(ey, ey, 0.20 - elev)


SURF = surface()


def surface_stats():
    g = np.arange(0.0, 0.58, 0.002)
    Y, Z = np.meshgrid(g, g, indexing="ij")
    s = SURF(Y, Z)
    gy, gz = np.gradient(s, 0.002)
    slope = np.degrees(np.arctan(np.hypot(gy, gz)))
    return dict(depth_min=float(s.min()), depth_max=float(s.max()),
                slope_max_deg=float(slope.max()), slope_p90_deg=float(np.percentile(slope, 90)),
                slope_median_deg=float(np.median(slope)))


def antennas(case):
    """Physical (x, y, z) of sources (nshot, 1, 3) and receivers (nshot, 8, 3)."""
    coupled = "coupled" in case or "homog" in case

    def x_of(y, z):
        if not coupled:
            return AIR_X
        if is_line(case):                      # the whole line 1-3 cm above (or higher)
            zz = z + np.linspace(-0.5 * LINE_L, 0.5 * LINE_L, 41)
            s = float(SURF(np.full((1, 41), y), zz[None]).min())
        else:
            s = float(SURF(np.array([[y]]), np.array([[z]]))[0, 0])
        return round(np.floor((s - 0.01) / DX + 1e-9) * DX, 6)   # 1-3 cm above the surface
    src, rec = [], []
    for y, z in SHOTS_YZ:
        src.append([[x_of(y, z), y, z]])
        rec.append([[x_of(y + dy, z + dz), round(y + dy, 6), round(z + dz, 6)] for dy, dz in OFFSETS])
    return np.array(src), np.array(rec)


def line_points(dz):
    """Offsets along z and normalised moment weights of the finite antenna, spacing dz
    (symmetric about the centre, which is a node of every odd refinement)."""
    m = int(np.floor(0.5 * LINE_L / dz - 1e-9))
    off = np.arange(-m, m + 1) * dz
    w = np.cos(np.pi * off / LINE_L) ** 2
    return off, w / w.sum()


def expand_line(p, off):
    """(nshot, n, 3) antenna centres -> (nshot, n * len(off), 3) line points."""
    shift = np.stack([0 * off, 0 * off, off], 1)
    return (p[:, :, None, :] + shift[None, None]).reshape(p.shape[0], -1, 3)


def wavelet(t):
    a = (np.pi * FREQ * (t - T0)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def time_axis():
    return torch.arange(1, int(round(T_MAX / DT_OUT)) + 1, dtype=torch.float64) * DT_OUT


def sphere_fraction(n, h, sphere=SPHERE):
    """Volume fraction of the sphere in the node cells (signed-distance estimate,
    clip(1/2 - d/h, 0, 1)), computed in its bounding box."""
    xc, yc, zc, R = sphere[:4]
    f = torch.zeros(n, dtype=torch.float64, device=DEV)
    lo = [max(0, int(np.floor((c - R) / h)) - 2) for c in (xc, yc, zc)]
    hi = [min(m, int(np.ceil((c + R) / h)) + 3) for c, m in zip((xc, yc, zc), n)]
    ax = [torch.arange(a, b, dtype=torch.float64, device=DEV) * h for a, b in zip(lo, hi)]
    X, Y, Z = torch.meshgrid(*ax, indexing="ij")
    d = torch.sqrt((X - xc) ** 2 + (Y - yc) ** 2 + (Z - zc) ** 2) - R
    f[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = torch.clamp(0.5 - d / h, 0.0, 1.0)
    return f


def materials(case, h, mode, target=True, ground_only=False, sub=None, extra=0):
    """Node materials on spacing h (shape * k + extra nodes): cell-averaged ('avg') or
    binary; ``ground_only``: the ground model without air (the hybrid imposes the air)."""
    eps_g, sig_g = GROUND[ground_of(case)]
    k = int(round(DX / h))
    n = tuple(m * k + extra for m in shape_of(case))
    x = torch.arange(n[0], dtype=torch.float64, device=DEV) * h
    if is_homog(case) or ground_only:
        frac = torch.ones(n, dtype=torch.float64, device=DEV)
    elif mode == "bin":
        Y, Z = np.meshgrid(np.arange(n[1]) * h, np.arange(n[2]) * h, indexing="ij")
        s = torch.as_tensor(SURF(Y, Z), device=DEV)
        frac = (x[:, None, None] >= s[None]).double()
    else:
        sub = sub or (5 if k <= 3 else 3)
        off = (np.arange(sub) + 0.5) / sub - 0.5
        frac = torch.zeros(n, dtype=torch.float64, device=DEV)
        for oy in off:
            for oz in off:
                Y, Z = np.meshgrid(np.arange(n[1]) * h + oy * h, np.arange(n[2]) * h + oz * h,
                                   indexing="ij")
                s = torch.as_tensor(SURF(Y, Z), device=DEV)
                frac += torch.clamp((x[:, None, None] + 0.5 * h - s[None]) / h, 0.0, 1.0)
        frac /= sub * sub
    eps = 1.0 + frac * (eps_g - 1.0)
    sig = frac * sig_g
    if has_target(case) and target:
        sph = sphere_of(case)
        fs = sphere_fraction(n, h, sph)
        if mode == "bin":
            fs = (fs >= 0.5).double()
        eps = eps + frac * fs * (sph[4] - eps_g)
        sig = sig + frac * fs * (sph[5] - sig_g)
    return eps.float(), sig.float()


def native(case, k, mode):
    """Native DeepGPR (staircase) on 2 cm / k, one shot at a time; with a target also the
    model without it."""
    h = DX / k
    models = [materials(case, h, mode, target=True)]
    if has_target(case):
        models.append(materials(case, h, mode, target=False))
    pml = PML * k
    n = models[0][0].shape
    dt = 0.95 * DeepGPR.max_stable_time_step(h, n[0] + 2 * pml, n[1] + 2 * pml, n[2] + 2 * pml)
    nt = int(np.ceil(T_MAX / dt)) + 1
    w = torch.tensor(wavelet((np.arange(nt) + 0.5) * dt) * k, dtype=torch.float32)  # moment w DX
    src, rec = antennas(case)
    lw = None
    if is_line(case):                       # Ez nodes on the line, normalised cos^2 weights
        off, lw = line_points(h)
        src, rec = expand_line(src, off), expand_line(rec, off)
        w = torch.tensor(lw, dtype=torch.float32)[:, None] * w[None]          # (npts, nt)
    idx = lambda p: torch.tensor(np.rint(p / h - np.array([0, 0, 0.5])).astype(np.int32))  # noqa: E731
    for p in (src, rec):
        f = p / h - np.array([0, 0, 0.5])
        assert np.abs(f - np.rint(f)).max() < 1e-6, "antenna not on an Ez node"
    t = (torch.arange(nt, dtype=torch.float64) + 1) * dt
    outs, secs = [], []
    torch.cuda.reset_peak_memory_stats()
    for eps, sig in models:
        parts = []
        for i in range(src.shape[0]):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = DeepGPR.compute(device=DEV, dx=h, dt=dt,
                                  source_amplitudes=(w.reshape(1, -1, 1) if lw is None else w[..., None]),
                                  source_location=idx(src[i:i + 1]), receiver_location=idx(rec[i:i + 1]),
                                  eps_r=eps, sigma=sig, pmlthick=pml, save_wavefield_history=False)
            torch.cuda.synchronize()
            secs.append(time.perf_counter() - t0)
            d = out[-1].double().cpu()                                    # (1, nt, nrec*)
            if lw is not None:
                d = (d.reshape(d.shape[0], d.shape[1], -1, lw.size) * torch.as_tensor(lw)).sum(-1)
            parts.append(resample(d, t, time_axis()))
            del out
        outs.append(torch.cat(parts).numpy())
    meta = dict(h_mm=1e3 * h, dt=dt, nt=nt, cells=int(np.prod([m + 2 * pml for m in n])),
                seconds_per_shot=float(np.median(secs)),
                peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    del models
    torch.cuda.empty_cache()
    return outs, meta


def hybrid(case, r, order, dg_h_mm, lts=False):
    """HybridMaxwell3D with Yee spacing 2 cm / r; with a target the Yee materials are the
    averaged model on the Yee grid itself (``fdtd_refinement = 1``)."""
    eps_g, sig_g = GROUND[ground_of(case)]
    kw = dict(lts="auto") if lts else {}
    if is_homog(case):
        kw.update(air_eps_r=eps_g, air_sigma=sig_g)
    if has_target(case):
        h = DX / r
        models = [materials(case, h, "avg", target=True, ground_only=True),
                  materials(case, h, "avg", target=False, ground_only=True)]
        build = dict(dx=h, fdtd_refinement=1, pmlthick=PML * r)
        amp = DX / h                          # the dipole length dl is the model spacing
    else:
        g = (torch.full(SHAPE, eps_g), torch.full(SHAPE, sig_g))
        models = [tuple(t.to(DEV, torch.float32) for t in g)]
        build = dict(dx=DX, fdtd_refinement=r, pmlthick=PML)
        amp = 1.0
    t0 = time.perf_counter()
    sol = HybridMaxwell3D(models[0][0], models[0][1], build["dx"], surface=SURF, dg_band=BAND,
                          pmlthick=build["pmlthick"], flux=0.1,
                          fdtd_refinement=build["fdtd_refinement"], dg_order=order,
                          dg_h=dg_h_mm * 1e-3, nshot=1, device=DEV, dtype=torch.float32, **kw)
    setup = time.perf_counter() - t0
    src, rec = antennas(case)
    nt = int(np.ceil(T_MAX / sol.dt)) + 1
    w = torch.tensor(wavelet((np.arange(nt) + 0.5) * sol.dt) * amp, dtype=torch.float32,
                     device=DEV)[None]
    lw = None
    if is_line(case):
        off, lw = line_points(LINE_DZ_HYB)
        src, rec = expand_line(src, off), expand_line(rec, off)
        w = torch.tensor(lw, dtype=torch.float32, device=DEV)[:, None] * w        # (npts, nt)
    outs = [[] for _ in models]
    secs = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(src.shape[0]):
        prop = HybridPropagator(sol, src[i:i + 1], rec[i:i + 1])
        t_prop = prop.time_axis(nt).to(DEV)
        for j, (e, s) in enumerate(models):
            with torch.no_grad():
                if i == 0 and j == 0:
                    prop(e, s, w)                  # warm-up (kernels, graph capture)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                d = prop(e, s, w)
                torch.cuda.synchronize()
            secs.append(time.perf_counter() - t1)
            d = d.double()
            if lw is not None:
                d = (d.reshape(d.shape[0], d.shape[1], -1, lw.size) * torch.as_tensor(lw, device=DEV)).sum(-1)
            outs[j].append(resample(d, t_prop.double(), time_axis().to(DEV)).cpu())
        del prop
    info = sol.summary()
    meta = dict(r=r, order=order, dg_h_mm=dg_h_mm, lts=bool(lts), p=sol.lts, dt=sol.dt, nt=nt,
                yee_dofs=int(info["fdtd_dofs"]), dg_dofs=int(info["dg_dofs"]),
                setup_seconds=setup, seconds_per_shot=float(np.median(secs)),
                peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    del sol
    torch.cuda.empty_cache()
    return [torch.cat(o).numpy() for o in outs], meta


def sub_box(main_k):
    """The refined box around the sphere on the main grid 2 cm / main_k (one main cell of
    margin); the original box for main_k = 3."""
    if main_k == 3:
        return SUB_BOX
    h = DX / main_k
    c, R = SPHERE_AMR[:3], SPHERE_AMR[3]
    return tuple((round((np.floor((ci - R) / h + 1e-9) - 1) * h, 7),
                  round((np.ceil((ci + R) / h - 1e-9) + 1) * h, 7)) for ci in c)


def hybrid_sub(case, order, dg_h_mm, main_k=3):
    """The full method: HybridSubgridMaxwell3D with the fine model (2 cm / (3 main_k),
    averaged), main Yee 2 cm / main_k (block averages), the DG band and one ratio-3 box
    around the sphere (time ratio 1 when the band's step fits the box, else 3)."""
    from DeepGPR.subgrid import Subgrid
    from DeepGPR.subgrid.adjoint import SubgridPropagator
    from DeepGPR.subgrid.hybrid import HybridSubgridMaxwell3D
    box = sub_box(main_k)
    hf = DX / (SUB_COARSEN * main_k)
    models = [materials(case, hf, "avg", target=True, ground_only=True, extra=1),
              materials(case, hf, "avg", target=False, ground_only=True, extra=1)]
    t0 = time.perf_counter()
    for time_ratio in (1, None):          # 1 when the band's step fits the box, else sub-steps
        try:
            sol = HybridSubgridMaxwell3D(models[0][0], models[0][1], hf, coarsen=SUB_COARSEN,
                                         surface=SURF, dg_band=BAND,
                                         subgrids=[Subgrid(box, ratio=SUB_COARSEN, time_ratio=time_ratio)],
                                         pmlthick=PML * main_k, flux=0.1,
                                         dg_order=order, dg_h=dg_h_mm * 1e-3, nshot=1, device=DEV,
                                         dtype=torch.float32)
            break
        except ValueError as exc:
            if time_ratio is None or "CFL" not in str(exc):
                raise
    setup = time.perf_counter() - t0
    amp = DX / (SUB_COARSEN * hf)             # dipole length = the main spacing
    src, rec = antennas(case)
    nt = int(np.ceil(T_MAX / sol.dt)) + 1
    w = torch.tensor(wavelet((np.arange(nt) + 0.5) * sol.dt) * amp, dtype=torch.float32, device=DEV)[None]
    lw = None
    if is_line(case):
        off, lw = line_points(LINE_DZ_HYB)
        src, rec = expand_line(src, off), expand_line(rec, off)
        w = torch.tensor(lw, dtype=torch.float32, device=DEV)[:, None] * w
    t_prop = ((torch.arange(nt, dtype=torch.float64) + 1) * sol.dt).to(DEV)
    outs = [[] for _ in models]
    secs = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(src.shape[0]):
        prop = SubgridPropagator(sol, src[i:i + 1], rec[i:i + 1])
        for j, (e, s) in enumerate(models):
            with torch.no_grad():
                if i == 0 and j == 0:
                    prop(e, s, w)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                d = prop(e, s, w)
                torch.cuda.synchronize()
            secs.append(time.perf_counter() - t1)
            d = d.double()
            if lw is not None:
                d = (d.reshape(d.shape[0], d.shape[1], -1, lw.size) * torch.as_tensor(lw, device=DEV)).sum(-1)
            outs[j].append(resample(d, t_prop, time_axis().to(DEV)).cpu())
        del prop
    info = sol.summary()
    meta = dict(r=main_k, order=order, dg_h_mm=dg_h_mm, sub_box=box, time_ratio=time_ratio or SUB_COARSEN,
                p=1, dt=sol.dt, nt=nt,
                yee_dofs=int(info["fdtd_dofs"]), dg_dofs=int(info["dg_dofs"]),
                sub_nodes=[int(v) for v in info["sub_nodes"]], setup_seconds=setup,
                seconds_per_shot=float(np.median(secs)),
                peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    del sol
    torch.cuda.empty_cache()
    return [torch.cat(o).numpy() for o in outs], meta


def save(case, key, datas, meta):
    d = OUT / case
    d.mkdir(parents=True, exist_ok=True)
    extra = {"data0": datas[1]} if len(datas) > 1 else {}
    np.savez(d / f"{key}.npz", data=datas[0], meta=json.dumps(meta), **extra)
    print(f"  [{case}] {key}: {meta}", flush=True)


def load(case):
    runs = {}
    for p in sorted((OUT / case).glob("*.npz")):
        z = np.load(p)
        runs[p.stem] = (z["data"], json.loads(str(z["meta"])), z["data0"] if "data0" in z.files else None)
    return runs


def errors(d, ref, d0=None, ref0=None):
    t = time_axis().numpy()
    g = (np.clip((t - 1e-9) / 1e-9, 0, None) ** 2)[None, :, None]
    early = (t < 6e-9)[None, :, None]
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))  # noqa: E731
    out = dict(plain=rel(d, ref), gained=rel(d * g, ref * g),
               early=rel(d * early, ref * early), late=rel(d * ~early, ref * ~early))
    if d0 is not None and ref0 is not None:
        out["target"] = rel(d - d0, ref - ref0)
        out["background"] = rel(d0, ref0)
    return out


def report(case, ref_key=None):
    runs = load(case)
    if not runs:
        return
    natives = sorted([k for k in runs if k.startswith("native_avg")], key=lambda k: runs[k][1]["h_mm"])
    ref_key = ref_key or (natives[0] if natives else sorted(runs)[0])
    ref, _, ref0 = runs[ref_key]
    ns = T_MAX * 1e9
    tgt = ref0 is not None
    head = ("| run | target response | background | gained | ms per shot per simulated ns | peak GB | dt [ps] |"
            if tgt else
            "| run | plain | gained | t < 6 ns | t >= 6 ns | ms per shot per simulated ns | peak GB | dt [ps] |")
    lines = [f"# {case}: data error against `{ref_key}`", "", "surface: " + json.dumps(surface_stats()), "",
             head, "|---|---|---|---|---|---|---|" + ("" if tgt else "---|")]
    rows = []
    for k, (d, m, d0) in runs.items():
        e = errors(d, ref, d0, ref0)
        rows.append((k, e, 1e3 * m["seconds_per_shot"] / ns, m))
    for k, e, cost, m in sorted(rows, key=lambda r: r[2]):
        if tgt:
            lines.append(f"| {k} | {100 * e.get('target', np.nan):.2f} % | {100 * e.get('background', np.nan):.2f} % | "
                         f"{100 * e['gained']:.2f} % | {cost:.0f} | {m['peak_gb']:.1f} | {1e12 * m['dt']:.2f} |")
        else:
            lines.append(f"| {k} | {100 * e['plain']:.2f} % | {100 * e['gained']:.2f} % | {100 * e['early']:.2f} % | "
                         f"{100 * e['late']:.2f} % | {cost:.0f} | {m['peak_gb']:.1f} | {1e12 * m['dt']:.2f} |")
    (OUT / case / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        metrics = ("target", "background") if tgt else ("plain", "gained")
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for fam, color, mark in (("native_avg", "#c2410c", "s"), ("native_bin", "#9ca3af", "^"),
                                 ("hyb_", "#1d4ed8", "o"), ("hybsub", "#7c3aed", "D")):
            pts = sorted((c, e, k) for k, e, c, _ in rows if k.startswith(fam) and k != ref_key)
            for j, metric in enumerate(metrics):
                if pts:
                    ax[j].loglog([p[0] for p in pts], [100 * p[1][metric] for p in pts], mark + "-",
                                 color=color, label=fam)
                    for c, e, k in pts:
                        ax[j].annotate(k.replace("native_", "").replace("hyb_", ""), (c, 100 * e[metric]),
                                       fontsize=7)
        for j, metric in enumerate(metrics):
            ax[j].set_xlabel("ms per shot per simulated ns (RTX 4090)")
            ax[j].set_ylabel(f"data error ({metric}) [%]")
            ax[j].set_title(f"{case}, reference {ref_key}")
            ax[j].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / case / "pareto.png", dpi=110)
    except Exception as exc:  # pragma: no cover
        print("figure failed:", exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=CASES)
    ap.add_argument("what", choices=("native", "hybrid", "report", "info"))
    ap.add_argument("--k", type=int, nargs="*", default=[3, 5, 7, 9])
    ap.add_argument("--mode", nargs="*", default=["avg"])
    ap.add_argument("--spec", nargs="*", default=["3:1:20"], help="r:order:dg_h_mm[:lts]")
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    if a.what == "info":
        src, rec = antennas(a.case)
        s = SURF(rec[:, :, 1], rec[:, :, 2])
        print(surface_stats())
        print("source x", sorted(set(src[:, 0, 0].round(3))), "receiver height above ground [cm]:",
              np.round(100 * (s - rec[:, :, 0]).min(), 2), "-", np.round(100 * (s - rec[:, :, 0]).max(), 2),
              "shape", shape_of(a.case))
    elif a.what == "native":
        for k in a.k:
            for mode in a.mode:
                key = f"native_{mode}_k{k:02d}"
                if (OUT / a.case / f"{key}.npz").exists():
                    continue
                datas, meta = native(a.case, k, mode)
                save(a.case, key, datas, meta)
    elif a.what == "hybrid":
        for spec in a.spec:
            parts = spec.split(":")
            if parts[0] == "sub":
                main_k, order, hmm = int(parts[1]), int(parts[2]), float(parts[3])
                key = f"hybsub_r{main_k}_P{order}_h{hmm:g}"
                if (OUT / a.case / f"{key}.npz").exists():
                    continue
                datas, meta = hybrid_sub(a.case, order, hmm, main_k)
                save(a.case, key, datas, meta)
                continue
            r, order, hmm = int(parts[0]), int(parts[1]), float(parts[2])
            lts = len(parts) > 3 and parts[3] == "lts"
            key = f"hyb_r{r}_P{order}_h{hmm:g}" + ("_lts" if lts else "")
            if (OUT / a.case / f"{key}.npz").exists():
                continue
            datas, meta = hybrid(a.case, r, order, hmm, lts)
            save(a.case, key, datas, meta)
    else:
        report(a.case, a.ref)


if __name__ == "__main__":
    main()
