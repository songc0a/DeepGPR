"""3D joint eps_r / sigma FWI on complex terrain with the LTS hybrid (paper contribution 2).

The hybrid DGTD-FDTD solver with local time stepping (``lts="auto"``: the Yee
grid at its own step, the DG band and the interface planes sub-cycled) and its
exact discrete adjoint (autograd transpose of the sub-cycled step, through the
mortar coupling and the ghost planes) invert surface GPR data for permittivity
and conductivity on rugged ground. A second arm runs the same inversion with
global stepping (every unknown at the band's step, hand-written transpose) as the
reference: the two discrete schemes differ only by their time discretisation, so
their inversions should agree.

* Model (2 cm model grid, 0.56 m deep, 0.64 m x 0.64 m, 8-cell absorbers):
  rugged surface 0.17 m +- 6 cm (9 x 9 elevation samples: a sine pattern plus
  random roughness, PCHIP), ground eps_r 5, sigma 3 mS/m; a permittivity sphere
  (eps_r 7.5) and, 24 cm away, a conductivity sphere (40 mS/m), radius 9 cm,
  centres 0.38 m deep; a known reflector (eps_r 8 below 0.50 m, also in the
  initial model) that the waves reach through the anomalies.
* Observed data: an independent solver, native DeepGPR on 4 mm (dx/5) with
  cell-averaged materials at the surface (staircase, another grid, time step and
  surface treatment), the ground model being the 2 cm model upsampled (so the
  truth is representable on the inversion grid; see docs/HYBRID_DGTD.md 11.7).
* Acquisition: 3 x 3 Ez sources and 7 x 7 Ez receivers 4 cm below the model top
  (7-19 cm above the ground), 400 MHz Ricker, 12 ns; offsets below 12 cm muted;
  misfit on time-gained data ((t - 1 ns)^2, ``--gain``).
* Inversion solver: hybrid, Yee 2/3 cm (``fdtd_refinement = 3``), DG band
  0.08-0.28 m (P1 or P2, flux 0.1: central + 10 % upwind penalty, with LTS as a
  split damping), float32, one shot at a time.
* Inversion: perturbations on the 2 cm grid below the surface, L-BFGS (strong
  Wolfe) on normalised parameters ``eps = eps0 + m_e``, ``sigma = sigma0 +
  omega0 eps0 m_s`` (one unit of ``m_s`` changes the loss tangent at the peak
  frequency as much as one unit of ``m_e`` changes eps_r); stage 1 eps_r only,
  data low-passed to 250 MHz; stage 2 eps_r and sigma, full band. Within a stage
  the objective is divided by its value at the start of the stage.

    python examples/hybrid/fwi3d_lts.py --observed          # 4 mm reference data
    python examples/hybrid/fwi3d_lts.py --diagnose          # modelling error vs signal
    python examples/hybrid/fwi3d_lts.py --gradcheck         # adjoint vs finite differences
    python examples/hybrid/fwi3d_lts.py --arm lts           # inversion with LTS
    python examples/hybrid/fwi3d_lts.py --arm global        # same with global stepping
    python examples/hybrid/fwi3d_lts.py --report            # figures, GIF, report.md

Results in ``examples/hybrid/results/fwi3d_lts/``.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import DeepGPR  # noqa: E402
from DeepGPR import apply_filter  # noqa: E402
from DeepGPR.hybrid import HybridMaxwell3D, HybridPropagator, Surface3D, resample  # noqa: E402

OUT = ROOT / "examples" / "hybrid" / "results" / "fwi3d_lts"
DX = 0.02
SHAPE = (28, 32, 32)
PML = 8
FREQ, T_MAX = 4.0e8, 12e-9
T0 = 1.2 / FREQ
OBS_REFINE = 5
EPS_BG, SIG_BG = 5.0, 0.003
LAYER_X, EPS_LAYER = 0.50, 8.0
# (centre x (depth), y, z, radius, eps_r, sigma)
SPHERES = [(0.38, 0.20, 0.32, 0.09, 7.5, None), (0.38, 0.44, 0.32, 0.09, None, 0.04)]
_EY = np.linspace(0.0, 0.62, 9)
_Y, _Z = np.meshgrid(_EY, _EY, indexing="ij")
_ELEV = (0.05 * np.sin(2 * np.pi * _Y / 0.45) * np.cos(2 * np.pi * _Z / 0.55)
         + 0.025 * np.random.default_rng(3).uniform(-1.0, 1.0, _Y.shape))
SURFACE = Surface3D.from_grid(_EY, _EY, 0.17 - _ELEV)
# band top 2 cm below the antennas: with a single Yee cell between a point source /
# receiver and the band's interface plane the 2 cm grid of the AMR study produced large
# spurious late-time fields (35 % data error against 3.0 % with this band)
BAND = (0.08, 0.28)
ANT_X = 0.04
# y on 2 cm nodes, z odd multiples of 1 cm: Ez nodes of the 4 mm and the 2/3 cm grids
SRC_YZ = [(y, z) for y in (0.14, 0.32, 0.50) for z in (0.13, 0.31, 0.49)]
REC_YZ = [(round(y, 2), round(z, 2)) for y in np.arange(0.06, 0.55, 0.08)
          for z in np.arange(0.05, 0.54, 0.08)]
MUTE = 0.12
OMEGA_EPS0 = 2 * np.pi * FREQ * 8.8541878128e-12           # sigma scale [S/m]
# time gain (t - 1 ns)^GAIN on observed and simulated data: without it the normalised
# misfit is the strong early surface reflection (98 % of the energy before 6 ns) and the
# late anomaly / reflector arrivals (where sigma acts) hardly count: the anomaly signal
# is 1.65 % of the data without and 16 % with the quadratic gain
GAIN = 2.0
STAGES = ((8, False, 2.5e8), (24, True, None))              # (iterations, joint, low-pass)
DT_OUT = 2e-11


def wavelet(nt, dt):
    t = (np.arange(nt) + 0.5) * dt - T0
    a = (np.pi * FREQ * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def grid(h):
    r = int(round(DX / h))
    return tuple(n * r for n in SHAPE), r


def background(h=DX):
    n, _ = grid(h)
    eps = np.full(n, EPS_BG)
    eps[int(round(LAYER_X / h)):] = EPS_LAYER
    return eps, np.full(n, SIG_BG)


def true_ground(h=DX):
    eps, sig = background(h)
    n, _ = grid(h)
    x, y, z = np.meshgrid(*(np.arange(m) * h for m in n), indexing="ij")
    for cx, cy, cz, r, e, s in SPHERES:
        m = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r * r
        if e is not None:
            eps[m] = e
        if s is not None:
            sig[m] = s
    return eps, sig


def averaged(eps_g, sig_g, h, sub=3):
    """Cell-averaged air/ground materials on a grid of spacing ``h``."""
    n = eps_g.shape
    off = (np.arange(sub) + 0.5) / sub - 0.5
    frac = np.zeros(n)
    x = np.arange(n[0]) * h
    for oy in off:
        for oz in off:
            Y, Z = np.meshgrid(np.arange(n[1]) * h + oy * h, np.arange(n[2]) * h + oz * h,
                               indexing="ij")
            frac += np.clip((x[:, None, None] + 0.5 * h - SURFACE(Y, Z)[None]) / h, 0.0, 1.0)
    frac /= sub * sub
    return 1.0 + frac * (eps_g - 1.0), frac * sig_g


def antennas(shots=None):
    src = np.array([[[ANT_X, y, z]] for y, z in SRC_YZ])
    rec = np.array([[[ANT_X, y, z] for y, z in REC_YZ]] * len(SRC_YZ))
    if shots is not None:
        src, rec = src[shots], rec[shots]
    return src, rec


def mute_mask():
    src, rec = antennas()
    off = np.linalg.norm(rec[:, :, 1:] - src[:, :, 1:], axis=2)
    return torch.as_tensor(off >= MUTE, dtype=torch.float32)


def time_axis():
    return torch.arange(1, int(round(T_MAX / DT_OUT)) + 1, dtype=torch.float64) * DT_OUT


def gain():
    """``(nt,)`` time gain ``max(t - 1 ns, 0)^GAIN`` of the output time axis (ones for 0)."""
    t = time_axis()
    if GAIN <= 0:
        return torch.ones_like(t, dtype=torch.float32)
    return (torch.clamp((t - 1e-9) / 1e-9, min=0.0) ** GAIN).float()


def ground_mask():
    Y, Z = np.meshgrid(np.arange(SHAPE[1]) * DX, np.arange(SHAPE[2]) * DX, indexing="ij")
    return torch.as_tensor((np.arange(SHAPE[0]) * DX)[:, None, None] >= SURFACE(Y, Z)[None])


def sphere_masks():
    x, y, z = np.meshgrid(*(np.arange(m) * DX for m in SHAPE), indexing="ij")
    return [torch.as_tensor((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r * r)
            for cx, cy, cz, r, _, _ in SPHERES]


# --------------------------------------------------------------------------- data
def observed_data(device, background_model=False):
    """Native DeepGPR on 4 mm, cell-averaged surface (cached)."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("observed_bg.npz" if background_model else "observed.npz")
    if path.exists():
        return torch.as_tensor(np.load(path)["data"])
    h = DX / OBS_REFINE
    eps_g, sig_g = background() if background_model else true_ground()
    idx = [np.clip(np.floor(np.arange(n * OBS_REFINE) / OBS_REFINE + 0.5).astype(int), 0, n - 1)
           for n in SHAPE]
    eps, sig = averaged(eps_g[np.ix_(*idx)], sig_g[np.ix_(*idx)], h)
    pml = OBS_REFINE * PML
    n = eps.shape
    dt = 0.95 * DeepGPR.max_stable_time_step(h, n[0] + 2 * pml, n[1] + 2 * pml, n[2] + 2 * pml)
    nt = int(np.ceil(T_MAX / dt)) + 1
    w = torch.tensor(wavelet(nt, dt) * OBS_REFINE, dtype=torch.float32)   # moment w * DX
    src, rec = antennas()
    idx = lambda p: torch.tensor(np.rint(p / h - np.array([0, 0, 0.5])).astype(np.int32))  # noqa: E731
    t0 = time.time()
    t = (torch.arange(nt, dtype=torch.float64) + 1) * dt
    parts = []
    for i in range(src.shape[0]):           # one shot at a time (memory)
        out = DeepGPR.compute(device=device, dx=h, dt=dt, source_amplitudes=w.reshape(1, -1, 1),
                              source_location=idx(src[i:i + 1]), receiver_location=idx(rec[i:i + 1]),
                              eps_r=torch.tensor(eps, dtype=torch.float32),
                              sigma=torch.tensor(sig, dtype=torch.float32), pmlthick=pml,
                              save_wavefield_history=False)
        parts.append(resample(out[-1].double().cpu(), t, time_axis()))
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    data = torch.cat(parts).numpy()
    np.savez(path, data=data, dt=dt, seconds=time.time() - t0)
    print(f"observed data ({'background' if background_model else 'true'} model): 4 mm grid {n}, "
          f"dt {dt:.3e}, {nt} steps, {time.time() - t0:.0f} s", flush=True)
    return torch.as_tensor(data)


# --------------------------------------------------------------------------- solver
FLUX = 0.1      # interior DG flux: central + 10 % upwind penalty (see the modelling-error check)


def build_solver(arm, order, device, dtype=torch.float32, flux=None):
    """Hybrid solver set up for one shot at a time (the propagators share it)."""
    eps0, sig0 = background()
    kw = dict(lts="auto") if arm == "lts" else {}
    flux = FLUX if flux is None else flux
    return HybridMaxwell3D(eps0, sig0, DX, surface=SURFACE, dg_band=BAND, pmlthick=PML,
                           flux=flux, fdtd_refinement=3, dg_order=order, nshot=1,
                           device=device, dtype=dtype, **kw)


def shot_propagators(sol):
    src, rec = antennas()
    return [HybridPropagator(sol, src[i:i + 1], rec[i:i + 1]) for i in range(len(SRC_YZ))]


class Forward:
    """Shot-by-shot simulation on the common output time axis."""

    def __init__(self, sol, device):
        self.sol, self.device = sol, device
        self.props = shot_propagators(sol)
        self.nt = int(np.ceil(T_MAX / sol.dt)) + 1
        self.t_prop = self.props[0].time_axis(self.nt).to(device)
        self.t_out = time_axis().to(device)
        self.w = torch.tensor(wavelet(self.nt, sol.dt), dtype=sol.dtype, device=device)

    def __call__(self, eps, sig, i):
        d = self.props[i](eps.to(self.sol.dtype), sig.to(self.sol.dtype), self.w[None])
        return resample(d.to(torch.float64 if self.sol.dtype == torch.float64 else torch.float32),
                        self.t_prop.to(d.dtype), self.t_out.to(d.dtype))


def diagnose(order, device, arms=("global", "lts")):
    """Modelling error of the inversion solvers at the true model against the anomaly
    signal of the observed data."""
    mute = mute_mask()[:, None, :] * gain()[None, :, None]
    obs_t = observed_data(device) * mute
    obs_b = observed_data(device, background_model=True) * mute
    nrm = lambda a: float(a.norm())  # noqa: E731
    out = {"anomaly_signal": nrm(obs_t - obs_b) / nrm(obs_t), "gain": GAIN}
    print(f"anomaly signal ||d(m) - d(m0)|| / ||d(m)|| of the observed data: "
          f"{out['anomaly_signal']:.4f}", flush=True)
    prev = None
    for arm in arms:
        sol = build_solver(arm, order, device)
        fw = Forward(sol, device)
        res = {}
        t0 = time.time()
        for tag, model in (("true", true_ground()), ("background", background())):
            e, s = (torch.tensor(a, dtype=torch.float32, device=device) for a in model)
            with torch.no_grad():
                res[tag] = torch.cat([fw(e, s, i) for i in range(len(SRC_YZ))]).cpu().float() * mute
        secs = (time.time() - t0) / 2
        key = f"P{order}_{arm}_flux{FLUX}"
        out[key] = {"modelling_error": nrm(res["true"] - obs_t) / nrm(obs_t),
                    "own_anomaly_signal": nrm(res["true"] - res["background"]) / nrm(res["true"]),
                    "dt": sol.dt, "p": sol.lts, "forward_seconds_9_shots": secs}
        print(f"{key}: p={sol.lts} dt={sol.dt:.3e} modelling error {out[key]['modelling_error']:.4f} "
              f"(own anomaly signal {out[key]['own_anomaly_signal']:.4f}), 9-shot forward {secs:.1f} s",
              flush=True)
        if arm == "lts" and prev is not None:
            out[f"P{order}_lts_vs_global"] = nrm(res["true"] - prev) / nrm(prev)
            print(f"  LTS vs global data at the true model: {out[f'P{order}_lts_vs_global']:.2e}",
                  flush=True)
        prev = res["true"]
        del sol, fw
        torch.cuda.empty_cache()
    path = OUT / f"diagnose_P{order}_flux{FLUX}_gain{GAIN}.json"
    if path.exists():                      # keep the entries of earlier calls (other arms)
        old = json.loads(path.read_text())
        old.update(out)
        out = old
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)


def gradcheck(order, device):
    """Directional derivatives of the misfit (one shot, 8 ns, float64, LTS) by central
    finite differences against the discrete adjoint, for an eps and a sigma direction."""
    global T_MAX
    T_MAX = 8e-9
    sol = build_solver("lts", order, device, dtype=torch.float64)
    src, rec = antennas(slice(4, 5))
    prop = HybridPropagator(sol, src, rec)
    nt = int(np.ceil(T_MAX / sol.dt)) + 1
    w = torch.tensor(wavelet(nt, sol.dt), dtype=torch.float64, device=device)[None]
    te, ts = (torch.tensor(a, dtype=torch.float64, device=device) for a in true_ground())
    e0, s0 = (torch.tensor(a, dtype=torch.float64, device=device) for a in background())
    with torch.no_grad():
        dobs = prop(te, ts, w)
    J = lambda e, s: 0.5 * float(((prop(e, s, w) - dobs) ** 2).sum())  # noqa: E731
    e = e0.clone().requires_grad_(True)
    s = s0.clone().requires_grad_(True)
    t0 = time.time()
    (0.5 * ((prop(e, s, w) - dobs) ** 2).sum()).backward()
    t_grad = time.time() - t0
    gm = ground_mask().to(device)
    gen = torch.Generator().manual_seed(7)
    out = {"p": sol.lts, "dt": sol.dt, "nt": nt, "gradient_seconds": t_grad, "checks": {}}
    for par, g, scale in (("eps", e.grad, 1.0), ("sigma", s.grad, OMEGA_EPS0)):
        v = torch.randn(SHAPE, generator=gen, dtype=torch.float64).to(device) * gm
        adj = float((g * v).sum())
        rows = []
        for h in (1e-2, 1e-3, 1e-4):
            hh = h * scale
            with torch.no_grad():
                if par == "eps":
                    fd = (J(e0 + hh * v, s0) - J(e0 - hh * v, s0)) / (2 * hh)
                else:
                    fd = (J(e0, s0 + hh * v) - J(e0, s0 - hh * v)) / (2 * hh)
            rows.append({"h": h, "fd": fd, "adjoint": adj, "rel_err": abs(fd - adj) / abs(adj)})
            print(f"{par}: h={h:.0e} FD {fd:.10e} adjoint {adj:.10e} rel. error {rows[-1]['rel_err']:.2e}",
                  flush=True)
        out["checks"][par] = rows
    with open(OUT / f"gradcheck_P{order}.json", "w") as fh:
        json.dump(out, fh, indent=1)


# --------------------------------------------------------------------------- inversion
def run(arm, order, device, epochs=None, tag=None):
    tag = tag or f"{arm}_P{order}_flux{FLUX}_gain{GAIN}"
    sol = build_solver(arm, order, device)
    fw = Forward(sol, device)
    info = sol.summary()
    print(f"[{tag}] p={sol.lts} dt={sol.dt:.3e} nt={fw.nt} Yee {info['grid']} "
          f"yee dofs {info['fdtd_dofs'] / 1e6:.2f} M, dg dofs {info['dg_dofs'] / 1e6:.2f} M", flush=True)
    mute = (mute_mask()[:, None, :] * gain()[None, :, None]).to(device)   # mute and time gain
    dobs = observed_data(device).to(device, torch.float32) * mute
    fs = 1.0 / DT_OUT
    eps0, sig0 = (torch.tensor(a, dtype=torch.float32, device=device) for a in background())
    true_e, true_s = (torch.tensor(a, dtype=torch.float32) for a in true_ground())
    gmask = ground_mask().to(device)
    tm = ground_mask()
    sph = sphere_masks()
    me = torch.zeros(SHAPE, device=device, requires_grad=True)
    ms = torch.zeros(SHAPE, device=device, requires_grad=True)
    hist = {"misfit": [], "eps_err": [], "sig_err": [], "seconds": [], "evals": [],
            "eps_in_eps_sphere": [], "sig_in_sig_sphere": [], "sig_in_eps_sphere": [],
            "eps_in_sig_sphere": [], "eps": [], "sig": []}
    n_eval = [0]

    def models():
        eps = torch.clamp(eps0 + me * gmask, min=1.0)
        sig = torch.clamp(sig0 + OMEGA_EPS0 * ms * gmask, min=0.0)
        return eps, sig

    def err(m, t, m0):
        return float((m - t)[tm].norm() / (t - m0.cpu())[tm].norm())

    def filtered(a, cutoff):
        return a if cutoff is None else apply_filter(a, fs, cutoff)

    def objective(cutoff, backward=False, scale=1.0):
        norm = float((filtered(dobs, cutoff) ** 2).sum())
        total = 0.0
        for i in range(len(SRC_YZ)):
            eps, sig = models()
            d = fw(eps, sig, i) * mute[i:i + 1]
            J = 0.5 * ((filtered(d, cutoff) - filtered(dobs[i:i + 1], cutoff)) ** 2).sum() / norm
            if backward:
                (scale * J).backward()
            total += float(J)
        if backward:
            n_eval[0] += 1
        return total

    def record(J):
        eps, sig = (t.detach().cpu() for t in models())
        hist["misfit"].append(float(J))
        hist["eps_err"].append(err(eps, true_e, eps0))
        hist["sig_err"].append(err(sig, true_s, sig0))
        hist["eps_in_eps_sphere"].append(float(eps[sph[0]].mean()))
        hist["sig_in_sig_sphere"].append(float(sig[sph[1]].mean()))
        hist["sig_in_eps_sphere"].append(float(sig[sph[0]].mean()))
        hist["eps_in_sig_sphere"].append(float(eps[sph[1]].mean()))
        hist["eps"].append(eps.numpy().astype(np.float32))
        hist["sig"].append(sig.numpy().astype(np.float32))
        hist["seconds"].append(time.time() - t_start)
        hist["evals"].append(n_eval[0])
        print(f"  [{tag}] epoch {len(hist['misfit']) - 1:3d}  misfit {J:.4e}  eps err "
              f"{hist['eps_err'][-1]:.3f}  sig err {hist['sig_err'][-1]:.3f}  "
              f"eps@eps-sphere {hist['eps_in_eps_sphere'][-1]:.2f}  "
              f"sig@sig-sphere {1e3 * hist['sig_in_sig_sphere'][-1]:.1f} mS/m  "
              f"evals {n_eval[0]}  t {hist['seconds'][-1]:.0f} s", flush=True)

    def save():
        np.savez(OUT / f"run_{tag}.npz", **{k: np.array(v) for k, v in hist.items()},
                 true_eps=true_e.numpy(), true_sig=true_s.numpy(), mask=tm.numpy())
        meta = {k: hist[k] for k in hist if k not in ("eps", "sig")}
        meta.update(arm=arm, order=order, flux=FLUX, gain=GAIN, p=sol.lts, dt=sol.dt, nt=fw.nt,
                    grid=list(info["grid"]),
                    yee_dofs=info["fdtd_dofs"], dg_dofs=info["dg_dofs"], stages=stages)
        with open(OUT / f"run_{tag}.json", "w") as fh:
            json.dump(meta, fh, indent=1)

    t_start = time.time()
    with torch.no_grad():
        record(objective(None))
    stages = [list(s) for s in STAGES]
    if epochs is not None:
        stages = [[n, j, c] for n, (_, j, c) in zip(epochs, STAGES)]
    for iters, joint, cutoff in stages:
        params = [me, ms] if joint else [me]
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=1, history_size=10,
                                line_search_fn="strong_wolfe")
        with torch.no_grad():
            j_ref = objective(cutoff)
        if joint:
            # gradient balance at the start of the joint stage (diagnostic only)
            me.grad = None
            ms.grad = None
            objective(cutoff, backward=True, scale=1.0 / j_ref)
            hist.setdefault("grad_norms_joint_start", [float(me.grad.norm()), float(ms.grad.norm())])
            print(f"  [{tag}] joint stage: |dJ/dm_e| {hist['grad_norms_joint_start'][0]:.3e}  "
                  f"|dJ/dm_s| {hist['grad_norms_joint_start'][1]:.3e}", flush=True)
            me.grad = None
            ms.grad = None
        for _ in range(iters):
            def closure():
                opt.zero_grad()
                J = objective(cutoff, backward=True, scale=1.0 / j_ref) / j_ref
                return torch.tensor(J)
            opt.step(closure)
            with torch.no_grad():
                record(objective(None))
            save()
    save()
    return tag


# --------------------------------------------------------------------------- report
INK, MUTED = "#1f2328", "#59636e"
ARM_STYLE = {"lts": ("#1d4ed8", "hybrid + LTS"), "global": ("#c2410c", "hybrid, global step")}


def report(order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    runs = {}
    for arm in ("lts", "global"):
        p = OUT / f"run_{arm}_P{order}_flux{FLUX}_gain{GAIN}.npz"
        if p.exists():
            runs[arm] = (np.load(p), json.load(open(OUT / f"run_{arm}_P{order}_flux{FLUX}_gain{GAIN}.json")))
    if not runs:
        print("no runs found")
        return
    first = next(iter(runs.values()))[0]
    true_e, true_s = first["true_eps"], first["true_sig"]
    ix = int(round(SPHERES[0][0] / DX))
    iz = int(round(SPHERES[0][2] / DX))
    ext_v = (0, (SHAPE[1] - 1) * DX, (SHAPE[0] - 1) * DX, 0)
    ext_h = (0, (SHAPE[2] - 1) * DX, (SHAPE[1] - 1) * DX, 0)
    yy = np.arange(SHAPE[1]) * DX
    surf_line = SURFACE(yy, np.full(SHAPE[1], iz * DX))
    cols = [("true", None)] + [(a, r) for a, r in runs.items()]

    def panels(fig, gs, k):
        ims = []
        for c, (name, r) in enumerate(cols):
            for row, (key, true, vmin, vmax, cmap, unit) in enumerate(
                    (("eps", true_e, 4.5, 7.5, "viridis", "eps_r"),
                     ("sig", true_s * 1e3, 0.0, 40.0, "magma", "sigma [mS/m]"))):
                data = true if r is None else r[0][key][min(k, len(r[0][key]) - 1)] * (
                    1e3 if key == "sig" else 1.0)
                for s_i, (cut, ext, lab) in enumerate(
                        ((lambda a: a[:, :, iz], ext_v, f"x-y, z={iz * DX:.2f} m"),
                         (lambda a: a[ix], ext_h, f"y-z, x={ix * DX:.2f} m"))):
                    ax = fig.add_subplot(gs[2 * row + s_i, c])
                    im = ax.imshow(cut(data), extent=ext, vmin=vmin, vmax=vmax, cmap=cmap,
                                   aspect="auto")
                    if s_i == 0:
                        ax.plot(yy, surf_line, color="white", lw=0.8)
                    title = "true" if r is None else ARM_STYLE[name][1]
                    ax.set_title(f"{title}: {unit}\n{lab}", fontsize=7.5, color=INK)
                    ax.tick_params(labelsize=6, colors=MUTED)
                    if r is not None:
                        ims.append((im, r[0], key, cut))
        return ims

    # final-model figure + curves
    fig = plt.figure(figsize=(3.2 * len(cols) + 4.2, 9.5), facecolor="white")
    gs = fig.add_gridspec(4, len(cols) + 1, width_ratios=[1] * len(cols) + [1.4], hspace=0.5,
                          wspace=0.3)
    ims = panels(fig, gs, 10 ** 6)
    ax_m = fig.add_subplot(gs[0:2, len(cols)])
    ax_e = fig.add_subplot(gs[2:4, len(cols)])
    for arm, (r, meta) in runs.items():
        color, label = ARM_STYLE[arm]
        t = np.array(r["seconds"]) / 60.0
        ax_m.semilogy(t, r["misfit"] / r["misfit"][0], color=color, lw=1.5, label=label)
        ax_e.plot(t, r["eps_err"], color=color, lw=1.5, label=f"eps_r, {label}")
        ax_e.plot(t, r["sig_err"], color=color, lw=1.2, ls="--", label=f"sigma, {label}")
    ax_m.set_title("normalised misfit", fontsize=9, color=INK)
    ax_e.set_title("model error ||m - m_true|| / ||m_true - m_0||", fontsize=9, color=INK)
    for a in (ax_m, ax_e):
        a.set_xlabel("wall time [min]", fontsize=8, color=MUTED)
        a.tick_params(labelsize=7, colors=MUTED)
        a.legend(fontsize=6.5, frameon=False)
    fig.suptitle("3D joint eps_r / sigma FWI on rugged ground (final models)", fontsize=11, color=INK)
    fig.savefig(OUT / f"final_P{order}.png", dpi=130)
    plt.close(fig)

    # evolution GIF (epoch by epoch)
    n = max(len(r[0]["misfit"]) for r in runs.values())
    fig = plt.figure(figsize=(3.2 * len(cols) + 4.2, 9.5), facecolor="white")
    gs = fig.add_gridspec(4, len(cols) + 1, width_ratios=[1] * len(cols) + [1.4], hspace=0.5,
                          wspace=0.3)
    ims = panels(fig, gs, 0)
    ax_m = fig.add_subplot(gs[0:2, len(cols)])
    ax_e = fig.add_subplot(gs[2:4, len(cols)])
    lines = []
    for arm, (r, _) in runs.items():
        color, label = ARM_STYLE[arm]
        k = np.arange(len(r["misfit"]))
        ax_m.semilogy(k, r["misfit"] / r["misfit"][0], color=color, lw=1.5, label=label)
        ax_e.plot(k, r["eps_err"], color=color, lw=1.5, label=f"eps_r, {label}")
        ax_e.plot(k, r["sig_err"], color=color, lw=1.2, ls="--", label=f"sigma, {label}")
    for a in (ax_m, ax_e):
        lines.append(a.axvline(0, color=MUTED, lw=0.8))
        a.set_xlabel("epoch (L-BFGS iteration)", fontsize=8, color=MUTED)
        a.tick_params(labelsize=7, colors=MUTED)
        a.legend(fontsize=6.5, frameon=False)
    title = fig.suptitle("", fontsize=11, color=INK)

    def update(k):
        for im, r, key, cut in ims:
            arr = r[key]
            im.set_data(cut(arr[min(k, len(arr) - 1)] * (1e3 if key == "sig" else 1.0)))
        for ln in lines:
            ln.set_xdata([k, k])
        title.set_text(f"3D joint eps_r / sigma FWI on rugged ground - epoch {k}")
        return []

    FuncAnimation(fig, update, frames=n, interval=500).save(
        OUT / f"evolution_P{order}.gif", writer=PillowWriter(fps=2), dpi=75)
    plt.close(fig)

    # report.md
    lines = [f"# 3D joint eps_r / sigma FWI on rugged ground (P{order} band, Yee 2/3 cm)", ""]
    for name in (f"diagnose_P{order}_flux{FLUX}_gain{GAIN}.json", f"gradcheck_P{order}.json"):
        p = OUT / name
        if p.exists():
            lines += [f"## {name}", "", "```json", p.read_text(), "```", ""]
    lines += ["## Runs", "",
              "| arm | p | dt [s] | epochs | gradient evals | misfit (final/initial) | eps_r error | "
              "sigma error | eps_r in eps sphere | sigma in sigma sphere [mS/m] | sigma in eps sphere "
              "[mS/m] | eps_r in sigma sphere | wall time [min] |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, (r, meta) in runs.items():
        lines.append(
            f"| {ARM_STYLE[arm][1]} | {meta['p']} | {meta['dt']:.3e} | {len(r['misfit']) - 1} | "
            f"{int(r['evals'][-1])} | {r['misfit'][-1] / r['misfit'][0]:.3f} | {r['eps_err'][-1]:.3f} | "
            f"{r['sig_err'][-1]:.3f} | {r['eps_in_eps_sphere'][-1]:.2f} | "
            f"{1e3 * r['sig_in_sig_sphere'][-1]:.1f} | {1e3 * r['sig_in_eps_sphere'][-1]:.1f} | "
            f"{r['eps_in_sig_sphere'][-1]:.2f} | {r['seconds'][-1] / 60:.1f} |")
    if len(runs) == 2:
        a, b = runs["lts"][0], runs["global"][0]
        k = min(len(a["eps"]), len(b["eps"])) - 1
        tm = a["mask"]
        de = np.linalg.norm((a["eps"][k] - b["eps"][k])[tm]) / np.linalg.norm((b["eps"][k] - b["eps"][0])[tm])
        ds = np.linalg.norm((a["sig"][k] - b["sig"][k])[tm]) / np.linalg.norm((b["sig"][k] - b["sig"][0])[tm])
        lines += ["", f"LTS vs global final models (epoch {k}): eps_r update difference {de:.3f}, "
                  f"sigma update difference {ds:.3f} (relative to the global arm's update)."]
    lines += ["", f"Figures: `final_P{order}.png`, `evolution_P{order}.gif`."]
    (OUT / f"report_P{order}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("lts", "global"))
    p.add_argument("--order", type=int, default=2, choices=(1, 2))
    p.add_argument("--epochs", type=int, nargs=2, default=None)
    p.add_argument("--observed", action="store_true")
    p.add_argument("--diagnose", action="store_true")
    p.add_argument("--gradcheck", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--flux", type=float, default=FLUX,
                   help="interior DG flux weight (0 = central; LTS currently needs 0)")
    p.add_argument("--diagnose-arms", nargs="+", default=["global", "lts"])
    p.add_argument("--gain", type=float, default=GAIN, help="time-gain power (0: none)")
    args = p.parse_args()
    globals()["FLUX"] = args.flux
    globals()["GAIN"] = args.gain
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    if args.observed:
        observed_data(device)
        observed_data(device, background_model=True)
    if args.diagnose:
        diagnose(args.order, device, args.diagnose_arms)
    if args.gradcheck:
        gradcheck(args.order, device)
    if args.arm:
        run(args.arm, args.order, device, args.epochs)
    if args.report:
        report(args.order)


if __name__ == "__main__":
    main()
