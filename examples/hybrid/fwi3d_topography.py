"""Small 3D dual-parameter (eps_r, sigma) FWI on undulating ground.

* True model (2 cm grid, 0.5 m deep, 0.64 m x 0.64 m): irregular surface
  around 0.16 m (gridded elevation samples, PCHIP), ground eps_r = 5,
  sigma = 3 mS/m, a high-permittivity sphere (eps_r 7) and a conductive
  sphere (sigma 30 mS/m), radius 10 cm, and a known deeper reflector
  (eps_r 8 below 0.46 m, also in the initial model).
* Observed data: an independent solver - native DeepGPR at dx/5 (4 mm) with
  cell-averaged materials at the surface (different grid, surface treatment
  and time step than either inversion solver; about 0.25 % from the converged
  solution in the Phase 4 benchmark). The ground model (spheres, reflector) is
  the 2 cm model upsampled, so the truth is representable on the inversion
  grid; sampling it directly on 4 mm shifts the reflector by half a model cell
  and dominated the late-time data error of every solver (14 %).
* Acquisition: 4 Ez sources (2 x 2) and 7 x 7 Ez receivers 4 cm below the
  top of the model (at least 8 cm above the ground), 400 MHz Ricker;
  traces within 0.15 m of the source muted.
* Inversion: ground model on the 2 cm grid from the background (reflector
  included); L-BFGS (strong Wolfe) on normalised parameters; stage 1 eps_r
  only with data low-passed to 300 MHz, stage 2 eps_r and sigma, full band.
  Within a stage the objective is divided by its value at the start of the
  stage (L-BFGS is scale-invariant except for its first step and its absolute
  tolerances, which stalled on the raw normalised misfit of order 1e-4).
  One epoch = one L-BFGS iteration.
* Solvers: ``hybrid_dgtd`` (P1 DGTD band through the surface + Yee at
  ``2 cm / --refine``, discrete adjoint with the fused CUDA kernels; with
  ``--refine 3`` the shots are processed one at a time) or ``fdtd`` (native
  DeepGPR 3D with the staircase surface, on ``2 cm / --refine``).

Every epoch is a frame of ``results/fwi3d_<run>.gif``; ``--compare`` renders
the runs side by side.

    python examples/hybrid/fwi3d_topography.py --diagnose      # modelling error vs signal
    python examples/hybrid/fwi3d_topography.py --solver hybrid_dgtd --refine 3
    python examples/hybrid/fwi3d_topography.py --solver fdtd
    python examples/hybrid/fwi3d_topography.py --solver fdtd --refine 3
    python examples/hybrid/fwi3d_topography.py --compare
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

import DeepGPR  # noqa: E402
from DeepGPR import apply_filter  # noqa: E402
from DeepGPR.hybrid import Surface3D, make_propagator, resample  # noqa: E402

OUT = ROOT / "examples" / "hybrid" / "results"
DX = 0.02
SHAPE = (25, 32, 32)
PML = 8
FREQ, T_MAX = 4.0e8, 11e-9
OBS_REFINE = 5
T0 = 1.2 / FREQ
EPS_BG, SIG_BG = 5.0, 0.003
LAYER_X, EPS_LAYER = 0.46, 8.0
SPHERES = [(0.34, 0.20, 0.31, 0.10, 7.0, None), (0.34, 0.44, 0.31, 0.10, None, 0.03)]
_EY = np.linspace(0, 0.62, 5)
_EZ = np.linspace(0, 0.62, 5)
_ELEV = np.array([[0.00, 0.02, 0.03, 0.01, -0.01],
                  [0.02, 0.04, 0.01, -0.02, -0.03],
                  [0.01, 0.00, -0.03, -0.04, -0.01],
                  [-0.02, -0.03, -0.01, 0.02, 0.03],
                  [-0.03, -0.01, 0.02, 0.04, 0.02]])
SURFACE = Surface3D.from_grid(_EY, _EZ, 0.16 - _ELEV)
ANT_X = 0.04
BAND = (0.08, 0.24)
# y on 2 cm nodes, z on Ez nodes (odd multiples of 1 cm): nodes of the dx/3 grid too.
SRC_YZ = [(0.16, 0.17), (0.16, 0.45), (0.44, 0.17), (0.44, 0.45)]
REC_YZ = [(round(y, 2), round(z, 2)) for y in np.arange(0.08, 0.57, 0.08)
          for z in np.arange(0.07, 0.56, 0.08)]
MUTE = 0.15
STAGES = ((6, False, 3.0e8), (12, True, None))     # (iterations, joint, low-pass cutoff)


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


def antennas():
    src = np.array([[[ANT_X, y, z]] for y, z in SRC_YZ])
    rec = np.array([[[ANT_X, y, z] for y, z in REC_YZ]] * len(SRC_YZ))
    return src, rec


def mute_mask():
    src, rec = antennas()
    off = np.linalg.norm(rec[:, :, 1:] - src[:, :, 1:], axis=2)
    return torch.as_tensor(off >= MUTE, dtype=torch.float32)


def time_axis(dt_out=2e-11):
    return torch.arange(1, int(T_MAX / dt_out) + 1, dtype=torch.float64) * dt_out


def observed_data(device, background_model=False):
    path = OUT / ("fwi3d_observed_bg.npz" if background_model else "fwi3d_observed.npz")
    if path.exists():
        z = np.load(path)
        return torch.as_tensor(z["data"])
    h = DX / OBS_REFINE
    # Ground model of the 2 cm inversion grid upsampled to the fine grid (a model
    # pixel is the cell centred on its node), so the true model is representable
    # on the inversion grid; only the surface treatment differs (cell averages).
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
    out = DeepGPR.compute(device=device, dx=h, dt=dt, source_amplitudes=w.reshape(1, -1, 1),
                          source_location=idx(src), receiver_location=idx(rec),
                          eps_r=torch.tensor(eps, dtype=torch.float32),
                          sigma=torch.tensor(sig, dtype=torch.float32), pmlthick=pml,
                          save_wavefield_history=False)
    t = (torch.arange(nt, dtype=torch.float64) + 1) * dt
    data = resample(out[-1].double().cpu(), t, time_axis()).numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(path, data=data)
    return torch.as_tensor(data)


def ground_mask():
    n = SHAPE
    Y, Z = np.meshgrid(np.arange(n[1]) * DX, np.arange(n[2]) * DX, indexing="ij")
    return torch.as_tensor((np.arange(n[0]) * DX)[:, None, None] >= SURFACE(Y, Z)[None])


def build(solver, device, refine=1, shots=None):
    """Propagator for ``solver`` on Yee spacing ``DX / refine``; ``shots`` selects
    a subset of the sources (one propagator per shot keeps hybrid memory low)."""
    eps0, sig0 = (torch.tensor(a, dtype=torch.float32) for a in background())
    src, rec = antennas()
    if shots is not None:
        src, rec = src[shots], rec[shots]
    if solver == "fdtd":
        n = [m * refine + 2 * PML * refine for m in SHAPE]
        dt = 0.95 * DeepGPR.max_stable_time_step(DX / refine, *n)
        # On the refined grid the full wavefield history would need ~100 GB: use
        # DeepGPR's temporally sub-sampled history (interval floor(1/(4 f_max dt))).
        extra = {"model_gradient_sampling_interval": "auto"} if refine > 1 else {}
        return make_propagator("fdtd", eps0, sig0, DX, src, rec, surface=SURFACE, pmlthick=PML,
                               dt=dt, device=device, fdtd_refinement=refine, **extra)
    return make_propagator("hybrid_dgtd", eps0, sig0, DX, src, rec, surface=SURFACE,
                           pmlthick=PML, device=device, dtype=torch.float32, dg_band=BAND,
                           fdtd_refinement=refine)


def shot_propagators(solver, device, refine):
    """One batched propagator, or (hybrid with ``refine > 1``) one per shot sharing
    the solver set-up (gradients accumulated shot by shot)."""
    if solver == "hybrid_dgtd" and refine > 1:
        from DeepGPR.hybrid import HybridPropagator
        first = build(solver, device, refine, shots=slice(0, 1))
        src, rec = antennas()
        props = [first] + [HybridPropagator(first.solver, src[i:i + 1], rec[i:i + 1])
                           for i in range(1, len(SRC_YZ))]
        return props, [slice(i, i + 1) for i in range(len(SRC_YZ))]
    return [build(solver, device, refine)], [slice(None)]


def run(args, device):
    solver = args.solver
    props, groups = shot_propagators(solver, device, args.refine)
    prop = props[0]
    t_axis = time_axis()
    nt = int(np.ceil(T_MAX / prop.dt)) + 1
    t_prop = prop.time_axis(nt)
    w = torch.tensor(wavelet(nt, prop.dt), dtype=torch.float32, device=device)

    def simulate(eps, sig, i):
        d = props[i](eps, sig, w[None].expand(1, -1)).to(torch.float32)
        return resample(d, t_prop.to(device), t_axis.to(device))

    if args.inverse_crime:
        # Observed data from the inversion solver itself at the true model.
        te, ts = (torch.tensor(a, dtype=torch.float32, device=device) for a in true_ground())
        with torch.no_grad():
            dobs_all = torch.cat([simulate(te, ts, i) for i in range(len(props))]).cpu()
    else:
        dobs_all = observed_data(device).to(torch.float32)
    mute = mute_mask()[:, None, :].to(device)
    dobs = dobs_all.to(device) * mute
    fs = 1.0 / float(t_axis[1] - t_axis[0])
    eps0, sig0 = (torch.tensor(a, dtype=torch.float32, device=device) for a in background())
    true_e, true_s = (torch.tensor(a, dtype=torch.float32) for a in true_ground())
    gmask = ground_mask().to(device)
    E_SCALE, S_SCALE = 1.0, 0.01
    me = torch.zeros(SHAPE, device=device, requires_grad=True)
    ms = torch.zeros(SHAPE, device=device, requires_grad=True)
    tm = ground_mask()
    err = lambda m, t, m0: float((m - t)[tm].norm() / (t - m0.cpu())[tm].norm())  # noqa: E731
    history = {"misfit": [], "eps_err": [], "sig_err": [], "eps": [], "sig": [], "seconds": []}

    def models():
        eps = torch.clamp(eps0 + E_SCALE * me * gmask, min=1.0)
        sig = torch.clamp(sig0 + S_SCALE * ms * gmask, min=0.0)
        return eps, sig

    def filtered(a, cutoff):
        return a if cutoff is None else apply_filter(a, fs, cutoff)

    def objective(cutoff, backward=False, scale=1.0):
        """Normalised misfit summed over the shot groups; with ``backward`` the
        gradient of ``scale * J`` is accumulated group by group."""
        norm = float((filtered(dobs, cutoff) ** 2).sum())
        total = 0.0
        for i, g in enumerate(groups):
            eps, sig = models()
            d = simulate(eps, sig, i) * mute[g]
            obs = filtered(dobs[g], cutoff)
            J = 0.5 * ((filtered(d, cutoff) - obs) ** 2).sum() / norm
            if backward:
                (scale * J).backward()
                if len(groups) > 1 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            total += float(J)
        return total

    def record(J):
        eps, sig = (t.detach().cpu() for t in models())
        history["misfit"].append(float(J))
        history["eps_err"].append(err(eps, true_e, eps0))
        history["sig_err"].append(err(sig, true_s, sig0))
        history["eps"].append(eps.numpy().astype(np.float32))
        history["sig"].append(sig.numpy().astype(np.float32))
        history["seconds"].append(time.time() - t_start)
        print(f"  epoch {len(history['misfit']) - 1:3d}  misfit {J:.4e}  eps err "
              f"{history['eps_err'][-1]:.3f}  sig err {history['sig_err'][-1]:.3f}  "
              f"t {history['seconds'][-1]:.0f} s", flush=True)

    t_start = time.time()
    with torch.no_grad():
        record(objective(None))
    stages = STAGES if args.epochs is None else tuple(
        (n, j, c) for n, (_, j, c) in zip(args.epochs, STAGES))
    for iters, joint, cutoff in stages:
        params = [me, ms] if joint else [me]
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=1, history_size=10,
                                line_search_fn="strong_wolfe")
        with torch.no_grad():
            j_ref = objective(cutoff)
        for _ in range(iters):
            def closure():
                opt.zero_grad()
                J = objective(cutoff, backward=True, scale=1.0 / j_ref) / j_ref
                return torch.tensor(J)
            opt.step(closure)
            with torch.no_grad():
                record(objective(None))
    tag = solver + (f"_r{args.refine}" if args.refine > 1 else "") + (
        "_crime" if args.inverse_crime else "")
    path = OUT / f"fwi3d_{tag}.npz"
    np.savez(path, **{k: np.array(v) for k, v in history.items()},
             true_eps=true_e.numpy(), true_sig=true_s.numpy(), mask=tm.numpy())
    with open(OUT / f"fwi3d_{tag}.json", "w") as fh:
        json.dump({k: history[k] for k in ("misfit", "eps_err", "sig_err", "seconds")}, fh,
                  indent=1)
    return path


def diagnose(device):
    """Modelling error of each solver at the true model against the anomaly signal."""
    mute = mute_mask()[:, None, :]
    obs_t = observed_data(device) * mute
    obs_b = observed_data(device, background_model=True) * mute
    n = lambda a: float(a.norm())  # noqa: E731
    out = {"anomaly_signal": n(obs_t - obs_b) / n(obs_t)}
    print(f"anomaly signal in the observed data ||d(m) - d(m0)|| / ||d(m)||: "
          f"{out['anomaly_signal']:.4f}")
    for solver, kw in (("fdtd", {}), ("hybrid_dgtd", {}), ("hybrid_dgtd_r3", {"fdtd_refinement": 3})):
        if solver.startswith("hybrid"):
            eps0, sig0 = (torch.tensor(a, dtype=torch.float32) for a in background())
            src, rec = antennas()
            prop = make_propagator("hybrid_dgtd", eps0, sig0, DX, src, rec, surface=SURFACE,
                                   pmlthick=PML, device=device, dtype=torch.float32,
                                   dg_band=BAND, **kw)
        else:
            prop = build(solver, device)
        nt = int(np.ceil(T_MAX / prop.dt)) + 1
        w = torch.tensor(wavelet(nt, prop.dt), dtype=torch.float32, device=device)
        res = {}
        for tag, model in (("true", true_ground()), ("background", background())):
            e, s_ = (torch.tensor(a, dtype=torch.float32, device=device) for a in model)
            with torch.no_grad():
                d = prop(e, s_, w[None]).to(torch.float32)
            res[tag] = resample(d, prop.time_axis(nt).to(device), time_axis().to(device)).cpu() * mute
        out[solver] = {"modelling_error": n(res["true"] - obs_t) / n(obs_t),
                       "own_anomaly_signal": n(res["true"] - res["background"]) / n(res["true"])}
        print(f"{solver:15s} modelling error at the true model {out[solver]['modelling_error']:.4f}"
              f"  (its own anomaly signal {out[solver]['own_anomaly_signal']:.4f})", flush=True)
        del prop
        torch.cuda.empty_cache()
    with open(OUT / "fwi3d_diagnose.json", "w") as fh:
        json.dump(out, fh, indent=1)


INK, MUTED = "#1f2328", "#59636e"
COLORS = {"fdtd": "#c2410c", "fdtd_r3": "#0f766e", "hybrid_dgtd": "#7c3aed",
          "hybrid_dgtd_r3": "#1d4ed8"}
LABELS = {"fdtd": "staircase FDTD, 2 cm (native)", "fdtd_r3": "staircase FDTD, 2/3 cm (native)",
          "hybrid_dgtd": "hybrid, Yee 2 cm", "hybrid_dgtd_r3": "hybrid, Yee 2/3 cm"}


def gif(solvers, path):
    runs = [np.load(OUT / f"fwi3d_{s}.npz") for s in solvers]
    true_e, true_s = runs[0]["true_eps"], runs[0]["true_sig"]
    n = max(len(r["misfit"]) for r in runs)
    iz = int(round(0.31 / DX))
    ix = int(round(0.34 / DX))
    surf_line = SURFACE(np.arange(SHAPE[1]) * DX, np.full(SHAPE[1], iz * DX))
    rows = [("eps", true_e, 4.5, 7.0, "viridis", "eps_r"),
            ("sig", true_s * 1e3, 0.0, 22.0, "magma", "sigma (mS/m)")]
    ncol = 1 + len(runs)
    fig = plt.figure(figsize=(3.1 * ncol + 3.6, 9.2), facecolor="white")
    gs = fig.add_gridspec(4, ncol + 1, width_ratios=[1] * ncol + [1.25], hspace=0.45, wspace=0.3)
    ims = []
    ext_v = (0, (SHAPE[1] - 1) * DX, (SHAPE[0] - 1) * DX, 0)
    ext_h = (0, (SHAPE[2] - 1) * DX, (SHAPE[1] - 1) * DX, 0)
    for r_i, (key, true, vmin, vmax, cmap, unit) in enumerate(rows):
        for c in range(ncol):
            for s_i, (cut, ext, lab) in enumerate(((lambda a: a[:, :, iz], ext_v, "x-y slice z=0.31 m"),
                                                    (lambda a: a[ix], ext_h, "y-z slice x=0.34 m"))):
                ax = fig.add_subplot(gs[2 * r_i + s_i, c])
                if c == 0:
                    data = true
                    title = f"true {unit}"
                else:
                    data = runs[c - 1][key][0] * (1e3 if key == "sig" else 1.0)
                    title = LABELS[solvers[c - 1]]
                im = ax.imshow(cut(data), extent=ext, vmin=vmin, vmax=vmax, cmap=cmap,
                               aspect="auto")
                if s_i == 0:
                    ax.plot(np.arange(SHAPE[1]) * DX, surf_line, color="white", lw=0.8)
                ax.set_title(f"{title}\n{lab}", fontsize=7.5, color=INK)
                ax.tick_params(labelsize=6, colors=MUTED)
                if c > 0:
                    ims.append((im, c - 1, key, cut))
    ax_m = fig.add_subplot(gs[0:2, ncol])
    ax_e = fig.add_subplot(gs[2:4, ncol])
    lines = []
    for r, s in zip(runs, solvers):
        k = np.arange(len(r["misfit"]))
        ax_m.semilogy(k, r["misfit"] / r["misfit"][0], color=COLORS[s], lw=1.5, label=LABELS[s])
        ax_e.plot(k, r["eps_err"], color=COLORS[s], lw=1.5, label=f"eps_r ({LABELS[s]})")
        ax_e.plot(k, r["sig_err"], color=COLORS[s], lw=1.2, ls="--", label=f"sigma ({LABELS[s]})")
        lines.append(ax_m.axvline(0, color=MUTED, lw=0.8))
        lines.append(ax_e.axvline(0, color=MUTED, lw=0.8))
    ax_m.set_title("normalised misfit", fontsize=9, color=INK)
    ax_e.set_title("model error ||m - m_true|| / ||m_true - m_0||", fontsize=9, color=INK)
    for a in (ax_m, ax_e):
        a.set_xlabel("epoch (L-BFGS iteration)", fontsize=8, color=MUTED)
        a.tick_params(labelsize=7, colors=MUTED)
        a.legend(fontsize=6, frameon=False)
    title = fig.suptitle("", fontsize=11, color=INK)

    def update(k):
        for im, ri, key, cut in ims:
            arr = runs[ri][key]
            kk = min(k, len(arr) - 1)
            im.set_data(cut(arr[kk] * (1e3 if key == "sig" else 1.0)))
        for ln in lines:
            ln.set_xdata([k, k])
        title.set_text(f"3D FWI on undulating ground - epoch {k}")
        return []

    anim = FuncAnimation(fig, update, frames=n, interval=500)
    anim.save(path, writer=PillowWriter(fps=2), dpi=80)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solver", choices=("hybrid_dgtd", "fdtd"), default="hybrid_dgtd")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--gif-only", action="store_true")
    p.add_argument("--refine", type=int, default=1,
                   help="Yee spacing DX / refine (hybrid: fdtd_refinement; fdtd: finer grid)")
    p.add_argument("--epochs", type=int, nargs=2, default=None,
                   help="iterations of stage 1 and stage 2 (default 6 12)")
    p.add_argument("--diagnose", action="store_true",
                   help="modelling error of each solver vs the anomaly signal")
    p.add_argument("--inverse-crime", action="store_true",
                   help="observed data from the inversion solver itself (setup check)")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.compare:
        tags = [t for t in ("fdtd", "fdtd_r3", "hybrid_dgtd_r3") if (OUT / f"fwi3d_{t}.npz").exists()]
        gif(tags, OUT / "fwi3d_compare.gif")
        return
    if args.diagnose:
        diagnose(device)
        return
    if not args.gif_only:
        run(args, device)
    if not args.inverse_crime:
        tag = args.solver + (f"_r{args.refine}" if args.refine > 1 else "")
        gif([tag], OUT / f"fwi3d_{tag}.gif")


if __name__ == "__main__":
    main()
