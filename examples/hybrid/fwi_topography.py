"""Dual-parameter (eps_r, sigma) FWI on undulating ground with the hybrid solver.

* True model: irregular surface, ground eps_r = 5, sigma = 2 mS/m, a
  high-permittivity disc (eps_r = 6.5) and a conductive disc (sigma = 20 mS/m),
  radius 0.25 m (contrasts kept below the cycle-skipping limit of the
  reflection that crosses them).
* Observed data: an independent solver - native DeepGPR FDTD at dx/4 with
  cell-averaged eps/sigma at the surface and the discs sampled on that grid
  (no inverse crime: a different grid, a different surface treatment and a
  different time step than either inversion solver).
* Acquisition: 14 sources and 33 receivers about 0.1 m above the ground
  (multi-offset surface GPR), 300 MHz Ricker.
* A known reflector (eps_r 5 -> 9 at 2 m depth) is in the true and in the
  initial model: its reflection crosses the anomalies twice, so the data carry
  travel-time and amplitude information about their interiors, not only the
  edge reflections.
* Inversion: the ground model on the 2.5 cm model grid, starting from the
  background (layer included, no anomalies); L-BFGS with strong-Wolfe line
  search on normalised parameters (eps_r in units of 1, sigma in 10 mS/m);
  stage 1: eps_r only, data low-passed to 180 MHz; stage 2: eps_r and sigma,
  full band. Normalised L2 misfit over offsets >= 0.3 m (near-source traces
  muted); bounds eps_r >= 1, sigma >= 0. One epoch = one L-BFGS iteration.
* Solver: ``--solver hybrid_dgtd`` (body-fitted DGTD band + Yee at dx/3, discrete
  adjoint) or ``--solver fdtd`` (native DeepGPR with the staircase surface).

Every epoch is a frame of ``results/fwi_<case>_<solver>.gif``.

    python examples/hybrid/fwi_topography.py --solver hybrid_dgtd --epochs 60
    python examples/hybrid/fwi_topography.py --solver fdtd --epochs 60
    python examples/hybrid/fwi_topography.py --compare      # GIF of both runs side by side
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

import torch.nn.functional as F  # noqa: E402

import DeepGPR  # noqa: E402
from DeepGPR import apply_filter  # noqa: E402
from DeepGPR.hybrid import Surface, make_propagator, resample  # noqa: E402

OUT = ROOT / "examples" / "hybrid" / "results"
DX = 0.025
DEPTH, WIDTH = 2.4, 4.0
PML = 12
FREQ, T0, T_MAX = 3.0e8, 1.2 / 3.0e8, 35e-9
EPS_BG, SIG_BG = 5.0, 0.002
LAYER_DEPTH, EPS_LAYER = 2.0, 9.0   # known deep reflector (also in the initial model)
ELEV_Y = np.arange(0.0, 4.01, 0.5)
ELEV_Z = np.array([0.00, 0.10, 0.22, 0.12, -0.06, -0.15, -0.05, 0.10, 0.05])
CASES = {
    "topo_dual": dict(surface=Surface.from_elevation(ELEV_Y, ELEV_Z, datum=0.62),
                      discs=[(1.40, 1.35, 0.25, 6.5, None), (1.40, 2.70, 0.25, None, 0.02)],
                      label="irregular topography, eps and sigma anomalies"),
    "topo_eps": dict(surface=Surface.from_elevation(ELEV_Y, ELEV_Z, datum=0.62),
                     discs=[(1.40, 2.0, 0.25, 6.5, None)],
                     label="irregular topography, eps anomaly"),
    "flat_eps": dict(surface=Surface.flat(0.62), discs=[(1.40, 2.0, 0.25, 6.5, None)],
                     label="flat ground, eps anomaly"),
}
BAND = (0.25, 1.05)       # DGTD band (physical depth) [m]
SRC_Y = np.arange(0.40, 3.61, 0.25)
REC_Y = np.arange(0.40, 3.61, 0.10)
HEIGHT = 0.10
MUTE = 0.3              # receivers closer than this to the source are muted [m]
SCALE_EPS, SCALE_SIG = 1.0, 0.01   # normalisation of the inversion parameters


def wavelet(nt, dt):
    t = (np.arange(nt) + 0.5) * dt - T0
    a = (np.pi * FREQ * t) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a))


def background(dx):
    """Ground eps_r = 5 over a known reflector (eps_r = 9) at 2 m depth."""
    nx, ny = int(round(DEPTH / dx)), int(round(WIDTH / dx))
    eps = torch.full((nx, ny), EPS_BG, dtype=torch.float64)
    x = torch.arange(nx, dtype=torch.float64)[:, None] * dx
    eps = torch.where(x + 0.5 * dx > LAYER_DEPTH,
                      torch.clamp((x + 0.5 * dx - LAYER_DEPTH) / dx, 0, 1) * (EPS_LAYER - EPS_BG)
                      + EPS_BG, eps).expand(nx, ny).clone()
    return eps, torch.full((nx, ny), SIG_BG, dtype=torch.float64)


def true_ground(dx, case):
    eps, sig = background(dx)
    nx, ny = eps.shape
    x = torch.arange(nx, dtype=torch.float64)[:, None] * dx
    y = torch.arange(ny, dtype=torch.float64)[None, :] * dx
    for xc, yc, r, e, s in case["discs"]:
        inside = (x - xc) ** 2 + (y - yc) ** 2 <= r * r
        if e is not None:
            eps[inside] = e
        if s is not None:
            sig[inside] = s
    return eps, sig


def antennas(case):
    """Sources/receivers ~0.1 m above the ground, on 2.5 cm nodes (all solvers)."""
    on = lambda ys: np.round(np.stack([case["surface"](ys) - HEIGHT, ys], 1) / DX) * DX  # noqa: E731
    src = on(SRC_Y)[:, None, :]
    rec = np.broadcast_to(on(REC_Y)[None], (len(SRC_Y), len(REC_Y), 2)).copy()
    return src, rec


def observed_data(case, device):
    """Independent reference: native FDTD at dx/4 with cell-averaged surface."""
    k = 4
    dx = DX / k
    eps, sig = true_ground(dx, case)
    nx, ny = eps.shape
    x = np.arange(nx)[:, None] * dx
    frac = np.zeros((nx, ny))
    for s in range(8):
        y = (np.arange(ny) + (s + 0.5) / 8 - 0.5) * dx
        frac += np.clip((x + 0.5 * dx - case["surface"](y)[None]) / dx, 0.0, 1.0)
    f = torch.as_tensor(frac / 8)
    eps, sig = f * eps + (1 - f), f * sig
    dt = 0.95 * DeepGPR.max_stable_time_step(dx, 10, 10, 1)
    nt = int(round(T_MAX / dt))
    src, rec = antennas(case)
    sl = torch.tensor(np.concatenate([np.rint(src / dx), np.zeros((len(src), 1, 1))], 2),
                      dtype=torch.int, device=device)
    rl = torch.tensor(np.concatenate([np.rint(rec / dx), np.zeros(rec.shape[:2] + (1,))], 2),
                      dtype=torch.int, device=device)
    d = DeepGPR.compute(device=device, dx=dx, dt=dt,
                        source_amplitudes=wavelet(nt, dt).float().reshape(1, -1, 1).to(device),
                        source_location=sl, receiver_location=rl, eps_r=eps.float(),
                        sigma=sig.float(), pmlthick=PML * k, save_wavefield_history=False)[-1]
    return d.double(), (torch.arange(nt, dtype=torch.float64) + 1) * dt


def build(solver, case, device):
    eps0, sig0 = background(DX)
    src, rec = antennas(case)
    if solver == "hybrid_dgtd":
        prop = make_propagator("hybrid_dgtd", eps0, sig0, DX, src, rec, surface=case["surface"],
                               pmlthick=PML, dg_band=BAND, dg_order=1, mesh="delaunay",
                               fdtd_refinement=3, device=device, dtype=torch.float32)
    else:
        prop = make_propagator("fdtd", eps0, sig0, DX, src, rec, surface=case["surface"],
                               pmlthick=PML, device=device)
    return prop, eps0, sig0


def ground_mask(case):
    nx, ny = int(round(DEPTH / DX)), int(round(WIDTH / DX))
    x = np.arange(nx)[:, None] * DX
    return x >= case["surface"](np.arange(ny) * DX)[None]


def run(args, device):
    case = CASES[args.case]
    dobs_fine, t_fine = observed_data(case, device)
    prop, eps0, sig0 = build(args.solver, case, device)
    nt = int(round(T_MAX / prop.dt))
    t_axis = prop.time_axis(nt)
    dobs = resample(dobs_fine, t_fine, t_axis).to(torch.float32)
    # Mute receivers within 0.3 m of the source: they record the (log-singular,
    # grid-dependent) near field of the point source.
    src, rec = antennas(case)
    keep = torch.as_tensor(np.abs(rec[:, :, 1] - src[:, :, 1]) >= MUTE, device=device)
    mute = keep[:, None, :].to(torch.float32)
    dobs = dobs * mute
    w = wavelet(nt, prop.dt)
    true_e, true_s = true_ground(DX, case)
    mask = torch.as_tensor(ground_mask(case))
    history = {"loss": [], "eps_err": [], "sig_err": [], "seconds": [], "evals": [],
               "eps": [], "sig": []}
    # Model error relative to the anomalies: ||m - m_true|| / ||m_true - m_0|| in the ground.
    err = lambda m, t, m0: float((m - t)[mask].norm() / (t - m0)[mask].norm())  # noqa: E731
    fs = 1.0 / prop.dt
    ground = mask.to(device)
    pair_x = (ground[1:, :] & ground[:-1, :]).float()
    pair_y = (ground[:, 1:] & ground[:, :-1]).float()

    def tv(m):
        """Isotropic TV over pairs of ground nodes (air values are not parameters)."""
        gx = F.pad((m[1:, :] - m[:-1, :]) * pair_x, (0, 0, 0, 1))
        gy = F.pad((m[:, 1:] - m[:, :-1]) * pair_y, (0, 1))
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8).sum()

    # Normalised parameters: eps = eps0 + SCALE_EPS u_e, sigma = sig0 + SCALE_SIG u_s.
    E0, S0 = eps0.to(device), sig0.to(device)
    u_e = torch.zeros_like(E0, requires_grad=True)
    u_s = torch.zeros_like(S0, requires_grad=True)
    evals = [0]

    def models(joint):
        eps = (E0 + SCALE_EPS * u_e).clamp(min=1.0)
        sig = (S0 + SCALE_SIG * u_s).clamp(min=0.0) if joint else S0
        return eps, sig

    def objective(joint, cutoff):
        eps, sig = models(joint)
        d = prop(eps, sig, w[None]).float() * mute
        obs = dobs
        if cutoff is not None:
            d, obs = apply_filter(d, fs, cutoff), apply_filter(dobs, fs, cutoff)
        loss = ((d - obs) ** 2).sum() / (obs ** 2).sum()
        if args.tv > 0:
            loss = loss + args.tv * (tv(u_e) + (tv(u_s) if joint else 0.0))
        evals[0] += 1
        return loss

    peak = 0
    stages = [(False, args.low_cut, args.eps_only), (True, None, args.epochs - args.eps_only)]
    epoch = 0
    for joint, cutoff, iters in stages:
        joint = joint and args.params == "both"
        params = [u_e, u_s] if joint else [u_e]
        if args.optimizer == "lbfgs":
            opt = torch.optim.LBFGS(params, lr=1.0, history_size=10, max_iter=1,
                                    line_search_fn="strong_wolfe")
        else:
            opt = torch.optim.Adam(params, lr=args.lr)
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            n0 = evals[0]

            def closure():
                opt.zero_grad()
                loss = objective(joint, cutoff)
                loss.backward()
                return loss

            loss = opt.step(closure)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak = max(peak, torch.cuda.max_memory_allocated(device))
            secs = time.perf_counter() - t0
            with torch.no_grad():
                eps_now, sig_now = (t.detach().cpu() for t in models(True))
            history["loss"].append(float(loss))
            history["eps_err"].append(err(eps_now, true_e, eps0))
            history["sig_err"].append(err(sig_now, true_s, sig0))
            history["seconds"].append(secs)
            history["evals"].append(evals[0] - n0)
            history["eps"].append(eps_now.numpy().astype(np.float32))
            history["sig"].append(sig_now.numpy().astype(np.float32))
            print(f"{args.solver} epoch {epoch:3d} loss {float(loss):.4e} eps err "
                  f"{history['eps_err'][-1]:.3f} sigma err {history['sig_err'][-1]:.3f} "
                  f"({secs:.1f} s, {evals[0] - n0} evaluations)", flush=True)
            epoch += 1
    tag = f"{args.case}_{args.solver}"
    np.savez_compressed(OUT / f"fwi_{tag}.npz", loss=history["loss"], eps=np.stack(history["eps"]),
                        sig=np.stack(history["sig"]), eps_err=history["eps_err"],
                        sig_err=history["sig_err"], seconds=history["seconds"],
                        evals=history["evals"],
                        true_eps=true_e.numpy(), true_sig=true_s.numpy(),
                        mask=mask.numpy(), eps_only=args.eps_only, low_epochs=args.eps_only,
                        low_cut=args.low_cut, joint=int(args.params == "both"))
    summary = dict(case=args.case, solver=args.solver, params=args.params, epochs=args.epochs,
                   dt=prop.dt, nt=nt,
                   final_loss=history["loss"][-1], final_eps_err=history["eps_err"][-1],
                   final_sig_err=history["sig_err"][-1],
                   mean_epoch_seconds=float(np.mean(history["seconds"])),
                   evaluations=int(sum(history["evals"])),
                   seconds_per_evaluation=float(sum(history["seconds"]) / sum(history["evals"])),
                   optimizer=args.optimizer, peak_gpu_mb=peak / 2**20)
    with open(OUT / f"fwi_{tag}.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    gif([tag], OUT / f"fwi_{tag}.gif", case)


# ---------------------------------------------------------------- figures
INK, INK2 = "#0b0b0b", "#52514e"
SOLVER_LABEL = {"hybrid_dgtd": "hybrid DGTD-FDTD", "fdtd": "Yee-FDTD (staircase surface)"}


def gif(tags, path, case):
    """Per-iteration GIF: eps_r, eps_r error, sigma (true + each run), misfit and
    model-error curves."""
    runs = [np.load(OUT / f"fwi_{t}.npz") for t in tags]
    labels = [SOLVER_LABEL.get(t.split("_", 2)[-1], t) for t in tags]
    true_e, true_s, mask = runs[0]["true_eps"], runs[0]["true_sig"], runs[0]["mask"]
    nep = min(len(r["loss"]) for r in runs)
    ys = np.linspace(0, WIDTH, 800)
    ext = [-DX / 2, WIDTH - DX / 2, DEPTH - DX / 2, -DX / 2]
    ncol = 1 + len(runs)
    joint = int(runs[0]["joint"]) if "joint" in runs[0] else 1
    rows = (
        ("eps", lambda r, k: r["eps"][k], true_e, 4.5, 7.0, "viridis", "eps_r"),
        ("eps_err", lambda r, k: r["eps"][k] - true_e, None, -1.5, 1.5, "RdBu_r",
         "eps_r - true eps_r"),
        ("sig", lambda r, k: r["sig"][k] * 1e3, true_s * 1e3, 0.0, 22.0, "magma", "sigma (mS/m)"),
    )[: 3 if joint else 2]
    fig = plt.figure(figsize=(3.4 * ncol + 1.0, 3.0 * len(rows) + 2.6), constrained_layout=True)
    gs = fig.add_gridspec(len(rows) + 1, 2 * ncol, height_ratios=[1] * len(rows) + [0.9])
    show = lambda a: np.where(mask, a, np.nan)  # noqa: E731
    panels = []
    for i, (key, getter, true, vmin, vmax, cmap, unit) in enumerate(rows):
        axes = [fig.add_subplot(gs[i, 2 * c:2 * c + 2]) for c in range(ncol)]
        if true is not None:
            im = axes[0].imshow(show(true), extent=ext, vmin=vmin, vmax=vmax, cmap=cmap)
            axes[0].set_title(f"true {unit}", fontsize=9, color=INK)
        else:
            axes[0].axis("off")
            axes[0].text(0.5, 0.5, "error of the\ninverted eps_r\n(blue: too low,\nred: too high)",
                         ha="center", va="center", fontsize=9, color=INK2,
                         transform=axes[0].transAxes)
        for c, r in enumerate(runs):
            im = axes[c + 1].imshow(show(getter(r, 0)), extent=ext, vmin=vmin, vmax=vmax,
                                    cmap=cmap)
            panels.append((im, getter, r, axes[c + 1], labels[c], unit))
        for ax in axes:
            if ax.axison:
                ax.plot(ys, case["surface"](ys), color="#f0efec", lw=0.8)
                ax.set_xlim(0, WIDTH)
                ax.set_ylim(DEPTH, 0)
                ax.tick_params(labelsize=7, colors=INK2)
        fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01).ax.tick_params(labelsize=7)
    colors = ["#2a78d6", "#eb6834"]
    ax_m = fig.add_subplot(gs[len(rows), :ncol])
    ax_e = fig.add_subplot(gs[len(rows), ncol:])
    stage = int(runs[0]["eps_only"])
    markers = []
    for c, r in enumerate(runs):
        ax_m.semilogy(r["loss"][:nep], color=colors[c], lw=1.5, label=labels[c])
        ax_e.plot(r["eps_err"][:nep], color=colors[c], lw=1.5, label=f"{labels[c]}: eps_r")
        if joint:
            ax_e.plot(r["sig_err"][:nep], color=colors[c], lw=1.0, ls="--",
                      label=f"{labels[c]}: sigma")
        markers.append((ax_m.plot([0], [r["loss"][0]], "o", color=colors[c], ms=5)[0],
                        ax_e.plot([0], [r["eps_err"][0]], "o", color=colors[c], ms=5)[0], r))
    for ax, title in ((ax_m, "data misfit (normalised L2)"),
                      (ax_e, "model error ||m - m_true|| / ||m_true - m_0||")):
        ax.axvline(stage - 0.5, color=INK2, lw=0.6, ls=":")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("iteration", fontsize=8, color=INK2)
        ax.tick_params(labelsize=7, colors=INK2)
        ax.grid(color="#e6e5e1", lw=0.5)
        ax.legend(fontsize=7)
    title = fig.suptitle("", fontsize=10, color=INK)

    def update(k):
        for im, getter, r, ax, label, unit in panels:
            im.set_data(show(getter(r, k)))
            ax.set_title(f"{label}\niteration {k}", fontsize=8, color=INK)
        for mk_m, mk_e, r in markers:
            mk_m.set_data([k], [r["loss"][k]])
            mk_e.set_data([k], [r["eps_err"][k]])
        joint = int(runs[0]["joint"]) if "joint" in runs[0] else 1
        phase = ("stage 1: eps_r only, data < "
                 f"{float(runs[0]['low_cut']) / 1e6:.0f} MHz") if k < stage else \
            ("stage 2: eps_r + sigma, full band" if joint else "stage 2: eps_r, full band")
        title.set_text(f"FWI, {case['label']} - iteration {k} ({phase})")

    FuncAnimation(fig, update, frames=nep).save(path, writer=PillowWriter(fps=4), dpi=72)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="topo_dual", choices=list(CASES))
    p.add_argument("--solver", default="hybrid_dgtd", choices=["hybrid_dgtd", "fdtd"])
    p.add_argument("--epochs", type=int, default=50, help="total iterations")
    p.add_argument("--eps-only", type=int, default=15,
                   help="iterations of stage 1 (eps_r only, low-passed data)")
    p.add_argument("--low-cut", type=float, default=1.8e8,
                   help="low-pass cut-off [Hz] of stage 1")
    p.add_argument("--optimizer", default="lbfgs", choices=["lbfgs", "adam"])
    p.add_argument("--lr", type=float, default=0.05, help="Adam step (normalised units)")
    p.add_argument("--tv", type=float, default=0.0, help="TV weight (normalised units)")
    p.add_argument("--params", default="both", choices=["both", "eps"],
                   help="invert eps_r and sigma (stage 2) or eps_r only")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--gif-only", action="store_true", help="re-render the GIF of --solver")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.compare:
        tags = [f"{args.case}_hybrid_dgtd", f"{args.case}_fdtd"]
        gif(tags, OUT / f"fwi_{args.case}_compare.gif", CASES[args.case])
        return
    if args.gif_only:
        tag = f"{args.case}_{args.solver}"
        gif([tag], OUT / f"fwi_{tag}.gif", CASES[args.case])
        return
    run(args, device)


if __name__ == "__main__":
    main()
