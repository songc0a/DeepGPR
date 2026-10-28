"""3D dual-parameter (eps_r, sigma) FWI with the subgridded FDTD and its discrete adjoint.

Model (model grid h = 1 cm, 0.9 m cube, soil eps_r 4, 2 mS/m) with two
anomalies at 0.5 m depth: A (eps_r 12, 20 mS/m) and B (conductivity only,
30 mS/m). 9 shots (3 x 3 Ez dipoles) and a grid of receivers on one plane.
Two presets:

* ``resolved`` (default): radius 6 cm, 18 cm apart, Ricker 400 MHz, antennas on
  x = 0.20 m, 6 x 6 receivers, small total-variation term;
* ``strong``: as ``resolved`` with radius 5 cm and strong contrasts (A: eps_r 25,
  50 mS/m; B: 200 mS/m), where the main grid's block averages misrepresent the
  scattering most;
* ``small``: radius 4 cm, 12 cm apart, Ricker 250 MHz, antennas on x = 0.12 m,
  7 x 7 receivers, no regularisation (anomalies far below the resolution: both
  inversions fit the data with the same smooth, weak perturbations).

Observed data. The response of the anomalies is computed on a uniform grid at
the model spacing (true minus background model), so neither inversion commits
an inverse crime in the anomalies. Raw uniform-grid data are dominated by the
3 cm main grid's background modelling error (numerical dispersion of the
direct wave; for ``small``: misfit 7e-3 against an anomaly signal of 4e-5,
measured), which neither inversion can remove because it arises outside the
refined region. The observed data of each inversion are therefore that
inversion's own background prediction plus the uniform-grid anomaly response
(``--data fine``: the raw uniform-grid data instead). The misfit is normalised
by the energy of the anomaly response (1 for the background model). Receivers
within 10 cm of their shot's source are muted (grid-dependent near field).

Inversions (L-BFGS, both parameters from the background, updates restricted to
the region of interest = the subgrid box, identical settings):

* ``coarse``:  main grid only (3 cm, block-averaged model);
* ``subgrid``: the main grid with a ratio-3 subgrid over the region of interest
  (1 cm, the model itself), gradients by the exact discrete adjoint.

Reported (``examples/subgrid/results/fwi3d/``): misfit history, the modelling
floor (misfit of the true model on each solver), model errors (point-wise and
after Gaussian smoothing over 3 / 6 cm), the integrated anomalies
``sum(m - m0)`` around A and B relative to the true ones, time per gradient,
peak memory; ``fwi3d_<preset>.png`` shows slices through the anomalies.

    python examples/subgrid/fwi3d_subgrid.py [--preset resolved|strong|small] [--iters 30]
                                             [--start background|smooth] [--param objects]

``--param objects`` inverts only the values of the anomalies (shapes known): the
recovered eps_r/sigma show the bias that each solver's modelling error causes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from DeepGPR.subgrid import Subgrid, SubgridMaxwell3D, fine_source_samples
from DeepGPR.subgrid.adjoint import SubgridPropagator

OUT = Path(__file__).resolve().parent / "results" / "fwi3d"
H = 0.01
SHAPE = (91, 91, 91)
EPS0, SIG0 = 4.0, 0.002
SCALE_EPS, SCALE_SIG = 1.0, 0.01
DTYPE = torch.float32

PRESETS = {
    "resolved": dict(f0=4e8, nt=260, radius=0.06, ya=0.39, yb=0.57, plane=0.20,
                     rec=np.arange(0.255, 0.61, 0.07), tv=3e-6, eps_a=12.0, sig_a=0.02, sig_b=0.03,
                     box=((0.38, 0.62), (0.27, 0.69), (0.30, 0.60))),
    "strong": dict(f0=4e8, nt=260, radius=0.05, ya=0.39, yb=0.57, plane=0.20,
                   rec=np.arange(0.255, 0.61, 0.07), tv=3e-6, eps_a=25.0, sig_a=0.05, sig_b=0.2,
                   box=((0.38, 0.62), (0.27, 0.69), (0.30, 0.60))),
    "small": dict(f0=2.5e8, nt=300, radius=0.04, ya=0.42, yb=0.54, plane=0.12,
                  rec=np.arange(0.24, 0.67, 0.07), tv=0.0, eps_a=12.0, sig_a=0.02, sig_b=0.03,
                  box=((0.36, 0.66), (0.30, 0.66), (0.30, 0.60))),
}
CFG = dict(PRESETS["resolved"])


def true_model():
    eps = np.full(SHAPE, EPS0)
    sig = np.full(SHAPE, SIG0)
    X, Y, Z = np.meshgrid(*[np.arange(n) * H for n in SHAPE], indexing="ij")
    r2 = CFG["radius"] ** 2
    a = ((X - 0.5) ** 2 + (Y - CFG["ya"]) ** 2 + (Z - 0.45) ** 2) <= r2
    b = ((X - 0.5) ** 2 + (Y - CFG["yb"]) ** 2 + (Z - 0.45) ** 2) <= r2
    eps[a], sig[a] = CFG["eps_a"], CFG["sig_a"]
    sig[b] = CFG["sig_b"]
    return eps, sig


def acquisition():
    ys = zs = (0.30, 0.45, 0.60)
    x = CFG["plane"]
    src = [[[x, y, z]] for y in ys for z in zs]
    g = CFG["rec"]
    rec = [[[x, y, z] for y in g for z in g]] * len(src)
    return np.array(src), np.array(rec)


def offset_mute(src, rec, min_offset=0.1):
    """Receivers closer than ``min_offset`` to their shot's source are muted (the point
    source's near field depends on the grid)."""
    off = np.linalg.norm(rec - src, axis=-1)            # (nshot, nrec)
    return torch.as_tensor(off >= min_offset, dtype=DTYPE)[:, None, :]


def ricker(nt, dt):
    f0 = CFG["f0"]
    t = np.arange(nt) * dt - 1.2 / f0
    a = (np.pi * f0 * t) ** 2
    return torch.as_tensor((1 - 2 * a) * np.exp(-a))


def roi_mask():
    m = np.zeros(SHAPE, dtype=bool)
    sl = tuple(slice(int(round(b[0] / H)) + 1, int(round(b[1] / H))) for b in CFG["box"])
    m[sl] = True
    return m


def observed(src, rec, dt, targets=True):
    eps, sig = true_model() if targets else (np.full(SHAPE, EPS0), np.full(SHAPE, SIG0))
    sol = SubgridMaxwell3D(torch.as_tensor(eps), torch.as_tensor(sig), H, coarsen=1,
                           pmlthick=30, dt=dt / 3, dipole_length=3 * H, nshot=len(src),
                           device="cuda", dtype=DTYPE)
    nt = CFG["nt"]
    w = ricker(nt, dt)
    wf = fine_source_samples(w[None], 3)[0]
    wf = torch.cat([torch.zeros(1, dtype=wf.dtype), wf, torch.zeros(3, dtype=wf.dtype)])[:3 * nt]
    t0 = time.perf_counter()
    d = sol.run(wf, src, rec)["receivers"][:, 2::3]
    return d, time.perf_counter() - t0


def tv_term(u, mask):
    """Isotropic TV of ``u`` over pairs of region-of-interest nodes."""
    m = mask
    gx = (u[1:] - u[:-1]) * (m[1:] & m[:-1])
    gy = (u[:, 1:] - u[:, :-1]) * (m[:, 1:] & m[:, :-1])
    gz = (u[:, :, 1:] - u[:, :, :-1]) * (m[:, :, 1:] & m[:, :, :-1])
    g2 = gx[:, :-1, :-1] ** 2 + gy[:-1, :, :-1] ** 2 + gz[:-1, :-1, :] ** 2
    return torch.sqrt(g2 + 1e-8).sum()


def metrics(eps, sig):
    from scipy.ndimage import gaussian_filter
    te, ts = true_model()
    mask = roi_mask()
    out = {}
    out["eps_err"] = float(np.linalg.norm((eps - te)[mask]) / np.linalg.norm((te - EPS0)[mask]))
    out["sig_err"] = float(np.linalg.norm((sig - ts)[mask]) / np.linalg.norm((ts - SIG0)[mask]))
    for s in (3, 6):
        for name, a, b, m0 in (("eps", te, eps, EPS0), ("sig", ts, sig, SIG0)):
            A, B = gaussian_filter(a, s), gaussian_filter(b, s)
            out[f"{name}_err_smooth{s}cm"] = float(np.linalg.norm((B - A)[mask])
                                                   / np.linalg.norm((A - m0)[mask]))
    X, Y, Z = np.meshgrid(*[np.arange(n) * H for n in SHAPE], indexing="ij")
    for name, y in (("A", CFG["ya"]), ("B", CFG["yb"])):
        near = ((X - 0.5) ** 2 + (Y - y) ** 2 + (Z - 0.45) ** 2) <= (1.5 * CFG["radius"]) ** 2
        for p, a, b, m0 in (("eps", te, eps, EPS0), ("sig", ts, sig, SIG0)):
            true_int = (a - m0)[near].sum()
            out[f"{name}_int_d{p}"] = float((b - m0)[near].sum() / true_int) if abs(true_int) > 0 \
                else float((b - m0)[near].sum() * H ** 3)
    return out


def start_model(kind_start):
    """Initial model: the background, or the true model smoothed with a 6 cm Gaussian."""
    if kind_start == "background":
        return np.full(SHAPE, EPS0), np.full(SHAPE, SIG0)
    from scipy.ndimage import gaussian_filter
    te, ts = true_model()
    return gaussian_filter(te, 6.0), gaussian_filter(ts, 6.0)


def invert(kind, src, rec, anomaly, raw, iters, mute, start="background"):
    E0 = torch.full(SHAPE, EPS0, dtype=DTYPE, device="cuda")
    S0 = torch.full(SHAPE, SIG0, dtype=DTYPE, device="cuda")
    Ei, Si = (torch.as_tensor(a, dtype=DTYPE, device="cuda") for a in start_model(start))
    sgs = [Subgrid(CFG["box"], ratio=3)] if kind == "subgrid" else []
    sol = SubgridMaxwell3D(E0, S0, H, coarsen=3, subgrids=sgs, pmlthick=10, nshot=len(src),
                           device="cuda", dtype=DTYPE)
    prop = SubgridPropagator(sol, src, rec)
    w = ricker(CFG["nt"], sol.dt).to("cuda", DTYPE)
    mask = torch.as_tensor(roi_mask(), device="cuda")
    mute = mute.to("cuda")
    with torch.no_grad():
        bg = prop(E0, S0, w) * mute
        et, st = (torch.as_tensor(a, dtype=DTYPE, device="cuda") for a in true_model())
        resp_true = prop(et, st, w) * mute - bg
    dobs = (raw if raw is not None else bg + anomaly) * mute
    norm = float(((dobs - bg) ** 2).sum())
    floor = float(((bg + resp_true - dobs) ** 2).sum() / norm)
    u_e = torch.zeros(SHAPE, dtype=DTYPE, device="cuda", requires_grad=True)
    u_s = torch.zeros(SHAPE, dtype=DTYPE, device="cuda", requires_grad=True)

    def models():
        eps = (Ei + SCALE_EPS * u_e * mask).clamp(min=1.0)
        sig = (Si + SCALE_SIG * u_s * mask).clamp(min=0.0)
        return eps, sig

    with torch.no_grad():
        e_i, s_i = models()
        start_misfit = float(((prop(e_i, s_i, w) * mute - dobs) ** 2).sum() / norm)
    opt = torch.optim.LBFGS([u_e, u_s], lr=1.0, history_size=10, max_iter=1,
                            line_search_fn="strong_wolfe")
    hist = {"loss": [], "misfit": [], "seconds": [], "evals": []}
    evals, last = [0], {}
    torch.cuda.reset_peak_memory_stats()

    def closure():
        opt.zero_grad()
        eps, sig = models()
        d = prop(eps, sig, w) * mute
        misfit = ((d - dobs) ** 2).sum() / norm
        loss = misfit
        if CFG["tv"] > 0:
            loss = loss + CFG["tv"] * (tv_term(u_e, mask) + tv_term(u_s, mask))
        loss.backward()
        evals[0] += 1
        last["misfit"] = float(misfit)
        return loss

    print(f"{kind}: modelling floor (true-model misfit) {floor:.4f}", flush=True)
    for it in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n0 = evals[0]
        loss = opt.step(closure)
        torch.cuda.synchronize()
        hist["loss"].append(float(loss))
        hist["misfit"].append(last["misfit"])
        hist["seconds"].append(time.perf_counter() - t0)
        hist["evals"].append(evals[0] - n0)
        if (it + 1) % 5 == 0 or it == 0:
            print(f"  {kind} it {it + 1:2d}: loss {hist['loss'][-1]:.4f} misfit "
                  f"{hist['misfit'][-1]:.4f} ({hist['seconds'][-1]:.1f} s, "
                  f"{hist['evals'][-1]} evals)", flush=True)
    with torch.no_grad():
        eps, sig = models()
        final = float(((prop(eps, sig, w) * mute - dobs) ** 2).sum() / norm)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    e, s = eps.double().cpu().numpy(), sig.double().cpu().numpy()
    summ = dict(modelling_floor=floor, start_misfit=start_misfit, final_misfit=final,
                start=start, seconds=float(sum(hist["seconds"])),
                evaluations=int(sum(hist["evals"])),
                seconds_per_evaluation=float(sum(hist["seconds"]) / max(1, sum(hist["evals"]))),
                peak_GB=peak, history=hist, solver=sol.summary(), **metrics(e, s))
    return summ, e, s


def invert_objects(kind, src, rec, anomaly, iters, mute):
    """Parametric dual-parameter inversion: the anomaly shapes are known, their values
    (eps_r and sigma of A, sigma of B) are not; L-BFGS on the three values from the
    background. Any bias of the recovered values is the forward modelling error."""
    E0 = torch.full(SHAPE, EPS0, dtype=DTYPE, device="cuda")
    S0 = torch.full(SHAPE, SIG0, dtype=DTYPE, device="cuda")
    sgs = [Subgrid(CFG["box"], ratio=3)] if kind == "subgrid" else []
    sol = SubgridMaxwell3D(E0, S0, H, coarsen=3, subgrids=sgs, pmlthick=10, nshot=len(src),
                           device="cuda", dtype=DTYPE)
    prop = SubgridPropagator(sol, src, rec)
    w = ricker(CFG["nt"], sol.dt).to("cuda", DTYPE)
    mute = mute.to("cuda")
    X, Y, Z = np.meshgrid(*[np.arange(n) * H for n in SHAPE], indexing="ij")
    r2 = CFG["radius"] ** 2
    ia = torch.as_tensor(((X - 0.5) ** 2 + (Y - CFG["ya"]) ** 2 + (Z - 0.45) ** 2) <= r2,
                         dtype=DTYPE, device="cuda")
    ib = torch.as_tensor(((X - 0.5) ** 2 + (Y - CFG["yb"]) ** 2 + (Z - 0.45) ** 2) <= r2,
                         dtype=DTYPE, device="cuda")
    with torch.no_grad():
        bg = prop(E0, S0, w) * mute
    dobs = bg + anomaly
    norm = float((anomaly ** 2).sum())
    theta = torch.zeros(3, dtype=torch.float64, device="cuda", requires_grad=True)
    scale = torch.tensor([1.0, 0.01, 0.01], dtype=torch.float64, device="cuda")
    opt = torch.optim.LBFGS([theta], lr=1.0, history_size=10, max_iter=1,
                            line_search_fn="strong_wolfe")
    hist = []

    def values():
        v = theta * scale
        return EPS0 + v[0], SIG0 + v[1], SIG0 + v[2]

    def closure():
        opt.zero_grad()
        ea, sa, sb = values()
        eps = (E0 + (ea - EPS0).to(DTYPE) * ia).clamp(min=1.0)
        sig = (S0 + (sa - SIG0).to(DTYPE) * ia + (sb - SIG0).to(DTYPE) * ib).clamp(min=0.0)
        loss = ((prop(eps, sig, w) * mute - dobs) ** 2).sum() / norm
        loss.backward()
        return loss

    t0 = time.perf_counter()
    for it in range(iters):
        loss = opt.step(closure)
        ea, sa, sb = (float(v) for v in values())
        hist.append(dict(misfit=float(loss), eps_a=ea, sig_a=sa, sig_b=sb))
    secs = time.perf_counter() - t0
    ea, sa, sb = (float(v) for v in values())
    out = dict(eps_a=ea, sig_a=sa, sig_b=sb, misfit=hist[-1]["misfit"], seconds=secs, history=hist,
               rel_err_eps_a=(ea - CFG["eps_a"]) / (CFG["eps_a"] - EPS0),
               rel_err_sig_a=(sa - CFG["sig_a"]) / (CFG["sig_a"] - SIG0),
               rel_err_sig_b=(sb - CFG["sig_b"]) / (CFG["sig_b"] - SIG0))
    print(f"{kind} (objects): eps_A {ea:.2f} (true {CFG['eps_a']}), sigma_A {1e3 * sa:.1f} mS/m "
          f"(true {1e3 * CFG['sig_a']:.0f}), sigma_B {1e3 * sb:.1f} mS/m (true "
          f"{1e3 * CFG['sig_b']:.0f}); misfit {out['misfit']:.4f}, {secs:.0f} s", flush=True)
    del sol, prop
    torch.cuda.empty_cache()
    return out


def figure(models, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    true_e, true_s = true_model()
    cols = [("true", true_e, true_s)] + [(k, e, s) for k, (e, s) in models.items()]
    fig, ax = plt.subplots(2, len(cols), figsize=(3.4 * len(cols), 6.2), constrained_layout=True)
    ix = int(round(0.5 / H))
    ext = [0, SHAPE[2] * H, SHAPE[1] * H, 0]
    for j, (name, e, s) in enumerate(cols):
        a = ax[0, j].imshow(e[ix], vmin=EPS0 - 1, vmax=CFG["eps_a"], cmap="viridis", extent=ext)
        b = ax[1, j].imshow(1e3 * s[ix], vmin=0, vmax=1e3 * max(CFG["sig_a"], CFG["sig_b"]),
                            cmap="magma", extent=ext)
        ax[0, j].set_title(f"{name}: eps_r (x = 0.5 m)")
        ax[1, j].set_title(f"{name}: sigma [mS/m]")
        for r in range(2):
            ax[r, j].set_xlabel("z [m]")
            ax[r, j].set_ylabel("y [m]")
            (y0, y1), (z0, z1) = CFG["box"][1], CFG["box"][2]
            ax[r, j].plot([z0, z1, z1, z0, z0], [y0, y0, y1, y1, y0], "w--", lw=0.8)
    fig.colorbar(a, ax=ax[0, :], shrink=0.8)
    fig.colorbar(b, ax=ax[1, :], shrink=0.8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--preset", choices=sorted(PRESETS), default="resolved")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--data", choices=("transplant", "fine"), default="transplant")
    p.add_argument("--start", choices=("background", "smooth"), default="background",
                   help="initial model: background or the true model smoothed over 6 cm")
    p.add_argument("--param", choices=("voxels", "objects"), default="voxels",
                   help="voxel FWI, or the values of the anomalies of known shape")
    args = p.parse_args()
    CFG.clear()
    CFG.update(PRESETS[args.preset])
    OUT.mkdir(parents=True, exist_ok=True)
    src, rec = acquisition()
    probe = SubgridMaxwell3D(torch.full(SHAPE, EPS0), torch.full(SHAPE, SIG0), H, coarsen=3,
                             pmlthick=10, cuda_kernels=False)
    d_true, t_obs = observed(src, rec, probe.dt)
    d_bg, _ = observed(src, rec, probe.dt, targets=False)
    anomaly = (d_true - d_bg).cuda()
    print(f"uniform 1 cm grid: {t_obs:.1f} s per run", flush=True)
    res = {"observed_seconds": t_obs, "data": args.data, "preset": args.preset,
           "config": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in CFG.items()}}
    models = {}
    mute = offset_mute(src, rec)
    raw = d_true.cuda() if args.data == "fine" else None
    if args.param == "objects":
        for kind in ("coarse", "subgrid"):
            res[kind] = invert_objects(kind, src, rec, anomaly * mute.cuda(), args.iters, mute)
        (OUT / f"fwi3d_{args.preset}_objects.json").write_text(json.dumps(res, indent=1))
        print("wrote", OUT)
        return
    for kind in ("coarse", "subgrid"):
        summ, e, s = invert(kind, src, rec, anomaly, raw, args.iters, mute, args.start)
        res[kind] = summ
        models[kind] = (e, s)
        print(f"{kind}: floor {summ['modelling_floor']:.4f}, final misfit {summ['final_misfit']:.4f}, "
              f"eps err {summ['eps_err']:.3f} (smooth 6 cm {summ['eps_err_smooth6cm']:.3f}), "
              f"sigma err {summ['sig_err']:.3f} (smooth 6 cm {summ['sig_err_smooth6cm']:.3f}), "
              f"A: int deps {summ['A_int_deps']:.2f} dsig {summ['A_int_dsig']:.2f}, "
              f"B: int dsig {summ['B_int_dsig']:.2f}; {summ['seconds_per_evaluation']:.2f} s/gradient, "
              f"peak {summ['peak_GB']:.2f} GB", flush=True)
    tag = args.preset + ("" if args.data == "transplant" else "_rawdata") + (
        "" if args.start == "background" else "_smoothstart")
    (OUT / f"fwi3d_{tag}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(OUT / f"fwi3d_{tag}_models.npz", **{f"{k}_{q}": v[i] for k, v in models.items()
                                                            for i, q in enumerate(("eps", "sig"))})
    figure(models, OUT / f"fwi3d_{tag}.png")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
