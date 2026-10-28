"""DGTD rugged surface + FDTD subsurface with a Huygens subgrid + dual-parameter FWI (3D).

Model (model grid h = 1 cm, 0.9 m cube): air over a rugged surface
(``d = 0.15 + 0.03 sin(2 pi y / 0.6) cos(2 pi z / 0.6)`` m), soil eps_r 6,
3 mS/m, two anomalies 12 cm apart at 0.585 m depth: A (radius 4 cm, eps_r 16,
20 mS/m) and B (radius 4 cm, 40 mS/m). 4 shots and 5 x 5 receivers 6 cm above
the mean surface (in air), Ricker 300 MHz, ``Ez``.

Solvers (all on the same model-grid parameters):

* ``hybrid``:            DGTD band around the surface + 3 cm Yee grid below/above;
* ``hybrid+subgrid``:    the same with a ratio-3 subgrid (1 cm) over the anomaly zone;
* ``staircase+subgrid``: 3 cm Yee everywhere (the surface as a 3 cm staircase of
  block averages, no DG band) with the same subgrid;
* reference:             the same DG band, a uniform 1 cm Yee grid with the model
  itself (``HybridMaxwell3D`` at h, DG elements as in ``hybrid``).

Observed data: the reference's anomaly response (true minus background model)
added to each solver's own background prediction (as in ``fwi3d_subgrid.py``).
Inversion: L-BFGS on eps_r and sigma in the region below the DG band, from the
background (``--start background``) or the true model smoothed over 6 cm
(``--start smooth``), total-variation term, 20 iterations by default.

    python examples/subgrid/fwi3d_dgtd_subgrid.py [--iters 20 --start smooth] [--param objects]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from DeepGPR.hybrid import HybridMaxwell3D, Surface3D
from DeepGPR.subgrid import Subgrid, SubgridMaxwell3D
from DeepGPR.subgrid.adjoint import SubgridPropagator
from DeepGPR.subgrid.hybrid import HybridSubgridMaxwell3D

OUT = Path(__file__).resolve().parent / "results" / "fwi3d_dgtd"
H = 0.01
SHAPE = (91, 91, 91)
EPS0, SIG0 = 6.0, 0.003
SURF = Surface3D.sine(0.15, 0.03, 0.6, 0.6)
BAND = (0.09, 0.24)
BOX = ((0.48, 0.69), (0.30, 0.60), (0.30, 0.60))
SUB = dict(is_os_sep=2, pml_cells=4)
F0, NT = 3e8, 1200
RADIUS, YA, YB, XC, ZC = 0.04, 0.39, 0.51, 0.585, 0.45
SCALE_EPS, SCALE_SIG, TV = 1.0, 0.01, 3e-6
DTYPE = torch.float32


def surface_depth(y, z):
    return SURF(np.asarray(y), np.asarray(z))


def true_model(targets=True):
    eps = np.full(SHAPE, EPS0)
    sig = np.full(SHAPE, SIG0)
    if targets:
        X, Y, Z = np.meshgrid(*[np.arange(n) * H for n in SHAPE], indexing="ij")
        a = ((X - XC) ** 2 + (Y - YA) ** 2 + (Z - ZC) ** 2) <= RADIUS ** 2
        b = ((X - XC) ** 2 + (Y - YB) ** 2 + (Z - ZC) ** 2) <= RADIUS ** 2
        eps[a], sig[a] = 16.0, 0.02
        sig[b] = 0.04
    return eps, sig


def with_air(eps, sig):
    """Model with air above the surface (for the solver without DG band)."""
    Y, Z = np.meshgrid(np.arange(SHAPE[1]) * H, np.arange(SHAPE[2]) * H, indexing="ij")
    d = surface_depth(Y, Z)
    x = (np.arange(SHAPE[0]) * H)[:, None, None]
    air = torch.as_tensor(x < d[None], device=eps.device)
    return torch.where(air, torch.ones_like(eps), eps), torch.where(air, torch.zeros_like(sig), sig)


def acquisition():
    ys = zs = (0.36, 0.54)
    src = [[[0.06, y, z]] for y in ys for z in zs]
    g = np.arange(0.27, 0.64, 0.09)
    rec = [[[0.06, y, z] for y in g for z in g]] * len(src)
    return np.array(src), np.array(rec)


def offset_mute(src, rec, min_offset=0.1):
    off = np.linalg.norm(rec - src, axis=-1)
    return torch.as_tensor(off >= min_offset, dtype=DTYPE)[:, None, :]


def ricker(nt, dt):
    t = np.arange(nt) * dt - 1.2 / F0
    a = (np.pi * F0 * t) ** 2
    return torch.as_tensor((1 - 2 * a) * np.exp(-a))


def update_mask():
    m = np.zeros(SHAPE, dtype=bool)
    m[int(round(BAND[1] / H)) + 3:-6, 6:-6, 6:-6] = True
    return m


def metrics(eps, sig):
    from scipy.ndimage import gaussian_filter
    te, ts = true_model()
    mask = update_mask()
    out = {"eps_err": float(np.linalg.norm((eps - te)[mask]) / np.linalg.norm((te - EPS0)[mask])),
           "sig_err": float(np.linalg.norm((sig - ts)[mask]) / np.linalg.norm((ts - SIG0)[mask]))}
    for name, a, b, m0 in (("eps", te, eps, EPS0), ("sig", ts, sig, SIG0)):
        A, B = gaussian_filter(a, 6), gaussian_filter(b, 6)
        out[f"{name}_err_smooth6cm"] = float(np.linalg.norm((B - A)[mask]) / np.linalg.norm((A - m0)[mask]))
    X, Y, Z = np.meshgrid(*[np.arange(n) * H for n in SHAPE], indexing="ij")
    for name, y in (("A", YA), ("B", YB)):
        near = ((X - XC) ** 2 + (Y - y) ** 2 + (Z - ZC) ** 2) <= (1.5 * RADIUS) ** 2
        for p, a, b, m0 in (("eps", te, eps, EPS0), ("sig", ts, sig, SIG0)):
            ti = (a - m0)[near].sum()
            out[f"{name}_int_d{p}"] = float((b - m0)[near].sum() / ti) if abs(ti) > 0 else 0.0
    return out


def build(kind, nshot, dt=None):
    e = torch.full(SHAPE, EPS0, dtype=DTYPE)
    s = torch.full(SHAPE, SIG0, dtype=DTYPE)
    if kind == "staircase+subgrid":
        e, s = with_air(e, s)
        return SubgridMaxwell3D(e, s, H, coarsen=3, subgrids=[Subgrid(BOX, **SUB)], pmlthick=10,
                                nshot=nshot, dt=dt, device="cuda", dtype=DTYPE)
    sgs = [Subgrid(BOX, **SUB)] if kind == "hybrid+subgrid" else []
    return HybridSubgridMaxwell3D(e, s, H, coarsen=3, surface=SURF, dg_band=BAND, subgrids=sgs,
                                  pmlthick=10, nshot=nshot, device="cuda", dtype=DTYPE, dt=dt)


def reference_response(src, rec, dt):
    """Anomaly response of the reference (DG band + uniform 1 cm Yee), 3 cm dipole."""
    out = []
    for targets in (True, False):
        e, s = (torch.as_tensor(a) for a in true_model(targets))
        sol = HybridMaxwell3D(e, s, H, surface=SURF, dg_band=BAND, pmlthick=30, dg_h=3 * H,
                              dt=dt, nshot=len(src), device="cuda", dtype=DTYPE)
        t0 = time.perf_counter()
        out.append(3.0 * sol.run(ricker(NT, dt), src, rec)["receivers"])
        secs = time.perf_counter() - t0
        del sol
        torch.cuda.empty_cache()
    return out[0] - out[1], secs


def start_model(start, device):
    if start == "background":
        e, s = np.full(SHAPE, EPS0), np.full(SHAPE, SIG0)
    else:
        from scipy.ndimage import gaussian_filter
        te, ts = true_model()
        e, s = gaussian_filter(te, 6.0), gaussian_filter(ts, 6.0)
    return (torch.as_tensor(e, dtype=DTYPE, device=device),
            torch.as_tensor(s, dtype=DTYPE, device=device))


def tv_term(u, m):
    gx = (u[1:] - u[:-1]) * (m[1:] & m[:-1])
    gy = (u[:, 1:] - u[:, :-1]) * (m[:, 1:] & m[:, :-1])
    gz = (u[:, :, 1:] - u[:, :, :-1]) * (m[:, :, 1:] & m[:, :, :-1])
    return torch.sqrt(gx[:, :-1, :-1] ** 2 + gy[:-1, :, :-1] ** 2 + gz[:-1, :-1, :] ** 2 + 1e-8).sum()


def invert(kind, src, rec, anomaly, mute, iters, start, dt):
    sol = build(kind, len(src), dt)
    prop = SubgridPropagator(sol, src, rec, use_graph=(kind == "staircase+subgrid"))
    w = ricker(NT, sol.dt).to("cuda", DTYPE)
    E0 = torch.full(SHAPE, EPS0, dtype=DTYPE, device="cuda")
    S0 = torch.full(SHAPE, SIG0, dtype=DTYPE, device="cuda")
    Ei, Si = start_model(start, "cuda")
    mask = torch.as_tensor(update_mask(), device="cuda")
    with torch.no_grad():
        bg = prop(E0, S0, w) * mute
        et, st = (torch.as_tensor(a, dtype=DTYPE, device="cuda") for a in true_model())
        floor_resp = prop(et, st, w) * mute - bg
    dobs = bg + anomaly
    norm = float((anomaly ** 2).sum())
    floor = float(((floor_resp - anomaly) ** 2).sum() / norm)
    u_e = torch.zeros(SHAPE, dtype=DTYPE, device="cuda", requires_grad=True)
    u_s = torch.zeros(SHAPE, dtype=DTYPE, device="cuda", requires_grad=True)

    def models():
        return ((Ei + SCALE_EPS * u_e * mask).clamp(min=1.0),
                (Si + SCALE_SIG * u_s * mask).clamp(min=0.0))

    opt = torch.optim.LBFGS([u_e, u_s], lr=1.0, history_size=10, max_iter=1,
                            line_search_fn="strong_wolfe")
    hist, evals, last = {"misfit": [], "seconds": []}, [0], {}
    torch.cuda.reset_peak_memory_stats()

    def closure():
        opt.zero_grad()
        eps, sig = models()
        misfit = ((prop(eps, sig, w) * mute - dobs) ** 2).sum() / norm
        loss = misfit + TV * (tv_term(u_e, mask) + tv_term(u_s, mask))
        loss.backward()
        evals[0] += 1
        last["misfit"] = float(misfit)
        return loss

    print(f"{kind}: modelling floor {floor:.4f}", flush=True)
    for it in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.step(closure)
        torch.cuda.synchronize()
        hist["misfit"].append(last["misfit"])
        hist["seconds"].append(time.perf_counter() - t0)
        print(f"  {kind} it {it + 1:2d}: misfit {last['misfit']:.4f} ({hist['seconds'][-1]:.0f} s)",
              flush=True)
    with torch.no_grad():
        eps, sig = models()
    e_np, s_np = eps.double().cpu().numpy(), sig.double().cpu().numpy()
    summ = dict(modelling_floor=floor, final_misfit=hist["misfit"][-1] if hist["misfit"] else None,
                evaluations=evals[0], seconds=float(sum(hist["seconds"])),
                seconds_per_evaluation=float(sum(hist["seconds"]) / max(1, evals[0])),
                peak_GB=torch.cuda.max_memory_allocated() / 2 ** 30, history=hist, dt=sol.dt,
                **metrics(e_np, s_np))
    del sol, prop
    torch.cuda.empty_cache()
    return summ, e_np, s_np


def invert_objects(kind, src, rec, anomaly, mute, iters, dt):
    """Values of the anomalies of known shape (eps_r, sigma of A; sigma of B)."""
    sol = build(kind, len(src), dt)
    prop = SubgridPropagator(sol, src, rec, use_graph=(kind == "staircase+subgrid"))
    w = ricker(NT, sol.dt).to("cuda", DTYPE)
    E0 = torch.full(SHAPE, EPS0, dtype=DTYPE, device="cuda")
    S0 = torch.full(SHAPE, SIG0, dtype=DTYPE, device="cuda")
    X, Y, Z = np.meshgrid(*[np.arange(n) * H for n in SHAPE], indexing="ij")
    ia = torch.as_tensor(((X - XC) ** 2 + (Y - YA) ** 2 + (Z - ZC) ** 2) <= RADIUS ** 2,
                         dtype=DTYPE, device="cuda")
    ib = torch.as_tensor(((X - XC) ** 2 + (Y - YB) ** 2 + (Z - ZC) ** 2) <= RADIUS ** 2,
                         dtype=DTYPE, device="cuda")
    with torch.no_grad():
        bg = prop(E0, S0, w) * mute
        et, st = (torch.as_tensor(a, dtype=DTYPE, device="cuda") for a in true_model())
        floor = float((((prop(et, st, w) * mute - bg) - anomaly) ** 2).sum() / (anomaly ** 2).sum())
    dobs = bg + anomaly
    norm = float((anomaly ** 2).sum())
    theta = torch.zeros(3, dtype=torch.float64, device="cuda", requires_grad=True)
    scale = torch.tensor([1.0, 0.01, 0.01], dtype=torch.float64, device="cuda")
    opt = torch.optim.LBFGS([theta], lr=1.0, history_size=10, max_iter=1,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        v = theta * scale
        eps = (E0 + v[0].to(DTYPE) * ia).clamp(min=1.0)
        sig = (S0 + v[1].to(DTYPE) * ia + v[2].to(DTYPE) * ib).clamp(min=0.0)
        loss = ((prop(eps, sig, w) * mute - dobs) ** 2).sum() / norm
        loss.backward()
        return loss

    print(f"{kind} (objects): modelling floor {floor:.4f}", flush=True)
    t0 = time.perf_counter()
    for _ in range(iters):
        loss = opt.step(closure)
    secs = time.perf_counter() - t0
    v = (theta * scale).detach().cpu().numpy()
    out = dict(modelling_floor=floor, misfit=float(loss), seconds=secs,
               eps_a=max(EPS0 + v[0], 1.0), sig_a=max(SIG0 + v[1], 0.0), sig_b=max(SIG0 + v[2], 0.0))
    print(f"{kind} (objects): eps_A {out['eps_a']:.2f} (true 16), sigma_A {1e3 * out['sig_a']:.1f} "
          f"(true 20), sigma_B {1e3 * out['sig_b']:.1f} mS/m (true 40); misfit {out['misfit']:.4f}, "
          f"{secs:.0f} s", flush=True)
    del sol, prop
    torch.cuda.empty_cache()
    return out


def figure(models, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    te, ts = true_model()
    cols = [("true", te, ts)] + [(k, e, s) for k, (e, s) in models.items()]
    fig, ax = plt.subplots(2, len(cols), figsize=(3.3 * len(cols), 6.0), constrained_layout=True)
    ix = int(round(XC / H))
    ext = [0, SHAPE[2] * H, SHAPE[1] * H, 0]
    for j, (name, e, s) in enumerate(cols):
        ax[0, j].imshow(e[ix], vmin=EPS0 - 1, vmax=16, cmap="viridis", extent=ext)
        ax[1, j].imshow(1e3 * s[ix], vmin=0, vmax=40, cmap="magma", extent=ext)
        ax[0, j].set_title(f"{name}: eps_r")
        ax[1, j].set_title(f"{name}: sigma [mS/m]")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--start", choices=("background", "smooth"), default="background")
    p.add_argument("--kinds", default="hybrid,hybrid+subgrid,staircase+subgrid")
    p.add_argument("--param", choices=("voxels", "objects"), default="voxels")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    src, rec = acquisition()
    mute = offset_mute(src, rec).cuda()
    probe = build("hybrid", len(src))
    dt = probe.dt
    del probe
    anomaly, t_ref = reference_response(src, rec, dt)
    anomaly = anomaly.cuda() * mute
    print(f"dt {dt:.3e} s, {NT} steps; reference run {t_ref:.0f} s", flush=True)
    res, models = {"dt": dt, "nt": NT, "reference_seconds": t_ref, "start": args.start}, {}
    if args.param == "objects":
        for kind in args.kinds.split(","):
            res[kind] = invert_objects(kind, src, rec, anomaly, mute, args.iters, dt)
        tag = "" if args.kinds == "hybrid,hybrid+subgrid,staircase+subgrid" else "_" + args.kinds.replace(",", "_").replace("+", "")
        (OUT / f"fwi3d_dgtd_objects{tag}.json").write_text(json.dumps(res, indent=1, default=float))
        print("wrote", OUT)
        return
    for kind in args.kinds.split(","):
        summ, e, s = invert(kind, src, rec, anomaly, mute, args.iters, args.start, dt)
        res[kind] = summ
        models[kind] = (e, s)
        print(f"{kind}: floor {summ['modelling_floor']:.4f}, misfit {summ['final_misfit']}, eps err "
              f"{summ['eps_err']:.3f} (6 cm {summ['eps_err_smooth6cm']:.3f}), sigma err "
              f"{summ['sig_err']:.3f} (6 cm {summ['sig_err_smooth6cm']:.3f}), A int deps "
              f"{summ['A_int_deps']:.2f} dsig {summ['A_int_dsig']:.2f}, B int dsig "
              f"{summ['B_int_dsig']:.2f}; {summ['seconds_per_evaluation']:.1f} s/gradient, peak "
              f"{summ['peak_GB']:.1f} GB", flush=True)
    tag = args.start
    (OUT / f"fwi3d_dgtd_{tag}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(OUT / f"fwi3d_dgtd_{tag}_models.npz",
                        **{f"{k}_{q}": v[i] for k, v in models.items() for i, q in enumerate(("eps", "sig"))})
    figure(models, OUT / f"fwi3d_dgtd_{tag}.png")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
