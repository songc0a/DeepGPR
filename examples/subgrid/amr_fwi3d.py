"""Adjoint-error-driven dynamic subgridding in 3D dual-parameter FWI (AMR-FWI).

Model, acquisition and data of ``fwi3d_subgrid.py`` (preset ``resolved`` or
``strong``: two anomalies at 0.5 m depth, 400 MHz, 9 shots, anomaly-response
data from a uniform 1 cm grid), but without prior knowledge of where the anomalies
are: the model may change anywhere in the subsurface below the antennas
(x > 0.33 m, the whole y-z extent) and refinement is admissible there.

Schedule (L-BFGS, the history reset at every new layout):

1. cycle 0: main grid only (3 cm), ``--iters`` iterations;
2. after every cycle: the differential DWR indicator ``|eta(m) - eta(m0)|`` of the
   current model and residual (:func:`DeepGPR.subgrid.amr.indicator`), Doerfler
   marking (``theta``) in the admissible region, together with the a priori
   criteria on the current model, gives the subgrid boxes of the next cycle
   (:func:`DeepGPR.subgrid.amr.regrid`); the observed data of the new layout are
   its own background prediction plus the uniform-grid anomaly response.

Compared (``--modes``) with the main grid only (``coarse``) and with the fixed box of
``fwi3d_subgrid.py`` placed with prior knowledge (``static``), all with the same
update region, iterations and L-BFGS restarts.

    python examples/subgrid/amr_fwi3d.py [--preset resolved|strong --cycles 3 --iters 10 --theta 0.5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fwi3d_subgrid as F  # noqa: E402

from DeepGPR.subgrid import SubgridMaxwell3D  # noqa: E402
from DeepGPR.subgrid.adjoint import SubgridPropagator  # noqa: E402
from DeepGPR.subgrid.amr import indicator, regrid  # noqa: E402

OUT = F.OUT.parent / "amr_fwi3d"
X_MIN = 0.33            # updates and refinement below the antennas


def update_mask():
    m = np.zeros(F.SHAPE, dtype=bool)
    i0 = int(round(X_MIN / F.H)) + 1
    m[i0:-6, 6:-6, 6:-6] = True
    return m


def allowed_main():
    n = tuple((s - 1) // 3 + 1 for s in F.SHAPE)
    a = np.zeros(n, dtype=bool)
    a[int(np.ceil(X_MIN / 0.03)) + 1:, :, :] = True
    return a


def figure(cycles, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    te, ts = F.true_model()
    n = len(cycles)
    fig, ax = plt.subplots(3, n + 1, figsize=(3.3 * (n + 1), 9.0), constrained_layout=True)
    ix, ixm = int(round(0.5 / F.H)), int(round(0.5 / 0.03))
    ext = [0, F.SHAPE[2] * F.H, F.SHAPE[1] * F.H, 0]
    ax[0, 0].imshow(te[ix], vmin=F.EPS0 - 1, vmax=12, cmap="viridis", extent=ext)
    ax[1, 0].imshow(1e3 * ts[ix], vmin=0, vmax=30, cmap="magma", extent=ext)
    ax[0, 0].set_title("true eps_r (x = 0.5 m)")
    ax[1, 0].set_title("true sigma [mS/m]")
    ax[2, 0].axis("off")
    for j, c in enumerate(cycles, start=1):
        ax[0, j].imshow(c["eps"][ix], vmin=F.EPS0 - 1, vmax=12, cmap="viridis", extent=ext)
        ax[1, j].imshow(1e3 * c["sig"][ix], vmin=0, vmax=30, cmap="magma", extent=ext)
        ax[0, j].set_title(f"cycle {j - 1}: eps_r")
        ax[1, j].set_title(f"cycle {j - 1}: sigma")
        if c.get("eta") is not None:
            e = c["eta"][ixm]
            ax[2, j].imshow(e / max(e.max(), 1e-30), cmap="gray_r", extent=ext)
            ax[2, j].set_title(f"indicator after cycle {j - 1} (x = 0.5 m)")
        else:
            ax[2, j].axis("off")
        for r in range(3):
            for sg in c["boxes"]:
                (y0, y1), (z0, z1) = sg[1], sg[2]
                ax[r, j].plot([z0, z1, z1, z0, z0], [y0, y0, y1, y1, y0], "c--", lw=0.9)
            for nb in c.get("next_boxes", []):
                (y0, y1), (z0, z1) = nb[1], nb[2]
                ax[r, j].plot([z0, z1, z1, z0, z0], [y0, y0, y1, y1, y0], "r:", lw=0.9)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_schedule(mode, args, src, rec, mute, anomaly):
    """``mode``: ``"amr"`` (dynamic layout), ``"coarse"`` (main grid only) or ``"static"``
    (the box of ``fwi3d_subgrid.py``, placed with prior knowledge); identical update
    region, iterations and L-BFGS restarts."""
    E0 = torch.full(F.SHAPE, F.EPS0, dtype=F.DTYPE, device="cuda")
    S0 = torch.full(F.SHAPE, F.SIG0, dtype=F.DTYPE, device="cuda")
    mask = torch.as_tensor(update_mask(), device="cuda")
    u_e = torch.zeros(F.SHAPE, dtype=F.DTYPE, device="cuda", requires_grad=True)
    u_s = torch.zeros(F.SHAPE, dtype=F.DTYPE, device="cuda", requires_grad=True)

    def models():
        return ((E0 + F.SCALE_EPS * u_e * mask).clamp(min=1.0),
                (S0 + F.SCALE_SIG * u_s * mask).clamp(min=0.0))

    allowed = allowed_main()
    layout = [F.Subgrid(F.CFG["box"], ratio=3)] if mode == "static" else []
    cycles, log = [], []
    norm = float((anomaly ** 2).sum())
    t_all = time.perf_counter()
    for cyc in range(args.cycles):
        sol = SubgridMaxwell3D(E0, S0, F.H, coarsen=3, subgrids=layout, pmlthick=10,
                               nshot=len(src), device="cuda", dtype=F.DTYPE)
        prop = SubgridPropagator(sol, src, rec)
        w = F.ricker(F.CFG["nt"], sol.dt).to("cuda", F.DTYPE)
        with torch.no_grad():
            bg = prop(E0, S0, w) * mute
        dobs = bg + anomaly
        opt = torch.optim.LBFGS([u_e, u_s], lr=1.0, history_size=10, max_iter=1,
                                line_search_fn="strong_wolfe")
        evals, last = [0], {}

        def closure():
            opt.zero_grad()
            eps, sig = models()
            d = prop(eps, sig, w) * mute
            misfit = ((d - dobs) ** 2).sum() / norm
            loss = misfit + F.CFG["tv"] * (F.tv_term(u_e, mask) + F.tv_term(u_s, mask))
            loss.backward()
            evals[0] += 1
            last["misfit"] = float(misfit)
            return loss

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for it in range(args.iters):
            opt.step(closure)
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        with torch.no_grad():
            eps, sig = models()
            d = prop(eps, sig, w) * mute
        e_np, s_np = eps.double().cpu().numpy(), sig.double().cpu().numpy()
        met = F.metrics(e_np, s_np)
        rec_c = dict(cycle=cyc, boxes=[[list(b) for b in sg.box] for sg in layout],
                     misfit=last["misfit"], seconds=secs, evaluations=evals[0],
                     seconds_per_evaluation=secs / max(1, evals[0]),
                     fine_nodes=[int(np.prod(sg.yee.shape)) for sg in sol.static.subs], **met)
        print(f"[{mode}] cycle {cyc}: {len(layout)} box(es), misfit {last['misfit']:.4f}, eps err "
              f"{met['eps_err']:.3f} (6 cm {met['eps_err_smooth6cm']:.3f}), sigma err "
              f"{met['sig_err']:.3f} (6 cm {met['sig_err_smooth6cm']:.3f}), A int deps "
              f"{met['A_int_deps']:.2f} dsig {met['A_int_dsig']:.2f}, B int dsig "
              f"{met['B_int_dsig']:.2f}; {secs:.0f} s ({evals[0]} evals)", flush=True)
        cyc_store = dict(eps=e_np, sig=s_np, boxes=[sg.box for sg in layout], eta=None)
        if mode == "amr" and cyc + 1 < args.cycles:
            t1 = time.perf_counter()
            iprop = SubgridPropagator(sol, src, rec, indicator=True)
            residual = (d - dobs) * mute / norm
            eta = indicator(iprop, eps, sig, w, residual, eps0=E0, sig0=S0)
            rg = regrid(e_np, s_np, F.H, 3, eta, allowed=allowed, theta=args.theta,
                        f_max=6e8, ppw=10, keep_points=np.concatenate([src.reshape(-1, 3),
                                                                       rec.reshape(-1, 3)]),
                        current=layout)
            layout = rg.plan.subgrids
            rec_c.update(indicator_seconds=time.perf_counter() - t1, marked=rg.info["marked_nodes"],
                         next_boxes=[[list(b) for b in sg.box] for sg in layout], changed=rg.changed)
            cyc_store.update(eta=eta, next_boxes=[sg.box for sg in layout])
            print(f"  indicator {rec_c['indicator_seconds']:.1f} s, {rg.info['marked_nodes']} marked "
                  f"nodes -> {len(layout)} box(es): "
                  f"{[[tuple(round(v, 2) for v in b) for b in sg.box] for sg in layout]}", flush=True)
        cycles.append(cyc_store)
        log.append(rec_c)
        del sol, prop
        torch.cuda.empty_cache()
    return dict(cycles=log, total_seconds=time.perf_counter() - t_all), cycles


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--preset", choices=("resolved", "strong"), default="resolved")
    p.add_argument("--modes", default="amr,coarse,static")
    args = p.parse_args()
    F.CFG.clear()
    F.CFG.update(F.PRESETS[args.preset])
    OUT.mkdir(parents=True, exist_ok=True)
    src, rec = F.acquisition()
    mute = F.offset_mute(src, rec).cuda()
    probe = SubgridMaxwell3D(torch.full(F.SHAPE, F.EPS0), torch.full(F.SHAPE, F.SIG0), F.H,
                             coarsen=3, pmlthick=10, cuda_kernels=False)
    d_true, _ = F.observed(src, rec, probe.dt)
    d_bg, _ = F.observed(src, rec, probe.dt, targets=False)
    anomaly = (d_true - d_bg).cuda() * mute
    path = OUT / f"amr_fwi3d_{args.preset}.json"
    res = json.loads(path.read_text()) if path.exists() else {}     # keep other modes' results
    res.update({"theta": args.theta, "iters_per_cycle": args.iters, "cycles": args.cycles,
                "preset": args.preset})
    for mode in args.modes.split(","):
        r, cycles = run_schedule(mode, args, src, rec, mute, anomaly)
        res[mode] = r
        if mode == "amr":
            figure(cycles, OUT / f"amr_fwi3d_{args.preset}.png")
            np.savez_compressed(OUT / f"amr_fwi3d_{args.preset}_models.npz", eps=cycles[-1]["eps"],
                                sig=cycles[-1]["sig"])
        print(f"[{mode}] total {r['total_seconds']:.0f} s", flush=True)
    path.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
