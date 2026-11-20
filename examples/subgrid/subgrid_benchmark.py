"""Speed of the subgridded (HSG) step: forward and gradient wall time per simulated ns
of :class:`~DeepGPR.subgrid.adjoint.SubgridPropagator`, against the main grid alone and
against uniform FDTD at the subgrids' spacing.

Models:

* ``SP_*``: the 3D sand-pit benchmark of ``amr_sandpit3d.py`` (model grid 0.75 cm,
  shape (91, 160, 100), Yee main grid 2.25 cm with 10-cell CPML, 450 MHz, 16 ns,
  y dipoles 1.5 cm above the sand), the 9 training shots in one propagation with 44
  receivers each. ``SP_static``: one 0.75 cm box over the whole update region (the
  ``static_full`` method), ``SP_oracle``: the boxes around the targets of M1 (the
  ``oracle`` method), ``SP_coarse``: the main grid alone, ``SP_fine``: uniform 0.75 cm
  FDTD (``fine``; one shot per propagation, as in the study).
* ``T_*``: the rugged-terrain model of ``amr_terrain3d.py`` (hybrid: Yee 2 cm, P1 band
  0.08-0.28 m, flux 0.1, global time step; 1.0 m deep, 16 ns, one shot per
  propagation, 7 x 7 receivers). ``T_static``: one 2/3 cm box over the whole
  admissible region (the ``static`` method), ``T_coarse``: no box.

    python examples/subgrid/subgrid_benchmark.py --configs SP_static SP_coarse \\
        --out results.json [--save DIR] [--forward-only]

The minimum of ``--reps`` timed runs after a warm-up (compilation, graph capture) is
reported; ``--save`` writes the data and the gradients for cross-version checks.
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
sys.path.insert(0, str(HERE.parents[0] / "hybrid"))
sys.path.insert(0, str(HERE.parents[1] / "src"))

DEV = "cuda"


def config_SP(kind):
    import amr_sandpit3d as A
    from DeepGPR.subgrid import Subgrid, SubgridMaxwell3D
    src, rec, train, _ = A.acquisition()
    keep = np.concatenate([src.reshape(-1, 3), rec.reshape(-1, 3)])
    allowed = A.allowed_main()
    boxes = []
    if kind == "static":
        boxes = A.plan_from_flags(allowed, keep)
    elif kind == "oracle":
        from scipy.ndimage import binary_dilation
        e0, s0 = A.model_arrays(None)
        te, ts = A.model_arrays("M1")
        m = (np.abs(te - e0) > 0.1) | (np.abs(ts - s0) > 1e-3)
        n = tuple((s - 1) // A.R + 1 for s in A.SHAPE)
        mm = np.zeros(n, dtype=bool)
        mm[tuple((np.argwhere(m) // A.R).T)] = True
        boxes = A.plan_from_flags(binary_dilation(mm, iterations=1) & allowed, keep)
    e, s = A.background()
    shots = train[:1] if kind == "fine" else train
    if kind == "fine":
        sol = SubgridMaxwell3D(e, s, A.H, coarsen=1, pmlthick=20, dt=A.DT / A.R,
                               dipole_length=A.R * A.H, component=A.COMP, nshot=len(shots),
                               device=DEV, dtype=torch.float32)
        w = A.resampled(A.ricker(A.NT, A.DT), A.R)
    else:
        sol = SubgridMaxwell3D(e, s, A.H, coarsen=A.R, pmlthick=10, dt=A.DT,
                               subgrids=[Subgrid(tuple(map(tuple, b))) for b in boxes],
                               dipole_length=A.R * A.H, component=A.COMP, nshot=len(shots),
                               device=DEV, dtype=torch.float32)
        w = A.ricker(A.NT, A.DT)

    def wav(nt, dt):
        return w[:nt].to(DEV, torch.float32)[None]
    return sol, src[shots], rec[shots], e.to(DEV), s.to(DEV), wav


def config_T(kind):
    import amr_terrain3d as T
    import fwi3d_lts as F
    from DeepGPR.subgrid import Subgrid
    from DeepGPR.subgrid.hybrid import HybridSubgridMaxwell3D
    T.configure()
    T._setup_fine()
    e0, s0 = (torch.tensor(a, dtype=torch.float32, device=DEV) for a in F.background())
    boxes = [T.whole_box()] if kind == "static" else []
    sol = HybridSubgridMaxwell3D(T.up(e0), T.up(s0), F.DX / T.R, coarsen=T.R, surface=F.SURFACE,
                                 dg_band=F.BAND, subgrids=[Subgrid(b, ratio=T.R, **T.SUB) for b in boxes],
                                 pmlthick=F.PML, flux=F.FLUX, nshot=1, device=DEV,
                                 dtype=torch.float32)
    src, rec = F.antennas()

    def wav(nt, dt):
        return torch.tensor(F.wavelet(nt, dt), dtype=torch.float32, device=DEV)[None]
    return sol, src[4:5], rec[4:5], T.up(e0), T.up(s0), wav


CONFIGS = {"SP_static": lambda: config_SP("static"), "SP_oracle": lambda: config_SP("oracle"),
           "SP_coarse": lambda: config_SP("coarse"), "SP_fine": lambda: config_SP("fine"),
           "T_static": lambda: config_T("static"), "T_coarse": lambda: config_T("coarse")}
T_SIM = {"SP": 16e-9, "T": 16e-9}


def setup(name, t_sim=None):
    from DeepGPR.subgrid.adjoint import SubgridPropagator
    sol, src, rec, eps, sig, wav = CONFIGS[name]()
    prop = SubgridPropagator(sol, src, rec)
    t_sim = t_sim or T_SIM[name.split("_")[0]]
    nt = int(round(t_sim / sol.dt))
    w = wav(nt, sol.dt)
    gd = torch.randn((sol.nshot, nt, prop.rec.count),
                     generator=torch.Generator().manual_seed(0)).to(DEV, torch.float32)
    subs = sol.static.subs
    info = dict(dt=sol.dt, nt=nt, t_sim=t_sim, nshot=sol.nshot, nrec=prop.rec.count,
                main=list(map(int, (sol.static.hyb.yee if hasattr(sol.static, "hyb") else sol.static.main).shape)),
                boxes=[list(map(int, sg.yee.shape)) for sg in subs],
                fine_mnodes=sum(int(np.prod(sg.yee.shape)) for sg in subs) / 1e6)
    return sol, prop, eps, sig, w, gd, info


def run_forward(prop, eps, sig, w):
    with torch.no_grad():
        return prop(eps, sig, w)


def run_gradient(prop, eps, sig, w, gd):
    e = eps.clone().requires_grad_(True)
    s = sig.clone().requires_grad_(True)
    d = prop(e, s, w)
    d.backward(gd)
    return d.detach(), e.grad, s.grad


def wall(names, reps, save, forward_only=False, t_sim=None):
    out = {}
    for name in names:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        sol, prop, eps, sig, w, gd, info = setup(name, t_sim)
        print(f"{name}: {json.dumps(info)}", flush=True)
        rec = dict(info)
        ns = info["t_sim"] * 1e9
        for kind in (("forward",) if forward_only else ("forward", "gradient")):
            f = (lambda: run_forward(prop, eps, sig, w)) if kind == "forward" else (
                lambda: run_gradient(prop, eps, sig, w, gd))
            res = f()                                   # warm-up: compile, capture
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ts = []
            for _ in range(reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                res = f()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            t = min(ts)
            rec[kind] = dict(ms_per_ns=1e3 * t / ns, ms_per_step=1e3 * t / info["nt"],
                             all_ms_per_ns=[1e3 * x / ns for x in ts],
                             peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
            print(f"  {kind:8s} {rec[kind]['ms_per_ns']:9.1f} ms/ns  {rec[kind]['ms_per_step']:8.3f} ms/step"
                  f"  (runs {', '.join(f'{x:.1f}' for x in rec[kind]['all_ms_per_ns'])})"
                  f"  peak {rec[kind]['peak_gb']:.2f} GB", flush=True)
            if save:
                Path(save).mkdir(parents=True, exist_ok=True)
                if kind == "forward":
                    np.save(Path(save) / f"{name}_data.npy", res.cpu().numpy())
                else:
                    d, ge, gs = res
                    np.savez(Path(save) / f"{name}_grad.npz", data=d.cpu().numpy(),
                             eps=ge.cpu().numpy(), sigma=gs.cpu().numpy())
        out[name] = rec
        del prop, sol
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["SP_static", "SP_coarse"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--forward-only", action="store_true")
    ap.add_argument("--tsim", type=float, default=None, help="simulated time [s] (default: the model's)")
    args = ap.parse_args()
    res = wall(args.configs, args.reps, args.save, args.forward_only, args.tsim)
    res["_meta"] = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                        time=time.strftime("%Y-%m-%d %H:%M:%S"), reps=args.reps)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
