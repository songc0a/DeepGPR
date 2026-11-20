"""3D speed of the hybrid DGTD-FDTD step: forward and gradient wall time per simulated ns,
against pure Yee FDTD on the same grid (same kernels, same adjoint machinery).

Models:

* ``A_*``: the step-(a) FWI model (``fwi3d_lts.py``: 0.56 x 0.64 x 0.64 m, rugged
  surface, P1 band 0.08-0.28 m with 2 cm elements, Yee 2/3 cm, flux 0.1), central shot,
  7 x 7 receivers. ``A_global`` (one global step), ``A_lts`` (local time stepping,
  ``lts="auto"``), ``A_fdtd`` (pure Yee staircase on the same 2/3 cm grid).
* ``B_*``: the ground-coupled wet-ground model of ``coupled3d.py`` (``wet_coupled_line_amr``,
  0.66 m deep), P1 band 0.06-0.34 m with 1 cm elements, Yee 4 mm, 4 cm line antennas
  1-3 cm above the ground (41 points each), central shot with 8 receivers.
  ``B_global``, ``B_fdtd`` (pure Yee 4 mm, cell-averaged materials).

    python examples/hybrid/glue_benchmark3d.py wall --configs A_global A_lts A_fdtd \\
        --out results.json [--save DIR]
    python examples/hybrid/glue_benchmark3d.py prof --configs A_global

``wall``: minimum of ``--reps`` timed runs after a warm-up (compilation, graph capture);
``--save`` writes the data and the gradients (``eps``, ``sigma``) for cross-version
checks. ``prof``: GPU kernel time per step by category (torch.profiler), hybrid
kernels against PyTorch (ATen) kernels.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

DEV = "cuda"


def config_A(kind):
    import fwi3d_lts as F
    from DeepGPR.hybrid import HybridMaxwell3D
    eps0, sig0 = F.background()
    if kind == "fdtd":
        sol = HybridMaxwell3D(eps0, sig0, F.DX, surface=F.SURFACE, dg_band=None, pmlthick=F.PML,
                              fdtd_refinement=3, nshot=1, device=DEV, dtype=torch.float32)
    else:
        sol = F.build_solver("lts" if kind == "lts" else "global", 1, DEV)
    src, rec = F.antennas([4])
    eps = torch.tensor(eps0, dtype=torch.float32, device=DEV)
    sig = torch.tensor(sig0, dtype=torch.float32, device=DEV)

    def wav(nt, dt):
        return torch.tensor(F.wavelet(nt, dt), dtype=torch.float32, device=DEV)[None]
    return sol, src, rec, eps, sig, wav


def config_B(kind):
    import coupled3d as c3
    from DeepGPR.hybrid import HybridMaxwell3D
    case, r = "wet_coupled_line_amr", 5
    h = c3.DX / r
    if kind == "fdtd":
        eps, sig = c3.materials(case, h, "avg", target=True)
        sol = HybridMaxwell3D(eps, sig, h, surface=None, dg_band=None, pmlthick=c3.PML * r,
                              fdtd_refinement=1, nshot=1, device=DEV, dtype=torch.float32)
    else:
        eps, sig = c3.materials(case, h, "avg", target=True, ground_only=True)
        sol = HybridMaxwell3D(eps, sig, h, surface=c3.SURF, dg_band=c3.BAND,
                              pmlthick=c3.PML * r, flux=0.1, fdtd_refinement=1, dg_order=1,
                              dg_h=0.01, nshot=1, device=DEV, dtype=torch.float32)
    src, rec = c3.antennas(case)
    off, lw = c3.line_points(c3.LINE_DZ_HYB if kind != "fdtd" else h)
    src, rec = c3.expand_line(src[4:5], off), c3.expand_line(rec[4:5], off)
    amp = c3.DX / h

    def wav(nt, dt):
        w = torch.tensor(c3.wavelet((np.arange(nt) + 0.5) * dt) * amp, dtype=torch.float32,
                         device=DEV)
        return torch.tensor(lw, dtype=torch.float32, device=DEV)[:, None] * w[None]
    return sol, src, rec, eps.to(DEV, torch.float32), sig.to(DEV, torch.float32), wav


CONFIGS = {"A_global": lambda: config_A("global"), "A_lts": lambda: config_A("lts"),
           "A_fdtd": lambda: config_A("fdtd"), "B_global": lambda: config_B("global"),
           "B_fdtd": lambda: config_B("fdtd")}

HYBRID_KERNELS = (r"yee_|yee2_|terms_|terms2_|dgupd|upd2_|ltsupd|slab_|fold_|eta_|eta2_|\bz_|z2_|"
                  r"glue_|inject_|sample_|trace_|trace2_|ghost_|ghost2_|pen_|xh_|\bcombine|scatter2")


def setup(name, t_sim):
    from DeepGPR.hybrid import HybridPropagator
    sol, src, rec, eps, sig, wav = CONFIGS[name]()
    prop = HybridPropagator(sol, src, rec)
    nt = int(np.ceil(t_sim / sol.dt)) + 1
    w = wav(nt, sol.dt)
    g = torch.Generator(device="cpu").manual_seed(0)
    nrec = prop.rec_ops.count
    gd = torch.randn((prop.static.nshot, nt, nrec), generator=g).to(DEV, torch.float32)
    s = sol.summary()
    info = dict(dt=sol.dt, nt=nt, lts=getattr(sol, "lts", 1), yee=list(map(int, s.get("grid", []))),
                fdtd_mdofs=s.get("fdtd_dofs", 0) / 1e6, dg_mdofs=s.get("dg_dofs", 0) / 1e6,
                state_mb=__import__("DeepGPR.hybrid.graphs", fromlist=["x"]).state_bytes(sol.static) / 2 ** 20)
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


def wall(names, t_sim, reps, save, forward_only=False):
    out = {}
    for name in names:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        sol, prop, eps, sig, w, gd, info = setup(name, t_sim)
        print(f"{name}: {json.dumps(info)}", flush=True)
        rec = dict(info)
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
            rec[kind] = dict(ms_per_ns=1e3 * t / (t_sim * 1e9), ms_per_step=1e3 * t / info["nt"],
                             all_ms_per_ns=[1e3 * x / (t_sim * 1e9) for x in ts],
                             peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
            print(f"  {kind:8s} {rec[kind]['ms_per_ns']:8.1f} ms/ns  {rec[kind]['ms_per_step']:7.3f} ms/step"
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


def prof(names, t_sim):
    import os
    from torch.profiler import ProfilerActivity, profile
    os.environ["DEEPGPR_HYBRID_DISABLE_GRAPHS"] = "1"   # the profiler does not see into graphs
    out = {}
    for name in names:
        torch.cuda.empty_cache()
        sol, prop, eps, sig, w, gd, info = setup(name, t_sim)
        nt = info["nt"]
        print(f"{name}: {json.dumps(info)}", flush=True)
        rec = dict(info)
        for kind in ("forward", "gradient"):
            f = (lambda: run_forward(prop, eps, sig, w)) if kind == "forward" else (
                lambda: run_gradient(prop, eps, sig, w, gd))
            f()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as pr:
                f()
                torch.cuda.synchronize()
            agg = collections.defaultdict(lambda: [0.0, 0])
            for ev in pr.key_averages():
                t = getattr(ev, "device_time_total", None)
                if t is None:
                    t = ev.cuda_time_total
                if t <= 0:
                    continue
                cat = "hybrid kernels" if re.search(HYBRID_KERNELS, ev.key) else "PyTorch (ATen) kernels"
                if re.search(r"Memcpy|Memset", ev.key):
                    cat = "memcpy/memset"
                agg[cat][0] += t / 1e3
                agg[cat][1] += ev.count
                agg["  " + ev.key[:60]][0] += t / 1e3
                agg["  " + ev.key[:60]][1] += ev.count
            tot = sum(v[0] for k, v in agg.items() if not k.startswith("  "))
            rec[kind] = {k: dict(ms_per_ns=v[0] / (t_sim * 1e9), calls_per_step=v[1] / nt)
                         for k, v in agg.items()}
            print(f"  {kind}: GPU kernel time {tot / (t_sim * 1e9):.1f} ms/ns", flush=True)
            for k, v in sorted([kv for kv in agg.items() if not kv[0].startswith("  ")],
                               key=lambda kv: -kv[1][0]):
                print(f"    {k:28s} {v[0] / (t_sim * 1e9):8.1f} ms/ns {100 * v[0] / tot:5.1f} %"
                      f" {v[1] / nt:7.1f} calls/step", flush=True)
            top = sorted([kv for kv in agg.items() if kv[0].startswith("  ")], key=lambda kv: -kv[1][0])
            for k, v in top[:14]:
                print(f"    {k:62s} {v[0] / (t_sim * 1e9):8.1f} ms/ns {v[1] / nt:6.1f}/step", flush=True)
        out[name] = rec
        del prop, sol
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("wall", "prof"))
    ap.add_argument("--configs", nargs="+", default=["A_global", "A_lts", "A_fdtd"])
    ap.add_argument("--tsim", type=float, default=2e-9)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--forward-only", action="store_true")
    args = ap.parse_args()
    res = wall(args.configs, args.tsim, args.reps, args.save, args.forward_only) if args.mode == "wall" else prof(
        args.configs, args.tsim)
    res["_meta"] = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                        time=time.strftime("%Y-%m-%d %H:%M:%S"), tsim=args.tsim, reps=args.reps)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
