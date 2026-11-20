"""Local time stepping of the 3D hybrid on the GPU: accuracy and cost.

Cases of ``phase4_benchmark.py`` (2 cm model grid, DG band 0.20-0.44 m, P1, adaptive
column mesh, float32, one shot, 16 ns). For every case the hybrid with Yee refinement
``r = 1`` and central fluxes (the LTS needs them; both arms use them) is run with the
global leapfrog (``dt`` set by the band) and with the local time stepping
(``lts="auto"``: Yee at ``lts_cfl`` of its own limit, capped by the side-UPML rate,
the band at ``dt / p``); the receivers are compared with the converged native
reference ``ref_7`` of the phase-4 benchmark (when present) and with each other,
and the forward and gradient costs are measured (global: hand-written step
transpose; LTS: autograd transpose of the step).

    python examples/hybrid/lts_benchmark3d.py [--cases B3_sine A3_flat] [--tmax 16e-9]
                                              [--grad-tmax 2.5e-9] [--nu 0.3] [--overlap 1]
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
import models3d as M  # noqa: E402

from DeepGPR.hybrid import HybridPropagator  # noqa: E402

OUT = M.ROOT / "examples" / "hybrid" / "results"
PHASE4 = OUT / "phase4"


def timed(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0, (torch.cuda.max_memory_allocated() - base) / 2 ** 30


def reference(case):
    path = PHASE4 / f"{case}_ref_7.npz"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)["data"]


def forward(sol, tmax):
    nt = int(np.ceil(tmax / sol.dt)) + 1
    src, rec = M.antennas()
    w = torch.tensor(M.wavelet(nt, sol.dt))
    res, secs, mem = timed(lambda: sol.run(w, src, rec))
    data = res["receivers"][0].double().cpu().numpy()
    t = (np.arange(nt) + 1) * sol.dt
    return M.resample_np(data, t), dict(seconds=secs, ms_per_step=1e3 * secs / nt, nt=nt,
                                        dt=sol.dt, peak_gb=mem)


def gradient(sol, tmax):
    nt = int(np.ceil(tmax / sol.dt)) + 1
    src, rec = M.antennas()
    w = torch.tensor(M.wavelet(nt, sol.dt), device="cuda")
    prop = HybridPropagator(sol, src, rec)
    eps, sig = (torch.tensor(a, dtype=torch.float64, device="cuda") for a in M.ground_model(sol._case))
    e, s = eps.clone().requires_grad_(True), sig.clone().requires_grad_(True)

    def run():
        d = prop(e, s, w[None])
        (0.5 * (d ** 2).sum()).backward()
        return e.grad.clone(), s.grad.clone()

    g, secs, mem = timed(run)
    return g, dict(seconds=secs, ms_per_step=1e3 * secs / nt, nt=nt, peak_gb=mem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["B3_sine", "A3_flat", "C3_irregular"])
    ap.add_argument("--tmax", type=float, default=M.T_MAX)
    ap.add_argument("--grad-tmax", type=float, default=2.5e-9)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--overlap", type=int, default=1)
    ap.add_argument("--lts-cfl", type=float, default=0.9)
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--strength", type=float, default=0.4, help="DG side-UPML strength (both arms)")
    ap.add_argument("--skip-grad", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    log = open(OUT / "lts_benchmark3d.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} LTS benchmark: nu={args.nu} overlap={args.overlap} "
        f"lts_cfl={args.lts_cfl} r={args.refine}")
    for case in args.cases:
        ref = reference(case)
        mask = M.metric_mask()
        rows = {}
        for name, kw in (("global", {}), ("lts", dict(lts="auto", lts_nu=args.nu,
                                                       lts_overlap=args.overlap,
                                                       lts_cfl=args.lts_cfl,
                                                       dg_pml_strength=args.strength))):
            sol = M.build_hybrid(case, refinement=args.refine, flux="central",
                                 dg_pml_strength=args.strength, **{k: v for k, v in kw.items()
                                                                   if k != "dg_pml_strength"})
            sol._case = case
            info = {k: v for k, v in sol.summary().items() if isinstance(v, (int, float))}
            data, finfo = forward(sol, args.tmax)
            row = dict(solver=info, forward=finfo)
            if ref is not None:
                row["error_vs_ref7"] = M.rel_error(data, ref, mask)
            rows[name] = row
            rows[name]["_data"] = data
            say(f"[{case}] {name}: p={sol.lts} dt={sol.dt:.3e} (fine {sol.dt_fine:.3e}, Yee limit "
                f"{sol.yee_dt_limit():.3e}) nt={finfo['nt']} forward {finfo['seconds']:.2f} s "
                f"({finfo['ms_per_step']:.2f} ms/step, {finfo['peak_gb']:.2f} GB)"
                + (f" error vs ref_7 {row['error_vs_ref7']:.4f}" if ref is not None else ""))
            if not args.skip_grad:
                g, ginfo = gradient(sol, args.grad_tmax)
                row["gradient"] = ginfo
                row["_grad"] = g
                say(f"[{case}] {name}: gradient over {args.grad_tmax * 1e9:.1f} ns: {ginfo['seconds']:.2f} s "
                    f"({ginfo['ms_per_step']:.2f} ms/step, {ginfo['peak_gb']:.2f} GB)")
            del sol
            torch.cuda.empty_cache()
        a, b = rows["global"]["_data"], rows["lts"]["_data"]
        diff = float(np.linalg.norm((a - b)[:, mask]) / np.linalg.norm(a[:, mask]))
        say(f"[{case}] LTS vs global receivers: {diff:.4f}; speed-up forward "
            f"{rows['global']['forward']['seconds'] / rows['lts']['forward']['seconds']:.2f}x"
            + (f", gradient {rows['global']['gradient']['seconds'] / rows['lts']['gradient']['seconds']:.2f}x"
               if not args.skip_grad else ""))
        if not args.skip_grad:
            ga, gb = rows["global"]["_grad"], rows["lts"]["_grad"]
            gdiff = [float((x - y).norm() / x.norm()) for x, y in zip(ga, gb)]
            say(f"[{case}] gradient LTS vs global (eps, sigma): {gdiff[0]:.4f}, {gdiff[1]:.4f}")
            rows["gradient_diff"] = gdiff
        rows["receiver_diff"] = diff
        np.savez(OUT / f"lts_benchmark3d_{case}.npz", global_=a, lts=b)
        for k in ("global", "lts"):
            rows[k].pop("_data", None)
            rows[k].pop("_grad", None)
        results[case] = rows
        (OUT / "lts_benchmark3d.json").write_text(json.dumps(
            dict(args=vars(args), results=results), indent=1))
    say("done")


if __name__ == "__main__":
    main()
