"""Cost of the 3D hybrid on the GPU: forward run and gradient (forward + adjoint).

B3 model of ``phase4_benchmark.py`` (sine surface, 2 cm model grid, DG band
0.20-0.44 m, P1, adaptive column mesh), 16 ns, one shot, float32, at Yee
refinement r = 1 and 3. Paths:

* ``native``         DeepGPR's CUDA FDTD (staircase) on dx/r, its own adjoint
* ``torch``          hybrid with the PyTorch operator (``cuda_kernels=False``), autograd VJP
* ``cuda-autograd``  hybrid with the fused kernels, autograd VJP of the step
                     (``DEEPGPR_HYBRID_AUTOGRAD_VJP=1``)
* ``cuda``           hybrid with the fused kernels and the hand-written step transpose
                     (the default; CUDA Graph replay is automatic and off at this size)

The gradient is that of ``J = 1/2 ||d||^2`` with respect to ``eps_r`` and
``sigma`` (checkpointing: sqrt(nt)). Reports seconds, ms per step and the peak
GPU memory; the gradients of the hybrid paths are compared with each other.

    python examples/hybrid/phase4_cuda_benchmark.py [--refine 1 3] [--skip torch] [--tmax 2.5e-9]

With r = 3 one state is 436 MB and the sqrt(nt) checkpointing keeps about
2 sqrt(nt) states, so the 16 ns gradient (2056 steps, ~40 GB) does not fit on a
24 GB GPU; ``--tmax`` shortens the window (per-step costs are what is compared).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models3d as M  # noqa: E402

import DeepGPR  # noqa: E402
from DeepGPR.hybrid import HybridPropagator, make_propagator  # noqa: E402

CASE = "B3_sine"
OUT = M.ROOT / "examples" / "hybrid" / "results"
ENV = {"cuda-autograd": {"DEEPGPR_HYBRID_AUTOGRAD_VJP": "1"}}


def timed(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0, (torch.cuda.max_memory_allocated() - base) / 2 ** 30


def hybrid_case(path, r, tmax):
    sol = M.build_hybrid(CASE, refinement=r, cuda_kernels=(path != "torch"))
    nt = int(np.ceil(tmax / sol.dt)) + 1
    w = M.wavelet(nt, sol.dt)
    src, rec = M.antennas()
    for k, v in ENV.get(path, {}).items():
        os.environ[k] = v
    try:
        _, t_fwd, m_fwd = timed(lambda: sol.run(w, src, rec))
        prop = HybridPropagator(sol, src, rec)
        eps, sig = (torch.tensor(a, dtype=torch.float64, device="cuda") for a in M.ground_model(CASE))
        e, s = eps.clone().requires_grad_(True), sig.clone().requires_grad_(True)
        wt = torch.tensor(w, device="cuda")

        def grad():
            d = prop(e, s, wt[None])
            (0.5 * (d ** 2).sum()).backward()
            return e.grad.clone(), s.grad.clone()
        g, t_grad, m_grad = timed(grad)
    finally:
        for k in ENV.get(path, {}):
            os.environ.pop(k, None)
    row = dict(path=path, r=r, nt=nt, dt=sol.dt, K=sol.static.dg.K, yee=list(sol.static.yee.shape),
               forward_s=t_fwd, forward_ms_step=1e3 * t_fwd / nt, forward_GB=m_fwd,
               gradient_s=t_grad, gradient_ms_step=1e3 * t_grad / nt, gradient_GB=m_grad)
    torch.cuda.empty_cache()
    return row, [t.cpu() for t in g]


def native_case(r, tmax):
    eps, sig = M.ground_model(CASE)
    h = M.DX / r
    n = eps.shape
    pml = M.PML * r
    dt = 0.95 * DeepGPR.max_stable_time_step(h, n[0] * r + 2 * pml, n[1] * r + 2 * pml,
                                             n[2] * r + 2 * pml)
    nt = int(np.ceil(tmax / dt)) + 1
    w = torch.tensor(M.wavelet(nt, dt), device="cuda")
    src, rec = M.antennas()
    extra = {"model_gradient_sampling_interval": "auto"} if r > 1 else {}
    prop = make_propagator("fdtd", torch.tensor(eps), torch.tensor(sig), M.DX, src, rec,
                           surface=M.CASES[CASE]["surface"], pmlthick=M.PML, dt=dt, device="cuda",
                           fdtd_refinement=r, **extra)
    e = torch.tensor(eps, dtype=torch.float32, device="cuda")
    s = torch.tensor(sig, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        _, t_fwd, m_fwd = timed(lambda: prop(e, s, w[None]))
    e2, s2 = e.clone().requires_grad_(True), s.clone().requires_grad_(True)

    def grad():
        d = prop(e2, s2, w[None])
        (0.5 * (d ** 2).sum()).backward()
    _, t_grad, m_grad = timed(grad)
    torch.cuda.empty_cache()
    return dict(path="native", r=r, nt=nt, dt=dt, forward_s=t_fwd, forward_ms_step=1e3 * t_fwd / nt,
                forward_GB=m_fwd, gradient_s=t_grad, gradient_ms_step=1e3 * t_grad / nt,
                gradient_GB=m_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--tmax", type=float, default=M.T_MAX)
    args = ap.parse_args()
    rows = []
    for r in args.refine:
        grads = {}
        for path in ("native", "torch", "cuda-autograd", "cuda"):
            if path in args.skip:
                continue
            if path == "native":
                row = native_case(r, args.tmax)
            else:
                row, grads[path] = hybrid_case(path, r, args.tmax)
            rows.append(row)
            print(f"r={r} {path:10s} nt {row['nt']:5d}  forward {row['forward_s']:7.2f} s "
                  f"({row['forward_ms_step']:6.2f} ms/step, {row['forward_GB']:.2f} GB)  gradient "
                  f"{row['gradient_s']:7.2f} s ({row['gradient_ms_step']:6.2f} ms/step, "
                  f"{row['gradient_GB']:.2f} GB)", flush=True)
        ref = grads.get("cuda")
        for path, g in grads.items():
            if ref is not None and path != "cuda":
                err = [float((a - b).norm() / b.norm()) for a, b in zip(g, ref)]
                print(f"  gradient {path} vs cuda: eps {err[0]:.1e}, sigma {err[1]:.1e}")
                rows.append(dict(path=path, r=r, gradient_vs_cuda=err))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / (f"phase4_cuda_benchmark_r{'_'.join(map(str, args.refine))}"
                 f"_{args.tmax * 1e9:g}ns.json")
    out.write_text(json.dumps(rows, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
