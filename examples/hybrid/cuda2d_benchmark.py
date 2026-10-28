"""Cost of the 2D hybrid on the GPU: forward run and gradient (forward + adjoint).

Model and acquisition of ``fwi_topography.py`` (case ``topo_dual``: 2.4 m x 4 m,
2.5 cm model grid, Delaunay DG band 0.25-1.05 m, P1, Yee at dx/3, 13 shots
batched, 35 ns, float32). Paths:

* ``native``         DeepGPR's CUDA FDTD (staircase) on dx, its own adjoint
* ``torch``          hybrid with the PyTorch step (``cuda_kernels=False``), autograd VJP
* ``cuda-autograd``  hybrid with the fused kernels, autograd VJP of the step
                     (``DEEPGPR_HYBRID_AUTOGRAD_VJP=1``)
* ``cuda``           hybrid with the fused kernels and the hand-written step transpose
                     (the default)

The gradient is that of ``J = 1/2 ||d||^2`` with respect to ``eps_r`` and
``sigma`` (sqrt(nt) checkpointing). A second part times a single-shot forward
run with and without CUDA Graph replay (small state: graphs are on).

    python examples/hybrid/cuda2d_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fwi_topography as FW  # noqa: E402

from DeepGPR.hybrid import HybridTM2D, graphs, make_propagator  # noqa: E402

OUT = FW.OUT / "cuda2d_benchmark.json"
ENV = {"cuda-autograd": {"DEEPGPR_HYBRID_AUTOGRAD_VJP": "1"}}
CASE = FW.CASES["topo_dual"]


def timed(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0, (torch.cuda.max_memory_allocated() - base) / 2 ** 30


def propagator(path):
    eps0, sig0 = FW.background(FW.DX)
    src, rec = FW.antennas(CASE)
    if path == "native":
        prop = make_propagator("fdtd", eps0, sig0, FW.DX, src, rec, surface=CASE["surface"],
                               pmlthick=FW.PML, device="cuda")
    else:
        prop = make_propagator("hybrid_dgtd", eps0, sig0, FW.DX, src, rec,
                               surface=CASE["surface"], pmlthick=FW.PML, dg_band=FW.BAND,
                               dg_order=1, mesh="delaunay", fdtd_refinement=3, device="cuda",
                               dtype=torch.float32, cuda_kernels=(path != "torch"))
    return prop, eps0, sig0


def case(path):
    prop, eps0, sig0 = propagator(path)
    nt = int(round(FW.T_MAX / prop.dt))
    w = FW.wavelet(nt, prop.dt).to("cuda", torch.float32)
    for k, v in ENV.get(path, {}).items():
        os.environ[k] = v
    try:
        e = eps0.to("cuda", torch.float32)
        s = sig0.to("cuda", torch.float32)
        with torch.no_grad():
            _, t_fwd, m_fwd = timed(lambda: prop(e, s, w[None]))
        e2, s2 = e.clone().requires_grad_(True), s.clone().requires_grad_(True)

        def grad():
            d = prop(e2, s2, w[None])
            (0.5 * (d ** 2).sum()).backward()
            return e2.grad.clone(), s2.grad.clone()
        g, t_grad, m_grad = timed(grad)
    finally:
        for k in ENV.get(path, {}):
            os.environ.pop(k, None)
    row = dict(path=path, nt=nt, dt=prop.dt, forward_s=t_fwd, forward_ms_step=1e3 * t_fwd / nt,
               forward_GB=m_fwd, gradient_s=t_grad, gradient_ms_step=1e3 * t_grad / nt,
               gradient_GB=m_grad)
    torch.cuda.empty_cache()
    return row, [t.cpu() for t in g]


def graph_case():
    """Single-shot forward run (small state): CUDA Graph replay against the eager step."""
    eps0, sig0 = FW.background(FW.DX)
    sol = HybridTM2D(eps0, sig0, FW.DX, surface=CASE["surface"], pmlthick=FW.PML,
                     dg_band=FW.BAND, dg_order=1, mesh="delaunay", fdtd_refinement=3,
                     device="cuda", dtype=torch.float32)
    src, rec = FW.antennas(CASE)
    nt = int(round(FW.T_MAX / sol.dt))
    w = FW.wavelet(nt, sol.dt)
    rows = {}
    for env in (None, "1"):
        if env:
            os.environ["DEEPGPR_HYBRID_DISABLE_GRAPHS"] = env
        try:
            res = sol.run(w, src[:1], rec[:1])
            rows["eager" if env else "graph"] = res["seconds"]
        finally:
            os.environ.pop("DEEPGPR_HYBRID_DISABLE_GRAPHS", None)
    return dict(nt=nt, state_MB=graphs.state_bytes(sol.static) / 2 ** 20,
                graph_ms_step=1e3 * rows["graph"] / nt, eager_ms_step=1e3 * rows["eager"] / nt)


def main():
    rows, grads = [], {}
    for path in ("native", "torch", "cuda-autograd", "cuda"):
        row, g = case(path)
        rows.append(row)
        if path != "native":
            grads[path] = g
        print(f"{path:14s} nt {row['nt']:5d}  forward {row['forward_s']:7.2f} s "
              f"({row['forward_ms_step']:6.3f} ms/step, {row['forward_GB']:.2f} GB)  gradient "
              f"{row['gradient_s']:7.2f} s ({row['gradient_ms_step']:6.3f} ms/step, "
              f"{row['gradient_GB']:.2f} GB)", flush=True)
    for path in ("torch", "cuda-autograd"):
        err = [float((a - b).norm() / b.norm()) for a, b in zip(grads[path], grads["cuda"])]
        print(f"  gradient {path} vs cuda: eps {err[0]:.1e}, sigma {err[1]:.1e}")
        rows.append(dict(path=path, gradient_vs_cuda=err))
    gr = graph_case()
    print(f"single shot (state {gr['state_MB']:.1f} MB): graph {gr['graph_ms_step']:.3f} ms/step, "
          f"eager {gr['eager_ms_step']:.3f} ms/step", flush=True)
    rows.append(dict(path="single-shot forward", **gr))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
