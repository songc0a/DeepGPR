"""Experiment A: model-generation cost of SIREN vs Tucker-SIREN (no FDTD).

One iteration = generate ``(eps_r, sigma)`` on the full grid, contract with a
fixed random cotangent (a stand-in for the FDTD gradient, same for all
methods), back-propagate, Adam step. Per configuration (own process):

* parameters, weight / gradient / Adam-state bytes, analytic FLOPs (``flops.py``),
* CUDA-event times of network evaluation, Tucker reconstruction, mapping,
  backward and optimizer step (5 warm-up + 20 timed iterations, mean/std),
  a decomposed backward (reconstruction vs network), synchronized wall clock,
* the CUDA kernel time and GEMM FLOPs reported by ``torch.profiler``,
* peak allocated / reserved memory and the memory held by the autograd graph.

Usage::

    python bench_generation.py run [--only NAME ...] [--quick]
    python bench_generation.py _worker '<json spec>'
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))

from flops import method_counts  # noqa: E402
from networks import PropertyMapping, SirenField, TuckerSirenField, siren_parameter_count  # noqa: E402
from runner import GIB, RESULTS, PhaseTimer, cuda_stats, optimizer_state_bytes, parameter_bytes, run_specs, worker_main  # noqa: E402

from DeepGPR.tucker import tucker_reconstruct  # noqa: E402

SIZES = {"50^3": (50, 50, 50), "100^3": (100, 100, 100), "120x120x100": (120, 120, 100), "160^3": (160, 160, 160), "200^3": (200, 200, 200)}
CHUNK = 262144


def tucker_siren_parameters(shape, rank, hidden, layers) -> int:
    nets = 2 * sum(siren_parameter_count(1, rank, hidden, layers) for _ in shape)
    return nets + 2 * rank ** len(shape)


def matching_width(target: int, count) -> int:
    return min(range(4, 1024), key=lambda w: abs(count(w) - target))


def generation_specs(quick: bool = False) -> List[Dict[str, Any]]:
    sizes = {"32^3": (32, 32, 32), "48^3": (48, 48, 48)} if quick else SIZES
    specs = []
    for label, shape in sizes.items():
        cells = math.prod(shape)
        base = dict(size=label, shape=list(shape))
        specs.append(dict(name=f"{label}|siren_4x128_full", group="A_fixed", method="siren", hidden=128, layers=4, chunk=0, **base))
        if cells > CHUNK:
            specs.append(dict(name=f"{label}|siren_4x128_chunk{CHUNK}", group="A_fixed", method="siren", hidden=128, layers=4, chunk=CHUNK, **base))
        for rank in (8, 16, 24):
            specs.append(dict(name=f"{label}|tsiren_4x128_r{rank}", group="A_fixed", method="tucker_siren", hidden=128, layers=4, rank=rank, **base))
        for width in (16, 32, 64):
            for rank in (8, 12, 16, 24):
                specs.append(dict(name=f"{label}|tsiren_2x{width}_r{rank}", group="B_small", method="tucker_siren", hidden=width, layers=2, rank=rank, **base))
        # C: matched parameter budgets (cores included).
        p_small = tucker_siren_parameters(shape, 16, 32, 2)
        w = matching_width(p_small, lambda w: siren_parameter_count(3, 2, w, 4))
        specs.append(dict(name=f"{label}|matched_siren_4x{w}", group="C_matched", method="siren", hidden=w, layers=4, chunk=0, match="tsiren_2x32_r16", **base))
        if cells > CHUNK:
            specs.append(dict(name=f"{label}|matched_siren_4x{w}_chunk{CHUNK}", group="C_matched", method="siren", hidden=w, layers=4, chunk=CHUNK, match="tsiren_2x32_r16", **base))
        p_siren = siren_parameter_count(3, 2, 128, 4)
        for rank in (8, 16, 24):
            w = matching_width(p_siren, lambda w, r=rank: tucker_siren_parameters(shape, r, w, 2))
            specs.append(dict(name=f"{label}|matched_tsiren_2x{w}_r{rank}", group="C_matched", method="tucker_siren", hidden=w, layers=2, rank=rank, match="siren_4x128", **base))
    return specs


def _build(spec, device):
    mapping = PropertyMapping()
    shape = tuple(spec["shape"])
    torch.manual_seed(spec.get("seed", 0))
    if spec["method"] == "siren":
        return SirenField(shape, mapping, spec["hidden"], spec["layers"], chunk=spec.get("chunk", 0), device=device).to(device)
    return TuckerSirenField(shape, mapping, spec["rank"], spec["hidden"], spec["layers"], device=device).to(device)


def _generate(field, timer=None):
    """Return (eps, sigma) and mark network / reconstruction / mapping phases."""
    if isinstance(field, SirenField):
        out = field.logits()
        if timer:
            timer.mark("network")
        t_eps, t_sigma = out[:, 0].reshape(field.shape), out[:, 1].reshape(field.shape)
        if timer:
            timer.mark("reconstruction")
    else:
        fe, fs = field.factor_outputs()
        if timer:
            timer.mark("network")
        t_eps = tucker_reconstruct(field.eps_field.core, fe)
        t_sigma = tucker_reconstruct(field.sigma_field.core, fs)
        if timer:
            timer.mark("reconstruction")
    eps, sigma = field.mapping.eps(t_eps), field.mapping.sigma(t_sigma)
    if timer:
        timer.mark("mapping")
    return eps, sigma


def _decomposed_backward(field, w_eps, w_sigma):
    """Backward split into (mapping + reconstruction) and network parts."""
    timer = PhaseTimer()
    timer.mark("start")
    if isinstance(field, SirenField):
        out = field.logits()
        leaf = out.detach().requires_grad_(True)
        timer.mark("forward")
        eps = field.mapping.eps(leaf[:, 0].reshape(field.shape))
        sigma = field.mapping.sigma(leaf[:, 1].reshape(field.shape))
        ((eps * w_eps).sum() + (sigma * w_sigma).sum()).backward()
        timer.mark("backward_reconstruction_mapping")
        out.backward(leaf.grad)
        timer.mark("backward_network")
    else:
        fe, fs = field.factor_outputs()
        leaves_e = [f.detach().requires_grad_(True) for f in fe]
        leaves_s = [f.detach().requires_grad_(True) for f in fs]
        timer.mark("forward")
        eps = field.mapping.eps(tucker_reconstruct(field.eps_field.core, leaves_e))
        sigma = field.mapping.sigma(tucker_reconstruct(field.sigma_field.core, leaves_s))
        ((eps * w_eps).sum() + (sigma * w_sigma).sum()).backward()
        timer.mark("backward_reconstruction_mapping")
        torch.autograd.backward([*fe, *fs], [l.grad for l in (*leaves_e, *leaves_s)])
        timer.mark("backward_network")
    return timer.durations()


def worker(spec: Dict[str, Any]) -> Dict[str, Any]:
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = bool(spec.get("tf32", False))
    torch.backends.cudnn.allow_tf32 = bool(spec.get("tf32", False))
    shape = tuple(spec["shape"])
    result: Dict[str, Any] = dict(name=spec["name"], counts=method_counts(spec, shape))
    try:
        field = _build(spec, device)
        generator = torch.Generator(device="cpu").manual_seed(1)
        w_eps = torch.randn(shape, generator=generator).to(device)
        w_sigma = (1e3 * torch.randn(shape, generator=generator)).to(device)  # sigma ~ 1e-3: comparable scale
        harness_bytes = 2 * w_eps.numel() * 4 + sum(b.numel() * b.element_size() for b in field.buffers())
        optimizer = torch.optim.Adam(field.parameters(), lr=1e-4)
        result.update(parameters=field.parameter_breakdown(), weight_bytes=parameter_bytes(field), harness_bytes=harness_bytes)

        def iteration(timer=None):
            if timer:
                timer.mark("start")
            optimizer.zero_grad(set_to_none=True)
            eps, sigma = _generate(field, timer)
            loss = (eps * w_eps).sum() + (sigma * w_sigma).sum()
            loss.backward()
            if timer:
                timer.mark("backward")
            optimizer.step()
            if timer:
                timer.mark("optimizer")

        for _ in range(5):
            iteration()
        torch.cuda.synchronize()
        result["optimizer_state_bytes"] = optimizer_state_bytes(optimizer)
        result["gradient_bytes"] = sum(p.grad.numel() * p.grad.element_size() for p in field.parameters() if p.grad is not None)

        # Memory: one iteration from a reset peak; graph memory after forward.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        optimizer.zero_grad(set_to_none=True)
        eps, sigma = _generate(field)
        torch.cuda.synchronize()
        result["graph_and_output_bytes"] = torch.cuda.memory_allocated() - before
        result["forward_peak_bytes_above_baseline"] = torch.cuda.max_memory_allocated() - before
        ((eps * w_eps).sum() + (sigma * w_sigma).sum()).backward()
        optimizer.step()
        del eps, sigma
        torch.cuda.synchronize()
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["baseline_allocated_bytes"] = before

        # Timing: CUDA events per phase + synchronized wall clock.
        phases: Dict[str, List[float]] = {}
        walls = []
        for _ in range(int(spec.get("repeats", 20))):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            timer = PhaseTimer()
            iteration(timer)
            durations = timer.durations()
            walls.append(time.perf_counter() - t0)
            for key, value in durations.items():
                phases.setdefault(key, []).append(value)
        result["cuda_event_seconds"] = {k: cuda_stats(v) for k, v in phases.items()}
        result["wall_seconds"] = cuda_stats(walls)

        decomposed: Dict[str, List[float]] = {}
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            for key, value in _decomposed_backward(field, w_eps, w_sigma).items():
                decomposed.setdefault(key, []).append(value)
        result["decomposed_backward_seconds"] = {k: cuda_stats(v) for k, v in decomposed.items()}

        # Kernel time and GEMM FLOPs from the profiler (one iteration).
        from torch.profiler import ProfilerActivity, profile

        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_flops=True) as prof:
            iteration()
            torch.cuda.synchronize()
        from torch.autograd import DeviceType

        kernels = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
        # Pure kernel execution time (the CUDA-event time also contains the
        # gaps in which the GPU waits for kernel launches).
        result["profiler_cuda_kernel_seconds"] = sum(e.time_range.elapsed_us() for e in kernels) / 1e6
        result["profiler_kernel_launches"] = len(kernels)
        result["profiler_flops"] = sum(getattr(e, "flops", 0) or 0 for e in prof.key_averages())
    except RuntimeError as error:  # CUDA OOM is a RuntimeError subclass in torch 1.11 and 2.x
        if "out of memory" not in str(error):
            raise
        result["oom"] = True
        result["error"] = str(error).splitlines()[0][:300]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "list", "_worker"])
    parser.add_argument("spec", nargs="?")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "_worker":
        worker_main(worker)
        return
    specs = generation_specs(args.quick)
    if args.only:
        specs = [s for s in specs if any(token in s["name"] for token in args.only)]
    if args.command == "list":
        for s in specs:
            print(s["name"])
        return
    out = RESULTS / ("quick" if args.quick else "full") / "generation.json"
    run_specs(Path(__file__), "generation", specs, out, force=args.force)


if __name__ == "__main__":
    main()
