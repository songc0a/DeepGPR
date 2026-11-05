"""Profile the Tucker reconstruction used by Tucker-SIREN (baseline first).

Baseline: ``DeepGPR.tucker.tucker_reconstruct`` as shipped (mode products in
increasing ``I_n / r_n`` order). Alternatives: every order of the three mode
products and a single ``torch.einsum``. Per variant: forward and
forward + backward CUDA-event time (20 repeats after warm-up), peak allocated
memory above the inputs, and the profiler's kernel time / launch count.

Usage: ``python profile_reconstruction.py`` (writes results/full/reconstruction_profile.json).
"""

from __future__ import annotations

import itertools
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from DeepGPR.tucker import tucker_reconstruct  # noqa: E402
from DeepGPR.tucker._tensor_ops import mode_dot  # noqa: E402


def ordered(core, factors, order):
    result = core
    for mode in order:
        result = mode_dot(result, factors[mode], mode)
    return result


def einsum(core, factors):
    return torch.einsum("abc,ia,jb,kc->ijk", core, *factors)


def measure(fn, inputs, repeats=20):
    weight = None
    for _ in range(3):
        out = fn()
        weight = torch.randn_like(out) if weight is None else weight
        (out * weight).sum().backward()
    torch.cuda.synchronize()
    fwd, both = [], []
    for _ in range(repeats):
        for t in inputs:
            t.grad = None
        a, b, c = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        a.record()
        out = fn()
        b.record()
        (out * weight).sum().backward()
        c.record()
        torch.cuda.synchronize()
        fwd.append(a.elapsed_time(b))
        both.append(a.elapsed_time(c))
        del out
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    (out * weight).sum().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = fn()
        (out * weight).sum().backward()
        torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
    return dict(
        forward_ms=statistics.fmean(fwd), forward_ms_std=statistics.stdev(fwd),
        forward_backward_ms=statistics.fmean(both), forward_backward_ms_std=statistics.stdev(both),
        peak_extra_mib=peak / 2**20, kernel_ms=sum(e.time_range.elapsed_us() for e in kernels) / 1e3, launches=len(kernels),
        output_mib=out.numel() * 4 / 2**20,
    )


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    results = {}
    for shape in ((120, 120, 100), (200, 200, 200)):
        for rank in (8, 16, 24):
            torch.manual_seed(0)
            core = torch.randn((rank,) * 3, device=device, requires_grad=True)
            factors = [torch.randn(n, rank, device=device, requires_grad=True) for n in shape]
            inputs = [core, *factors]
            key = f"{shape[0]}x{shape[1]}x{shape[2]}|r{rank}"
            entry = {"baseline_deepgpr": measure(lambda: tucker_reconstruct(core, factors), inputs)}
            for order in itertools.permutations(range(3)):
                entry["order_" + "".join(map(str, order))] = measure(lambda o=order: ordered(core, factors, o), inputs)
            entry["einsum"] = measure(lambda: einsum(core, factors), inputs)
            results[key] = entry
            print(key, {k: round(v["forward_backward_ms"], 3) for k, v in entry.items()}, flush=True)
    out = HERE / "results" / "full" / "reconstruction_profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(environment=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0)), results=results), indent=1))


if __name__ == "__main__":
    main()
