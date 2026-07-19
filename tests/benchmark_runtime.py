from __future__ import annotations

import argparse
import ctypes
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

try:
    import resource
except ImportError:
    resource = None


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC_PATH = str(REPO_ROOT / "src")
while SRC_PATH in sys.path:
    sys.path.remove(SRC_PATH)
sys.path.insert(0, SRC_PATH)

import DeepGPR


STORAGE_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_rss_bytes() -> int:
    if resource is not None:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    success = ctypes.windll.psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    )
    if not success:
        raise ctypes.WinError()
    return int(counters.PeakWorkingSetSize)


def make_inputs(args: argparse.Namespace, device: torch.device):
    torch.manual_seed(args.seed)
    nx, ny = args.nx, args.ny
    x = torch.arange(nx, dtype=torch.float32, device=device)[:, None]
    y = torch.arange(ny, dtype=torch.float32, device=device)[None, :]
    anomaly = torch.exp(
        -0.5
        * (
            ((x - 0.57 * nx) / max(0.12 * nx, 1.0)) ** 2
            + ((y - 0.61 * ny) / max(0.14 * ny, 1.0)) ** 2
        )
    )
    eps_r = (4.0 + 0.7 * anomaly).requires_grad_(not args.forward_only)
    sigma = (3.0e-4 + 2.0e-4 * anomaly).requires_grad_(not args.forward_only)

    source = DeepGPR.wavelet.ricker(
        args.frequency,
        args.nt,
        args.dt,
        args.peak_time,
        device=device,
    ).reshape(1, args.nt, 1)

    source_location = torch.zeros(
        (args.shots, 1, 3), dtype=torch.int32, device=device
    )
    shot_x = torch.linspace(
        args.pml + 2,
        nx - args.pml - 3,
        args.shots,
        device=device,
    ).round().to(torch.int32)
    source_location[:, 0, 0] = shot_x
    source_location[:, 0, 1] = args.pml + 2

    receiver_location = torch.zeros(
        (args.shots, args.receivers, 3), dtype=torch.int32, device=device
    )
    receiver_x = torch.linspace(
        args.pml + 2,
        nx - args.pml - 3,
        args.receivers,
        device=device,
    ).round().to(torch.int32)
    receiver_location[:, :, 0] = receiver_x
    receiver_location[:, :, 1] = args.pml + 2
    return eps_r, sigma, source, source_location, receiver_location


def run_once(args: argparse.Namespace, device: torch.device):
    eps_r, sigma, source, source_location, receiver_location = make_inputs(args, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    synchronize(device)
    start = time.perf_counter()
    result = DeepGPR.compute(
        device=device,
        dx=args.dx,
        dt=args.dt,
        source_amplitudes=source,
        source_location=source_location,
        receiver_location=receiver_location,
        eps_r=eps_r,
        sigma=sigma,
        pmlthick=args.pml,
        source_direction=2,
        receiver_component=2,
        model_gradient_sampling_interval=args.sampling_interval,
        wavefield_storage_dtype=STORAGE_DTYPES[args.storage],
        fdtd_order=args.order,
        mode=2,
        debug=False,
    )
    synchronize(device)
    forward_seconds = time.perf_counter() - start

    backward_seconds = 0.0
    loss = result[-1].square().mean()
    if not args.forward_only:
        start = time.perf_counter()
        loss.backward()
        synchronize(device)
        backward_seconds = time.perf_counter() - start

    snapshot = {
        "receiver": result[-1].detach().cpu(),
        "loss": loss.detach().cpu(),
        "grad_eps_r": None if eps_r.grad is None else eps_r.grad.detach().cpu(),
        "grad_sigma": None if sigma.grad is None else sigma.grad.detach().cpu(),
    }
    metrics = {
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "total_seconds": forward_seconds + backward_seconds,
        "peak_rss_bytes": peak_rss_bytes(),
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        ),
    }
    del result, eps_r, sigma, source, source_location, receiver_location, loss
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, snapshot


def median(rows: list[dict[str, float]], key: str) -> float:
    return statistics.median(row[key] for row in rows)


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.float()
    candidate = candidate.float()
    difference = candidate - reference
    denominator = torch.linalg.vector_norm(reference)
    relative_l2 = torch.linalg.vector_norm(difference) / denominator.clamp_min(1.0e-30)
    return {
        "max_abs_error": float(difference.abs().max()),
        "mean_abs_error": float(difference.abs().mean()),
        "relative_l2_error": float(relative_l2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the local DeepGPR CPU or CUDA backend."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--nt", type=int, default=500)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--receivers", type=int, default=24)
    parser.add_argument("--pml", type=int, default=10)
    parser.add_argument("--dx", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=2.0e-11)
    parser.add_argument("--frequency", type=float, default=3.5e8)
    parser.add_argument("--peak-time", type=float, default=3.0e-9)
    parser.add_argument("--order", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--storage", choices=tuple(STORAGE_DTYPES), default="fp32")
    parser.add_argument("--sampling-interval", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--save-result", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--torch-profile", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this PyTorch environment.")
        torch.cuda.set_device(device)

    print(f"DeepGPR package: {Path(DeepGPR.__file__).resolve()}")
    print(f"Native library: {DeepGPR.get_deepgpr_library_path(device)}")
    print(
        f"case: device={device}, {args.nx}x{args.ny}, nt={args.nt}, "
        f"shots={args.shots}, receivers={args.receivers}, order={args.order}, "
        f"storage={args.storage}, interval={args.sampling_interval}"
    )

    for _ in range(args.warmup):
        run_once(args, device)
    rows_and_snapshots = [run_once(args, device) for _ in range(args.repeats)]
    rows = [item[0] for item in rows_and_snapshots]
    snapshot = rows_and_snapshots[-1][1]

    summary = {
        "forward_seconds_median": median(rows, "forward_seconds"),
        "backward_seconds_median": median(rows, "backward_seconds"),
        "total_seconds_median": median(rows, "total_seconds"),
        "peak_rss_bytes_max": max(row["peak_rss_bytes"] for row in rows),
        "cuda_peak_allocated_bytes_max": max(
            row["cuda_peak_allocated_bytes"] for row in rows
        ),
        "cuda_peak_reserved_bytes_max": max(
            row["cuda_peak_reserved_bytes"] for row in rows
        ),
    }
    if args.reference is not None:
        reference = torch.load(args.reference, map_location="cpu", weights_only=False)
        comparisons = {}
        for key in ("receiver", "grad_eps_r", "grad_sigma"):
            if reference.get(key) is not None and snapshot[key] is not None:
                comparisons[key] = error_metrics(reference[key], snapshot[key])
        summary["comparison"] = comparisons
    print(json.dumps(summary, indent=2))

    if args.save_result is not None:
        args.save_result.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"summary": summary, **snapshot}, args.save_result)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.torch_profile is not None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            profile_memory=True,
            record_shapes=True,
        ) as profiler:
            run_once(args, device)
        args.torch_profile.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.torch_profile))
        sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
        print(profiler.key_averages().table(sort_by=sort_key, row_limit=30))
        print(f"Chrome trace: {args.torch_profile.resolve()}")


if __name__ == "__main__":
    main()
