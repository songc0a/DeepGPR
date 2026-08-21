"""Benchmark FP32/FP16 histories against fused GPU block-INT8 histories.

Run from the repository root after rebuilding ``deepgpr.so``::

    python tests/benchmark_wavefield_compression.py --device cuda:0

Use ``--profile-kernels`` for intrusive CUDA-kernel timing. Nsight Compute is
still required for scoreboard stalls, cache hit rates, occupancy, and registers.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
import DeepGPR


MODES = (
    ("fp32", torch.float32, "none"),
    ("fp16", torch.float16, "none"),
    ("int8", torch.float32, "int8"),
)


def make_inputs(args, requires_grad=True):
    device = torch.device(args.device)
    eps_r = torch.full((args.nx, args.ny), 5.0, device=device)
    sigma = torch.full_like(eps_r, 1.0e-3)
    eps_r.requires_grad_(requires_grad)
    sigma.requires_grad_(requires_grad)
    source = DeepGPR.wavelet.ricker(
        args.frequency, args.nt, args.dt, args.peak_time, device=device
    ).reshape(1, args.nt, 1)
    source_location = torch.zeros(
        (args.shots, 1, 3), dtype=torch.int32, device=device
    )
    receiver_location = torch.zeros(
        (args.shots, args.receivers, 3), dtype=torch.int32, device=device
    )
    shot_x = torch.linspace(
        args.pml + 2, args.nx - args.pml - 3, args.shots, device=device
    ).round().to(torch.int32)
    receiver_x = torch.linspace(
        args.pml + 2, args.nx - args.pml - 3, args.receivers, device=device
    ).round().to(torch.int32)
    source_location[:, 0, 0] = shot_x
    source_location[:, 0, 1] = args.pml + 2
    receiver_location[:, :, 0] = receiver_x
    receiver_location[:, :, 1] = args.pml + 2
    return eps_r, sigma, source, source_location, receiver_location


def compute_kwargs(args, storage_dtype, compression, tensors):
    eps_r, sigma, source, source_location, receiver_location = tensors
    return dict(
        device=args.device,
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
        wavefield_storage_dtype=storage_dtype,
        wavefield_compression=compression,
        wavefield_compression_block_size=(args.block_x, args.block_y)
        if compression == "int8"
        else None,
        fdtd_order=args.order,
        mode=2,
    )


def run_once(args, storage_dtype, compression, capture=False):
    device = torch.device(args.device)
    tensors = make_inputs(args)
    kwargs = compute_kwargs(args, storage_dtype, compression, tensors)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = DeepGPR.compute(**kwargs)
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - start
    loss = result[-1].square().mean()
    start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - start
    peak_bytes = torch.cuda.max_memory_allocated(device)
    history = result[0].detach()
    row = {
        "forward": forward_seconds,
        "backward": backward_seconds,
        "total": forward_seconds + backward_seconds,
        "peak_bytes": peak_bytes,
        # R_saved uses exactly the same packed layout and is retained internally.
        "history_bytes": 2 * history.numel() * history.element_size(),
    }
    if capture:
        row.update(
            history=history.cpu(),
            data=result[-1].detach().cpu(),
            grad_eps=tensors[0].grad.detach().cpu(),
            grad_sigma=tensors[1].grad.detach().cpu(),
        )
    del result, loss, kwargs, tensors
    gc.collect()
    torch.cuda.empty_cache()
    return row


def relative_l2(candidate, reference):
    return float((candidate - reference).norm() / reference.norm().clamp_min(1.0e-30))


def cosine(candidate, reference):
    return float(
        torch.dot(candidate.flatten(), reference.flatten())
        / (candidate.norm() * reference.norm()).clamp_min(1.0e-30)
    )


def profile_mode(args, storage_dtype, compression):
    try:
        from torch.profiler import ProfilerActivity, profile
    except ImportError:
        return {}
    device = torch.device(args.device)
    tensors = make_inputs(args)
    kwargs = compute_kwargs(args, storage_dtype, compression, tensors)
    with profile(activities=[ProfilerActivity.CUDA]) as forward_profile:
        result = DeepGPR.compute(**kwargs)
        torch.cuda.synchronize(device)
    loss = result[-1].square().mean()
    with profile(activities=[ProfilerActivity.CUDA]) as backward_profile:
        loss.backward()
        torch.cuda.synchronize(device)

    def aggregate(profiler, needle=None):
        total_us = 0.0
        for event in profiler.key_averages():
            if needle is not None and needle not in event.key:
                continue
            total_us += float(getattr(event, "self_device_time_total", 0.0))
        return total_us * 1.0e-6

    forward_kernels = aggregate(forward_profile)
    compression = aggregate(forward_profile, "quantize_")
    backward_kernels = aggregate(backward_profile)
    gradient = aggregate(backward_profile, "accumulate_material_gradients")
    return {
        "compression_kernel_seconds": compression,
        "fdtd_kernel_seconds": max(0.0, forward_kernels - compression),
        "gradient_kernel_seconds": gradient,
        "adjoint_kernel_seconds": max(0.0, backward_kernels - gradient),
    }


def median_rows(rows):
    result = dict(rows[0])
    for key in ("forward", "backward", "total", "peak_bytes"):
        result[key] = statistics.median(row[key] for row in rows)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nx", type=int, default=384)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--nt", type=int, default=800)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--receivers", type=int, default=32)
    parser.add_argument("--pml", type=int, default=20)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=1.5e-11)
    parser.add_argument("--frequency", type=float, default=4.0e8)
    parser.add_argument("--peak-time", type=float, default=2.5e-9)
    parser.add_argument("--order", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--sampling-interval", type=int, default=1)
    parser.add_argument("--block-x", type=int, default=8)
    parser.add_argument("--block-y", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-kernels", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch runtime is required.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"native library: {DeepGPR.get_deepgpr_library_path(device)}")

    results = {}
    for name, storage_dtype, compression in MODES:
        for _ in range(args.warmup):
            run_once(args, storage_dtype, compression)
        rows = [run_once(args, storage_dtype, compression) for _ in range(args.repeats)]
        results[name] = median_rows(rows)
        validation = run_once(args, storage_dtype, compression, capture=True)
        results[name].update(
            history=validation["history"],
            data=validation["data"],
            grad_eps=validation["grad_eps"],
            grad_sigma=validation["grad_sigma"],
        )
        if args.profile_kernels:
            results[name].update(profile_mode(args, storage_dtype, compression))

    reference = results["fp32"]
    reference_history_bytes = reference["history_bytes"]
    history_shape = (
        (args.nt + args.sampling_interval - 1) // args.sampling_interval,
        args.shots,
        args.nx,
        args.ny,
        1,
    )
    print(
        "mode   forward_s backward_s total_s peak_MiB history_MiB ratio "
        "eps_rel eps_cos sigma_rel sigma_cos"
    )
    for name, _, _ in MODES:
        row = results[name]
        eps_rel = relative_l2(row["grad_eps"], reference["grad_eps"])
        sigma_rel = relative_l2(row["grad_sigma"], reference["grad_sigma"])
        eps_cos = cosine(row["grad_eps"], reference["grad_eps"])
        sigma_cos = cosine(row["grad_sigma"], reference["grad_sigma"])
        print(
            f"{name:5s} {row['forward']:.6f} {row['backward']:.6f} "
            f"{row['total']:.6f} {row['peak_bytes'] / 2**20:.2f} "
            f"{row['history_bytes'] / 2**20:.2f} "
            f"{reference_history_bytes / row['history_bytes']:.3f} "
            f"{eps_rel:.6e} {eps_cos:.8f} {sigma_rel:.6e} {sigma_cos:.8f}"
        )
        actual_gbps = row["history_bytes"] / row["backward"] / 1.0e9
        logical_gbps = reference_history_bytes / row["backward"] / 1.0e9
        print(
            f"      history-read lower bound={actual_gbps:.3f} GB/s, "
            f"logical decode throughput={logical_gbps:.3f} GB/s"
        )
        if args.profile_kernels:
            print(
                "      FDTD/compression/adjoint/gradient CUDA kernel time: "
                f"{row['fdtd_kernel_seconds']:.6f}/"
                f"{row['compression_kernel_seconds']:.6f}/"
                f"{row['adjoint_kernel_seconds']:.6f}/"
                f"{row['gradient_kernel_seconds']:.6f} s"
            )

    reconstructed = DeepGPR.decompress_wavefield_history(
        results["int8"]["history"].to(device),
        history_shape,
        (args.block_x, args.block_y, 1),
    ).cpu()
    fp32_history = reference["history"].float()
    print(
        "INT8 wavefield RMSE/relative-L2/max-abs: "
        f"{float(torch.sqrt(torch.mean((reconstructed - fp32_history) ** 2))):.6e}/"
        f"{relative_l2(reconstructed, fp32_history):.6e}/"
        f"{float((reconstructed - fp32_history).abs().max()):.6e}"
    )


if __name__ == "__main__":
    main()
