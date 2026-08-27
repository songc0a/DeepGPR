"""Benchmark pure FDTD and saved-wavefield forward/backward modes.

Formal timings use CUDA events without a profiler attached.  Nsight runs should
use ``--profile-run --warmup 1 --repeat 1 --nvtx`` so they do not overwrite the
formal result files.
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
import DeepGPR


@dataclass(frozen=True)
class Mode:
    name: str
    storage_dtype: torch.dtype
    compression: str
    save_wavefield_history: bool
    forward_nvtx: str
    backward_nvtx: str | None
    conversion_backend: str = "auto"
    int8_reduction_backend: str = "current"


MODES = {
    "fdtd": Mode(
        "fdtd",
        torch.float32,
        "none",
        False,
        "DEEPGPR_FDTD_ONLY_FORWARD",
        None,
    ),
    "fp32": Mode(
        "fp32",
        torch.float32,
        "none",
        True,
        "DEEPGPR_FP32_FORWARD",
        "DEEPGPR_FP32_BACKWARD",
    ),
    "fp16": Mode(
        "fp16",
        torch.float16,
        "none",
        True,
        "DEEPGPR_FP16_FORWARD",
        "DEEPGPR_FP16_BACKWARD",
    ),
    "fp16_legacy": Mode(
        "fp16_legacy",
        torch.float16,
        "none",
        True,
        "DEEPGPR_FP16_LEGACY_FORWARD",
        "DEEPGPR_FP16_LEGACY_BACKWARD",
        "legacy",
    ),
    "fp16_native": Mode(
        "fp16_native",
        torch.float16,
        "none",
        True,
        "DEEPGPR_FP16_NATIVE_FORWARD",
        "DEEPGPR_FP16_NATIVE_BACKWARD",
        "native_scalar",
    ),
    "fp16_vec2": Mode(
        "fp16_vec2",
        torch.float16,
        "none",
        True,
        "DEEPGPR_FP16_VEC2_FORWARD",
        "DEEPGPR_FP16_VEC2_BACKWARD",
        "native_vec2",
    ),
    "bf16_legacy": Mode(
        "bf16_legacy",
        torch.bfloat16,
        "none",
        True,
        "DEEPGPR_BF16_LEGACY_FORWARD",
        "DEEPGPR_BF16_LEGACY_BACKWARD",
        "legacy",
    ),
    "bf16_native": Mode(
        "bf16_native",
        torch.bfloat16,
        "none",
        True,
        "DEEPGPR_BF16_NATIVE_FORWARD",
        "DEEPGPR_BF16_NATIVE_BACKWARD",
        "native_scalar",
    ),
    "bf16_vec2": Mode(
        "bf16_vec2",
        torch.bfloat16,
        "none",
        True,
        "DEEPGPR_BF16_VEC2_FORWARD",
        "DEEPGPR_BF16_VEC2_BACKWARD",
        "native_vec2",
    ),
    "int8": Mode(
        "int8",
        torch.float32,
        "int8",
        True,
        "DEEPGPR_INT8_FORWARD",
        "DEEPGPR_INT8_BACKWARD",
        "legacy",
        "auto",
    ),
    "int8_current": Mode(
        "int8_current",
        torch.float32,
        "int8",
        True,
        "DEEPGPR_INT8_CURRENT_FORWARD",
        "DEEPGPR_INT8_CURRENT_BACKWARD",
        "legacy",
        "current",
    ),
    "int8_cub": Mode(
        "int8_cub",
        torch.float32,
        "int8",
        True,
        "DEEPGPR_INT8_CUB_FORWARD",
        "DEEPGPR_INT8_CUB_BACKWARD",
        "legacy",
        "cub_block",
    ),
    "int8_warp": Mode(
        "int8_warp",
        torch.float32,
        "int8",
        True,
        "DEEPGPR_INT8_WARP_FORWARD",
        "DEEPGPR_INT8_WARP_BACKWARD",
        "legacy",
        "warp_shuffle",
    ),
}


def make_inputs(args):
    device = torch.device(args.device)
    eps_r = torch.full(
        (args.nx, args.ny), 5.0, device=device, requires_grad=True
    )
    sigma = torch.full_like(eps_r, 1.0e-3, requires_grad=True)
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


def compute_kwargs(args, mode, tensors):
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
        save_wavefield_history=mode.save_wavefield_history,
        wavefield_storage_dtype=mode.storage_dtype,
        wavefield_conversion_backend=mode.conversion_backend,
        int8_reduction_backend=mode.int8_reduction_backend,
        use_async_offload=args.async_offload,
        wavefield_compression=mode.compression,
        wavefield_compression_block_size=(args.block_x, args.block_y)
        if mode.compression == "int8"
        else None,
        fdtd_order=args.order,
        mode=2,
    )


def _range_push(enabled, name):
    if enabled:
        torch.cuda.nvtx.range_push(name)


def _range_pop(enabled):
    if enabled:
        torch.cuda.nvtx.range_pop()


def run_once(args, mode, forward_only, capture=False):
    device = torch.device(args.device)
    tensors = make_inputs(args)
    kwargs = compute_kwargs(args, mode, tensors)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    total_start.record()
    forward_start.record()
    _range_push(args.nvtx, mode.forward_nvtx)
    try:
        result = DeepGPR.compute(**kwargs)
    finally:
        _range_pop(args.nvtx)
    forward_end.record()

    loss = None
    if forward_only:
        total_end.record()
    else:
        loss = result[-1].square().mean()
        backward_start.record()
        _range_push(args.nvtx, mode.backward_nvtx)
        try:
            loss.backward()
        finally:
            _range_pop(args.nvtx)
        backward_end.record()
        total_end.record()

    total_end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1.0e3
    forward_ms = forward_start.elapsed_time(forward_end)
    backward_ms = (
        0.0 if forward_only else backward_start.elapsed_time(backward_end)
    )
    total_ms = total_start.elapsed_time(total_end)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    history_bytes = (
        2 * result[0].numel() * result[0].element_size()
        if mode.save_wavefield_history
        else 0
    )
    row = {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "total_ms": total_ms,
        "wall_ms": wall_ms,
        "peak_bytes": peak_bytes,
        "history_bytes": history_bytes,
    }
    if capture:
        row["data"] = result[-1].detach().cpu()
        if not forward_only:
            row["grad_eps"] = tensors[0].grad.detach().cpu()
            row["grad_sigma"] = tensors[1].grad.detach().cpu()

    del result, loss, kwargs, tensors
    gc.collect()
    torch.cuda.empty_cache()
    return row


def summarize(rows):
    summary = {
        "peak_bytes": statistics.median(row["peak_bytes"] for row in rows),
        "history_bytes": rows[0]["history_bytes"],
    }
    for stage in ("forward", "backward", "total", "wall"):
        values = [row[f"{stage}_ms"] for row in rows]
        summary[f"{stage}_mean_ms"] = statistics.fmean(values)
        summary[f"{stage}_median_ms"] = statistics.median(values)
        summary[f"{stage}_min_ms"] = min(values)
        summary[f"{stage}_std_ms"] = statistics.pstdev(values)
    return summary


def relative_l2(candidate, reference):
    return float(
        (candidate - reference).norm()
        / reference.norm().clamp_min(1.0e-30)
    )


def cosine(candidate, reference):
    candidate = candidate.flatten().double()
    reference = reference.flatten().double()
    value = torch.dot(candidate, reference) / (
        candidate.norm() * reference.norm()
    ).clamp_min(1.0e-30)
    return float(value.clamp(-1.0, 1.0))


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _emit_and_save(lines, path, profile_run):
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if not profile_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def selected_modes(args):
    if args.mode == "defaults":
        names = ("fdtd", "fp32", "fp16", "int8") if args.forward_only else (
            "fp32",
            "fp16",
            "int8",
        )
        return [MODES[name] for name in names]
    if args.mode != "all":
        return [MODES[args.mode]]
    names = ("fdtd", "fp32", "fp16_legacy", "fp16_native", "fp16_vec2", "bf16_legacy", "bf16_native", "bf16_vec2", "int8_current", "int8_cub", "int8_warp") if args.forward_only else (
        "fp32",
        "fp16_legacy",
        "fp16_native",
        "fp16_vec2",
        "bf16_legacy",
        "bf16_native",
        "bf16_vec2",
        "int8_current",
        "int8_cub",
        "int8_warp",
    )
    return [MODES[name] for name in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--ny", type=int, default=384)
    parser.add_argument("--nt", type=int, default=1200)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--receivers", type=int, default=64)
    parser.add_argument("--pml", type=int, default=20)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=1.5e-11)
    parser.add_argument("--frequency", type=float, default=4.0e8)
    parser.add_argument("--peak-time", type=float, default=2.5e-9)
    parser.add_argument("--order", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--sampling-interval", type=int, default=1)
    parser.add_argument("--block-x", type=int, default=8)
    parser.add_argument("--block-y", type=int, default=8)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument(
        "--async-offload",
        action="store_true",
        help="benchmark the separate pinned-host history offload path",
    )
    parser.add_argument(
        "--mode", choices=("all", "defaults", *MODES), default="all"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", "--repeats", dest="repeat", type=int, default=20)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument(
        "--profile-run",
        action="store_true",
        help="do not overwrite formal benchmark result files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "profiling_results",
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be nonnegative and --repeat must be positive")
    if args.mode == "fdtd" and not args.forward_only:
        parser.error("--mode fdtd requires --forward-only")
    if args.async_offload and any(mode.compression == "int8" for mode in selected_modes(args)):
        parser.error("--async-offload is incompatible with INT8 modes")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch runtime is required.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    modes = selected_modes(args)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"native library: {DeepGPR.get_deepgpr_library_path(device)}")
    print(
        f"case: nx={args.nx} ny={args.ny} nt={args.nt} shots={args.shots} "
        f"receivers={args.receivers} sampling_interval={args.sampling_interval}"
    )
    print(
        f"run: {'forward-only' if args.forward_only else 'full'} "
        f"warmup={args.warmup} repeat={args.repeat} profiler={args.profile_run} "
        f"async_offload={args.async_offload}"
    )

    summaries = {}
    captures = {}
    for mode in modes:
        print(f"running mode={mode.name}", flush=True)
        for _ in range(args.warmup):
            run_once(args, mode, args.forward_only)
        rows = []
        for repeat_index in range(args.repeat):
            row = run_once(
                args,
                mode,
                args.forward_only,
                capture=repeat_index == 0,
            )
            if repeat_index == 0:
                captures[mode.name] = {
                    key: row.pop(key)
                    for key in ("data", "grad_eps", "grad_sigma")
                    if key in row
                }
            rows.append(row)
        summaries[mode.name] = summarize(rows)

    lines = [
        f"native_library={DeepGPR.get_deepgpr_library_path(device)}",
        (
            f"case nx={args.nx} ny={args.ny} nt={args.nt} shots={args.shots} "
            f"receivers={args.receivers} sampling_interval={args.sampling_interval}"
        ),
    ]

    if args.forward_only:
        lines.append(
            "mode,forward_mean_ms,forward_median_ms,forward_min_ms,"
            "forward_std_ms,wall_median_ms,peak_MiB,history_MiB"
        )
        csv_rows = []
        for mode in modes:
            row = summaries[mode.name]
            values = {
                "mode": mode.name,
                "forward_mean_ms": row["forward_mean_ms"],
                "forward_median_ms": row["forward_median_ms"],
                "forward_min_ms": row["forward_min_ms"],
                "forward_std_ms": row["forward_std_ms"],
                "wall_median_ms": row["wall_median_ms"],
                "peak_MiB": row["peak_bytes"] / 2**20,
                "history_MiB": row["history_bytes"] / 2**20,
            }
            csv_rows.append(values)
            lines.append(
                f"{mode.name},{values['forward_mean_ms']:.6f},"
                f"{values['forward_median_ms']:.6f},"
                f"{values['forward_min_ms']:.6f},"
                f"{values['forward_std_ms']:.6f},"
                f"{values['wall_median_ms']:.6f},"
                f"{values['peak_MiB']:.2f},{values['history_MiB']:.2f}"
            )

        if "fdtd" in summaries and "fp32" in summaries:
            rel = relative_l2(captures["fdtd"]["data"], captures["fp32"]["data"])
            max_abs = float(
                (captures["fdtd"]["data"] - captures["fp32"]["data"])
                .abs()
                .max()
            )
            lines.append(f"fdtd_vs_fp32_data_relative_l2={rel:.9e}")
            lines.append(f"fdtd_vs_fp32_data_max_abs={max_abs:.9e}")
            fdtd_ms = summaries["fdtd"]["forward_median_ms"]
            for name in ("fp32", "fp16_legacy", "fp16_native", "fp16_vec2", "bf16_legacy", "bf16_native", "bf16_vec2", "int8_current", "int8_cub", "int8_warp"):
                if name not in summaries:
                    continue
                overhead = summaries[name]["forward_median_ms"] - fdtd_ms
                percentage = overhead / fdtd_ms * 100.0
                lines.append(
                    f"{name}_history_overhead_ms={overhead:.6f} "
                    f"percentage={percentage:.3f}%"
                )

        output_path = args.output_dir / "forward_benchmark.txt"
        _emit_and_save(lines, output_path, args.profile_run)
        if not args.profile_run:
            _write_csv(
                args.output_dir / "forward_benchmark.csv",
                list(csv_rows[0]),
                csv_rows,
            )
    else:
        lines.append(
            "mode,forward_mean_ms,forward_median_ms,backward_mean_ms,"
            "backward_median_ms,total_mean_ms,total_median_ms,peak_MiB,"
            "history_MiB,wall_median_ms,eps_rel,eps_cos,sigma_rel,sigma_cos"
        )
        reference = captures.get("fp32")
        csv_rows = []
        for mode in modes:
            row = summaries[mode.name]
            capture = captures[mode.name]
            if reference is None:
                eps_rel = sigma_rel = float("nan")
                eps_cos = sigma_cos = float("nan")
            else:
                eps_rel = relative_l2(capture["grad_eps"], reference["grad_eps"])
                sigma_rel = relative_l2(
                    capture["grad_sigma"], reference["grad_sigma"]
                )
                eps_cos = cosine(capture["grad_eps"], reference["grad_eps"])
                sigma_cos = cosine(
                    capture["grad_sigma"], reference["grad_sigma"]
                )
            values = {
                "mode": mode.name,
                "forward_mean_ms": row["forward_mean_ms"],
                "forward_median_ms": row["forward_median_ms"],
                "backward_mean_ms": row["backward_mean_ms"],
                "backward_median_ms": row["backward_median_ms"],
                "total_mean_ms": row["total_mean_ms"],
                "total_median_ms": row["total_median_ms"],
                "peak_MiB": row["peak_bytes"] / 2**20,
                "history_MiB": row["history_bytes"] / 2**20,
                "wall_median_ms": row["wall_median_ms"],
                "eps_rel": eps_rel,
                "eps_cos": eps_cos,
                "sigma_rel": sigma_rel,
                "sigma_cos": sigma_cos,
            }
            csv_rows.append(values)
            lines.append(
                f"{mode.name},{values['forward_mean_ms']:.6f},"
                f"{values['forward_median_ms']:.6f},"
                f"{values['backward_mean_ms']:.6f},"
                f"{values['backward_median_ms']:.6f},"
                f"{values['total_mean_ms']:.6f},"
                f"{values['total_median_ms']:.6f},"
                f"{values['peak_MiB']:.2f},{values['history_MiB']:.2f},"
                f"{values['wall_median_ms']:.6f},"
                f"{eps_rel:.9e},{eps_cos:.9f},{sigma_rel:.9e},{sigma_cos:.9f}"
            )

        output_path = args.output_dir / "full_benchmark.txt"
        _emit_and_save(lines, output_path, args.profile_run)
        if not args.profile_run:
            _write_csv(
                args.output_dir / "full_benchmark.csv",
                list(csv_rows[0]),
                csv_rows,
            )


if __name__ == "__main__":
    main()
