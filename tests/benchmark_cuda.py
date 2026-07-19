from __future__ import annotations

import argparse
import gc
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch


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


class PowerSampler:
    """Sample NVIDIA power and utilization without adding a Python dependency."""

    def __init__(self, gpu: str, interval: float = 0.1) -> None:
        self.gpu = gpu
        self.interval = interval
        self.rows: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        command = [
            "nvidia-smi",
            "--id",
            self.gpu,
            "--query-gpu=power.draw,utilization.gpu,clocks.sm",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )
                values = [float(value.strip()) for value in result.stdout.split(",")]
                if len(values) == 3:
                    self.rows.append(tuple(values))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def summary(self) -> str:
        if not self.rows:
            return "power sampling unavailable"
        power = [row[0] for row in self.rows]
        utilization = [row[1] for row in self.rows]
        clocks = [row[2] for row in self.rows]
        return (
            f"power mean/max={statistics.mean(power):.1f}/{max(power):.1f} W, "
            f"GPU utilization mean/max={statistics.mean(utilization):.1f}/{max(utilization):.1f} %, "
            f"SM clock mean={statistics.mean(clocks):.0f} MHz"
        )


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def make_inputs(args: argparse.Namespace, device: torch.device):
    eps_r = torch.full((args.nx, args.ny), 5.0, device=device)
    sigma = torch.full((args.nx, args.ny), 1.0e-3, device=device)
    eps_r.requires_grad_(not args.forward_only)
    sigma.requires_grad_(not args.forward_only)

    source = DeepGPR.wavelet.ricker(
        args.frequency,
        args.nt,
        args.dt,
        args.peak_time,
        device=device,
    ).reshape(1, args.nt, 1)

    source_location = torch.zeros((args.shots, 1, 3), dtype=torch.int32, device=device)
    shot_x = torch.linspace(
        args.pml + 2,
        args.nx - args.pml - 3,
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
        args.nx - args.pml - 3,
        args.receivers,
        device=device,
    ).round().to(torch.int32)
    receiver_location[:, :, 0] = receiver_x
    receiver_location[:, :, 1] = args.pml + 2
    return eps_r, sigma, source, source_location, receiver_location


def run_once(args: argparse.Namespace, device: torch.device) -> tuple[float, float]:
    eps_r, sigma, source, source_location, receiver_location = make_inputs(args, device)
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
    if not args.forward_only:
        loss = result[-1].square().mean()
        start = time.perf_counter()
        loss.backward()
        synchronize(device)
        backward_seconds = time.perf_counter() - start

    del result, eps_r, sigma, source, source_location, receiver_location
    gc.collect()
    torch.cuda.empty_cache()
    return forward_seconds, backward_seconds


def median(values: list[float]) -> float:
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the local DeepGPR CUDA backend.")
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
    parser.add_argument("--storage", choices=tuple(STORAGE_DTYPES), default="fp32")
    parser.add_argument("--sampling-interval", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--sample-power", action="store_true")
    parser.add_argument(
        "--nvidia-smi-index",
        default="0",
        help="Physical GPU index used only by optional nvidia-smi sampling.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch environment.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    print(f"DeepGPR package: {Path(DeepGPR.__file__).resolve()}")
    print(f"Native library: {DeepGPR.get_deepgpr_library_path(device)}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        f"case: {args.nx}x{args.ny}, nt={args.nt}, shots={args.shots}, "
        f"order={args.order}, storage={args.storage}, interval={args.sampling_interval}"
    )

    for _ in range(args.warmup):
        run_once(args, device)

    sampler = PowerSampler(args.nvidia_smi_index) if args.sample_power else None
    if sampler is not None:
        sampler.start()
    timings = [run_once(args, device) for _ in range(args.repeats)]
    if sampler is not None:
        sampler.stop()

    forward = [row[0] for row in timings]
    backward = [row[1] for row in timings]
    total = [sum(row) for row in timings]
    print(f"forward median/min: {median(forward):.6f}/{min(forward):.6f} s")
    if not args.forward_only:
        print(f"backward median/min: {median(backward):.6f}/{min(backward):.6f} s")
    print(f"total median/min: {median(total):.6f}/{min(total):.6f} s")
    if sampler is not None:
        print(sampler.summary())


if __name__ == "__main__":
    main()
