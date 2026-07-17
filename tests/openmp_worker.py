from __future__ import annotations

import argparse
import os
import sys
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one OpenMP DeepGPR comparison case.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    loaded_package = Path(DeepGPR.__file__).resolve()
    expected_package = (REPO_ROOT / "src" / "DeepGPR").resolve()
    loaded_package.relative_to(expected_package)
    torch.set_num_threads(1)
    torch.manual_seed(2026)

    n_shots, nx, ny, nt = 4, 28, 36, 400
    dx, dt, pml = 0.02, 2.0e-11, 5
    x = torch.arange(nx, dtype=torch.float32)[:, None]
    y = torch.arange(ny, dtype=torch.float32)[None, :]
    anomaly = torch.exp(
        -0.5 * (((x - 15.0) / 3.5) ** 2 + ((y - 21.0) / 4.5) ** 2)
    )
    er = (4.0 + 0.7 * anomaly).requires_grad_(True)
    se = (3.0e-4 + 2.0e-4 * anomaly).requires_grad_(True)
    source = DeepGPR.wavelet.ricker(3.5e8, nt, dt, 3.0e-9).reshape(1, nt, 1)
    source_location = torch.zeros((n_shots, 1, 3), dtype=torch.int32)
    source_location[:, 0, 0] = torch.tensor([8, 12, 16, 20])
    source_location[:, 0, 1] = 9
    receiver_location = torch.zeros((n_shots, 3, 3), dtype=torch.int32)
    receiver_location[:, :, 0] = source_location[:, :, 0]
    receiver_location[:, :, 1] = torch.tensor([13, 17, 21])

    start = time.perf_counter()
    result = DeepGPR.compute(
        device="cpu",
        dx=dx,
        dt=dt,
        source_amplitudes=source,
        source_location=source_location,
        receiver_location=receiver_location,
        er=er,
        se=se,
        pmlthick=pml,
        fdtd_order=8,
        mode=2,
        model_gradient_sampling_interval=1,
        wavefield_storage_dtype=torch.float32,
        debug=True,
    )
    shot_weights = torch.tensor([1.0, 0.7, 1.3, 0.9]).reshape(n_shots, 1, 1)
    loss = 0.5 * (result[-1] * shot_weights).square().mean()
    loss.backward()
    elapsed = time.perf_counter() - start

    payload = {
        "requested_omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", "0")),
        "elapsed_seconds": elapsed,
        "receiver": result[-1].detach().cpu(),
        "grad_er": er.grad.detach().cpu(),
        "grad_se": se.grad.detach().cpu(),
        "loss": loss.detach().cpu(),
        "package": str(loaded_package),
        "library": DeepGPR.get_deepgpr_library_path("cpu"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)


if __name__ == "__main__":
    main()
