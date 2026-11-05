"""Experiment B: how well each parameterization represents the true 3D model.

Target: the 120 x 120 x 100 OverThrust-derived model of
``examples/tucker/benchmark_tucker.py`` (true eps_r and sigma). Every method is
fitted from scratch (no decomposition of the target is used to initialise
networks) with full-batch Adam on

    L = mean(((eps - eps_true) / s_eps)^2) + mean(((log10 sigma - log10 sigma_true) / s_lg)^2)

(``s`` = standard deviation of the true field; air sigma = 0 is compared
through the mapping's floor). The direct-Tucker reference is the truncated
HOSVD of the true unconstrained fields, then the same Adam refinement.
Per method: parameters, step time, peak memory, error curves, time to reach
error thresholds, final RMSE / relative L2 / SSIM (eps; log10 sigma).

Usage: ``python bench_representation.py run [--only ...] [--quick]``.
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
sys.path.insert(0, str(HERE.parent))

from metrics import model_errors  # noqa: E402
from networks import DirectTuckerField, PropertyMapping, SirenField, TuckerSirenField  # noqa: E402
from runner import RESULTS, optimizer_state_bytes, parameter_bytes, run_specs, worker_main  # noqa: E402

THRESHOLDS = (0.10, 0.05, 0.03, 0.02, 0.01)


def true_model(quick: bool, device):
    import benchmark_tucker as bt  # examples/tucker/benchmark_tucker.py (same model as the Tucker FWI)

    case = bt.build_case(quick, device, shots=1)
    return case["eps"], case["sigma"], case["air"]


def representation_specs(quick: bool) -> List[Dict[str, Any]]:
    steps = 400 if quick else 3000
    specs = [dict(name="siren_4x128", method="siren", hidden=128, layers=4, lr=1e-4, steps=steps)]
    specs.append(dict(name="siren_4x256", method="siren", hidden=256, layers=4, lr=1e-4, steps=steps))
    for rank in (8, 12, 16, 24):
        specs.append(dict(name=f"tsiren_2x32_r{rank}", method="tucker_siren", hidden=32, layers=2, rank=rank, lr=1e-3, steps=steps))
        specs.append(dict(name=f"direct_tucker_r{rank}", method="direct_tucker", rank=rank, lr=1e-3, steps=steps))
    # Capacity sweep at fixed rank: is the error limited by the rank or by the 1D networks?
    for width, layers in ((16, 2), (64, 2), (128, 2), (64, 3), (128, 4)):
        specs.append(dict(name=f"tsiren_{layers}x{width}_r16", method="tucker_siren", hidden=width, layers=layers, rank=16, lr=1e-3, steps=steps))
    specs.append(dict(name="tsiren_2x64_r24", method="tucker_siren", hidden=64, layers=2, rank=24, lr=1e-3, steps=steps))
    specs.append(dict(name="tsiren_2x64_r32", method="tucker_siren", hidden=64, layers=2, rank=32, lr=1e-3, steps=steps))
    specs.append(dict(name="direct_tucker_r32", method="direct_tucker", rank=32, lr=1e-3, steps=steps))
    # Learning-rate check of the two main methods (reported as search cost).
    for lr in (3e-5, 3e-4):
        specs.append(dict(name=f"siren_4x128_lr{lr:g}", method="siren", hidden=128, layers=4, lr=lr, steps=steps, lr_search=True))
    for lr in (3e-4, 3e-3):
        specs.append(dict(name=f"tsiren_2x32_r16_lr{lr:g}", method="tucker_siren", hidden=32, layers=2, rank=16, lr=lr, steps=steps, lr_search=True))
    return specs


def worker(spec: Dict[str, Any]) -> Dict[str, Any]:
    device = torch.device(spec.get("device", "cuda"))
    torch.backends.cuda.matmul.allow_tf32 = False
    cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    quick = bool(spec.get("quick"))
    eps_true, sigma_true, air = true_model(quick, device)
    shape = tuple(eps_true.shape)
    mapping = PropertyMapping(eps_min=1.0, eps_max=10.0, sigma_min=0.0, sigma_max=float(sigma_true.max()) * 1.25)
    log_true = torch.log10(sigma_true + 1e-9)
    s_eps, s_lg = float(eps_true.std()), float(log_true.std())
    torch.manual_seed(spec.get("seed", 0))
    if spec["method"] == "siren":
        field = SirenField(shape, mapping, spec["hidden"], spec["layers"], device=device)
    elif spec["method"] == "tucker_siren":
        field = TuckerSirenField(shape, mapping, spec["rank"], spec["hidden"], spec["layers"], device=device)
        with torch.no_grad():  # small random cores (fit from scratch)
            for prop in (field.eps_field, field.sigma_field):
                prop.core.normal_(0.0, 1.0 / math.sqrt(prop.core.numel()))
    else:
        start = time.perf_counter()
        field = DirectTuckerField(mapping.eps_inverse(eps_true), mapping.sigma_inverse(sigma_true), mapping, spec["rank"])
        hosvd_seconds = time.perf_counter() - start
    field = field.to(device)
    optimizer = torch.optim.Adam(field.parameters(), lr=spec["lr"])
    subsurface = (slice(None), slice(air, None), slice(None))

    def evaluate():
        with torch.no_grad():
            eps, sigma = field()
            e = model_errors(eps, eps_true, subsurface)
            s = model_errors(sigma, sigma_true, subsurface, log=True)
        return e, s

    curve: List[Dict[str, float]] = []
    reached: Dict[str, float] = {}
    sync()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    train_seconds = 0.0
    e0, s0 = evaluate()
    curve.append(dict(step=0, seconds=0.0, eps_rel_l2=e0["rel_l2"], eps_rmse=e0["rmse"], sigma_log_rmse=s0["rmse"]))
    for step in range(1, spec["steps"] + 1):
        sync()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        eps, sigma = field()
        loss = ((eps - eps_true) / s_eps).square().mean() + ((torch.log10(sigma + 1e-9) - log_true) / s_lg).square().mean()
        loss.backward()
        optimizer.step()
        sync()
        train_seconds += time.perf_counter() - t0
        if step % 25 == 0 or step == spec["steps"]:
            e, s = evaluate()
            curve.append(dict(step=step, seconds=train_seconds, loss=float(loss), eps_rel_l2=e["rel_l2"], eps_rmse=e["rmse"], sigma_log_rmse=s["rmse"]))
            for threshold in THRESHOLDS:
                key = f"{threshold:g}"
                if key not in reached and e["rel_l2"] <= threshold:
                    reached[key] = train_seconds
    e, s = evaluate()
    with torch.no_grad():
        eps, sigma = field()
    out_dir = RESULTS / ("quick" if quick else "full") / "representation_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"eps": eps.cpu(), "sigma": sigma.cpu()}, out_dir / f"{spec['name']}.pt")
    result = dict(
        name=spec["name"],
        parameters=field.parameter_breakdown(),
        weight_bytes=parameter_bytes(field),
        optimizer_state_bytes=optimizer_state_bytes(optimizer),
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None,
        train_seconds=train_seconds,
        seconds_per_step=train_seconds / spec["steps"],
        eps=e,
        sigma_log10=s,
        time_to_eps_rel_l2=reached,
        curve=curve,
    )
    if spec["method"] == "direct_tucker":
        result["hosvd_seconds"] = hosvd_seconds
        result["initial_hosvd_eps"] = e0
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
    specs = representation_specs(args.quick)
    for s in specs:
        s["quick"] = args.quick
    if args.only:
        specs = [s for s in specs if s["name"] in args.only]
    if args.command == "list":
        print("\n".join(s["name"] for s in specs))
        return
    out = RESULTS / ("quick" if args.quick else "full") / "representation.json"
    run_specs(Path(__file__), "representation", specs, out, force=args.force)


if __name__ == "__main__":
    main()
