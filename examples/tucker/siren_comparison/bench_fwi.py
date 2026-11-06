"""Experiment C: SIREN vs Tucker-SIREN (vs dense, direct Tucker) inside DeepGPR FWI.

Cases (identical physics, data and loss for every method):

* ``3d``: the 120 x 120 x 100 OverThrust-derived model of
  ``examples/tucker/benchmark_tucker.py`` (dx 2 cm, dt 30 ps, 500 MHz,
  nt 600, 4 shots, 10-cell CPML, mode 3), uncompressed default history.
* ``2d``: the 2D FWI notebook at its own size (``2DFWImodel.npy``,
  200 x 100, 50 shots, nt 2000).

Wavefield Tucker sketching is OFF in every run (only the model
parameterization differs). Loss ``0.5 sum ||d - d_obs||^2 / sum ||d_obs||^2``.

Per run: pre-fit to the common initial model (time reported separately), Adam
with the learning rate chosen by a short search (cost reported), per-iteration
time split into model generation / FDTD forward / adjoint / network backward /
optimizer step, model errors (RMSE, SSIM for eps and log10 sigma), peak GPU
memory, checkpoints every 10 iterations (runs resume).

Usage::

    python bench_fwi.py search --case 3d          # learning-rate search
    python bench_fwi.py run --case 3d [--only ...]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parent))

import DeepGPR  # noqa: E402
from metrics import model_errors  # noqa: E402
from networks import (  # noqa: E402
    DenseField,
    DirectTuckerField,
    FixedTopField,
    PropertyMapping,
    gaussian_smooth,
    SirenField,
    TuckerSirenField,
    prefit_field,
    prefit_tucker_siren,
)
from runner import RESULTS, PhaseTimer, load_results, optimizer_state_bytes, parameter_bytes, run_specs, worker_main  # noqa: E402

LR_CANDIDATES = {
    "dense": (0.003, 0.01, 0.03, 0.1),
    "siren": (3e-6, 1e-5, 3e-5, 1e-4),
    "tucker_siren": (3e-5, 1e-4, 3e-4, 1e-3),
    "direct_tucker": (3e-4, 1e-3, 3e-3, 1e-2),
}
SEARCH_ITERATIONS = 15

# Protocols. v1: the first round (box-filtered start, constant 3D sigma start,
# sigma mapping from 0 S/m, air inverted, joint eps/sigma from iteration 1).
# v2 (after review): start = Gaussian-smoothed true model (sigma 10 cells in
# 3D, 20 in 2D, as in examples/{2.2DFWI,3.3DFWI}.ipynb and the SIREN-IFWI
# notebooks), air fixed at its true values and excluded from every
# parameterization, sigma mapping range from the subsurface (0.5 min .. 1.5 max),
# eps only for the first third of the iterations, then eps and sigma.
PROTOCOLS = ("v1", "v2")
ITERATIONS = {"v1": {"3d": 100, "2d": 150}, "v2": {"3d": 150, "2d": 200}}


def result_dir(quick: bool, protocol: str = "v1") -> Path:
    base = "quick" if quick else "full"
    return RESULTS / (base if protocol == "v1" else f"{base}_{protocol}")


def initial_models(case_name: str, case: Dict[str, Any], protocol: str):
    if protocol == "v1":
        return case["eps_init"], case["sigma_init"]
    width = 10.0 if case_name == "3d" else 20.0
    eps0 = gaussian_smooth(case["eps"], width)
    sigma0 = gaussian_smooth(case["sigma"], width)
    air = case.get("air", 0)
    if case_name == "3d" and air:
        eps0[:, :air] = case["eps"][:, :air]
        sigma0[:, :air] = case["sigma"][:, :air]
    return eps0, sigma0


def make_mapping(case: Dict[str, Any], region, protocol: str) -> PropertyMapping:
    if protocol == "v1":
        return PropertyMapping(eps_min=1.0, eps_max=10.0, sigma_min=0.0, sigma_max=float(case["sigma"].max()) * 1.25)
    sub = case["sigma"][region]
    return PropertyMapping(eps_min=1.0, eps_max=10.0, sigma_min=0.5 * float(sub.min()), sigma_max=1.5 * float(sub.max()))


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------


def load_case(case_name: str, quick: bool, device):
    import benchmark_tucker as bt

    if case_name == "3d":
        case = bt.build_case(quick, device)
        observed = bt.observed_data(case, quick)
        shots = [bt.shot_kwargs(case, s) for s in range(case["shots"])]
        air = case["air"]
        region = (slice(None), slice(air, None), slice(None))
        depth = 40 if not quick else 12
        zone = (slice(4, -4), slice(air, air + depth), slice(4, -4))
    else:
        case = bt.build_case_2d(device, quick)
        cache = RESULTS / ("quick" if quick else "full") / "cache" / f"observed2d_{case['nt']}.pt"  # data: same for every protocol
        options = dict(case["kwargs"], source_location=case["source_location"], receiver_location=case["receiver_location"])
        if cache.exists():
            data = torch.load(cache).to(device)
        else:
            with torch.no_grad():
                data = DeepGPR.compute(eps_r=case["eps"], sigma=case["sigma"], save_wavefield_history=False, **options)[-1]
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data.cpu(), cache)
        observed, shots = [data], [options]
        region = (slice(None), slice(None))
        zone = (slice(None), slice(None))
    return case, observed, shots, region, zone


def build_field(spec, case, mapping, device):
    """Pre-fit (or load the cached pre-fit of) one parameterization.

    The pre-fit depends only on (case, method, seed); it is cached so the
    learning-rate search and the main run start from the same state, and its
    originally measured time is reported with every run that uses it.
    """
    quick = bool(spec.get("quick"))
    label = spec["name"].split("|")[-2] if "|search|" in spec["name"] else spec["name"].split("|")[1]
    cache = result_dir(quick, spec.get("protocol", "v1")) / "prefit" / f"{spec['case']}|{label}|seed{spec.get('seed', 0)}.pt"
    field, info = _build_field(spec, case, mapping, device)
    if cache.exists():
        state = torch.load(cache, map_location=device)
        field.load_state_dict(state["field"])
        info = dict(state["info"], prefit_cached=True)
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(field=field.state_dict(), info=info), cache)
    return field, info


def _build_field(spec, case, mapping, device):
    protocol = spec.get("protocol", "v1")
    eps_init, sigma_init = case["eps_start"], case["sigma_start"]
    air = case.get("air", 0) if (protocol == "v2" and spec["case"] == "3d") else 0
    # v2: the air layer is known and fixed; only the subsurface is parameterized.
    eps_sub, sigma_sub = eps_init[:, air:], sigma_init[:, air:]
    shape = tuple(eps_sub.shape)
    t_eps0 = mapping.eps_inverse(eps_sub)
    t_sig0 = mapping.sigma_inverse(sigma_sub)
    torch.manual_seed(spec.get("seed", 0))
    started = time.perf_counter()
    info: Dict[str, Any] = {}
    method = spec["method"]
    if method == "dense":
        field = DenseField(t_eps0, t_sig0, mapping)
    elif method == "direct_tucker":
        field = DirectTuckerField(t_eps0, t_sig0, mapping, spec["rank"])
    elif method == "siren":
        field = SirenField(shape, mapping, spec.get("hidden", 128), spec.get("layers", 4), device=device).to(device)
        info["prefit_history"] = prefit_field(field, t_eps0, t_sig0, steps=spec.get("prefit_steps", 3000), lr=1e-3)
    else:
        field = TuckerSirenField(shape, mapping, spec["rank"], spec.get("hidden", 32), spec.get("layers", 2), device=device).to(device)
        info["prefit_eps"] = prefit_tucker_siren(field.eps_field, t_eps0, steps=spec.get("prefit_steps", 1500))
        info["prefit_sigma"] = prefit_tucker_siren(field.sigma_field, t_sig0, steps=spec.get("prefit_steps", 1500))
    field = field.to(device)
    if air:
        field = FixedTopField(field, eps_init[:, :air], sigma_init[:, :air]).to(device)
    torch.cuda.synchronize()
    info["prefit_seconds"] = time.perf_counter() - started
    with torch.no_grad():
        eps0, sig0 = field()
        info["prefit_eps_vs_init"] = model_errors(eps0, eps_init)
        info["prefit_sigma_vs_init"] = model_errors(sig0, sigma_init, log=True)
    return field, info


# ---------------------------------------------------------------------------
# FWI loop
# ---------------------------------------------------------------------------


def fwi(spec: Dict[str, Any]) -> Dict[str, Any]:
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    quick = bool(spec.get("quick"))
    protocol = spec.get("protocol", "v1")
    case, observed, shots, region, zone = load_case(spec["case"], quick, device)
    case["eps_start"], case["sigma_start"] = initial_models(spec["case"], case, protocol)
    mapping = make_mapping(case, region, protocol)
    scale = sum(float(d.square().sum()) for d in observed)
    field, info = build_field(spec, case, mapping, device)
    optimizer = torch.optim.Adam(field.parameters(), lr=spec["lr"])
    out_dir = result_dir(quick, protocol) / "fwi_runs" / spec["case"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "initial.pt").exists():
        torch.save({"eps": case["eps_start"].cpu(), "sigma": case["sigma_start"].cpu()}, out_dir / "initial.pt")
    # v2: eps only for the first third of the iterations (examples/2.2DFWI.ipynb).
    eps_only_iterations = spec["iterations"] // 3 if protocol == "v2" else 0
    if protocol == "v2" and spec.get("lr_search"):
        eps_only_iterations = ITERATIONS["v2"][spec["case"]] // 3
    ckpt_path = out_dir / f"{spec['name']}.ckpt"
    history: List[Dict[str, Any]] = []
    start_iteration = 0
    if ckpt_path.exists() and not spec.get("lr_search"):
        state = torch.load(ckpt_path, map_location=device)
        field.load_state_dict(state["field"])
        optimizer.load_state_dict(state["optimizer"])
        history, info = state["history"], state["info"]
        start_iteration = len(history)

    def evaluate():
        with torch.no_grad():
            eps, sigma = field()
        return dict(
            eps=model_errors(eps, case["eps"], region),
            eps_zone=model_errors(eps, case["eps"], zone),
            sigma_log=model_errors(sigma, case["sigma"], region, log=True),
        )

    if start_iteration == 0:
        info["initial_errors"] = evaluate()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    iterations = spec["iterations"]
    for iteration in range(start_iteration, iterations):
        timer = PhaseTimer()
        torch.cuda.synchronize()
        wall0 = time.perf_counter()
        timer.mark("start")
        optimizer.zero_grad(set_to_none=True)
        eps, sigma = field()
        timer.mark("generation")
        eps_leaf = eps.detach().requires_grad_(True)
        sig_leaf = sigma.detach().requires_grad_(True)
        fdtd_forward = adjoint = 0.0
        loss_total = 0.0
        for options, data_obs in zip(shots, observed):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            c = torch.cuda.Event(enable_timing=True)
            a.record()
            data = DeepGPR.compute(eps_r=eps_leaf, sigma=sig_leaf, **options)[-1]
            loss = 0.5 * (data - data_obs).square().sum() / scale
            b.record()
            loss.backward()
            c.record()
            torch.cuda.synchronize()
            fdtd_forward += a.elapsed_time(b) / 1e3
            adjoint += b.elapsed_time(c) / 1e3
            loss_total += float(loss)
        timer.mark("fdtd")
        if iteration < eps_only_iterations:
            # eps only: sigma parameters receive no gradient (separate fields)
            # or only the eps-driven part (the shared SIREN network).
            torch.autograd.backward([eps], [eps_leaf.grad])
        else:
            torch.autograd.backward([eps, sigma], [eps_leaf.grad, sig_leaf.grad])
        timer.mark("network_backward")
        optimizer.step()
        timer.mark("optimizer")
        phases = timer.durations()
        record = dict(
            iteration=iteration,
            loss=loss_total,
            generation_s=phases["generation"],
            fdtd_forward_s=fdtd_forward,
            adjoint_s=adjoint,
            network_backward_s=phases["network_backward"],
            optimizer_s=phases["optimizer"],
            iteration_s=phases["total"],
            wall_s=time.perf_counter() - wall0,
        )
        if iteration % 5 == 4 or iteration == iterations - 1:
            record["errors"] = evaluate()
        history.append(record)
        if not spec.get("lr_search") and (iteration % 10 == 9 or iteration == iterations - 1):
            torch.save(dict(field=field.state_dict(), optimizer=optimizer.state_dict(), history=history, info=info), ckpt_path)
    result = dict(
        name=spec["name"],
        info=info,
        parameters=field.parameter_breakdown(),
        weight_bytes=parameter_bytes(field),
        optimizer_state_bytes=optimizer_state_bytes(optimizer),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        history=history,
        resumed_from=start_iteration,
    )
    if not spec.get("lr_search"):
        with torch.no_grad():
            eps, sigma = field()
        torch.save({"eps": eps.cpu(), "sigma": sigma.cpu()}, out_dir / f"{spec['name']}_final.pt")
    return result


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


def methods(case: str) -> List[Dict[str, Any]]:
    out = [dict(label="dense", method="dense"), dict(label="siren_4x128", method="siren", hidden=128, layers=4)]
    for rank in (8, 12, 16, 24):
        out.append(dict(label=f"tsiren_2x32_r{rank}", method="tucker_siren", hidden=32, layers=2, rank=rank))
    out.append(dict(label="direct_tucker_r16", method="direct_tucker", rank=16))
    return out


def search_specs(case: str, quick: bool, protocol: str = "v1") -> List[Dict[str, Any]]:
    specs = []
    for m in methods(case):
        for lr in LR_CANDIDATES[m["method"]]:
            specs.append(dict(m, name=f"{case}|search|{m['label']}|lr{lr:g}", case=case, lr=lr, iterations=SEARCH_ITERATIONS if not quick else 4, lr_search=True, quick=quick, seed=0, protocol=protocol))
    return specs


def chosen_learning_rates(case: str, quick: bool, protocol: str = "v1") -> Dict[str, float]:
    results = load_results(result_dir(quick, protocol) / "fwi_search.json")
    best: Dict[str, tuple] = {}
    for name, r in results.items():
        if not name.startswith(f"{case}|search|") or "error" in r:
            continue
        label = name.split("|")[2]
        losses = [h["loss"] for h in r["history"]]
        # Mean of the last three losses: penalises oscillating learning rates.
        final = sum(losses[-3:]) / len(losses[-3:])
        if all(math.isfinite(v) for v in losses) and (label not in best or final < best[label][0]):
            best[label] = (final, r["spec"]["lr"])
    return {label: lr for label, (_, lr) in best.items()}


def run_specs_for(case: str, quick: bool, seeds=(0, 1, 2), protocol: str = "v1") -> List[Dict[str, Any]]:
    lrs = chosen_learning_rates(case, quick, protocol)
    iterations = ITERATIONS[protocol][case] if not quick else 6
    specs = []
    for m in methods(case):
        if m["label"] not in lrs:
            continue
        deterministic = m["method"] in ("dense", "direct_tucker")
        for seed in ((0,) if deterministic else seeds):
            specs.append(dict(m, name=f"{case}|{m['label']}|seed{seed}", case=case, lr=lrs[m["label"]], iterations=iterations, seed=seed, quick=quick, protocol=protocol))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["search", "run", "list", "_worker"])
    parser.add_argument("spec", nargs="?")
    parser.add_argument("--case", default="3d", choices=["2d", "3d"])
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--protocol", default="v1", choices=PROTOCOLS)
    args = parser.parse_args()
    if args.command == "_worker":
        worker_main(fwi)
        return
    out_dir = result_dir(args.quick, args.protocol)
    if args.command == "search":
        specs, out, kind = search_specs(args.case, args.quick, args.protocol), out_dir / "fwi_search.json", "fwi-search"
    else:
        specs, out, kind = run_specs_for(args.case, args.quick, tuple(args.seeds), args.protocol), out_dir / "fwi.json", "fwi"
    if args.only:
        specs = [s for s in specs if any(token in s["name"] for token in args.only)]
    if args.command == "list":
        print("\n".join(s["name"] for s in specs))
        return
    run_specs(Path(__file__), kind, specs, out, force=args.force)


if __name__ == "__main__":
    main()
