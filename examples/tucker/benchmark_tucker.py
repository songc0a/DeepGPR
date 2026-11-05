"""Reproducible benchmarks of the two Tucker modules on a 3D GPR model.

Model: the OverThrust section of the 3D FWI example (``examples/OverThrust.npy``),
resampled to 110 x 100 cells, extruded over 120 cells along x with a smooth
lateral dip (so the model is genuinely 3D) and covered by 10 air cells:
``eps_r`` is ``(120, 120, 100)``, ``sigma = 5e-4 eps_r`` (0 in air).

Subcommands (each configuration runs in its own process so that peak CUDA
memory is measured in isolation)::

    python examples/tucker/benchmark_tucker.py model      # Module A
    python examples/tucker/benchmark_tucker.py wavefield  # Module B
    python examples/tucker/benchmark_tucker.py fwi        # inversions
    python examples/tucker/benchmark_tucker.py report     # Markdown tables
    ... --quick   (40 x 40 x 32 smoke version)

Results: ``examples/tucker/results/<scale>/*.json`` and ``report.md``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import DeepGPR  # noqa: E402
from DeepGPR.solver.sketched import WavefieldSketchStats  # noqa: E402
from DeepGPR.tucker import (  # noqa: E402
    TuckerMaterialModel,
    TuckerModelConfig,
    TuckerSketchConfig,
    TuckerTensor,
    cosine_similarity,
    hosvd,
    relative_error,
    tucker_parameter_count,
    tucker_reconstruct,
)

RESULTS = Path(__file__).resolve().parent / "results"
GIB = 1024**3
MIB = 1024**2


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------


def scale_name(quick: bool) -> str:
    return "quick" if quick else "full"


def build_case(
    quick: bool, device: torch.device, shots: Optional[int] = None, nt: Optional[int] = None
) -> Dict[str, Any]:
    """True/initial models and solver keywords (see the module docstring)."""
    if quick:
        nx, depth, nz, air, default_nt, nshots = 40, 30, 32, 10, 300, 2
    else:
        nx, depth, nz, air, default_nt, nshots = 120, 110, 100, 10, 600, 4
    nshots = shots or nshots
    nt = nt or default_nt
    section = torch.tensor(np.load(ROOT / "examples" / "OverThrust.npy"), dtype=torch.float32)
    section = F.interpolate(section[None, None], size=(depth, nz), mode="bilinear", align_corners=True)[0, 0]
    rows = torch.arange(depth, dtype=torch.float32)
    eps = torch.empty(nx, depth, nz)
    for ix in range(nx):
        # Smooth lateral dip: up to +-8 % of the depth across the model.
        shift = 0.08 * depth * math.sin(math.pi * (ix / (nx - 1) - 0.5))
        source_rows = (rows - shift).clamp(0, depth - 1)
        low = source_rows.floor().long()
        high = (low + 1).clamp(max=depth - 1)
        weight = (source_rows - low.float())[:, None]
        eps[ix] = section[low] * (1 - weight) + section[high] * weight
    eps = torch.cat([torch.ones(nx, air, nz), eps], dim=1)
    sigma = 5e-4 * eps
    sigma[:, :air] = 0.0
    smooth = eps[None, None]
    for _ in range(4):
        smooth = F.avg_pool3d(smooth, 9, stride=1, padding=4, count_include_pad=False)
    eps_init = smooth[0, 0]
    eps_init[:, :air] = 1.0
    sigma_init = torch.full_like(sigma, float(sigma[:, air:].mean()))
    sigma_init[:, :air] = 0.0
    dx, dt, freq = 0.02, 3.0e-11, 5.0e8
    wavelet = DeepGPR.wavelet.ricker(freq, nt, dt, 1.2 / freq).reshape(1, nt, 1)
    source_location = torch.zeros((nshots, 1, 3), dtype=torch.int32)
    source_location[:, 0, 0] = torch.linspace(nx * 0.3, nx * 0.7, nshots).round().int() if nshots > 1 else nx // 2
    source_location[:, 0, 1] = air
    source_location[:, 0, 2] = torch.linspace(nz * 0.3, nz * 0.7, nshots).round().int() if nshots > 1 else nz // 2
    gx = torch.arange(4, nx - 4, max(nx // 10, 2))
    gz = torch.arange(4, nz - 4, max(nz // 10, 2))
    grid = torch.stack(torch.meshgrid(gx, gz, indexing="ij"), dim=-1).reshape(-1, 2)
    receiver_location = torch.zeros((nshots, grid.shape[0], 3), dtype=torch.int32)
    receiver_location[:, :, 0] = grid[:, 0]
    receiver_location[:, :, 1] = air
    receiver_location[:, :, 2] = grid[:, 1]
    kwargs = dict(
        device=device,
        dx=dx,
        dt=dt,
        source_amplitudes=wavelet.to(device),
        pmlthick=10,
        mode=3,
        source_direction=2,
        receiver_component=2,
    )
    return dict(
        eps=eps.to(device), sigma=sigma.to(device), eps_init=eps_init.to(device), sigma_init=sigma_init.to(device),
        air=air, nt=nt, shots=nshots, source_location=source_location.to(device),
        receiver_location=receiver_location.to(device), kwargs=kwargs,
    )


def shot_kwargs(case: Dict[str, Any], shot: int) -> Dict[str, Any]:
    options = dict(case["kwargs"])
    options["source_location"] = case["source_location"][shot : shot + 1]
    options["receiver_location"] = case["receiver_location"][shot : shot + 1]
    return options


def observed_data(case: Dict[str, Any], quick: bool) -> List[torch.Tensor]:
    cache = RESULTS / scale_name(quick) / "cache" / f"observed_{case['shots']}_{case['nt']}.pt"
    if cache.exists():
        return [t.to(case["eps"].device) for t in torch.load(cache)]
    data = []
    with torch.no_grad():
        for shot in range(case["shots"]):
            out = DeepGPR.compute(eps_r=case["eps"], sigma=case["sigma"], save_wavefield_history=False, **shot_kwargs(case, shot))
            data.append(out[-1].clone())
    atomic_save([t.cpu() for t in data], cache)
    return data


def atomic_save(obj: Any, path: Path) -> None:
    """``torch.save`` through a temporary file (parallel workers share caches)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def fresh_peak() -> int:
    gc.collect()
    sync()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


# ---------------------------------------------------------------------------
# Wavefield-history benchmark (Module B)
# ---------------------------------------------------------------------------


def checkpointed_shot(eps, sigma, options, segment_steps, sketch, collected=None):
    """User-level temporal checkpointing (tests/checkpoint_example.ipynb pattern).

    ``collected`` receives one :class:`WavefieldSketchStats` per sketched
    segment call (forward recomputations during backward included).
    """
    from torch.utils.checkpoint import checkpoint

    source = options["source_amplitudes"]
    common = {k: v for k, v in options.items() if k != "source_amplitudes"}
    state = ()
    pieces = []
    for start in range(0, source.shape[1], segment_steps):
        chunk = source[:, start : start + segment_steps].contiguous()

        def advance(er, se, src, *boundary):
            work = tuple(t.clone() for t in boundary)
            stats = None
            if sketch is not None and collected is not None and torch.is_grad_enabled():
                stats = WavefieldSketchStats()
                collected.append(stats)
            result = DeepGPR.compute(
                eps_r=er, sigma=se, source_amplitudes=src,
                E=work[:3] or None, H=work[3:6] or None, PML=work[6:] or None,
                wavefield_sketch=sketch, wavefield_sketch_stats=stats, **common,
            )
            return (*result[1], *result[2], *result[3], result[-1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs = checkpoint(advance, eps, sigma, chunk, *state, preserve_rng_state=False)
        state = outputs[:30]
        pieces.append(outputs[30])
    return torch.cat(pieces, dim=1)


def sketch_config(spec: Dict[str, Any], measure: bool = False) -> Optional[TuckerSketchConfig]:
    if "sketch" not in spec:
        return None
    options = dict(spec["sketch"])
    options["measure_error"] = measure
    return TuckerSketchConfig(**options)


def wavefield_worker(spec: Dict[str, Any], quick: bool) -> Dict[str, Any]:
    device = torch.device("cuda")
    case = build_case(quick, device, shots=1, nt=spec.get("nt"))
    observed = observed_data(case, quick)[0]
    options = shot_kwargs(case, 0)
    if spec.get("rhs_history"):
        options["wavefield_rhs_history"] = spec["rhs_history"]
    sketch = sketch_config(spec)
    stats = WavefieldSketchStats(synchronize=True) if sketch is not None else None
    eps = case["eps_init"].clone().requires_grad_(True)
    sigma = case["sigma_init"].clone().requires_grad_(True)
    scale = observed.square().sum()

    # Warm-up (CUDA context, native resource cache, allocator).
    with torch.no_grad():
        DeepGPR.compute(eps_r=case["eps_init"], sigma=case["sigma_init"], save_wavefield_history=False, **options)
    base = fresh_peak()
    t0 = time.perf_counter()
    collected: List[WavefieldSketchStats] = []
    if spec.get("checkpoint"):
        data = checkpointed_shot(eps, sigma, options, spec["checkpoint"], sketch, collected)
    else:
        outputs = DeepGPR.compute(eps_r=eps, sigma=sigma, wavefield_sketch=sketch, wavefield_sketch_stats=stats, **options)
        data = outputs[-1]
        del outputs
    loss = 0.5 * (data - observed).square().sum() / scale
    sync()
    t1 = time.perf_counter()
    forward_peak = torch.cuda.max_memory_allocated() - base
    loss.backward()
    sync()
    t2 = time.perf_counter()
    result: Dict[str, Any] = dict(
        name=spec["name"],
        loss=float(loss),
        forward_s=t1 - t0,
        backward_s=t2 - t1,
        total_s=t2 - t0,
        forward_peak_gib=forward_peak / GIB,
        peak_allocated_gib=(torch.cuda.max_memory_allocated() - base) / GIB,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / GIB,
    )
    del data, loss
    grads = {"eps": eps.grad.detach(), "sigma": sigma.grad.detach()}
    # Uncompressed history sizes (float32, Ex/Ey/Ez frames on the extended grid).
    pml = 2 * options["pmlthick"]
    cells = math.prod(size + pml for size in case["eps"].shape)
    frame_bytes = 3 * cells * 4
    nt = case["nt"]
    result["raw_e_frame_bytes"] = frame_bytes
    if "sketch" not in spec:
        if spec.get("checkpoint"):
            # One recomputed segment's default (E-only + final frame) history at a time.
            result["history_bytes"] = (spec["checkpoint"] + 1) * frame_bytes
        elif spec.get("rhs_history") == "stored":
            result["history_bytes"] = 2 * nt * frame_bytes
        else:
            result["history_bytes"] = (nt + 1) * frame_bytes
    # Gradients are cached per configuration; errors against the reference
    # configuration are computed by ``report`` (workers may run in parallel).
    atomic_save({k: v.cpu() for k, v in grads.items()}, gradient_cache(quick, spec["name"]))
    result["reference"] = spec.get("reference", "none")
    result["nt"] = nt
    if collected:
        # Checkpoint + sketch: one compressed history per recomputed segment,
        # alive one at a time; report the largest.
        stats = max(collected, key=lambda item: item.size.stored_bytes)
    if stats is not None:
        size = stats.size
        compressed = [v for v in size.components.values() if v[2]]
        mask_bytes = len(compressed) * -(-cells // 8)
        tucker_values_bytes = sum(v[0] for v in compressed) - mask_bytes
        compressed_raw = sum(v[1] for v in compressed)
        result["theoretical_ratio"] = compressed_raw / tucker_values_bytes if tucker_values_bytes else None
        result["actual_ratio_vs_er"] = size.raw_bytes / size.stored_bytes
        result["actual_ratio_vs_default"] = (nt + 1) * frame_bytes / size.stored_bytes
        if collected:
            for key in ("compress_seconds", "decompress_seconds", "forward_native_seconds", "backward_native_seconds"):
                setattr(stats, key, sum(getattr(item, key) for item in collected))
        result.update(
            history_bytes=size.stored_bytes,
            raw_er_bytes=size.raw_bytes,
            map_bytes=size.map_bytes,
            segment_buffer_bytes=stats.segment_buffer_bytes,
            segments=stats.segments,
            native_forward_s=stats.forward_native_seconds,
            compress_s=stats.compress_seconds,
            native_backward_s=stats.backward_native_seconds,
            decompress_s=stats.decompress_seconds,
            components={name: dict(stored=v[0], raw=v[1], compressed=v[2], reason=v[3]) for name, v in size.components.items()},
        )
        # Error pass (separate: measuring costs time).
        del eps, sigma, grads
        measure = sketch_config(spec, measure=True)
        assert measure is not None
        if measure.time_block == 1 or (measure.segment_steps or 16) % measure.time_block == 0:
            stats2 = WavefieldSketchStats()
            with torch.no_grad():
                eps2 = case["eps_init"].clone().requires_grad_(True)
            outputs = DeepGPR.compute(eps_r=eps2, sigma=case["sigma_init"], wavefield_sketch=measure, wavefield_sketch_stats=stats2, **options)
            result["wavefield_rel_l2"] = {k: v.relative_l2 for k, v in stats2.errors.items()}
            result["wavefield_max_frame_rel"] = {k: v.max_frame_relative for k, v in stats2.errors.items()}
            del outputs, eps2
    return result


def gradient_cache(quick: bool, name: str) -> Path:
    safe = name.replace("+", "_plus_").replace("/", "_")
    return RESULTS / scale_name(quick) / "cache" / f"gradients_{safe}.pt"


def add_gradient_errors(results: Dict[str, Any], quick: bool) -> None:
    """Relative L2 error and cosine of every cached gradient vs its reference."""
    for name, result in results.items():
        if name.startswith("_") or "error" in result:
            continue
        reference_name = result.get("reference", "none")
        path, reference_path = gradient_cache(quick, name), gradient_cache(quick, reference_name)
        if not (path.exists() and reference_path.exists()):
            continue
        grads, reference = torch.load(path), torch.load(reference_path)
        for key in ("eps", "sigma"):
            result[f"grad_{key}_rel_l2"] = relative_error(grads[key], reference[key])
            result[f"grad_{key}_cosine"] = cosine_similarity(grads[key], reference[key])


def wavefield_specs(quick: bool) -> List[Dict[str, Any]]:
    seg = 8
    specs: List[Dict[str, Any]] = [
        dict(name="none"),
        dict(name="none_ER_stored", rhs_history="stored"),
        dict(name="checkpoint_60", checkpoint=60),
        dict(name="checkpoint_150", checkpoint=150),
    ]
    for rank in (8, 12, 16, 24):
        specs.append(dict(name=f"tucker_r{rank}", sketch=dict(ranks=rank, oversampling=8, segment_steps=seg)))
    for rank in (12, 16, 24):
        specs.append(dict(name=f"tucker_r{rank}_fromE", sketch=dict(ranks=rank, oversampling=8, segment_steps=seg, rhs="from_e")))
    specs.append(dict(name="tucker_r16_tb8", sketch=dict(ranks=16, oversampling=8, segment_steps=seg, time_block=8, time_rank=4)))
    specs.append(dict(name="tucker_r24_tb8", sketch=dict(ranks=24, oversampling=8, segment_steps=seg, time_block=8, time_rank=4)))
    specs.append(dict(name="tucker_r16_seg32", sketch=dict(ranks=16, oversampling=8, segment_steps=32)))
    specs.append(dict(name="checkpoint_150+tucker_r16", checkpoint=150, sketch=dict(ranks=16, oversampling=8, segment_steps=seg)))
    # Twice as many steps: the uncompressed history (~34 GB) cannot fit on a
    # 24 GB card; exact checkpointing is the reference.
    long = dict(nt=1200 if not quick else 600, reference="long_checkpoint_150")
    specs += [
        dict(name="long_none", **long),
        dict(name="long_checkpoint_150", checkpoint=150, **long),
        dict(name="long_tucker_r16", sketch=dict(ranks=16, oversampling=8, segment_steps=seg), **long),
        dict(name="long_tucker_r16_fromE", sketch=dict(ranks=16, oversampling=8, segment_steps=seg, rhs="from_e"), **long),
        dict(name="long_tucker_r16_tb8", sketch=dict(ranks=16, oversampling=8, segment_steps=seg, time_block=8, time_rank=4), **long),
        dict(name="long_checkpoint_150+tucker_r16", checkpoint=150, sketch=dict(ranks=16, oversampling=8, segment_steps=seg), **long),
    ]
    return specs


# ---------------------------------------------------------------------------
# Model benchmark (Module A)
# ---------------------------------------------------------------------------


def model_worker(spec: Dict[str, Any], quick: bool) -> Dict[str, Any]:
    device = torch.device("cuda")
    case = build_case(quick, device, shots=1)
    shape = tuple(case["eps"].shape)
    rank = spec["rank"]
    ranks = (rank,) * 3
    result: Dict[str, Any] = dict(name=spec["name"], shape=shape, ranks=ranks)
    count = tucker_parameter_count(shape, ranks)
    result.update(count._asdict())
    # Representation error of the true and initial models (best HOSVD fit).
    for key in ("eps", "eps_init", "sigma"):
        core, factors = hosvd(case[key].double(), ranks)
        result[f"{key}_hosvd_rel_l2"] = relative_error(tucker_reconstruct(core, factors), case[key].double())
    # Reconstruction time and memory (float32, CUDA, no TF32).
    torch.backends.cuda.matmul.allow_tf32 = False
    module = TuckerTensor.from_tensor(case["eps_init"], ranks).to(device)
    for _ in range(3):
        module().sum().backward()
    sync()
    repeats = 20
    t0 = time.perf_counter()
    for _ in range(repeats):
        value = module()
    sync()
    t1 = time.perf_counter()
    for _ in range(repeats):
        module().square().sum().backward()
    sync()
    t2 = time.perf_counter()
    result["reconstruct_ms"] = 1e3 * (t1 - t0) / repeats
    result["reconstruct_fwd_bwd_ms"] = 1e3 * (t2 - t1) / repeats
    result.update({f"module_{k}": v for k, v in module.memory_report().items()})
    # One FDTD forward + backward through the Tucker model (exact history).
    observed = observed_data(case, quick)[0]
    options = shot_kwargs(case, 0)
    model = TuckerMaterialModel.from_models(
        case["eps_init"], case["sigma_init"],
        TuckerModelConfig(ranks=ranks, mapping="softplus", bounds=(1.0, None)),
        TuckerModelConfig(ranks=ranks, mapping="softplus", bounds=(0.0, None), scale=1e-3),
    )
    base = fresh_peak()
    t0 = time.perf_counter()
    eps, sigma = model()
    data = DeepGPR.compute(eps_r=eps, sigma=sigma, **options)[-1]
    loss = 0.5 * (data - observed).square().sum() / observed.square().sum()
    sync()
    t1 = time.perf_counter()
    loss.backward()
    sync()
    t2 = time.perf_counter()
    result.update(
        fdtd_forward_s=t1 - t0, fdtd_backward_s=t2 - t1,
        fdtd_peak_allocated_gib=(torch.cuda.max_memory_allocated() - base) / GIB,
    )
    del value
    return result


def model_specs(quick: bool) -> List[Dict[str, Any]]:
    return [dict(name=f"tucker_r{r}", rank=r) for r in (8, 12, 16, 24)]


# ---------------------------------------------------------------------------
# FWI benchmark (both modules)
# ---------------------------------------------------------------------------


def fwi_worker(spec: Dict[str, Any], quick: bool) -> Dict[str, Any]:
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    case = build_case(quick, device)
    observed = observed_data(case, quick)
    air = case["air"]
    tucker = spec.get("model_rank")
    ranks = (tucker,) * 3 if tucker else 8
    model = TuckerMaterialModel.from_models(
        case["eps_init"], case["sigma_init"],
        TuckerModelConfig(enabled=bool(tucker), ranks=ranks, mapping="softplus", bounds=(1.0, None)),
        TuckerModelConfig(enabled=False, trainable=False, mapping="softplus", bounds=(0.0, None), scale=1e-3),
    )
    sketch = sketch_config(spec)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.LBFGS(
        parameters, lr=1.0, max_iter=spec.get("max_iter", 20), max_eval=spec.get("max_eval", 25),
        history_size=10, line_search_fn="strong_wolfe",
    )
    history: List[Dict[str, float]] = []
    start = time.perf_counter()
    scale = sum(float(d.square().sum()) for d in observed)

    def closure():
        optimizer.zero_grad()
        total = 0.0
        t0 = time.perf_counter()
        for shot in range(case["shots"]):
            eps, sigma = model()
            data = DeepGPR.compute(eps_r=eps, sigma=sigma, wavefield_sketch=sketch, **shot_kwargs(case, shot))[-1]
            loss = 0.5 * (data - observed[shot]).square().sum() / scale
            loss.backward()
            total += float(loss)
        sync()
        with torch.no_grad():
            eps = model.epsilon()
            rmse = float((eps[:, air:] - case["eps"][:, air:]).square().mean().sqrt())
        history.append(dict(loss=total, eps_rmse=rmse, seconds=time.perf_counter() - start, eval_s=time.perf_counter() - t0))
        return torch.tensor(total, device=device)

    base = fresh_peak()
    optimizer.step(closure)
    initial_rmse = float((case["eps_init"][:, air:] - case["eps"][:, air:]).square().mean().sqrt())
    return dict(
        name=spec["name"], evaluations=len(history), history=history,
        initial_loss=history[0]["loss"], final_loss=min(h["loss"] for h in history),
        initial_eps_rmse=initial_rmse, final_eps_rmse=history[-1]["eps_rmse"],
        mean_eval_s=sum(h["eval_s"] for h in history) / len(history),
        peak_allocated_gib=(torch.cuda.max_memory_allocated() - base) / GIB,
        trainable_parameters=sum(p.numel() for p in parameters),
    )


def fwi_specs(quick: bool) -> List[Dict[str, Any]]:
    sketch = dict(ranks=16, oversampling=8, segment_steps=8)
    return [
        dict(name="dense+none"),
        dict(name="dense+tucker_sketch_r16", sketch=sketch),
        dict(name="tucker_r8+none", model_rank=8),
        dict(name="tucker_r12+none", model_rank=12),
        dict(name="tucker_r16+none", model_rank=16),
        dict(name="tucker_r24+none", model_rank=24),
        dict(name="tucker_r16+tucker_sketch_r16", model_rank=16, sketch=sketch),
    ]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

WORKERS = {"wavefield": (wavefield_worker, wavefield_specs), "model": (model_worker, model_specs), "fwi": (fwi_worker, fwi_specs)}


def run_suite(kind: str, quick: bool, only: Optional[List[str]] = None) -> None:
    out_dir = RESULTS / scale_name(quick)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{kind}.json"
    _, specs = WORKERS[kind]
    for spec in specs(quick):
        if only and spec["name"] not in only:
            continue
        print(f"[{kind}] {spec['name']} ...", flush=True)
        command = [sys.executable, __file__, "_worker", kind, json.dumps(spec)] + (["--quick"] if quick else [])
        process = subprocess.run(command, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
        if process.returncode != 0:
            print(process.stdout[-2000:], process.stderr[-4000:])
            result = dict(name=spec["name"], error=process.stderr.strip().splitlines()[-1] if process.stderr else "failed")
        else:
            result = json.loads(process.stdout.strip().splitlines()[-1])
            print("   ", {k: v for k, v in result.items() if not isinstance(v, (dict, list))}, flush=True)
        result["gpu"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        merge_result(path, spec["name"], result)


def merge_result(path: Path, name: str, result: Dict[str, Any]) -> None:
    """Add one result to a JSON file shared by parallel drivers (file lock)."""
    import fcntl

    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results = json.loads(path.read_text()) if path.exists() else {}
        results[name] = result
        results["_environment"] = environment()
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(results, indent=1))
        os.replace(temporary, path)


def environment() -> Dict[str, Any]:
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return dict(
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        git=git, date=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-2 or abs(value) >= 1e4:
            return f"{value:.2e}"
        return f"{value:.{digits}g}"
    return str(value)


def report(quick: bool) -> None:
    out_dir = RESULTS / scale_name(quick)
    lines = [f"# Tucker benchmark ({scale_name(quick)})", ""]
    wave_path = out_dir / "wavefield.json"
    if wave_path.exists():
        wave = json.loads(wave_path.read_text())
        add_gradient_errors(wave, quick)
        merged = json.dumps(wave, indent=1)
        (out_dir / "wavefield_with_errors.json").write_text(merged)
        env = wave.get("_environment", {})
        lines += [f"Environment: {env}", "", "## Wavefield history (1 shot)", ""]
        base = wave.get("none", {})
        if base.get("history_bytes") is None:
            base = {}
        lines.append(
            "| method | history MiB | theoretical ratio | actual vs E+R | actual vs default | peak GiB | peak vs default "
            "| fwd s | bwd s | total s | compress s | decompress s | e_g eps | cos eps | e_g sigma | cos sigma | max e_W |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, r in wave.items():
            if name.startswith("_"):
                continue
            if "error" in r:
                lines.append(f"| {name} | failed: {r['error']} |")
                continue
            hist = r.get("history_bytes")
            # Default history of the same run length: E-only frames + final frame.
            default_bytes = (r["nt"] + 1) * r["raw_e_frame_bytes"] if "nt" in r else None
            ratio = default_bytes / hist if hist and default_bytes else None
            same_length = base.get("nt") == r.get("nt")
            peak_ratio = (
                r["peak_allocated_gib"] / base["peak_allocated_gib"]
                if same_length and base.get("peak_allocated_gib")
                else None
            )
            ew = r.get("wavefield_rel_l2")
            lines.append(
                f"| {name} | {_fmt(hist / MIB) if hist else '-'} | {_fmt(r.get('theoretical_ratio'))} | "
                f"{_fmt(r.get('actual_ratio_vs_er'))} | {_fmt(ratio)} | {_fmt(r['peak_allocated_gib'])} | {_fmt(peak_ratio)} | "
                f"{_fmt(r['forward_s'])} | {_fmt(r['backward_s'])} | {_fmt(r['total_s'])} | {_fmt(r.get('compress_s'))} | {_fmt(r.get('decompress_s'))} | "
                f"{_fmt(r.get('grad_eps_rel_l2'))} | {_fmt(r.get('grad_eps_cosine'), 7)} | {_fmt(r.get('grad_sigma_rel_l2'))} | {_fmt(r.get('grad_sigma_cosine'), 7)} | "
                f"{_fmt(max(ew.values())) if ew else '-'} |"
            )
        lines.append("")
    model_path = out_dir / "model.json"
    if model_path.exists():
        model = json.loads(model_path.read_text())
        lines += ["## Model parameterization", ""]
        lines.append("| ranks | params | ratio | HOSVD err true eps | err init eps | err sigma | reconstruct ms | rec. fwd+bwd ms | FDTD fwd s | FDTD bwd s | peak GiB |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for name, r in model.items():
            if name.startswith("_") or "error" in r:
                continue
            lines.append(
                f"| {tuple(r['ranks'])} | {r['total']} | {_fmt(r['compression_ratio'])} | {_fmt(r['eps_hosvd_rel_l2'])} | {_fmt(r['eps_init_hosvd_rel_l2'])} | "
                f"{_fmt(r['sigma_hosvd_rel_l2'])} | {_fmt(r['reconstruct_ms'])} | {_fmt(r['reconstruct_fwd_bwd_ms'])} | {_fmt(r['fdtd_forward_s'])} | "
                f"{_fmt(r['fdtd_backward_s'])} | {_fmt(r['fdtd_peak_allocated_gib'])} |"
            )
        lines.append("")
    fwi_path = out_dir / "fwi.json"
    if fwi_path.exists():
        fwi = json.loads(fwi_path.read_text())
        lines += ["## FWI (L-BFGS, eps only)", ""]
        lines.append("| method | trainable | evals | initial loss | best loss | eps RMSE start | eps RMSE end | s / eval | peak GiB |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for name, r in fwi.items():
            if name.startswith("_"):
                continue
            if "error" in r:
                lines.append(f"| {name} | failed: {r['error']} |")
                continue
            lines.append(
                f"| {name} | {r['trainable_parameters']} | {r['evaluations']} | {_fmt(r['initial_loss'])} | {_fmt(r['final_loss'])} | "
                f"{_fmt(r['initial_eps_rmse'])} | {_fmt(r['final_eps_rmse'])} | {_fmt(r['mean_eval_s'])} | {_fmt(r['peak_allocated_gib'])} |"
            )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["model", "wavefield", "fwi", "report", "all", "_worker"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", nargs="*")
    options = parser.parse_args()
    if options.command == "_worker":
        kind, spec = options.args[0], json.loads(options.args[1])
        result = WORKERS[kind][0](spec, options.quick)
        print(json.dumps(result))
        return
    if options.command == "report":
        report(options.quick)
        return
    kinds = ["wavefield", "model", "fwi"] if options.command == "all" else [options.command]
    for kind in kinds:
        run_suite(kind, options.quick, options.only)
    report(options.quick)


if __name__ == "__main__":
    main()
