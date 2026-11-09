"""Model-size study: Tucker-compressed history vs exact checkpointing, and the largest 3D FWI.

The 3D benchmark model (``benchmark_tucker.build_case``) is enlarged at fixed
dx (2 cm) by a factor ``g`` per axis (cells ~ g^3, record ``600 g`` steps so
that the waves cross the larger domain). For every size the fastest exact
method that fits is measured (single-level checkpointing on the GPU with
``sqrt(2 nt)``-step segments, or the boundary states in host memory) and the
Tucker-compressed history (``rhs="from_e"``, ranks 16 / 24 / 32). Memory
estimates calibrated on Section 9.5-9.6 of
``docs/tucker_model_wavefield_compression.md`` skip configurations that cannot
fit (GPU 24 GB, host memory capped at ``HOST_LIMIT``); a method that ran out of
GPU memory is not tried at larger sizes. Finally a few dense FWI iterations
(Adam state on the GPU) are run at the largest size the compressed gradient
reached. All runs use one idle GPU (picked once). Results:
``results/full/size.json``, ``size_fwi.json``, ``size_report.md``.

Usage: ``python size_study.py [--quick] [--gpu N]``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import benchmark_tucker as bt  # noqa: E402

FACTORS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.25, 4.5, 5.0)
FACTORS_QUICK = (1.0, 1.25)
RANKS = (16, 24, 32)
GPU_LIMIT = 23.0e9  # bytes that may be reserved on a 24 GB card
HOST_LIMIT = 150e9  # host memory the exact reference may use on the shared server
# Bytes per cell (calibrated): solver state and coefficients, one stored E frame
# (3 components), one boundary state, the Tucker run with 4-step segments.
BASE, FRAME, STATE, TUCKER4 = 150, 12, 30, 200


def size(g: float, quick: bool):
    """(cells with CPML, nt)."""
    if quick:
        nx, depth, nz, air, nt0 = 40, 30, 32, 10, 300
    else:
        nx, depth, nz, air, nt0 = 120, 110, 100, 10, 600
    nx, depth, nz = (int(round(v * g)) for v in (nx, depth, nz))
    return (nx + 20) * (depth + air + 20) * (nz + 20), int(round(nt0 * g))


def plan(g: float, quick: bool):
    """Exact reference spec (or a skip reason) and the Tucker segment length."""
    cells, nt = size(g, quick)
    s_opt = bt.optimal_checkpoint_steps(nt)
    ckpt = cells * (BASE + s_opt * FRAME + math.ceil(nt / s_opt) * STATE)
    name = f"Z{g:g}"
    if ckpt <= GPU_LIMIT:
        reference = dict(name=f"{name}_checkpoint_opt", checkpoint=s_opt, estimate_gb=ckpt / 1e9)
    else:
        reference, reason = None, f"single-level checkpoint needs ~{ckpt / 1e9:.0f} GB on the GPU"
        for steps in (40, 24, 16, 8, 4):
            gpu, host = cells * (BASE + steps * FRAME), cells * math.ceil(nt / steps) * STATE
            if gpu <= GPU_LIMIT:
                if host <= HOST_LIMIT:
                    reference = dict(name=f"{name}_offload_checkpoint", offload_checkpoint=steps, estimate_gb=gpu / 1e9, host_estimate_gb=host / 1e9)
                else:
                    reason += f"; host-offloaded needs ~{host / 1e9:.0f} GB host memory with {steps}-step segments (limit {HOST_LIMIT / 1e9:.0f})"
                break
        if reference is None:
            reference = dict(name=f"{name}_reference", skipped=reason)
    segment = 4 if cells * TUCKER4 <= GPU_LIMIT else 2
    return cells, nt, reference, segment


def run_worker(kind: str, spec: dict, gpu: str) -> dict:
    env = dict(os.environ, PYTHONPATH=str(HERE.parents[1] / "src"), CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512")
    command = [sys.executable, str(HERE / "benchmark_tucker.py"), "_worker", kind, json.dumps(spec)] + (["--quick"] if spec.get("quick") else [])
    start = time.time()
    process = subprocess.run(command, capture_output=True, text=True, env=env)
    if process.returncode != 0:
        tail = process.stderr.strip().splitlines()
        message = next((line for line in reversed(tail) if "Error" in line), tail[-1] if tail else "failed")
        return dict(name=spec["name"], error=message[:400], oom="out of memory" in process.stderr, wall_s=time.time() - start)
    result = json.loads(process.stdout.strip().splitlines()[-1])
    result["wall_s"] = time.time() - start
    return result


def idle_gpu(preferred: str = "") -> str:
    """A GPU with < 1 GB in use and < 5 % utilisation at two checks one minute apart."""
    while True:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        rows = [tuple(int(v) for v in line.split(",")) for line in out.strip().splitlines()]
        idle = [str(i) for i, used, util in rows if used < 1000 and util < 5]
        if preferred:
            idle = [g for g in idle if g == preferred]
        if idle:
            time.sleep(60)
            out = subprocess.run(["nvidia-smi", "-i", idle[0], "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
            used, util = (int(v) for v in out.strip().split(","))
            if used < 1000 and util < 5:
                return idle[0]
        time.sleep(120)


def log(path: Path, message: str) -> None:
    with open(path, "a") as handle:
        handle.write(time.strftime("%F %T ") + message + "\n")
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--gpu", default="", help="use only this GPU (default: the first idle one, then stay on it)")
    parser.add_argument("--no-wait", action="store_true", help="do not wait for an idle GPU (tests; timings not exclusive)")
    args = parser.parse_args()
    out_dir = bt.RESULTS / bt.scale_name(args.quick)
    out_dir.mkdir(parents=True, exist_ok=True)
    path, fwi_path, status = out_dir / "size.json", out_dir / "size_fwi.json", out_dir / "size_status.log"
    gpu = args.gpu if (args.no_wait and args.gpu) else idle_gpu(args.gpu)
    log(status, f"start on GPU {gpu}")
    done = json.loads(path.read_text()) if path.exists() else {}
    failed_ranks, largest = set(), None
    for g in (FACTORS_QUICK if args.quick else FACTORS):
        cells, nt, reference, segment = plan(g, args.quick)
        group = dict(domain_factor=g, nt=nt, quick=args.quick)
        specs = []
        if "skipped" in reference:
            bt.merge_result(path, reference["name"], dict(name=reference["name"], skipped=reference["skipped"], cells=cells, nt=nt))
        else:
            ref = dict(reference, **group, repeats=2 if cells < 20e6 else 1)
            specs.append(ref)
        for rank in RANKS:
            if rank in failed_ranks:
                continue
            specs.append(dict(name=f"Z{g:g}_tucker_r{rank}", sketch=dict(ranks=rank, oversampling=8, segment_steps=segment, rhs="from_e", recovery_device="host"),
                              reference=reference["name"], repeats=2 if cells < 60e6 else 1, **group))
        if not any("sketch" in s for s in specs):
            break
        any_tucker = False
        for spec in specs:
            previous = done.get(spec["name"], {})
            if previous and "error" not in previous:
                result = previous
            else:
                if not args.no_wait:
                    idle_gpu(gpu)  # wait until our GPU is idle again (shared host)
                log(status, f"{spec['name']}: {cells / 1e6:.1f} M cells, nt {nt} ...")
                result = run_worker("size", spec, gpu)
                result.update(cells=cells, nt=nt, gpu=gpu)
                bt.merge_result(path, spec["name"], result)
                log(status, f"   -> {'ERROR ' + result['error'][:160] if 'error' in result else '%.1f s, peak %.2f GiB (reserved %.2f), host %.1f GiB' % (result['total_s'], result['peak_allocated_gib'], result['peak_reserved_gib'], result['host_peak_gib'])}")
            if "sketch" in spec:
                if "error" in result:
                    if result.get("oom"):
                        failed_ranks.add(spec["sketch"]["ranks"])
                else:
                    any_tucker = True
                    largest = (g, spec["sketch"]["ranks"], spec["sketch"]["segment_steps"], cells, nt)
        # Gradient errors for this size, then free the disk (large gradients).
        results = json.loads(path.read_text())
        bt.add_gradient_errors(results, args.quick)
        path.write_text(json.dumps(results, indent=1))
        for spec in specs:
            cache = bt.gradient_cache(args.quick, spec["name"])
            if cache.exists():
                cache.unlink()
        if not any_tucker:
            log(status, f"no Tucker rank fits at g = {g:g}; stopping the size sweep")
            break
    # The largest model: a few dense FWI iterations with the compressed history.
    if largest is not None:
        fwi_done = json.loads(fwi_path.read_text()) if fwi_path.exists() else {}
        sizes = [g for g in (FACTORS_QUICK if args.quick else FACTORS) if g <= largest[0]]
        for g in reversed(sizes):
            cells, nt, _, segment = plan(g, args.quick)
            rank = 24 if 24 not in failed_ranks else 16
            spec = dict(name=f"F{g:g}_tucker_r{rank}", domain_factor=g, nt=nt, quick=args.quick, iterations=3,
                        sketch=dict(ranks=rank, oversampling=8, segment_steps=2 if g >= largest[0] else segment, rhs="from_e", recovery_device="host"))
            if spec["name"] in fwi_done and "error" not in fwi_done[spec["name"]]:
                break
            log(status, f"FWI {spec['name']}: {cells / 1e6:.1f} M cells ...")
            result = run_worker("sizefwi", spec, gpu if args.no_wait else idle_gpu(gpu))
            bt.merge_result(fwi_path, spec["name"], result)
            log(status, f"   -> {'ERROR ' + result['error'][:160] if 'error' in result else 'peak %.2f GiB, %.0f s per iteration' % (result['peak_reserved_gib'], result['seconds_per_iteration'])}")
            if "error" not in result:
                break
    write_report(out_dir)
    log(status, "DONE")


def write_report(out_dir: Path) -> None:
    results = json.loads((out_dir / "size.json").read_text()) if (out_dir / "size.json").exists() else {}
    fwi = json.loads((out_dir / "size_fwi.json").read_text()) if (out_dir / "size_fwi.json").exists() else {}
    lines = ["# Model-size study (Tucker-compressed history vs exact checkpointing)", "",
             "| run | cells (M, with CPML) | nt | default history (GiB) | GPU peak alloc / reserved (GiB) | host peak (GiB) | time per gradient (s) | e_g eps | e_g sigma | max frame error |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in sorted(results.items(), key=lambda kv: (float(kv[0].split("_")[0][1:]) if kv[0][1:2].isdigit() else 0, kv[0])):
        if name.startswith("_"):
            continue
        cells, nt = r.get("cells", 0), r.get("nt", 0)
        default = (nt + 1) * cells * 12 / bt.GIB
        if "skipped" in r:
            lines.append(f"| {name} | {cells / 1e6:.1f} | {nt} | {default:.0f} | skipped: {r['skipped']} | | | | | |")
            continue
        if "error" in r:
            lines.append(f"| {name} | {cells / 1e6:.1f} | {nt} | {default:.0f} | failed: {r['error'][:100]} | | | | | |")
            continue
        ew = r.get("wavefield_rel_l2")
        lines.append(
            f"| {name} | {cells / 1e6:.1f} | {nt} | {default:.0f} | {r['peak_allocated_gib']:.2f} / {r['peak_reserved_gib']:.2f} | {r['host_peak_gib']:.1f} | {r['total_s']:.1f} | "
            f"{bt._fmt(r.get('grad_eps_rel_l2'))} | {bt._fmt(r.get('grad_sigma_rel_l2'))} | {bt._fmt(max(ew.values())) if ew else '-'} |"
        )
    if fwi:
        lines += ["", "## Dense FWI iterations with the compressed history", "",
                  "| run | cells (M, with CPML) | nt | GPU peak alloc / reserved (GiB) | host peak (GiB) | s per iteration | loss | eps RMSE |", "|---|---|---|---|---|---|---|---|"]
        for name, r in fwi.items():
            if "error" in r:
                lines.append(f"| {name} | failed: {r['error'][:120]} | | | | | | |")
                continue
            h = r["history"]
            lines.append(f"| {name} | {r['cells'] / 1e6:.1f} | {r['nt']} | {r['peak_allocated_gib']:.2f} / {r['peak_reserved_gib']:.2f} | {r['host_peak_gib']:.1f} | "
                         f"{r['seconds_per_iteration']:.0f} | {' -> '.join('%.3g' % x['loss'] for x in h[1:])} | {h[0]['eps_rmse']:.3f} -> {h[-1]['eps_rmse']:.3f} |")
    (out_dir / "size_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
