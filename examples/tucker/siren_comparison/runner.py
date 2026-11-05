"""Process isolation, result files and timing helpers for the SIREN experiments.

Every configuration runs in its own Python process (``_worker``), so CUDA
peak-memory statistics and allocator caches never leak between configurations.
Results are merged into one JSON file per experiment under a file lock;
configurations that already succeeded are skipped on re-runs.
"""

from __future__ import annotations

import fcntl
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
GIB = 1024**3
MIB = 1024**2


def environment() -> Dict[str, Any]:
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    host = os.uname().nodename
    return dict(
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        host=host,
        git=git,
        date=time.strftime("%Y-%m-%d %H:%M:%S"),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
    )


def load_results(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def merge_result(path: Path, name: str, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results = load_results(path)
        results[name] = result
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(results, indent=1))
        os.replace(temporary, path)


def run_specs(script: Path, kind: str, specs: List[Dict[str, Any]], out: Path, force: bool = False, extra_env=None) -> None:
    """Run each spec in its own process; skip specs that already succeeded."""
    done = load_results(out)
    for spec in specs:
        name = spec["name"]
        if not force and name in done and "error" not in done[name]:
            print(f"[{kind}] {name}: done, skipped", flush=True)
            continue
        print(f"[{kind}] {name} ...", flush=True)
        env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}:{HERE}")
        env.update(extra_env or {})
        started = time.time()
        process = subprocess.run(
            [sys.executable, str(script), "_worker", json.dumps(spec)], capture_output=True, text=True, env=env
        )
        if process.returncode != 0:
            tail = process.stderr.strip().splitlines()
            result = dict(name=name, spec=spec, error=tail[-1] if tail else "failed")
            print("   failed:", result["error"][:200], flush=True)
        else:
            result = json.loads(process.stdout.strip().splitlines()[-1])
            result["spec"] = spec
            result["environment"] = environment()
        result["wall_seconds_process"] = time.time() - started
        merge_result(out, name, result)


def cuda_stats(values: List[float]) -> Dict[str, float]:
    return dict(
        mean=statistics.fmean(values),
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
        min=min(values),
        median=statistics.median(values),
        n=len(values),
    )


class PhaseTimer:
    """CUDA-event timestamps on the current stream plus synchronized wall clock."""

    def __init__(self) -> None:
        self.events: List = []
        self.names: List[str] = []

    def mark(self, name: str) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events.append(event)
        self.names.append(name)

    def durations(self) -> Dict[str, float]:
        torch.cuda.synchronize()
        out = {}
        for i in range(1, len(self.events)):
            out[self.names[i]] = self.events[i - 1].elapsed_time(self.events[i]) / 1e3
        out["total"] = self.events[0].elapsed_time(self.events[-1]) / 1e3
        return out


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def parameter_bytes(module: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def worker_main(worker: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
    """Entry point of ``script.py _worker '<json spec>'``."""
    spec = json.loads(sys.argv[2])
    result = worker(spec)
    print(json.dumps(result))


__all__ = [
    "GIB",
    "MIB",
    "PhaseTimer",
    "RESULTS",
    "ROOT",
    "cuda_stats",
    "environment",
    "load_results",
    "merge_result",
    "optimizer_state_bytes",
    "parameter_bytes",
    "run_specs",
    "worker_main",
]
