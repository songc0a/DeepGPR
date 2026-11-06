"""Round 4 (protocol v4) on a shared host: run the bench_fwi tasks on GPUs that are idle.

Each task (stage-A search, stage-B search, one FWI run) is started on a GPU
from ``--gpus`` with enough free memory for it (``need_mib``: 2D 13000, 3D
19000, 3D SIREN 23000; at two checks 30 s apart) once its dependencies are
done; a GPU runs one of our tasks at a time and may be shared with other
processes (v4 is about accuracy; timings come from rounds 1-3). Every
task is run a second time after it exits (bench_fwi skips finished
configurations and retries failed ones, e.g. after an OOM caused by another
user). Usage::

    python schedule_v4.py [--gpus 3,5,7]   # log: results/full_v4/logs/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "results" / "full_v4" / "logs"
LABELS = ["dense", "siren_4x128", "tsiren_2x32_r8", "tsiren_2x32_r12", "tsiren_2x32_r16", "tsiren_2x32_r24", "direct_tucker_r16"]
DETERMINISTIC = ("dense", "direct_tucker_r16")


def need_mib(task) -> int:
    if task["name"].startswith("2d"):
        return 13000
    if int(__import__("os").environ.get("FWI_CHECKPOINT_STEPS", "0")):  # exact checkpointing: ~2.7 GiB (+5 GiB SIREN graph)
        return 10000 if "siren_4x128" in task["name"] else 5000
    return 23000 if "siren_4x128" in task["name"] else 19000


def token(label: str) -> str:
    return "|dense|" if label == "dense" else label


def tasks():
    out = []
    for label in LABELS:  # stage-A searches first: they unlock everything else
        out.append(dict(name=f"3d_A_{label}", args=["search", "--case", "3d", "--only", token(label)], deps=[]))
    out.append(dict(name="2d_A", args=["search", "--case", "2d"], deps=[]))
    out.append(dict(name="2d_B", args=["search_sigma", "--case", "2d"], deps=["2d_A"]))
    out.append(dict(name="2d_runs", args=["run", "--case", "2d"], deps=["2d_B"]))
    for label in LABELS:
        out.append(dict(name=f"3d_B_{label}", args=["search_sigma", "--case", "3d", "--only", token(label)], deps=[f"3d_A_{label}"]))
    for seed in (0, 1, 2):  # seeds outer: every method gets a run early
        for label in LABELS:
            if seed and label in DETERMINISTIC:
                continue
            out.append(dict(name=f"3d_run_{label}_s{seed}", args=["run", "--case", "3d", "--only", f"{label}|seed{seed}"], deps=[f"3d_B_{label}"]))
    return out


def free_mib(gpu: str) -> int:
    out = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip() or 0)


def status(message: str) -> None:
    with open(LOGS / "status", "a") as handle:
        handle.write(time.strftime("%F %T ") + message + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="3,5,7")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    LOGS.mkdir(parents=True, exist_ok=True)
    pending, done = tasks(), set()
    running = {}  # gpu -> [task, process, pass]
    idle_since = {}
    status(f"start, {len(pending)} tasks, GPUs {gpus}")
    while pending or running:
        for gpu, (task, process, attempt) in list(running.items()):
            if process.poll() is None:
                continue
            status(f"{task['name']} gpu{gpu} pass{attempt} exit {process.returncode}")
            del running[gpu]
            if attempt == 1:
                pending.insert(0, dict(task, retry=True))
            else:
                done.add(task["name"])
        for gpu in gpus:
            if gpu in running:
                idle_since.pop(gpu, None)
                continue
            free = free_mib(gpu)
            ready = [t for t in pending if all(d in done for d in t["deps"]) and need_mib(t) <= free]
            if not ready:
                idle_since.pop(gpu, None)
                continue
            if time.time() - idle_since.setdefault(gpu, time.time()) < 30:
                continue
            task = ready[0]
            pending.remove(task)
            attempt = 2 if task.get("retry") else 1
            log = open(LOGS / f"{task['name']}.log", "a")
            command = [sys.executable, str(HERE / "bench_fwi.py"), *task["args"], "--protocol", "v4"]
            env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES=gpu)
            running[gpu] = [task, subprocess.Popen(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env), attempt]
            idle_since.pop(gpu, None)
            status(f"{task['name']} gpu{gpu} pass{attempt} start")
        time.sleep(10)
    status("ALL DONE")


if __name__ == "__main__":
    main()
