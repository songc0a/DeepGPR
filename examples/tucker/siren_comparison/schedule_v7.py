"""Round 7 (protocol v7, independent sigma) on a shared host: run the bench_fwi tasks on idle GPUs.

Same dispatcher as ``schedule_v6.py``, but a GPU also has to be idle in compute
(utilisation < 30 % at both checks), and any GPU of ``--gpus`` may be used.
Tasks: learning-rate searches (dense, shared SIREN, two SIRENs), the three
500-iteration runs, dense with the amplitude-preserving sigma term and sigma
step 0.1, then two single-change dense variants.

Each task (stage-A search, stage-B search, one FWI run) is started on a GPU
from ``--gpus`` with enough free memory for it (``need_mib``: 2D 13000, 3D
19000, 3D SIREN 23000; at two checks 30 s apart) once its dependencies are
done; a GPU runs one of our tasks at a time and may be shared with other
processes (timings of shared runs are not comparable). Every
task is run a second time after it exits (bench_fwi skips finished
configurations and retries failed ones, e.g. after an OOM caused by another
user). Usage::

    python schedule_v7.py [--gpus 0,1,2,3,4,5,6,7]   # log: results/full_v7/logs/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "results" / "full_v7" / "logs"
LABELS = ["dense", "siren_4x128", "siren2_4x90"]
TUNE = ["tune", "--case", "3d", "--only", "dense", "--loss", "mssim", "--schedule", "cosine", "--iterations", "500", "--sigma-start", "250"]


def need_mib(task) -> int:  # peaks on the 4 x 4 x 2 m model: 9.7 GiB, SIREN 11.6 GiB
    return 16000 if "siren" in task["name"] else 14000


def token(label: str) -> str:
    return "|dense|" if label == "dense" else label


def tasks():
    out = [dict(name=f"3d_A_{label}", args=["search", "--case", "3d", "--only", token(label)], deps=[]) for label in LABELS]
    run = ["run", "--case", "3d", "--sigma-scale", "1", "--seeds", "0", "--only"]
    out.append(dict(name="3d_run_dense", args=run + ["|dense|"], deps=["3d_A_dense"]))
    out.append(dict(name="3d_tune_dense_amp_s0.1", args=TUNE + ["--sigma-loss", "l2", "--sigma-loss-gain", "2", "--sigma-scale", "0.1", "--tag", "v7_amp_s0.1"], deps=["3d_A_dense"]))
    out.append(dict(name="3d_run_siren_4x128", args=run + ["siren_4x128|seed0"], deps=["3d_A_siren_4x128"]))
    out.append(dict(name="3d_run_siren2_4x90", args=run + ["siren2_4x90|seed0"], deps=["3d_A_siren2_4x90"]))
    # single-change variants (only when a GPU is free)
    out.append(dict(name="3d_tune_dense_s0.1", args=TUNE + ["--sigma-scale", "0.1", "--tag", "v7_s0.1"], deps=["3d_A_dense"]))
    out.append(dict(name="3d_tune_dense_amp", args=TUNE + ["--sigma-loss", "l2", "--sigma-loss-gain", "2", "--tag", "v7_amp"], deps=["3d_A_dense"]))
    return out


def gpu_state(gpu: str):
    out = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    free, util = (int(v) for v in out.strip().split(","))
    return free, util


def free_mib(gpu: str) -> int:
    out = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip() or 0)


def status(message: str) -> None:
    with open(LOGS / "status", "a") as handle:
        handle.write(time.strftime("%F %T ") + message + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tasks", default="", help="comma-separated task names to run (default: all)")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    LOGS.mkdir(parents=True, exist_ok=True)
    pending, done = tasks(), set()
    if args.tasks:
        wanted = set(args.tasks.split(","))
        done = {t["name"] for t in pending if t["name"] not in wanted}  # treat the others as finished
        pending = [t for t in pending if t["name"] in wanted]
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
            free, util = gpu_state(gpu)
            ready = [t for t in pending if all(d in done for d in t["deps"]) and need_mib(t) <= free] if util < 30 else []
            if not ready:
                idle_since.pop(gpu, None)
                continue
            if time.time() - idle_since.setdefault(gpu, time.time()) < 30:
                continue
            task = ready[0]
            pending.remove(task)
            attempt = 2 if task.get("retry") else 1
            log = open(LOGS / f"{task['name']}.log", "a")
            command = [sys.executable, str(HERE / "bench_fwi.py"), *task["args"], "--protocol", "v7"]
            env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
                       PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512")
            running[gpu] = [task, subprocess.Popen(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env), attempt]
            idle_since.pop(gpu, None)
            status(f"{task['name']} gpu{gpu} pass{attempt} start")
        time.sleep(10)
    status("ALL DONE")


if __name__ == "__main__":
    main()
