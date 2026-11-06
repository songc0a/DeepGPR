"""v4: reuse v3 FWI runs whose v4 configuration is identical (same lr, sigma scale 1).

With the same learning rate and sigma scale 1 a v4 run performs exactly the
v3 computation (the 2D v3 and v4 runs are bitwise identical, every loss of
every run). The v3 entry is copied with ``reused_from = "v3"``; the metrics
v4 adds (zone sigma, deep eps) are computed from the saved initial and final
models. Usage: ``python reuse_v3_runs.py --case 3d``.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import bench_fwi  # noqa: E402
from metrics import model_errors  # noqa: E402
from runner import load_results, merge_result  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="3d", choices=["2d", "3d"])
    args = parser.parse_args()
    case_name = args.case
    v3_dir, v4_dir = bench_fwi.result_dir(False, "v3"), bench_fwi.result_dir(False, "v4")
    lr3 = bench_fwi.chosen_learning_rates(case_name, False, "v3")
    lr4 = bench_fwi.chosen_learning_rates(case_name, False, "v4")
    scale4 = bench_fwi.chosen_sigma_scales(case_name, False, "v4")
    v3 = load_results(v3_dir / "fwi.json")
    done4 = load_results(v4_dir / "fwi.json")
    case, _, _, region, zone = bench_fwi.load_case(case_name, False, torch.device("cpu"), "v4")
    deep = (slice(None), slice(zone[1].stop, None), slice(None)) if case_name == "3d" else None
    initial = torch.load(v4_dir / "fwi_runs" / case_name / "initial.pt")
    initial3 = torch.load(v3_dir / "fwi_runs" / case_name / "initial.pt")
    assert torch.equal(initial["eps"], initial3["eps"]) and torch.equal(initial["sigma"], initial3["sigma"])

    def extra(eps, sigma):
        out = {"sigma_log_zone": model_errors(sigma, case["sigma"].cpu(), zone, log=True)}
        if deep is not None:
            out["eps_deep"] = model_errors(eps, case["eps"].cpu(), deep)
        return out

    for name, r in sorted(v3.items()):
        if not name.startswith(f"{case_name}|") or "history" not in r:
            continue
        label = name.split("|")[1]
        if label not in lr4 or lr4[label] != lr3.get(label) or scale4.get(label) != 1.0:
            continue
        if name in done4 and "error" not in done4[name]:
            print(name, "already in v4")
            continue
        result = copy.deepcopy(r)
        result["spec"] = dict(result["spec"], protocol="v4", sigma_scale=1.0)
        result["reused_from"] = "v3"
        final = torch.load(v3_dir / "fwi_runs" / case_name / f"{name}_final.pt")
        result["info"]["initial_errors"].update(extra(initial["eps"], initial["sigma"]))
        result["history"][-1]["errors"].update(extra(final["eps"], final["sigma"]))
        result["metrics_added_from_final_model"] = sorted(extra(final["eps"], final["sigma"]))
        shutil.copy2(v3_dir / "fwi_runs" / case_name / f"{name}_final.pt", v4_dir / "fwi_runs" / case_name / f"{name}_final.pt")
        merge_result(v4_dir / "fwi.json", name, result)
        print(name, "reused from v3 (lr", lr4[label], ")")


if __name__ == "__main__":
    main()
