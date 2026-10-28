"""Phase 4 (3D) benchmark: staircase Yee vs hybrid DGTD-FDTD vs fine references.

For every case of :mod:`models3d`:

* ``staircase``  - native DeepGPR (CUDA) on the 2 cm grid, node-wise air/ground;
* ``staircase_3`` - the same on the 2/3 cm grid (the Yee spacing of ``hybrid_r3``);
* ``hybrid_r1``  - hybrid P1 band (adaptive column mesh), Yee on the 2 cm grid (CUDA library);
* ``hybrid_r3``  - hybrid P1 band (adaptive column mesh), Yee refined to 2/3 cm;
* ``hybrid_r3_terrain`` - as ``hybrid_r3`` on the terrain-following mesh;
* ``ref_3``/``ref_5``/``ref_7`` - native DeepGPR on 2/3 cm, 4 mm and 2.9 mm grids
  with cell-averaged materials. They converge at second order (measured), so
  ``ref_7`` is the reference and ``|ref_5 - ref_7|`` bounds its error.

Errors: relative L2 of the receiver gathers against ``ref_7`` (surface line with
offset >= 0.3 m, buried line). Results are cached in
``examples/hybrid/results/phase4/<case>_<solver>.npz``.

    python examples/hybrid/phase4_benchmark.py [--cases A3_flat B3_sine C3_irregular]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import models3d as M

OUT = M.ROOT / "examples" / "hybrid" / "results" / "phase4"
SOLVERS = ("staircase", "staircase_3", "hybrid_r1", "hybrid_r3", "hybrid_r3_terrain", "ref_3",
           "ref_5", "ref_7")


def run(case, solver):
    path = OUT / f"{case}_{solver}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["data"], z["info"].item()
    print(f"  running {case} / {solver}", flush=True)
    if solver == "staircase":
        data, info = M.run_native(*M.staircase_model(case), M.DX)
    elif solver == "staircase_3":
        data, info = M.run_native(*M.staircase_model(case, M.DX / 3), M.DX / 3)
    elif solver.startswith("ref_"):
        r = int(solver.split("_")[1])
        data, info = M.run_native(*M.averaged_model(case, M.DX / r), M.DX / r)
    else:
        # hybrid_r<r>: adaptive column mesh (default); hybrid_r<r>_terrain: terrain-following
        r = int(solver.split("_")[1][1:])
        mesh = "terrain" if solver.endswith("_terrain") else "adaptive"
        sol = M.build_hybrid(case, refinement=r, mesh=mesh)
        data, info = M.run_hybrid(sol)
        del sol
        torch.cuda.empty_cache()
    np.savez(path, data=data, info=np.array(info, dtype=object))
    return data, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=list(M.CASES))
    ap.add_argument("--solvers", nargs="+", default=list(SOLVERS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for case in args.cases:
        print(case, flush=True)
        res = {s: run(case, s) for s in args.solvers}
        ref = res["ref_7"][0] if "ref_7" in res else None
        rows = {}
        for s, (data, info) in res.items():
            row = dict(info)
            if ref is not None:
                row["error"] = M.rel_error(data, ref)
                n = M.REC_Y.size
                row["error_surface_line"] = M.rel_error(data, ref, np.concatenate(
                    [M.metric_mask()[:n], np.zeros(n, bool)]))
                row["error_buried_line"] = M.rel_error(data, ref, np.concatenate(
                    [np.zeros(n, bool), np.ones(n, bool)]))
            rows[s] = row
            print(f"  {s:11s} err {row.get('error', float('nan')):.4f} "
                  f"(surface {row.get('error_surface_line', float('nan')):.4f}, "
                  f"buried {row.get('error_buried_line', float('nan')):.4f})  "
                  f"t {info['seconds']:.1f} s  nt {info['nt']}", flush=True)
        summary[case] = rows
    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
