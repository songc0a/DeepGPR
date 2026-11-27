"""4 cm gradient check of the block-wise Tucker history: exact MS-SSIM eps gradient (two-level checkpointing)
vs block Tucker at several tolerances and the global Tucker sketch of the FWI, per shot and summed over shots.

    python check_block_gradient.py --model eps_init.npy --tag init_a --shots 0 2 4 6 8 10 12 14

Writes <out>/<tag>.json (per-shot metrics) and <out>/<tag>.npz (summed gradients of the shots, per-row
squared norms) after every shot; merge_block_gradient.py combines the parts of one model.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path.home() / "overthrust3d_fwi"
sys.path.insert(0, str(HERE))
import fwi_overthrust3d as F  # noqa: E402
import DeepGPR  # noqa: E402
import DeepGPR.solver.sketched as SK  # noqa: E402
from DeepGPR.tucker import BlockTuckerConfig, TuckerSketchConfig, surface_adjoint_keep_until  # noqa: E402

captured: list = []


class Capture(SK.BlockTuckerHistory):
    """Keeps a handle on the history the solver builds, for its size report."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        captured.append(self)


SK.BlockTuckerHistory = Capture


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="half_exact16")
    p.add_argument("--profile", default="half")
    p.add_argument("--model", required=True, help="model file in the run dir (eps_init.npy, snapshots/eps_it0030.npy)")
    p.add_argument("--shots", type=int, nargs="+", default=list(range(16)))
    p.add_argument("--tols", type=float, nargs="+", default=[0.03, 0.02, 0.01])
    p.add_argument("--global-ranks", type=int, nargs="*", default=[16])
    p.add_argument("--block", type=int, default=32)
    p.add_argument("--floor", type=float, default=1e-4)
    p.add_argument("--margin-ns", type=float, default=2.0)
    p.add_argument("--storage", default="cpu")
    p.add_argument("--outer-steps", default="", help="exact gradient: outer checkpoint levels, e.g. 480,96 (full size)")
    p.add_argument("--segment-steps", type=int, default=None, help="exact gradient: finest segment (default profile)")
    p.add_argument("--offload", action="store_true", help="exact gradient: E history of a segment in pinned host memory")
    p.add_argument("--pin", action="store_true", help="exact gradient: checkpoint states in pinned host memory")
    p.add_argument("--exact-from", default=None, help="tag of an earlier run in --out: load its saved exact gradients")
    p.add_argument("--source-tol-factor", type=float, default=None,
                   help="tolerance factor of the blocks around the source (tight_regions); default off")
    p.add_argument("--source-radius-cells", type=int, default=30, help="horizontal half-width of the source region")
    p.add_argument("--source-depth-cells", type=int, default=30, help="depth of the source region below the receiver row")
    p.add_argument("--payload-dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                   help="stored type of the block factors and cores")
    p.add_argument("--exact-last", action="store_true", help="compute the exact gradient after the compressed ones "
                   "(its offload caps the PyTorch pool, which would distort their GPU peaks)")
    p.add_argument("--out", default="gradcheck_block_4cm")
    p.add_argument("--tag", required=True)
    a = p.parse_args()

    run = HERE / a.run
    out = HERE / a.out
    out.mkdir(exist_ok=True)
    cfg = F.config(a.profile, shot_grid=4)
    device = torch.device("cuda", torch.cuda.current_device())  # indexed: set_per_process_memory_fraction needs it
    meta = json.loads((run / "meta.json").read_text())
    nt, air, pml, dx, dt = meta["nt"], cfg["air"], cfg["pml"], cfg["dx"], cfg["dt"]
    eps_true = torch.from_numpy(np.load(run / "eps_true.npy"))
    model = torch.from_numpy(np.load(run / a.model))
    if model.shape != eps_true.shape:  # ground-only snapshot
        full = eps_true.clone()
        full[:, air:] = model
        model = full
    shape = tuple(eps_true.shape)
    sources, receivers, lines = F.acquisition(cfg, shape)
    wavelet = DeepGPR.wavelet.ricker(cfg["freq"], nt, dt, 1.2 / cfg["freq"]).reshape(1, nt, 1).to(device)
    sigma = torch.full(shape, cfg["sigma"], device=device)
    sigma[:, :air] = 0.0
    misfit = F.ChunkedMSSIM(lines, cfg["loss_chunk_lines"])
    ground = (slice(None), slice(air, None), slice(None))
    target = (eps_true[ground] - model[ground]).double()  # host: keeps the GPU peaks those of the gradients
    outer = tuple(int(v) for v in a.outer_steps.split(",") if v)
    segment = a.segment_steps or cfg["exact_segment_steps"]

    keep = surface_adjoint_keep_until(model, air, dx, dt, nt, pml, margin=a.margin_ns * 1e-9)
    base = {k: v for k, v in cfg["sketch"].items() if k != "ranks"}
    sketches = {}
    suffix = "" if a.source_tol_factor is None else f"_src{a.source_tol_factor:g}"
    suffix += {"float16": "_fp16", "bfloat16": "_bf16"}.get(a.payload_dtype, "")
    for tol in a.tols:
        sketches[f"block{tol:g}{suffix}"] = BlockTuckerConfig(**base, block=(a.block,) * 3, origin=(0, pml + air, 0),
                                                              tolerance=tol, floor=a.floor, keep_until=keep,
                                                              storage_device=a.storage, payload_dtype=a.payload_dtype)

    def source_regions(s: int, tol: float) -> tuple:
        """Box around the source column, receiver row down to source_depth_cells (frame cells incl. CPML)."""
        x, z = (int(v) + pml for v in sources[s, 0, [0, 2]])
        r, top = a.source_radius_cells, pml + air
        return ((x - r, x + r + 1, top, top + a.source_depth_cells, z - r, z + r + 1, tol * a.source_tol_factor),)
    for r in a.global_ranks:
        sketches[f"r{r}"] = TuckerSketchConfig(**dict(cfg["sketch"], ranks=r))
    names = [*sketches, "exact"] if a.exact_last else ["exact", *sketches]
    model = model.to(device)

    sums = {n: torch.zeros(target.shape, dtype=torch.float64) for n in names}
    rows = {n: [] for n in names}  # per shot: per-row ||g_n - g_exact||^2 (exact: ||g_exact||^2)
    results = dict(run=a.run, profile=a.profile, model=a.model, shots=[], tols=a.tols, global_ranks=a.global_ranks,
                   block=a.block, floor=a.floor, margin_ns=a.margin_ns, storage=a.storage, payload_dtype=a.payload_dtype, nt=nt, dx=dx, dt=dt, air=air,
                   exact=dict(outer_steps=outer, segment_steps=segment, offload=a.offload, pin=a.pin, loaded_from=a.exact_from),
                   source_region=dict(tol_factor=a.source_tol_factor, radius_cells=a.source_radius_cells,
                                      depth_cells=a.source_depth_cells),
                   keep_until_rows=keep.reshape(-1).tolist(), per_shot={n: [] for n in names})
    print(f"{a.tag}: model {a.model}, shots {a.shots}, methods {names}", flush=True)
    for s in a.shots:
        opts = dict(device=device, dx=dx, dt=dt, source_amplitudes=wavelet, pmlthick=pml, mode=3, source_direction=2,
                    receiver_component=2, source_location=sources[s:s + 1].to(device),
                    receiver_location=receivers[s:s + 1].to(device))
        obs = torch.load(run / "observed" / f"shot{s:02d}.pt").to(device)
        grads = {}
        for n in names:
            if n == "exact" and a.exact_from:
                grads[n] = torch.from_numpy(np.load(out / f"{a.exact_from}_shot{s:02d}_exact.npy")).double()
                results["per_shot"][n].append(dict(shot=s, loaded_from=a.exact_from, seconds=None, peak_gib=None,
                                                   stored_bytes=None, raw_bytes=None))
                print(f"shot {s:2d} exact      loaded from {a.exact_from}", flush=True)
                continue
            sk = sketches.get(n)
            if a.source_tol_factor is not None and isinstance(sk, BlockTuckerConfig):
                sk = dataclasses.replace(sk, tight_regions=source_regions(s, sk.tolerance))
            e = model.clone().requires_grad_(True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                if n == "exact":
                    F.exact_gradient(e, sigma, opts, segment, lambda d: misfit(d, obs)[1], outer, a.offload, a.pin)
                else:
                    d = DeepGPR.compute(eps_r=e, sigma=sigma, wavefield_sketch=sk, **opts)[-1]
                    d.backward(misfit(d, obs)[1])
                    del d
            except torch.cuda.OutOfMemoryError as err:
                peak = torch.cuda.max_memory_allocated() / 2**30
                del e
                captured.clear()
                torch.cuda.empty_cache()
                results["per_shot"][n].append(dict(shot=s, oom=True, peak_gib=peak, error=str(err)[:300]))
                print(f"shot {s:2d} {n:10s} OUT OF MEMORY after {time.perf_counter() - t0:.0f} s, peak {peak:.2f} GiB", flush=True)
                continue
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            g = e.grad[ground].cpu().double()
            del e
            if n == "exact" and a.offload:
                torch.cuda.set_per_process_memory_fraction(1.0, device)  # undo the cap of the offloaded exact path
            stored = raw = None
            if captured:
                rep = captured[-1].size_report()
                stored, raw = rep.stored_bytes, rep.raw_bytes
                captured.clear()
            grads[n] = g
            np.save(out / f"{a.tag}_shot{s:02d}_{n}.npy", g.float().numpy())  # kept even if a later method fails
            del g
            torch.cuda.empty_cache()
            results["per_shot"][n].append(dict(shot=s, seconds=seconds, peak_gib=peak, stored_bytes=stored, raw_bytes=raw))
            extra = f", history {stored / 2**30:.2f} GiB ({raw / stored:.0f}x)" if stored else ""
            print(f"shot {s:2d} {n:10s} {seconds:5.0f} s peak {peak:.2f} GiB{extra}", flush=True)
        ref = grads.get("exact")
        for n in names:
            if n not in grads or ref is None:
                continue
            g = grads[n]
            rec = results["per_shot"][n][-1]
            if n == "exact":
                rec.update(rel=0.0, cos=1.0)
                rows[n].append((g ** 2).sum(dim=(0, 2)).numpy())
            else:
                diff = g - ref
                rec.update(rel=float(diff.norm() / ref.norm()), cos=float((g * ref).sum() / (g.norm() * ref.norm())))
                rows[n].append((diff ** 2).sum(dim=(0, 2)).numpy())
                del diff
            rec["corr"] = F.pearson(-g, target)
            sums[n] += g
            print(f"shot {s:2d} {n:10s} rel {rec['rel']:.4f} cos {rec['cos']:.6f} corr {rec['corr']:+.4f}", flush=True)
        del ref, grads
        results["shots"].append(s)
        (out / f"{a.tag}.json").write_text(json.dumps(results, indent=1))
        np.savez(out / f"{a.tag}.npz", shots=np.array(results["shots"]),
                 **{f"sum_{n}": sums[n].float().numpy() for n in names},
                 **{f"rows_{n}": np.stack(rows[n]) for n in names})
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
