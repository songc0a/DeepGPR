"""Exact vs block-wise Tucker history (adaptive ranks + adjoint shadow zones) on the 2 cm Overthrust model.

Every frame is compressed on its own (no time mode), so the frame the adjoint would read is obtained on the fly:
in one forward run each exact frame E^n the solver hands to the history is compressed and decompressed with
``BlockTuckerHistory``'s own routines (exactly what ``push`` stores and ``reconstruct_into`` returns) and, for
reference, with the global per-frame Tucker sketch of the FWI (DeepGPR's operators, recovered on the fly). The
compressed history is never kept; its size is the sum of the per-frame records.
Errors are measured where forward values enter the eps gradient: ground cells (receiver row and below, CPML
excluded) outside the adjoint shadow of that frame. dE = E^(n+1) - E^n is the R^n proxy of rhs="from_e".

    CUDA_VISIBLE_DEVICES=4 python block_wavefield_compare.py --tol 0.01 --out block_true_shot5_tol0.01
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

import fwi_overthrust3d as F

HERE = Path(__file__).resolve().parent
GIB = 1024**3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="run16_250MHz")
    p.add_argument("--profile", default="full250")
    p.add_argument("--shot", type=int, default=5)
    p.add_argument("--model", choices=("true", "init"), default="true")
    p.add_argument("--tol", type=float, required=True)
    p.add_argument("--block", type=int, default=32)
    p.add_argument("--floor", type=float, default=1e-4)
    p.add_argument("--margin-ns", type=float, default=2.0)
    p.add_argument("--global-ranks", type=int, nargs="*", default=[16, 64])
    p.add_argument("--slice-every", type=int, default=10)
    p.add_argument("--depth-every", type=int, default=30)
    p.add_argument("--depth-m", type=float, default=1.0)
    p.add_argument("--memory-gib", type=float, default=22.5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import DeepGPR
    import DeepGPR.solver.sketched as SK
    from DeepGPR.tucker import BlockTuckerConfig, TuckerSketchConfig, surface_adjoint_keep_until
    from DeepGPR.tucker.sketch import recover_many
    from DeepGPR.tucker.wavefield import _ComponentStore

    out = HERE / args.out
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_gib * GIB / total), device)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = F.config(args.profile, shot_grid=4)
    run = HERE / args.run_dir
    meta = json.loads((run / "meta.json").read_text())
    nt, air, pml, dt, dx = meta["nt"], cfg["air"], cfg["pml"], cfg["dt"], cfg["dx"]
    eps_host = torch.from_numpy(np.load(run / f"eps_{args.model}.npy"))
    shape = tuple(eps_host.shape)
    sources, receivers, _ = F.acquisition(cfg, shape)
    s = args.shot
    src = [int(v) for v in sources[s, 0]]
    wavelet = DeepGPR.wavelet.ricker(cfg["freq"], nt, dt, 1.2 / cfg["freq"]).reshape(1, nt, 1).to(device)
    kwargs = dict(device=device, dx=dx, dt=dt, source_amplitudes=wavelet, pmlthick=pml, mode=3, source_direction=2,
                  receiver_component=2, source_location=sources[s:s + 1].to(device), receiver_location=receivers[s:s + 1].to(device))
    sigma = torch.full(shape, cfg["sigma"], device=device)
    sigma[:, :air] = 0.0
    names = ("Ex", "Ey", "Ez")
    frames_total = nt + 1
    ext = tuple(n + 2 * pml for n in shape)
    phys = tuple(slice(pml, pml + n) for n in shape)
    zsrc = src[2] + pml
    ydepth = pml + air + int(round(args.depth_m / dx))
    keep = surface_adjoint_keep_until(eps_host, air, dx, dt, nt, pml, margin=args.margin_ns * 1e-9)
    sketch_fields = {k: v for k, v in cfg["sketch"].items() if k != "ranks"}
    bcfg = BlockTuckerConfig(**sketch_fields, block=(args.block,) * 3, origin=(0, pml + air, 0), tolerance=args.tol,
                             floor=args.floor, keep_until=keep, storage_device="cpu")
    info = dict(model=args.model, shot=s, source_cell=src, profile=args.profile, nt=nt, dt=dt, freq=cfg["freq"], dx=dx,
                shape=list(shape), pml=pml, air=air, zsrc_ext=zsrc, ydepth_ext=ydepth, depth_m=(ydepth - pml - air) * dx,
                slice_every=args.slice_every, depth_every=args.depth_every, block=args.block, tolerance=args.tol,
                floor=args.floor, margin_ns=args.margin_ns, global_ranks=args.global_ranks,
                keep_until_rows=keep.reshape(-1).tolist(), passes={})

    def forward(sketch, history_cls, attr) -> float:
        saved = getattr(SK, attr)
        setattr(SK, attr, history_cls)
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            e = eps_host.to(device).requires_grad_(True)
            d = DeepGPR.compute(eps_r=e, sigma=sigma, wavefield_sketch=sketch, **kwargs)[-1]
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            del d, e
            return seconds
        finally:
            setattr(SK, attr, saved)
            torch.cuda.empty_cache()

    from DeepGPR.tucker import BlockFrameStats, BlockTuckerHistory

    h = BlockTuckerHistory(names, ext, 1, frames_total, dataclasses.replace(bcfg, storage_device="cuda"), device)
    record_bytes = np.zeros((frames_total, 3))
    compress_seconds = [0.0]

    # ---- global Tucker operators (FWI sketch) for the same frames ----------------------------------------
    base = TuckerSketchConfig(**cfg["sketch"])
    ops = {}
    for r in args.global_ranks:
        ops[r] = {}
        for name in names:
            st = _ComponentStore(name, ext, 1, 1, dataclasses.replace(base, ranks=r), True, device, torch.float32)
            ops[r][name] = st.operator
            del st
    info["global_storage"] = {f"r{r}": frames_total * sum(op.stored_numel() for op in ops[r].values()) * 4 for r in args.global_ranks}

    # ---- pass 2: exact frames vs reconstructions -----------------------------------------------------------
    ground = torch.zeros(ext, dtype=torch.bool, device=device)
    ground[pml:pml + shape[0], pml + air:pml + shape[1], pml:pml + shape[2]] = True
    keep_dev = keep.to(device)
    methods = ["block", *[f"r{r}" for r in args.global_ranks]]
    # per frame: [region err^2, region ref^2, dE err^2, dE ref^2 (even frames: pair n, n+1), ground err^2, ground ref^2]
    acc = {m: np.zeros((frames_total, 6)) for m in methods}
    vsteps = list(range(0, frames_total, args.slice_every))
    dsteps = list(range(0, frames_total, args.depth_every))
    vindex = {n: i for i, n in enumerate(vsteps)}
    dindex = {n: i for i, n in enumerate(dsteps)}
    vert = {k: np.zeros((len(vsteps), shape[0], shape[1]), np.float32) for k in ["exact", *methods]}
    dep = {k: np.zeros((len(dsteps), shape[0], shape[2]), np.float16) for k in vert}

    def sq(t):
        return float(torch.linalg.vector_norm(t)) ** 2

    def reconstruct(m: str, name: str, n: int, w: torch.Tensor) -> torch.Tensor:
        if m == "block":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            record = h._compress(w, n, h.frame_stats[name][n])
            torch.cuda.synchronize()
            compress_seconds[0] += time.perf_counter() - t0
            record_bytes[n, names.index(name)] = 0 if record is None else record.nbytes()
            buf = torch.empty(ext, device=device)
            h._decompress(record, buf)
            return buf
        op = ops[int(m[1:])][name]
        block = recover_many([(op, op.sketch(w.reshape(1, *ext)), None)])[0]
        return block.reconstruct()[0].reshape(ext)

    class Compare:
        """Frames arrive in pairs (segment_steps=2): errors of both, dE = E^(n+1) - E^n of the pair."""

        def __init__(self, components, frame_shape, shots, frames, config, device, history_dtype=torch.float32):
            assert tuple(components) == names and tuple(frame_shape) == ext and frames == frames_total
            self.frames, self.frames_pushed = frames, 0

        def push(self, frames, start):
            count = next(iter(frames.values())).shape[0]
            regions = [ground & (keep_dev >= start + j) for j in range(count)]
            for name in names:
                ws = [frames[name][j, 0] for j in range(count)]
                for j, w in enumerate(ws):
                    n = start + j
                    if name == "Ez" and n in vindex:
                        vert["exact"][vindex[n]] = w[phys[0], phys[1], zsrc].cpu().numpy()
                    if name == "Ez" and n in dindex:
                        dep["exact"][dindex[n]] = w[phys[0], ydepth, phys[2]].cpu().numpy()
                    ref = sq(w[regions[j]]), sq(w[ground])
                    for m in methods:
                        acc[m][n, 1] += ref[0]
                        acc[m][n, 5] += ref[1]
                if count == 2:
                    dref = sq((ws[1] - ws[0])[regions[0]])
                for m in methods:
                    diffs = []
                    for j, w in enumerate(ws):
                        n = start + j
                        a = reconstruct(m, name, n, w)
                        if name == "Ez" and n in vindex:
                            vert[m][vindex[n]] = a[phys[0], phys[1], zsrc].cpu().numpy()
                        if name == "Ez" and n in dindex:
                            dep[m][dindex[n]] = a[phys[0], ydepth, phys[2]].cpu().numpy()
                        a.sub_(w)
                        acc[m][n, 0] += sq(a[regions[j]])
                        acc[m][n, 4] += sq(a[ground])
                        diffs.append(a)
                    if count == 2:
                        acc[m][start, 2] += sq((diffs[1] - diffs[0])[regions[0]])
                        acc[m][start, 3] += dref
                    del diffs
            self.frames_pushed += count
            if self.frames_pushed % 200 < count:
                k = self.frames_pushed
                msg = ", ".join(f"{m} E {np.sqrt(acc[m][:k, 0].sum() / max(acc[m][:k, 1].sum(), 1e-300)):.4f} "
                                f"dE {np.sqrt(acc[m][:k, 2].sum() / max(acc[m][:k, 3].sum(), 1e-300)):.4f}" for m in methods)
                print(f"  pass 2 frame {k}/{frames_total}: cumulative region error {msg}", flush=True)

        def size_report(self):
            return None

    seconds = forward(dataclasses.replace(base, segment_steps=2), Compare, "SketchedWavefieldHistory")
    raw = frames_total * 3 * int(np.prod(ext)) * 4
    stored = float(record_bytes.sum())
    info["passes"]["compare"] = dict(seconds=seconds, peak_gib=torch.cuda.max_memory_allocated() / GIB,
                                     block_compress_seconds=compress_seconds[0])
    info["storage"] = dict(block_stored_bytes=stored, raw_bytes=raw, block_ratio=raw / stored,
                           **{f"r{r}_ratio": raw / v for r, v in info["global_storage"].items() for r in [r[1:]]})
    fs = {name: np.array([[st.blocks, st.stored, st.shadow, st.floor, st.payload_bytes, st.mean_rank / max(st.stored, 1)]
                          for st in h.frame_stats[name]]) for name in names}
    print(f"forward + comparison: {seconds:.0f} s (block compression {compress_seconds[0]:.0f} s), peak "
          f"{torch.cuda.max_memory_allocated() / GIB:.2f} GiB; block history {stored / GIB:.2f} GiB of {raw / GIB:.1f} GiB raw "
          f"(compression {raw / stored:.0f} x)", flush=True)
    summary = {}
    for m in methods:
        a = acc[m]
        summary[m] = dict(region_rel_all=float(np.sqrt(a[:, 0].sum() / a[:, 1].sum())),
                          dE_rel_all=float(np.sqrt(a[:, 2].sum() / a[:, 3].sum())),
                          ground_rel_all=float(np.sqrt(a[:, 4].sum() / a[:, 5].sum())))
        print(m, summary[m], flush=True)
    info["summary"] = summary
    np.savez(out / "errors.npz", **{f"acc_{m}": acc[m] for m in methods}, **{f"stats_{n}": fs[n] for n in names},
             record_bytes=record_bytes)
    np.savez(out / "vertical_Ez.npz", steps=np.array(vsteps), **vert)
    np.savez(out / "depth_Ez.npz", steps=np.array(dsteps), **dep)
    (out / "info.json").write_text(json.dumps(info, indent=1, default=float))
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
