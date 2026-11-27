"""Design study for the block-wise Tucker history on real 2 cm Overthrust frames (true model, one shot).

Captures exact E frames (Ex, Ey, Ez) at a few times (frame n and n + 1, for the R^n = (E^(n+1) - ca E^n) / cb
proxy dE = E^(n+1) - E^n) from the solver's own frames, then compresses them with block Tucker (block sizes,
tolerances, floors, with / without adjoint shadow zones) and with the global per-frame Tucker sketch (r32, r64, as
the adjoint would read it). Errors are measured in the region whose forward values enter the eps gradient:
ground cells (receiver row and below, CPML excluded) that are not in the adjoint shadow at that frame.

    CUDA_VISIBLE_DEVICES=4 python block_design.py --out block_design_true_shot5
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
    p.add_argument("--times", type=float, nargs="+", default=[15, 30, 45, 60, 75])
    p.add_argument("--tolerances", type=float, nargs="+", default=[0.003, 0.01, 0.03, 0.1])
    p.add_argument("--blocks", type=int, nargs="+", default=[32, 16])
    p.add_argument("--floors", type=float, nargs="+", default=[1e-4, 1e-3])
    p.add_argument("--global-ranks", type=int, nargs="+", default=[32, 64])
    p.add_argument("--margin-ns", type=float, default=2.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import DeepGPR
    import DeepGPR.solver.sketched as SK
    from DeepGPR.tucker import BlockFrameStats, BlockTuckerConfig, BlockTuckerHistory, TuckerSketchConfig, surface_adjoint_keep_until
    from DeepGPR.tucker.sketch import recover_many
    from DeepGPR.tucker.wavefield import _ComponentStore

    out = HERE / args.out
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = F.config(args.profile, shot_grid=4)
    run = HERE / args.run_dir
    meta = json.loads((run / "meta.json").read_text())
    nt, air, pml, dt, dx = meta["nt"], cfg["air"], cfg["pml"], cfg["dt"], cfg["dx"]
    eps_host = torch.from_numpy(np.load(run / "eps_true.npy"))
    shape = tuple(eps_host.shape)
    sources, receivers, _ = F.acquisition(cfg, shape)
    s = args.shot
    wavelet = DeepGPR.wavelet.ricker(cfg["freq"], nt, dt, 1.2 / cfg["freq"]).reshape(1, nt, 1).to(device)
    kwargs = dict(device=device, dx=dx, dt=dt, source_amplitudes=wavelet, pmlthick=pml, mode=3, source_direction=2,
                  receiver_component=2, source_location=sources[s:s + 1].to(device), receiver_location=receivers[s:s + 1].to(device))
    sigma = torch.full(shape, cfg["sigma"], device=device)
    sigma[:, :air] = 0.0
    names = ("Ex", "Ey", "Ez")
    frames_total = nt + 1
    ext = tuple(n + 2 * pml for n in shape)
    wanted = sorted({n for t in args.times for n in (int(round(t * 1e-9 / dt)), int(round(t * 1e-9 / dt)) + 1)})

    # ---- capture exact frames ------------------------------------------------------------------------------
    captured = {}

    class Capture:
        def __init__(self, components, frame_shape, shots, frames, config, device, history_dtype=torch.float32):
            assert tuple(components) == names and tuple(frame_shape) == ext and frames == frames_total
            self.frames, self.frames_pushed = frames, 0

        def push(self, frames, start):
            for j in range(next(iter(frames.values())).shape[0]):
                if start + j in wanted:
                    captured[start + j] = {n: frames[n][j, 0].cpu() for n in names}
            self.frames_pushed += next(iter(frames.values())).shape[0]

        def size_report(self):
            return None

    original = SK.SketchedWavefieldHistory
    SK.SketchedWavefieldHistory = Capture
    t0 = time.perf_counter()
    e = eps_host.to(device).requires_grad_(True)
    d = DeepGPR.compute(eps_r=e, sigma=sigma, wavefield_sketch=TuckerSketchConfig(**dict(cfg["sketch"], segment_steps=1)), **kwargs)[-1]
    del d, e
    SK.SketchedWavefieldHistory = original
    torch.cuda.empty_cache()
    print(f"captured {len(captured)} frames in {time.perf_counter() - t0:.0f} s", flush=True)

    # ---- regions ---------------------------------------------------------------------------------------------
    keep = surface_adjoint_keep_until(eps_host, air, dx, dt, nt, pml, margin=args.margin_ns * 1e-9)
    keep_air_only = torch.where(keep >= 0, torch.full_like(keep, frames_total), keep)  # air masked, no time shadow
    rows = keep.reshape(-1)
    info = dict(shot=s, times=args.times, frames=wanted, keep_until_rows=rows.tolist(), dt=dt, nt=nt,
                shadow_start_ns={f"{d_m:.2f} m": float(rows[pml + air + int(round(d_m / dx))]) * dt * 1e9
                                 for d_m in (0.0, 0.5, 1.0, 2.0, 3.0, 3.7)})
    print("adjoint shadow starts (ns) at depth:", info["shadow_start_ns"], flush=True)
    ground = torch.zeros(ext, dtype=torch.bool, device=device)
    ground[pml:pml + shape[0], pml + air:pml + shape[1], pml:pml + shape[2]] = True
    keep_dev = keep.to(device)

    def region(n: int) -> torch.Tensor:
        return ground & (keep_dev >= n)

    raw_frame = 3 * int(np.prod(ext)) * 4

    def measure(n: int, rec: dict, rec1: dict) -> dict:
        m = region(n)
        e2 = r2 = de2 = dr2 = g2 = gr2 = 0.0
        for name in names:
            w0, w1 = captured[n][name].to(device), captured[n + 1][name].to(device)
            a0, a1 = rec[name], rec1[name]
            e2 += float(torch.linalg.vector_norm((a0 - w0)[m])) ** 2
            r2 += float(torch.linalg.vector_norm(w0[m])) ** 2
            de2 += float(torch.linalg.vector_norm(((a1 - a0) - (w1 - w0))[m])) ** 2
            dr2 += float(torch.linalg.vector_norm((w1 - w0)[m])) ** 2
            g2 += float(torch.linalg.vector_norm((a0 - w0)[ground])) ** 2
            gr2 += float(torch.linalg.vector_norm(w0[ground])) ** 2
        return dict(err=np.sqrt(e2 / r2), derr=np.sqrt(de2 / dr2), err_ground_all=np.sqrt(g2 / gr2),
                    region_energy_fraction=r2 / gr2)

    results = []
    # ---- block Tucker -----------------------------------------------------------------------------------------
    for b in args.blocks:
        for floor in args.floors:
            for tol in args.tolerances:
                for shadow in (True, False):
                    bc = BlockTuckerConfig(block=(b, b, b), origin=(0, pml + air, 0), tolerance=tol, floor=floor,
                                           keep_until=keep if shadow else keep_air_only, storage_device="cpu")
                    h = BlockTuckerHistory(names, ext, 1, frames_total, bc, device)
                    for t in args.times:
                        n = int(round(t * 1e-9 / dt))
                        rec, rec1, nbytes, stats0 = {}, {}, 0, BlockFrameStats()
                        torch.cuda.synchronize()
                        tc = time.perf_counter()
                        records = {}
                        for k, nn in ((0, n), (1, n + 1)):
                            for name in names:
                                st = stats0 if k == 0 else BlockFrameStats()
                                records[(k, name)] = h._compress(captured[nn][name].to(device), nn, st)
                        torch.cuda.synchronize()
                        t_comp = (time.perf_counter() - tc) / 2
                        tc = time.perf_counter()
                        for (k, name), r in records.items():
                            buf = torch.empty(ext, device=device)
                            h._decompress(r, buf)
                            (rec if k == 0 else rec1)[name] = buf
                            if k == 0:
                                nbytes += 0 if r is None else r.nbytes()
                        torch.cuda.synchronize()
                        t_dec = (time.perf_counter() - tc) / 2
                        row = dict(method=f"block{b}", tol=tol, floor=floor, shadow=shadow, t_ns=t, frame=n,
                                   bytes=nbytes, ratio=raw_frame / max(nbytes, 1), compress_s=t_comp, decompress_s=t_dec,
                                   blocks=stats0.blocks, stored=stats0.stored, shadow_blocks=stats0.shadow, floor_blocks=stats0.floor,
                                   mean_rank=stats0.mean_rank / max(stats0.stored, 1), **measure(n, rec, rec1))
                        results.append(row)
                        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
                        del rec, rec1, records
                    del h
                    torch.cuda.empty_cache()
    # ---- global Tucker (FWI sketch) -------------------------------------------------------------------------
    base = TuckerSketchConfig(**cfg["sketch"])
    for r in args.global_ranks:
        sk = dataclasses.replace(base, ranks=r)
        ops = {}
        for name in names:
            st = _ComponentStore(name, ext, 1, 1, sk, True, device, torch.float32)
            ops[name] = st.operator
            del st
        support = {name: torch.ones(ext, dtype=torch.bool, device=device) for name in names}
        per = sum(op.stored_numel() for op in ops.values()) * 4
        for t in args.times:
            n = int(round(t * 1e-9 / dt))
            rec, rec1 = {}, {}
            for k, nn in ((0, n), (1, n + 1)):
                frame = {name: captured[nn][name].to(device) for name in names}
                blocks = recover_many([(ops[nm], ops[nm].sketch(frame[nm].reshape(1, *ext)), None) for nm in names])
                for name, blk in zip(names, blocks):
                    (rec if k == 0 else rec1)[name] = blk.reconstruct()[0].reshape(ext)
            row = dict(method=f"global_r{r}", tol=None, floor=None, shadow=False, t_ns=t, frame=n, bytes=per,
                       ratio=raw_frame / per, **measure(n, rec, rec1))
            results.append(row)
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    info["results"] = results
    (out / "design.json").write_text(json.dumps(info, indent=1, default=float))
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
