"""Exact vs Tucker-recovered E frames, compressed and recovered on the fly (ranks whose history does not fit).

With ``time_block = 1`` (the FWI setting) every frame is sketched and recovered on its own with fixed per-component
random maps, so the stored history is not needed to know what the adjoint would read: the same operators
(built by DeepGPR's own ``_ComponentStore``) sketch each exact frame, ``recover_many`` recovers it, and the final
support mask is applied, exactly as ``SketchedWavefieldHistory.push`` / ``reconstruct_into`` do.
Pass A: the forward once to get the final support mask (cells nonzero in any frame) of Ex, Ey, Ez.
Pass B: the forward again; each frame is compressed, recovered and compared for every rank.
Storage of the history that the FWI would keep is computed with the same formula as ``storage_nbytes``.

    CUDA_VISIBLE_DEVICES=4 python wavefield_compare_online.py --ranks 16 48 64 --out wavefield_true_shot5_r48_r64
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
    p.add_argument("--ranks", type=int, nargs="+", default=[16, 48, 64])
    p.add_argument("--slice-every", type=int, default=10)
    p.add_argument("--depth-every", type=int, default=30)
    p.add_argument("--depth-m", type=float, default=1.0)
    p.add_argument("--memory-gib", type=float, default=22.5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import DeepGPR
    import DeepGPR.solver.sketched as SK
    from DeepGPR.tucker import TuckerSketchConfig
    from DeepGPR.tucker.sketch import recover_many
    from DeepGPR.tucker.wavefield import _ComponentStore

    out = HERE / args.out
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier comparison
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_gib * GIB / total), device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cfg = F.config(args.profile, shot_grid=4)
    run = HERE / args.run_dir
    meta = json.loads((run / "meta.json").read_text())
    nt, air, pml, dt = meta["nt"], cfg["air"], cfg["pml"], cfg["dt"]
    eps_host = torch.from_numpy(np.load(run / f"eps_{args.model}.npy"))
    shape = tuple(eps_host.shape)
    sources, receivers, _ = F.acquisition(cfg, shape)
    s = args.shot
    src = [int(v) for v in sources[s, 0]]
    wavelet = DeepGPR.wavelet.ricker(cfg["freq"], nt, cfg["dt"], 1.2 / cfg["freq"]).reshape(1, nt, 1).to(device)
    kwargs = dict(device=device, dx=cfg["dx"], dt=dt, source_amplitudes=wavelet, pmlthick=pml, mode=3,
                  source_direction=2, receiver_component=2,
                  source_location=sources[s:s + 1].to(device), receiver_location=receivers[s:s + 1].to(device))
    sigma = torch.full(shape, cfg["sigma"], device=device)
    sigma[:, :air] = 0.0
    base = TuckerSketchConfig(**cfg["sketch"])
    phys = tuple(slice(pml, pml + n) for n in shape)
    zsrc = src[2] + pml  # x-depth section through the source, perpendicular to the z dipole
    ydepth = pml + air + int(round(args.depth_m / cfg["dx"]))
    names = ("Ex", "Ey", "Ez")
    original = SK.SketchedWavefieldHistory

    def forward(sketch, history_cls) -> float:
        SK.SketchedWavefieldHistory = history_cls
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            e = eps_host.to(device).requires_grad_(True)  # requires_grad -> the sketched history path
            d = DeepGPR.compute(eps_r=e, sigma=sigma, wavefield_sketch=sketch, **kwargs)[-1]
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            del d, e  # drops the autograd graph (no backward)
            return seconds
        finally:
            SK.SketchedWavefieldHistory = original
            torch.cuda.empty_cache()

    info = dict(model=args.model, shot=s, source_cell=src, profile=args.profile, nt=nt, dt=dt, freq=cfg["freq"],
                dx=cfg["dx"], shape=list(shape), pml=pml, air=air, zsrc_ext=zsrc, ydepth_ext=ydepth,
                depth_m=(ydepth - pml - air) * cfg["dx"], slice_every=args.slice_every, depth_every=args.depth_every,
                ranks=args.ranks, mode="online (per-frame sketch + recovery, final support mask)", passes={})
    frames_total = nt + 1  # rhs="from_e": E^0 ... E^nt

    # ---- pass A: final support masks --------------------------------------------------------------------
    support = {}

    class Support:
        def __init__(self, components, frame_shape, shots, frames, config, device, history_dtype=torch.float32):
            assert tuple(components) == names and shots == 1 and frames == frames_total, (components, shots, frames)
            self.frames, self.frames_pushed = frames, 0
            for name in names:
                support[name] = torch.zeros(frame_shape, dtype=torch.bool, device=device)

        def push(self, frames, start):
            for name, t in frames.items():  # same running support as _ComponentStore.push
                support[name] |= (torch.amax(t, dim=(0, 1)) != 0) | (torch.amin(t, dim=(0, 1)) != 0)
            self.frames_pushed += next(iter(frames.values())).shape[0]

        def size_report(self):
            return None

    seconds = forward(dataclasses.replace(base, segment_steps=1), Support)
    info["passes"]["support"] = dict(seconds=seconds, peak_gib=torch.cuda.max_memory_allocated() / GIB,
                                     support_cells={n: int(m.sum()) for n, m in support.items()})
    print(f"pass A (support): {seconds:.0f} s, peak {torch.cuda.max_memory_allocated() / GIB:.2f} GiB, "
          f"{ {n: int(m.sum()) for n, m in support.items()} } of {int(np.prod(support['Ex'].shape))} cells", flush=True)

    # ---- pass B: compress, recover and compare every frame --------------------------------------------
    operators, storage = {}, {}
    for r in args.ranks:
        sk = dataclasses.replace(base, ranks=r)
        operators[r] = {}
        for name in names:
            st = _ComponentStore(name, tuple(support[name].shape), 1, 1, sk, True, device, torch.float32)
            assert st.operator is not None and st.time_block == 1, (r, name, st.reason)
            operators[r][name] = st.operator
            del st
        cells = int(np.prod(support["Ex"].shape))
        per_frame = sum(op.stored_numel() for op in operators[r].values()) * 4  # float32 core + factors
        stored = frames_total * per_frame + 3 * (-(-cells // 8))  # + packed support masks (storage_nbytes)
        raw = frames_total * 3 * cells * 4
        storage[r] = dict(stored_bytes=stored, raw_bytes=raw, ratio=raw / stored, per_frame_bytes=per_frame,
                          map_bytes=sum(op.map_nbytes() for op in operators[r].values()),
                          core=list(operators[r]["Ez"].stored_shape()[0]))
        print(f"r{r}: history {stored / GIB:.2f} GiB of {raw / GIB:.1f} GiB raw (compression {raw / stored:.0f} x), "
              f"maps {storage[r]['map_bytes'] / GIB:.3f} GiB", flush=True)
    info["storage"] = {f"r{r}": v for r, v in storage.items()}
    torch.cuda.empty_cache()
    err = {r: np.zeros((frames_total, 3)) for r in args.ranks}
    ref = np.zeros((frames_total, 3))
    vsteps = list(range(0, frames_total, args.slice_every))
    dsteps = list(range(0, frames_total, args.depth_every))
    vert = {k: np.zeros((len(vsteps), shape[0], shape[1]), np.float32) for k in ["exact", *[f"r{r}" for r in args.ranks]]}
    dep = {k: np.zeros((len(dsteps), shape[0], shape[2]), np.float16) for k in vert}
    def recover(r: int, frame: dict) -> dict:
        """Sketch + recover one frame of every component together (as SketchedWavefieldHistory.push)."""
        ops = operators[r]
        blocks = recover_many([(ops[name], ops[name].sketch(frame[name].reshape(1, *frame[name].shape)), None)
                               for name in names])
        return dict(zip(names, blocks))

    def norm_sq(t: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(t[phys])) ** 2  # float32 tree reduction, no full-size copy

    class Compare:
        def __init__(self, components, frame_shape, shots, frames, config, device, history_dtype=torch.float32):
            assert tuple(components) == names and shots == 1 and frames == frames_total, (components, shots, frames)
            self.frames = frames
            self.frames_pushed = 0

        def push(self, frames, start):
            count = next(iter(frames.values())).shape[0]
            for j in range(count):
                n = start + j
                frame = {name: frames[name][j, 0] for name in names}
                blocks = {r: recover(r, frame) for r in args.ranks}
                for c, name in enumerate(names):
                    w = frames[name][j, 0]
                    ref[n, c] = norm_sq(w)
                    if name == "Ez" and n in vindex:
                        vert["exact"][vindex[n]] = w[phys[0], phys[1], zsrc].cpu().numpy()
                    if name == "Ez" and n in dindex:
                        dep["exact"][dindex[n]] = w[phys[0], ydepth, phys[2]].cpu().numpy()
                    for r in args.ranks:
                        v = blocks[r][name].reconstruct()[0].reshape(w.shape).masked_fill_(~support[name], 0)
                        if name == "Ez" and n in vindex:
                            vert[f"r{r}"][vindex[n]] = v[phys[0], phys[1], zsrc].cpu().numpy()
                        if name == "Ez" and n in dindex:
                            dep[f"r{r}"][dindex[n]] = v[phys[0], ydepth, phys[2]].cpu().numpy()
                        err[r][n, c] = norm_sq(v.sub_(w))
                        del v
                del blocks
            self.frames_pushed += count
            if self.frames_pushed % 200 < count:
                print(f"  pass 2 frame {self.frames_pushed}/{frames_total}, "
                      + ", ".join(f"r{r} rel {np.sqrt(err[r][:self.frames_pushed].sum() / max(ref[:self.frames_pushed].sum(), 1e-300)):.4f}"
                                  for r in args.ranks), flush=True)

        def size_report(self):
            return None

    vindex = {n: i for i, n in enumerate(vsteps)}
    dindex = {n: i for i, n in enumerate(dsteps)}
    seconds = forward(dataclasses.replace(base, segment_steps=1), Compare)
    info["passes"]["compare"] = dict(seconds=seconds, peak_gib=torch.cuda.max_memory_allocated() / GIB)
    print(f"pass B: {seconds:.0f} s, peak {torch.cuda.max_memory_allocated() / GIB:.2f} GiB", flush=True)

    summary = {}
    for r in args.ranks:
        e2, r2 = err[r], ref
        summary[f"r{r}"] = dict(
            relative_l2_all=float(np.sqrt(e2.sum() / r2.sum())),
            relative_l2_component={name: float(np.sqrt(e2[:, c].sum() / r2[:, c].sum())) for c, name in enumerate(names)},
            max_frame_relative=float(np.nanmax(np.sqrt(e2.sum(1) / np.where(r2.sum(1) > 0, r2.sum(1), np.nan)))))
        print(f"r{r}: {summary[f'r{r}']}", flush=True)
    info["summary"] = summary
    np.savez(out / "errors.npz", ref=ref, **{f"err_r{r}": err[r] for r in args.ranks}, steps=np.arange(frames_total))
    np.savez(out / "vertical_Ez.npz", steps=np.array(vsteps), **vert)
    np.savez(out / "depth_Ez.npz", steps=np.array(dsteps), **dep)
    (out / "info.json").write_text(json.dumps(info, indent=1, default=str))
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
