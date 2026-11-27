"""Exact vs Tucker-recovered E history of one 2 cm Overthrust forward run (the FWI's sketch settings).

Pass 1 (once per rank): the FWI's own sketched forward (``compute(..., wavefield_sketch=...)``) builds the
compressed history; its Tucker blocks are then moved to host memory.
Pass 2: the same forward again; every exact frame E^n (Ex, Ey, Ez) the solver hands to the history is compared
with the reconstruction of that frame from each compressed history, exactly as the adjoint would read it
(``reconstruct_into``: Tucker block, then the final support mask). Saved: 3D relative errors of every frame
(physical domain incl. air, PML excluded) and Ez slices for plotting.

    CUDA_VISIBLE_DEVICES=4 python wavefield_compare.py --out wavefield_true_shot5
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
    p.add_argument("--ranks", type=int, nargs="+", default=[16, 32])
    p.add_argument("--slice-every", type=int, default=10)
    p.add_argument("--depth-every", type=int, default=30)
    p.add_argument("--depth-m", type=float, default=1.0)
    p.add_argument("--memory-gib", type=float, default=18.5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import DeepGPR
    import DeepGPR.solver.sketched as SK
    from DeepGPR.tucker import TuckerSketchConfig
    from DeepGPR.tucker.sketch import TuckerBlock
    from DeepGPR.tucker.wavefield import SketchedWavefieldHistory, unpack_bits

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

    # ---- pass 1: compressed histories ------------------------------------------------------------------
    histories, info = {}, dict(model=args.model, shot=s, source_cell=src, profile=args.profile, nt=nt, dt=dt,
                               freq=cfg["freq"], dx=cfg["dx"], shape=list(shape), pml=pml, air=air, zsrc_ext=zsrc,
                               ydepth_ext=ydepth, depth_m=(ydepth - pml - air) * cfg["dx"],
                               slice_every=args.slice_every, depth_every=args.depth_every, ranks=args.ranks, passes={})
    for r in args.ranks:
        sketch = dataclasses.replace(base, ranks=r)
        built = []

        class Keep(original):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                built.append(self)

        seconds = forward(sketch, Keep)
        h = built[0]
        report = h.size_report()
        host = {}
        for name in names:
            st = h.stores[name]
            assert st.operator is not None and st.time_block == 1 and st.support_bits is not None, name
            host[name] = dict(core=st.core.cpu(), factors={m: f.cpu() for m, f in st.factors.items()},
                              shape=st.operator.shape, support=st.support_bits.clone(), frame_shape=st.frame_shape)
        histories[r] = host
        info["passes"][f"r{r}"] = dict(sketch=dataclasses.asdict(sketch) | dict(compute_dtype=str(sketch.compute_dtype)),
                                       seconds=seconds, peak_gib=torch.cuda.max_memory_allocated() / GIB,
                                       stored_bytes=report.stored_bytes, raw_bytes=report.raw_bytes,
                                       frames=h.frames)
        print(f"pass 1 r{r}: {seconds:.0f} s, peak {torch.cuda.max_memory_allocated() / GIB:.2f} GiB, "
              f"history {report.stored_bytes / GIB:.2f} GiB of {report.raw_bytes / GIB:.1f} GiB raw", flush=True)
        del built, h
        torch.cuda.empty_cache()

    # ---- pass 2: exact frames vs reconstructions --------------------------------------------------------
    frames_total = nt + 1  # rhs="from_e": E^0 ... E^nt
    err = {r: np.zeros((frames_total, 3)) for r in args.ranks}
    ref = np.zeros((frames_total, 3))
    vsteps = list(range(0, frames_total, args.slice_every))
    dsteps = list(range(0, frames_total, args.depth_every))
    vert = {k: np.zeros((len(vsteps), shape[0], shape[1]), np.float32) for k in ["exact", *[f"r{r}" for r in args.ranks]]}
    dep = {k: np.zeros((len(dsteps), shape[0], shape[2]), np.float16) for k in vert}
    supports = {r: {name: unpack_bits(h[name]["support"].to(device), h[name]["frame_shape"]) for name in names}
                for r, h in histories.items()}

    def reconstruct(r: int, name: str, n: int) -> torch.Tensor:
        h = histories[r][name]
        block = TuckerBlock(h["core"][n].to(device, non_blocking=True),
                            {m: f[n].to(device, non_blocking=True) for m, f in h["factors"].items()}, h["shape"])
        v = block.reconstruct()[0].reshape(h["frame_shape"])  # shots = 1
        return v.masked_fill_(~supports[r][name], 0)

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
                for c, name in enumerate(names):
                    w = frames[name][j, 0]
                    ref[n, c] = norm_sq(w)
                    if name == "Ez" and n in vindex:
                        vert["exact"][vindex[n]] = w[phys[0], phys[1], zsrc].cpu().numpy()
                    if name == "Ez" and n in dindex:
                        dep["exact"][dindex[n]] = w[phys[0], ydepth, phys[2]].cpu().numpy()
                    for r in args.ranks:
                        v = reconstruct(r, name, n)
                        if name == "Ez" and n in vindex:
                            vert[f"r{r}"][vindex[n]] = v[phys[0], phys[1], zsrc].cpu().numpy()
                        if name == "Ez" and n in dindex:
                            dep[f"r{r}"][dindex[n]] = v[phys[0], ydepth, phys[2]].cpu().numpy()
                        err[r][n, c] = norm_sq(v.sub_(w))
                        del v
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
    print(f"pass 2: {seconds:.0f} s, peak {torch.cuda.max_memory_allocated() / GIB:.2f} GiB", flush=True)

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
