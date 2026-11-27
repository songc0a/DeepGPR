"""2.8 cm Overthrust model (53.6 M cells, mid250), everything on one 24 GB GPU: exact gradient by binomial
checkpointing (segments of 8 steps, 7 checkpoints) vs block Tucker (10 %, fp16, source region 0.1, history on the
GPU), alternating, twice each; gradients against the multi-level exact gradient of check_mid250_gpu.py.

Model: the 2 cm exact run's it-30 model resampled to the 2.8 cm grid (as eps_true in prepare), shot 5 of the
4 x 4 grid (the largest history at 2 cm). FWI residents on the device: eps, eps.grad, two Adam moments.

    CUDA_VISIBLE_DEVICES=0 python check_mid250_binomial.py
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom

HERE = Path.home() / "overthrust3d_fwi"
sys.path.insert(0, str(HERE))
import fwi_overthrust3d as F  # noqa: E402
import DeepGPR  # noqa: E402
import DeepGPR.solver.sketched as SK  # noqa: E402
from DeepGPR.tucker import BlockTuckerConfig, surface_adjoint_keep_until  # noqa: E402

captured: list = []


class Capture(SK.BlockTuckerHistory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        captured.append(self)


SK.BlockTuckerHistory = Capture
GIB = 2**30
run = HERE / "run_mid250"
out = run / "gradcheck"
out.mkdir(exist_ok=True)
cfg = F.config("mid250", shot_grid=4, block_tolerance=0.1, payload_dtype="float16")
device = torch.device("cuda", 0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
meta = json.loads((run / "meta.json").read_text())
nt, air, pml, dx, dt = meta["nt"], cfg["air"], cfg["pml"], cfg["dx"], cfg["dt"]
eps_true = torch.from_numpy(np.load(run / "eps_true.npy"))
shape = tuple(eps_true.shape)
nx, depth, nz = meta["ground_shape"]
# the 2 cm exact run's it-30 model (ground only, (x, depth, z)) on the 2.8 cm grid, same resampling as prepare
g2 = np.load(HERE / "run16_250MHz" / "snapshots" / "eps_it0030.npy").astype(np.float64)
g28 = zoom(g2, [t / n for t, n in zip((nx, depth, nz), g2.shape)], order=1, mode="nearest", grid_mode=True)
model = eps_true.clone()
model[:, air:] = torch.from_numpy(g28.astype(np.float32))
del g2, g28
sources, receivers, lines = F.acquisition(cfg, shape)
shot = 5
wavelet = DeepGPR.wavelet.ricker(cfg["freq"], nt, dt, 1.2 / cfg["freq"]).reshape(1, nt, 1).to(device)
sigma = torch.full(shape, cfg["sigma"], device=device)
sigma[:, :air] = 0.0
opts = dict(device=device, dx=dx, dt=dt, source_amplitudes=wavelet, pmlthick=pml, mode=3, source_direction=2,
            receiver_component=2, source_location=sources[shot:shot + 1].to(device),
            receiver_location=receivers[shot:shot + 1].to(device))
misfit = F.ChunkedMSSIM(lines, cfg["loss_chunk_lines"])
ground = (slice(None), slice(air, None), slice(None))
target = (eps_true[ground] - model[ground]).double()
t0 = time.perf_counter()
with torch.no_grad():
    obs = DeepGPR.compute(eps_r=eps_true.to(device), sigma=sigma, save_wavefield_history=False, **opts)[-1]
torch.cuda.synchronize()
forward_s = time.perf_counter() - t0
print(f"mid250: model {shape}, cells {meta['cells'] / 1e6:.1f} M with CPML, nt {nt}, size {meta['size_m']} m, "
      f"shot {shot} at {sources[shot, 0].tolist()}, {receivers.shape[1]} receivers; history-free forward {forward_s:.0f} s",
      flush=True)
keep = surface_adjoint_keep_until(model, air, dx, dt, nt, pml, margin=cfg["block_margin_ns"] * 1e-9)
x, z = (int(v) + pml for v in sources[shot, 0, [0, 2]])
r, top = cfg["block_source_radius_cells"], pml + air
region = (x - r, x + r + 1, top, top + cfg["block_source_depth_cells"], z - r, z + r + 1,
          cfg["block_tolerance"] * cfg["block_source_factor"])
base = dict({k: v for k, v in cfg["sketch"].items() if k != "ranks"}, block=(cfg["block_size"],) * 3,
            origin=(0, pml + air, 0), tolerance=cfg["block_tolerance"], floor=cfg["block_floor"], keep_until=keep,
            payload_dtype=cfg["block_payload_dtype"], tight_regions=(region,))
block = BlockTuckerConfig(**base, storage_device=str(device), device_chunk_bytes=512 * 2**20)
results = dict(profile="mid250", shape=list(shape), cells_with_cpml=meta["cells"], nt=nt, dx=dx, dt=dt, shot=shot,
               model="run16_250MHz it-30 resampled to 2.8 cm", history_free_forward_s=forward_s, runs=[])
grads = {}
for name in ("binomial", "block_gpu", "binomial", "block_gpu"):
    eps = model.to(device).requires_grad_(True)
    eps.grad = torch.zeros_like(eps)
    adam = [torch.zeros_like(eps), torch.zeros_like(eps)]  # exp_avg, exp_avg_sq of the FWI optimiser
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rec = dict(method=name)
    t0 = time.perf_counter()
    if name == "binomial":
        stats = {}
        F.binomial_gradient(eps, sigma, opts, 8, lambda d: misfit(d, obs)[1], 7, stats)
        rec.update(stats)
    else:
        d = DeepGPR.compute(eps_r=eps, sigma=sigma, wavefield_sketch=block, **opts)[-1]
        torch.cuda.synchronize()
        rec["forward_s"] = time.perf_counter() - t0
        rec["history_gib"] = captured[-1].size_report().stored_bytes / GIB
        d.backward(misfit(d, obs)[1])
        del d
    torch.cuda.synchronize()
    rec["seconds"] = time.perf_counter() - t0
    rec["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / GIB
    rec["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / GIB
    rec["host_maxrss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    g = eps.grad[ground].cpu().double()
    rec["corr_to_model_error"] = F.pearson(-g, target)
    if name in grads:
        rec["rel_to_own_first_run"] = float((g - grads[name]).norm() / grads[name].norm())
    else:
        grads[name] = g
    captured.clear()
    del eps, adam, g
    torch.cuda.empty_cache()
    results["runs"].append(rec)
    print(json.dumps(rec), flush=True)
    (out / "check_mid250_binomial.json").write_text(json.dumps(results, indent=1))
g_ml = torch.from_numpy(np.load(out / "shot05_exact.npy")).double()  # multi-level exact (offload + pin)
g_bi, g_bl = grads["binomial"], grads["block_gpu"]
cmp = dict(binomial_vs_multilevel_rel=float((g_bi - g_ml).norm() / g_ml.norm()),
           binomial_vs_multilevel_max_abs=float((g_bi - g_ml).abs().max()), multilevel_max_abs=float(g_ml.abs().max()),
           block_vs_binomial_rel=float((g_bl - g_bi).norm() / g_bi.norm()),
           block_vs_binomial_cos=float((g_bl * g_bi).sum() / (g_bl.norm() * g_bi.norm())))
results["comparison"] = cmp
print("comparison: " + json.dumps(cmp), flush=True)
(out / "check_mid250_binomial.json").write_text(json.dumps(results, indent=1))
print("CHECK_DONE", flush=True)
