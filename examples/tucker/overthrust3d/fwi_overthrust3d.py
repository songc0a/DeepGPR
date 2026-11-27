"""3D GPR FWI of the SEG/EAGE Overthrust model (eps only), Tucker-compressed history, shots split over GPUs.

Model: ``overthrust.vites`` (801 x 801 x 187 cells at 25 m, m/s, x fastest) cropped
to the central 780 x 780 columns, one native cell per 2 cm cell (15.6 m x 15.6 m
x 3.74 m deep); velocity mapped linearly to eps 3.24 .. 8.0 (the range of
``examples/OverThrust.npy``, the 2D section of the earlier runs); 10 air cells
(eps 1) and 10 CPML cells per side: 800 x 217 x 800 = 138.9 M cells, the size of
the largest FWI of the size study (Section 9.7 of
docs/tucker_model_wavefield_compression.md). sigma = 1 mS/m below the air in the
true and in the inverted model (fixed, not inverted).

Acquisition: 500 MHz Ricker (dx 2 cm, dt 30 ps as in the size study), 10 shots
in three staggered rows (3-4-3) over the surface, or with ``--shot-grid N`` an
N x N regular grid of shots (cell centres of an N x N partition of the surface,
e.g. N = 4: 16 shots 3.9 m apart); for every shot 78 x 78
receivers (every 20 cm) over the whole surface; record = 1.25 x the two-way
vertical time to the bottom of the slowest column + the wavelet delay.

Inversion (round-6/7 dense settings): start = true model Gaussian-smoothed with
sigma 10 cells (20 cm); MS-SSIM misfit of ``~/MSSIM`` (each receiver line along
z is one section; evaluated in chunks of lines with the exact chunk-wise
gradient), anisotropic TV with the notebook rule (TV gradient scaled to 0.1 of
the data gradient norm), Adam lr 0.03 with cosine decay to 0.003, eps clamped
to [1, 12], air fixed. Gradient history: Tucker sketch rank 16, oversampling 8,
2-step segments, rhs from_e, host recovery.

Data parallel (DDP-style) over shots: ``torchrun --nproc_per_node=N``; rank r
takes shots r, r+N, ...; the gradients are summed with ``all_reduce`` (gloo on
host copies: no NCCL buffers on the full GPUs), every rank takes the same Adam
step and rank 0 broadcasts eps after it. Resumable: rank 0 writes
``run/checkpoint.pt`` after every iteration.

Usage (in this folder; ``--quick``: model decimated 4x, 125 MHz, 3 iterations)::

    python fwi_overthrust3d.py prepare
    torchrun --standalone --nproc_per_node=2 fwi_overthrust3d.py run
    # 4 x 4 shots in run16/ (copy eps_true.npy, eps_init.npy, meta.json from run/ first)
    torchrun --standalone --nproc_per_node=2 fwi_overthrust3d.py run --shot-grid 4 --run-dir run16
    # same with Adam lr 0.3 -> 0.03 (observed/ can be copied from run16/)
    torchrun --standalone --nproc_per_node=2 fwi_overthrust3d.py run --shot-grid 4 --lr 0.3 --run-dir run16_lr0.3
    # gradient smoothed with a 5-cell Gaussian before Adam, lr 0.1 -> 0.01
    torchrun --standalone --nproc_per_node=2 fwi_overthrust3d.py run --shot-grid 4 --lr 0.1 --grad-smooth 5 --run-dir run16_lr0.1_s5
    # same physical model on a 4 cm grid (250 MHz, 19.8 M cells), exact gradient (two-level checkpointing,
    # boundary states in host memory), one GPU; --self-test first compares exact and Tucker r16 on one shot
    python fwi_overthrust3d.py prepare --profile half --run-dir half_exact16
    torchrun --standalone --nproc_per_node=1 fwi_overthrust3d.py run --profile half --gradient exact --shot-grid 4 --self-test --run-dir half_exact16
    # the 2 cm model at 250 MHz (50 iterations), Tucker rank chosen with --rank
    python fwi_overthrust3d.py prepare --profile full250 --run-dir run16_250MHz_r16
    torchrun --standalone --nproc_per_node=1 fwi_overthrust3d.py run --profile full250 --shot-grid 4 --rank 16 --run-dir run16_250MHz_r16
    # the 2 cm model at 250 MHz with the exact gradient: four-level checkpointing (480 / 96 / 16 steps), E history
    # of each 16-step segment and all checkpoint states in pinned host memory (~86 GiB host and 16 GiB GPU per
    # process, ~11-12 min per shot on an RTX 4090); bind to the GPUs' NUMA node (GPUs 4-7: CPUs 32-63,96-127)
    taskset -c 32-63,96-127 python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:29644 --rdzv_id=ot250 fwi_overthrust3d.py run --profile full250 --shot-grid 4 \
        --gradient exact --outer-steps 480,96 --segment-steps 16 --offload --pin --run-dir run16_250MHz

A second ``torchrun --standalone`` on the same host collides with the first
one's rendezvous port 29400; for single-GPU runs start the script directly
with RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=<free port>.

Every run logs ``gcorr`` (Pearson correlation of the negative data gradient
with true - current model) and ``ucorr`` (of the applied update with it) and
saves the summed data gradient in ``gradients/`` at iteration 0 and every
``gradient_every`` iterations.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
VITES = HERE / "data/raw/Overthrust_3D_CD1/3-D_Overthrust_Model_Disk1/3D-Velocity-Grid/overthrust.vites"
NATIVE = (187, 801, 801)  # (depth, y, x), x fastest
C = 0.299792458  # m/ns
GIB = 1024**3

FULL = dict(
    crop=780, decimate=1, block_mean=False, air=10, pml=10, dx=0.02, dt=3.0e-11, freq=5.0e8, eps_range=(3.24, 8.0),
    sigma=1e-3, smooth_cells=10.0, receiver_step=10, receiver_margin=5, iterations=200, lr=0.03, lr_final_factor=0.1,
    tv_ratio=0.1, eps_bounds=(1.0, 12.0), loss_chunk_lines=6, snapshot_every=25, slices_every=5, shot_grid=None,
    grad_smooth_cells=0.0, gradient="tucker", misfit="mssim", exact_segment_steps=40, exact_outer_steps=(), exact_offload=False, exact_pin=False, gradient_every=10,
    binomial_checkpoints=7,
    sketch=dict(ranks=16, oversampling=8, segment_steps=2, rhs="from_e", recovery_device="host"),
    block_tolerance=0.03, block_size=32, block_floor=1e-4, block_margin_ns=2.0,
    block_source_factor=0.1, block_source_radius_cells=30, block_source_depth_cells=30, block_payload_dtype="float32",
)
# Same 15.6 x 15.6 x 3.72 m model on a 4 cm grid: 2 x 2 x 2 block means of the velocity (the odd last native
# depth cell dropped), 250 MHz so that the cells per wavelength stay those of the 2 cm / 500 MHz run, dt 60 ps
# (same Courant number), 20 cm of air, receivers every 20 cm, start model smoothed over 20 cm.
HALF = dict(FULL, decimate=2, block_mean=True, air=5, dx=0.04, dt=6.0e-11, freq=2.5e8, smooth_cells=5.0,
            receiver_step=5, receiver_margin=2, iterations=50, loss_chunk_lines=12, snapshot_every=10)
QUICK = dict(FULL, decimate=4, air=3, dx=0.08, dt=1.2e-10, freq=1.25e8, smooth_cells=2.5, receiver_step=3,
             receiver_margin=2, iterations=3, snapshot_every=2, slices_every=1, loss_chunk_lines=4, exact_segment_steps=20,
             gradient_every=1, sketch=dict(FULL["sketch"], segment_steps=4))
# The 2 cm / 138.9 M-cell model at 250 MHz (the frequency at which the 4 cm runs converge), dt 30 ps.
FULL250 = dict(FULL, freq=2.5e8, iterations=50, snapshot_every=10)
# The same 15.6 x 15.6 x 3.75 m model on a 2.8 cm grid (53.6 M cells with the CPML, 0.39 x the 2 cm grid): the
# native velocities linearly resampled at the cell centres of the coarser grid (factor 1 / 1.4), 250 MHz, dt 42 ps
# (same Courant number), 19.6 cm of air, receivers every 19.6 cm, start model smoothed over 20 cm, and the
# block-Tucker source region the same ~0.6 m as at 2 cm.
MID250 = dict(FULL250, zoom=1 / 1.4, air=7, dx=0.028, dt=4.2e-11, smooth_cells=20 / 2.8, receiver_step=7,
              receiver_margin=4, block_source_radius_cells=21, block_source_depth_cells=21)
PROFILES = dict(full=FULL, full250=FULL250, mid250=MID250, half=HALF, quick=QUICK)
DEFAULT_DIR = dict(full="run", full250="run250", mid250="run_mid250", half="half", quick="quick")


def scaled250(dx: float) -> dict:
    """The 2 cm / 250 MHz model resampled to cell size ``dx`` like MID250: same physical size and record length,
    same Courant number, ~20 cm of air, receivers every ~20 cm, start model smoothed over 20 cm, block-Tucker
    source region ~0.6 m (MID250 = scaled250(0.028))."""
    return dict(FULL250, zoom=0.02 / dx, air=round(0.2 / dx), dx=dx, dt=3.0e-11 * dx / 0.02, smooth_cells=0.2 / dx,
                receiver_step=round(0.2 / dx), receiver_margin=round(0.1 / dx),
                block_source_radius_cells=round(0.6 / dx), block_source_depth_cells=round(0.6 / dx))


# d210 ... d240: 2.1-2.4 cm grids between full250 and mid250 (largest grid that fits one 24 GB GPU, all on the GPU)
for _dx in (0.021, 0.022, 0.0225, 0.023, 0.024):
    PROFILES[f"d{round(_dx * 1e4)}"] = scaled250(_dx)
    DEFAULT_DIR[f"d{round(_dx * 1e4)}"] = f"run_d{round(_dx * 1e4)}"
CHECKED_KEYS = ("dx", "freq", "shot_grid", "lr", "grad_smooth_cells", "gradient", "misfit", "block_tolerance",
                "block_payload_dtype")


def config(profile: str, shot_grid: int | None = None, lr: float | None = None, grad_smooth: float | None = None,
           gradient: str | None = None, misfit: str | None = None, iterations: int | None = None,
           block_tolerance: float | None = None, payload_dtype: str | None = None) -> dict:
    cfg = dict(PROFILES[profile], shot_grid=shot_grid, profile=profile)
    if lr is not None:
        cfg["lr"] = lr  # initial Adam rate; the cosine decay still ends at lr_final_factor * lr
    if grad_smooth is not None:
        cfg["grad_smooth_cells"] = grad_smooth
    if gradient is not None:
        cfg["gradient"] = gradient
    if misfit is not None:
        cfg["misfit"] = misfit
    if iterations is not None:
        cfg["iterations"] = iterations
    if block_tolerance is not None:
        cfg["block_tolerance"] = block_tolerance
    if payload_dtype is not None:
        cfg["block_payload_dtype"] = payload_dtype  # stored Tucker cores/factors (fp16: per-block scaled cores)
    return cfg


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-300))


def smooth_gradient(g: torch.Tensor, sigma: float) -> torch.Tensor:
    """3D Gaussian smoothing (sigma in cells, edges 'nearest') of the ground gradient, on the host."""
    from scipy.ndimage import gaussian_filter

    host = np.ascontiguousarray(g.detach().cpu().numpy(), dtype=np.float32)
    return torch.from_numpy(gaussian_filter(host, sigma=sigma, mode="nearest", truncate=4.0)).to(g.device)


def workdir(profile: str, name: str | None = None) -> Path:
    return HERE / (name or DEFAULT_DIR[profile])


# ---------------------------------------------------------------------------
# Model and acquisition
# ---------------------------------------------------------------------------


def prepare(profile: str, name: str | None = None) -> None:
    """eps_true / eps_init (.npy, (x, depth, z) with the air rows) and meta.json."""
    from scipy.ndimage import gaussian_filter

    cfg, out = config(profile), workdir(profile, name)
    out.mkdir(parents=True, exist_ok=True)
    v = np.fromfile(VITES, dtype=">f4").reshape(NATIVE)
    vmin, vmax = float(v.min()), float(v.max())
    start = (NATIVE[1] - cfg["crop"]) // 2
    v = v[:, start:start + cfg["crop"], start:start + cfg["crop"]]
    d = cfg["decimate"]
    if cfg.get("zoom"):  # non-integer coarsening: linear interpolation at the cell centres of the coarser grid
        from scipy.ndimage import zoom

        target = tuple(int(round(n * cfg["zoom"])) for n in v.shape)
        v = zoom(v.astype(np.float64), [t / n for t, n in zip(target, v.shape)], order=1, mode="nearest",
                 grid_mode=True)
    elif cfg["block_mean"] and d > 1:  # d x d x d block means (trailing cells that do not fill a block dropped)
        nd, nc = v.shape[0] // d, cfg["crop"] // d
        v = v[:nd * d, :nc * d, :nc * d].astype(np.float64).reshape(nd, d, nc, d, nc, d).mean(axis=(1, 3, 5))
    else:
        v = v[::d, ::d, ::d]
    v = np.ascontiguousarray(v.transpose(2, 0, 1)).astype(np.float32)  # (x, depth, y) = DeepGPR (x, y, z)
    e0, e1 = cfg["eps_range"]
    ground = (e0 + (v - vmin) / (vmax - vmin) * (e1 - e0)).astype(np.float32)
    smooth = gaussian_filter(ground, sigma=cfg["smooth_cells"], mode="nearest", truncate=4.0).astype(np.float32)
    nx, depth, nz = ground.shape
    air = np.ones((nx, cfg["air"], nz), dtype=np.float32)
    eps_true = np.concatenate([air, ground], axis=1)
    eps_init = np.concatenate([air, smooth], axis=1)
    np.save(out / "eps_true.npy", eps_true)
    np.save(out / "eps_init.npy", eps_init)
    two_way = float((2 * np.sqrt(ground.astype(np.float64)) * cfg["dx"] / C).sum(axis=1).max())  # ns, slowest column
    delay = 1.2 / cfg["freq"]
    nt = int(math.ceil((1.25 * two_way * 1e-9 + delay) / cfg["dt"]))
    cells = (nx + 2 * cfg["pml"]) * (depth + cfg["air"] + 2 * cfg["pml"]) * (nz + 2 * cfg["pml"])
    meta = dict(shape=list(eps_true.shape), ground_shape=[nx, depth, nz], cells=cells, nt=nt, two_way_ns=two_way,
                record_ns=nt * cfg["dt"] * 1e9, velocity_range=[vmin, vmax], size_m=[nx * cfg["dx"], depth * cfg["dx"], nz * cfg["dx"]],
                eps_true_range=[float(ground.min()), float(ground.max())], eps_init_rmse=float(np.sqrt(((smooth - ground) ** 2).mean())))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def acquisition(cfg: dict, shape) -> tuple:
    """Sources (S, 1, 3) and receivers (S, R, 3), receivers z-fastest on the surface row ``air``."""
    nx, _, nz = shape
    air = cfg["air"]
    positions = []
    if cfg.get("shot_grid"):
        n = cfg["shot_grid"]  # n x n regular grid, x outer / z inner, symmetric about the centre
        for i in range(n):
            for k in range(n):
                positions.append(((2 * i + 1) * nx // (2 * n), (2 * k + 1) * nz // (2 * n)))
    else:
        rows = [(1 / 6, 3), (1 / 2, 4), (5 / 6, 3)]  # (z fraction, shots in the row): staggered 3-4-3
        for zf, count in rows:
            for k in range(count):
                positions.append((int(round(nx * (2 * k + 1) / (2 * count))), int(round(nz * zf))))
    sources = torch.zeros((len(positions), 1, 3), dtype=torch.int32)
    for s, (ix, iz) in enumerate(positions):
        sources[s, 0] = torch.tensor([ix, air, iz], dtype=torch.int32)
    m, step = cfg["receiver_margin"], cfg["receiver_step"]
    rx = torch.arange(m, nx - m + 1, step)
    rz = torch.arange(m, nz - m + 1, step)
    grid = torch.stack(torch.meshgrid(rx, rz, indexing="ij"), dim=-1).reshape(-1, 2)
    receivers = torch.zeros((len(positions), grid.shape[0], 3), dtype=torch.int32)
    receivers[:, :, 0], receivers[:, :, 1], receivers[:, :, 2] = grid[:, 0], air, grid[:, 1]
    return sources, receivers, (rx.numel(), rz.numel())


# ---------------------------------------------------------------------------
# Misfit
# ---------------------------------------------------------------------------


class ChunkedMSSIM:
    """``Misfit_M_SSIM`` of ``~/MSSIM`` on receiver lines, evaluated in chunks of lines.

    The misfit is ``200 mean(-(L CS)) + 200`` over (lines, 5 kernels, time,
    receivers) with every trace scaled to its maximum; lines are independent, so
    the mean over all lines is the line-weighted mean of chunk means and the
    gradient is assembled chunk by chunk (the full 6084-trace x 2800-sample
    section would need ~5 GiB of convolution buffers on a full GPU).
    """

    def __init__(self, lines: tuple, chunk: int):
        for path in (os.environ.get("MSSIM_PARENT"), os.path.expanduser("~"), os.path.expanduser("~/Desktop")):
            if path and os.path.isdir(os.path.join(path, "MSSIM")):
                sys.path.insert(0, path)
                break
        from MSSIM import Misfit_M_SSIM

        self.mssim = Misfit_M_SSIM()
        self.lines, self.chunk = lines, chunk

    def sections(self, data: torch.Tensor) -> torch.Tensor:
        nx_r, nz_r = self.lines
        return data[0].reshape(data.shape[1], nx_r, nz_r).permute(1, 0, 2)  # (lines along x, T, receivers along z)

    def __call__(self, syn: torch.Tensor, obs: torch.Tensor):
        """Value and gradient with respect to ``syn`` (1, T, R)."""
        s, o = self.sections(syn.detach()), self.sections(obs)
        grad = torch.zeros_like(s)
        total, lines = 0.0, s.shape[0]
        for start in range(0, lines, self.chunk):
            part = s[start:start + self.chunk].contiguous().requires_grad_(True)
            value = (self.mssim.forward(o[start:start + self.chunk].contiguous(), part) - 200.0) * (part.shape[0] / lines)
            (g,) = torch.autograd.grad(value, part)
            grad[start:start + self.chunk] = g
            total += float(value)
        return total + 200.0, grad.permute(1, 0, 2).reshape(syn.shape)

    def full(self, syn: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.mssim.forward(self.sections(obs), self.sections(syn))


class L2Misfit:
    """0.5 |syn - obs|^2 / |obs|^2 per shot (value and gradient with respect to ``syn``)."""

    def __call__(self, syn: torch.Tensor, obs: torch.Tensor):
        scale = float(obs.square().sum())
        residual = syn.detach() - obs
        return float(0.5 * residual.square().sum()) / scale, residual / scale


def exact_gradient(eps: torch.Tensor, sigma: torch.Tensor, opts: dict, segment_steps: int, data_gradient,
                   outer_steps=(), offload: bool = False, pin: bool = False) -> torch.Tensor:
    """Exact (uncompressed) gradient by multi-level checkpointing; accumulates ``eps.grad``, returns the data.

    A history-free forward pass keeps the 30 solver state tensors (E, H, CPML) at
    every boundary of the coarsest level in host memory; ``data_gradient(data)``
    gives the cotangent of the receiver data. The backward pass walks the segments
    in reverse: a segment of an outer level is re-run history-free from its
    boundary to store the boundaries of the next finer level (``outer_steps``
    coarse to fine, then ``segment_steps``), and a finest segment is recomputed
    with the full wavefield history and differentiated, chaining the state
    cotangents back to the start. Without ``outer_steps`` this is the two-level
    scheme (as ``offloaded_checkpoint_gradients`` in
    ``examples/tucker/benchmark_tucker.py``); each outer level costs one more
    history-free forward pass and cuts the host memory to the sum over levels of
    (segment / finer segment) boundaries.

    At full size the fixed device memory of one differentiated segment (padded
    materials, state, adjoint state, update coefficients) is ~19 GiB, so
    ``offload`` keeps the segment's E history in pinned host memory
    (``use_async_offload``), and the state cotangents are handed to the native
    adjoint without its defensive copy (they are ours and discarded after the
    segment), which saves one state (~3.3 GiB at 2 cm). ``pin`` keeps the
    boundaries in pinned host memory (one flat block each: PyTorch's host
    allocator rounds every pinned block up to a power of two) for faster copies.
    """
    import DeepGPR
    import DeepGPR.solver.autograd as solver_autograd

    def own_in_place(gradient, shape, device, dtype):
        return torch.zeros(shape, device=device, dtype=dtype) if gradient is None else gradient.contiguous()

    source = opts["source_amplitudes"]
    common = {k: v for k, v in opts.items() if k != "source_amplitudes"}
    levels = [int(s) for s in outer_steps if s] + [int(segment_steps)]
    if offload:
        # The native async transfer slots (2 x 3 E components on the extended grid) are cudaMalloc'ed
        # per call and not cached above 64 MiB, so cap PyTorch's pool to leave them room (+1 GiB).
        pml = int(opts.get("pmlthick", 10))
        slots = 2 * 3 * 4 * np.prod([n + 2 * pml + 1 for n in eps.shape])
        total = torch.cuda.get_device_properties(eps.device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, (total - slots - 2**30) / total), eps.device)
    cotangent, states = None, None

    def to_host(state: tuple) -> tuple:
        if not pin:
            return tuple(t.to("cpu") for t in state)
        flat = torch.empty(sum(t.numel() for t in state), dtype=state[0].dtype, pin_memory=True)
        views, offset = [], 0
        for t in state:
            views.append(flat[offset:offset + t.numel()].view(t.shape).copy_(t))
            offset += t.numel()
        return tuple(views)

    def forward(a: int, b: int, step: int, boundary, pieces: list | None) -> list:
        """History-free run over [a, b) from the host ``boundary``; host boundaries every ``step`` steps."""
        state = None if boundary is None else tuple(t.to(eps.device) for t in boundary)
        boundaries = []
        with torch.no_grad():
            for start in range(a, b, step):
                boundaries.append(boundary if start == a else to_host(state))
                if pieces is None and start + step >= b:
                    break  # the state at b is not needed
                result = DeepGPR.compute(
                    eps_r=eps, sigma=sigma, source_amplitudes=source[:, start:min(start + step, b)].contiguous(),
                    save_wavefield_history=False, E=state[:3] if state else None, H=state[3:6] if state else None,
                    PML=state[6:] if state else None, **common)
                state = (*result[1], *result[2], *result[3])
                if pieces is not None:
                    pieces.append(result[-1].clone())
                del result
        return boundaries

    def backward(a: int, b: int, boundaries: list, depth: int) -> None:
        nonlocal states
        step = levels[depth]
        starts = list(range(a, b, step))
        for index in reversed(range(len(starts))):
            start, stop = starts[index], min(starts[index] + step, b)
            boundary, boundaries[index] = boundaries[index], None
            if depth + 1 < len(levels):
                backward(start, stop, forward(start, stop, levels[depth + 1], boundary, None), depth + 1)
                continue
            inputs = None if boundary is None else [t.to(eps.device).requires_grad_(True) for t in boundary]
            del boundary
            torch.cuda.empty_cache()  # offload: native transfer slots are cudaMalloc'ed; else: fragmentation
            if os.environ.get("EXACT_DEBUG"):
                free = torch.cuda.mem_get_info()[0]
                print(f"{time.time():.1f} exact segment {start}-{stop}: allocated {torch.cuda.memory_allocated() / 2**30:.2f}, reserved "
                      f"{torch.cuda.memory_reserved() / 2**30:.2f}, peak so far {torch.cuda.max_memory_allocated() / 2**30:.2f}, "
                      f"device free {free / 2**30:.2f} GiB", flush=True)
            result = DeepGPR.compute(
                eps_r=eps, sigma=sigma, source_amplitudes=source[:, start:stop].contiguous(),
                E=inputs[:3] if inputs else None, H=inputs[3:6] if inputs else None,
                PML=inputs[6:] if inputs else None, use_async_offload=offload, **common)
            outputs, grads = [result[-1]], [cotangent[:, start:stop].contiguous()]
            if states is not None:  # (no loop variables: one would keep the graph and its 30 inputs alive)
                keep = [g is not None and t.requires_grad for t, g in zip((*result[1], *result[2], *result[3]), states)]
                outputs += [t for t, k in zip((*result[1], *result[2], *result[3]), keep) if k]
                grads += [g for g, k in zip(states, keep) if k]
            states = None
            if offload:
                torch.cuda.empty_cache()
            original = solver_autograd._own_state_gradient
            solver_autograd._own_state_gradient = own_in_place
            try:
                torch.autograd.backward(outputs, grads)
            finally:
                solver_autograd._own_state_gradient = original
            states = None if inputs is None else [t.grad for t in inputs]
            del result, outputs, grads, inputs

    pieces = []
    boundaries = forward(0, source.shape[1], levels[0], None, pieces)
    data = torch.cat(pieces, dim=1)
    del pieces
    cotangent = data_gradient(data)
    backward(0, source.shape[1], boundaries, 0)
    return data


def binomial_gradient(eps: torch.Tensor, sigma: torch.Tensor, opts: dict, segment_steps: int, data_gradient,
                      checkpoints: int, stats: dict | None = None, persistent_history: bool = False) -> torch.Tensor:
    """Exact gradient by binomial (revolve-type) checkpointing, every state on the device; accumulates ``eps.grad``.

    The time axis is cut into segments of ``segment_steps``; a segment is recomputed from its start state with
    its E history on the device and differentiated, last segment first, chaining the state cotangents back. With
    c free checkpoint slots and r recomputations, eta(c, r) = C(c + r, c) segments can be reversed (Griewank &
    Walther, ACM TOMS 26(1), 2000, "revolve"): a range of n segments is split so that the right part has at most
    eta(c - 1, r) segments (one slot fewer) and the left part at most eta(c, r - 1) (one more recomputation),
    with r the smallest value for which eta(c, r) >= n. The first descent of this recursion is one forward sweep,
    which also records the receiver data; ``data_gradient(data)`` gives its cotangent. The solver advances
    states in place, so a checkpoint is cloned before it is advanced. ``stats`` receives step and checkpoint
    counts.
    """
    import DeepGPR
    import DeepGPR.solver.autograd as solver_autograd

    def own_in_place(gradient, shape, device, dtype):
        return torch.zeros(shape, device=device, dtype=dtype) if gradient is None else gradient.contiguous()

    # The native forward and backward each take a fresh (6, nx+1, ny+1, nz+1) scratch for the update coefficients
    # (2.38 GiB at 2.2 cm); allocating it twice per segment fragmented the default allocator until a later shot
    # found no contiguous block (OOM at shot 5 of the first d220 run). One zeroed scratch serves every call here,
    # and the cache is emptied at the start and end of the shot.
    scratch: dict = {}
    original_coefficients = solver_autograd._allocate_update_coefficients

    def reuse_coefficients(config):
        key = (config.nx, config.ny, config.nz, config.dtype, str(config.device))
        buf = scratch.get(key)
        if buf is None:
            buf = scratch[key] = torch.zeros((6, config.nx + 1, config.ny + 1, config.nz + 1), device=config.device,
                                             dtype=config.dtype)
        else:
            buf.zero_()
        return dict(zip(solver_autograd._UPDATE_COEFFICIENT_NAMES, buf.unbind(0)))

    # Optionally the same for the differentiated segment's E history (3 frames = 3.55 GiB at 2.2 cm) and its final E
    # frame: one flat buffer each, viewed to the requested shape. Off by default: held through the history-free
    # advances it raised the d220 S=3/C=3 peak from 21.5 to 22.7 GiB (third OOM); with max_split_size_mb these
    # sizes are reused from the cache anyway.
    original_histories = solver_autograd._allocate_histories
    flat: dict = {}

    def view_of(name: str, shape, config):
        numel = math.prod(shape)
        buf = flat.get(name)
        if buf is None or buf.numel() < numel:
            flat.pop(name, None)
            buf = flat[name] = torch.empty(numel, device=config.device, dtype=config.storage.dtype)
        return buf[:numel].view(shape)

    def reuse_histories(config, needs_model_gradient, use_async_offload, reconstruct_rhs=False):
        if (not config.save_wavefield_history or use_async_offload or not reconstruct_rhs or not needs_model_gradient
                or config.storage.compression == "int8"):
            return original_histories(config, needs_model_gradient, use_async_offload, reconstruct_rhs)
        cells = (config.nstep, *config.history_cells)
        shape = solver_autograd.saved_history_shape(config.mode, solver_autograd.saved_time_steps(
            config.nt, config.sampling_interval), *cells)
        final = view_of("final", solver_autograd.saved_history_shape(config.mode, 1, *cells), config)
        return view_of("e", shape, config), final, shape

    torch.cuda.empty_cache()
    solver_autograd._allocate_update_coefficients = reuse_coefficients
    if persistent_history:
        solver_autograd._allocate_histories = reuse_histories
    try:
        return _binomial(eps, sigma, opts, segment_steps, data_gradient, checkpoints, stats, own_in_place)
    finally:
        solver_autograd._allocate_update_coefficients = original_coefficients
        solver_autograd._allocate_histories = original_histories
        scratch.clear()
        flat.clear()
        torch.cuda.empty_cache()


def _binomial(eps, sigma, opts, segment_steps, data_gradient, checkpoints, stats, own_in_place):
    """Body of binomial_gradient (the schedule and the solver calls)."""
    import DeepGPR
    import DeepGPR.solver.autograd as solver_autograd

    source = opts["source_amplitudes"]
    common = {k: v for k, v in opts.items() if k != "source_amplitudes"}
    nt, size = source.shape[1], int(segment_steps)
    n = -(-nt // size)
    st = stats if stats is not None else {}
    st.update(segments=n, segment_steps=size, checkpoints=int(checkpoints), forward_steps=0, differentiated_steps=0,
              max_checkpoints_held=0)
    held, expected = [0], [n - 1]
    pieces: list = []
    cotangent, states = None, None

    def span(k: int) -> tuple:
        return k * size, min((k + 1) * size, nt)

    def split(a: int, b: int, c: int) -> int:
        r = 0
        while math.comb(c + r, c) < b - a:
            r += 1
        return b - min(math.comb(c - 1 + r, c - 1), b - a - 1)

    def run(state, k: int, history: bool):
        a, b = span(k)
        return DeepGPR.compute(eps_r=eps, sigma=sigma, source_amplitudes=source[:, a:b].contiguous(),
                               save_wavefield_history=history, E=state[:3] if state else None,
                               H=state[3:6] if state else None, PML=state[6:] if state else None, **common)

    def advance(state, k0: int, k1: int, collect: bool = False):
        """History-free run from the start of segment k0 to the start of k1 (``state`` itself is kept), in one
        solver call (a call per segment costs ~30-50 ms of setup at 2.2 cm, ~3 forward steps)."""
        state = None if state is None else tuple(t.clone() for t in state)
        if k1 <= k0:
            return state
        a, b = span(k0)[0], span(k1 - 1)[1]
        with torch.no_grad():
            result = DeepGPR.compute(eps_r=eps, sigma=sigma, source_amplitudes=source[:, a:b].contiguous(),
                                     save_wavefield_history=False, E=state[:3] if state else None,
                                     H=state[3:6] if state else None, PML=state[6:] if state else None, **common)
            state = (*result[1], *result[2], *result[3])
            if collect:
                pieces.append(result[-1].clone())
            del result
        st["forward_steps"] += b - a
        return state

    def differentiate(k: int, state) -> None:
        """Segment k from ``state`` (its last use) with the E history, backward with the chained cotangents."""
        nonlocal states
        if k != expected[0]:
            raise RuntimeError(f"binomial schedule reached segment {k}, expected {expected[0]}")
        expected[0] -= 1
        a, b = span(k)
        inputs = None if state is None else [t.requires_grad_(True) for t in state]
        del state
        result = run(inputs, k, True)
        outputs, grads = [result[-1]], [cotangent[:, a:b].contiguous()]
        if states is not None:
            keep = [g is not None and t.requires_grad for t, g in zip((*result[1], *result[2], *result[3]), states)]
            outputs += [t for t, kp in zip((*result[1], *result[2], *result[3]), keep) if kp]
            grads += [g for g, kp in zip(states, keep) if kp]
        states = None
        original = solver_autograd._own_state_gradient
        solver_autograd._own_state_gradient = own_in_place
        try:
            torch.autograd.backward(outputs, grads)
        finally:
            solver_autograd._own_state_gradient = original
        states = None if inputs is None else [t.grad for t in inputs]
        st["differentiated_steps"] += b - a
        del result, outputs, grads, inputs

    def backward(a: int, b: int, state_a, c: int, stored: dict) -> None:
        """Reverse segments [a, b) from ``state_a`` (the state at segment a) with c free checkpoint slots."""
        if b - a == 1:
            differentiate(a, state_a)
            return
        if c == 0:  # no slot left: recompute from state_a for every segment
            for k in range(b - 1, a, -1):
                differentiate(k, advance(state_a, a, k))
            differentiate(a, state_a)
            return
        m = split(a, b, c)
        state_m = stored.pop(m) if m in stored else advance(state_a, a, m)
        held[0] += 1
        st["max_checkpoints_held"] = max(st["max_checkpoints_held"], held[0])
        backward(m, b, state_m, c - 1, stored)
        del state_m
        held[0] -= 1
        backward(a, m, state_a, c, stored)

    # first descent: one forward sweep storing the checkpoints of the right-most path, and all receiver data
    stored, a, c, state = {}, 0, int(checkpoints), None
    while n - a > 1 and c > 0:
        m = split(a, n, c)
        state = advance(state, a, m, collect=True)
        stored[m] = state
        a, c = m, c - 1
    advance(state, a, n, collect=True)
    del state
    data = torch.cat(pieces, dim=1)
    pieces.clear()
    cotangent = data_gradient(data)
    backward(0, n, None, int(checkpoints), stored)
    if expected[0] != -1:
        raise RuntimeError(f"binomial schedule stopped before segment {expected[0]}")
    return data


def tv_gradient(m: torch.Tensor) -> torch.Tensor:
    """Gradient of the anisotropic TV sum |d m| over all axes (sign(0) = 0, as autograd of abs)."""
    g = torch.zeros_like(m)
    for axis in range(m.dim()):
        n = m.shape[axis]
        s = torch.sign(m.narrow(axis, 1, n - 1) - m.narrow(axis, 0, n - 1))
        g.narrow(axis, 0, n - 1).sub_(s)
        g.narrow(axis, 1, n - 1).add_(s)
        del s
    return g


# ---------------------------------------------------------------------------
# FWI
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def allreduce_host(t: torch.Tensor) -> torch.Tensor:
    host = t.detach().cpu()
    dist.all_reduce(host, op=dist.ReduceOp.SUM)
    return host.to(t.device)


def run(profile: str, self_test: bool, shot_grid: int | None = None, name: str | None = None, lr: float | None = None,
        grad_smooth: float | None = None, gradient: str | None = None, misfit_name: str | None = None,
        iterations: int | None = None, segment_steps: int | None = None, rank_override: int | None = None,
        outer_steps: tuple | None = None, offload: bool = False, pin: bool = False,
        block_tolerance: float | None = None, host_limit_gib: float | None = None, spill_dir: str | None = None,
        payload_dtype: str | None = None, stop_after: int | None = None, history_on_gpu: bool = False,
        checkpoints: int | None = None) -> None:
    import DeepGPR
    from DeepGPR.tucker import BlockTuckerConfig, TuckerSketchConfig, surface_adjoint_keep_until

    dist.init_process_group("gloo", timeout=datetime.timedelta(hours=8))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg, out = config(profile, shot_grid, lr, grad_smooth, gradient, misfit_name, iterations, block_tolerance,
                      payload_dtype), workdir(profile, name)
    if segment_steps is not None:
        cfg["exact_segment_steps"] = segment_steps
    if checkpoints is not None:
        cfg["binomial_checkpoints"] = checkpoints
    if rank_override is not None:
        cfg["sketch"] = dict(cfg["sketch"], ranks=rank_override)
    if outer_steps is not None:
        cfg["exact_outer_steps"] = tuple(outer_steps)
    if offload:
        cfg["exact_offload"] = True
    if pin:
        cfg["exact_pin"] = True
    meta = json.loads((out / "meta.json").read_text())
    nt, air = meta["nt"], cfg["air"]
    eps_true_host = torch.from_numpy(np.load(out / "eps_true.npy"))
    shape = tuple(eps_true_host.shape)
    sources, receivers, lines = acquisition(cfg, shape)
    shots = list(range(rank, sources.shape[0], world))
    n_shots = sources.shape[0]
    wavelet = DeepGPR.wavelet.ricker(cfg["freq"], nt, cfg["dt"], 1.2 / cfg["freq"]).reshape(1, nt, 1).to(device)
    kwargs = dict(device=device, dx=cfg["dx"], dt=cfg["dt"], source_amplitudes=wavelet, pmlthick=cfg["pml"], mode=3,
                  source_direction=2, receiver_component=2)

    def options(s: int) -> dict:
        return dict(kwargs, source_location=sources[s:s + 1].to(device), receiver_location=receivers[s:s + 1].to(device))

    sigma = torch.full(shape, cfg["sigma"], device=device)
    sigma[:, :air] = 0.0
    if rank == 0:
        log(f"world {world}, cells {meta['cells'] / 1e6:.1f} M, model {shape}, nt {nt} ({meta['record_ns']:.1f} ns), "
            f"{n_shots} shots, {receivers.shape[1]} receivers ({lines[0]} x {lines[1]}), DeepGPR {DeepGPR.__file__}")
        log(f"profile {profile} (dx {cfg['dx']} m, {cfg['freq'] / 1e6:.0f} MHz, dt {cfg['dt'] * 1e12:.0f} ps), "
            f"{cfg['iterations']} iterations, misfit {cfg['misfit']}, gradient {cfg['gradient']}"
            + (f" (levels {[*cfg['exact_outer_steps'], cfg['exact_segment_steps']]} steps, states in host memory"
               f"{', E history offloaded to pinned host memory' if cfg['exact_offload'] else ''}"
               f"{', pinned boundaries' if cfg['exact_pin'] else ''})" if cfg["gradient"] == "exact" else
               f" (binomial checkpointing on the GPU: {cfg['binomial_checkpoints']} checkpoints, segments of "
               f"{cfg['exact_segment_steps']} steps)" if cfg["gradient"] == "binomial" else
               f" (block Tucker {cfg['block_size']}^3, tolerance {cfg['block_tolerance']}, floor {cfg['block_floor']}, "
               f"adjoint shadow margin {cfg['block_margin_ns']} ns, source region +-{cfg['block_source_radius_cells']} x "
               f"{cfg['block_source_depth_cells']} cells at {cfg['block_source_factor']} x tolerance, history in "
               f"{'GPU memory (512 MiB buffers)' if history_on_gpu else 'host memory'}"
               f"{'' if cfg['block_payload_dtype'] == 'float32' else ' as ' + cfg['block_payload_dtype']}"
               f"{'' if host_limit_gib is None else f', beyond {host_limit_gib} GiB spilled to {spill_dir}'})"
               if cfg["gradient"] == "block" else
               f" (Tucker {cfg['sketch']})"))
        log(f"Adam lr {cfg['lr']} -> {cfg['lr'] * cfg['lr_final_factor']:.4g} (cosine), gradient smoothing sigma {cfg['grad_smooth_cells']} cells")
        log(f"run dir {out}, shot positions (x, z cells): {[tuple(int(v) for v in sources[s, 0, [0, 2]]) for s in range(n_shots)]}")
        (out / "acquisition.json").write_text(json.dumps(dict(
            shot_grid=cfg["shot_grid"], sources=sources[:, 0].tolist(), receiver_lines=list(lines),
            receiver_step=cfg["receiver_step"], receiver_margin=cfg["receiver_margin"])))

    # Observed data (true model, no history), one file per shot.
    obs_dir = out / "observed"
    obs_dir.mkdir(exist_ok=True)
    for s in shots:
        path = obs_dir / f"shot{s:02d}.pt"
        if path.exists():
            continue
        t0 = time.perf_counter()
        eps_true = eps_true_host.to(device)
        with torch.no_grad():
            data = DeepGPR.compute(eps_r=eps_true, sigma=sigma, save_wavefield_history=False, **options(s))[-1]
        torch.save(data.cpu(), path.with_suffix(f".{os.getpid()}.tmp"))
        os.replace(path.with_suffix(f".{os.getpid()}.tmp"), path)
        del eps_true, data
        torch.cuda.empty_cache()
        log(f"rank {rank}: observed shot {s} in {time.perf_counter() - t0:.0f} s")
    dist.barrier()
    observed = {s: torch.load(obs_dir / f"shot{s:02d}.pt") for s in shots}  # host; moved per shot
    misfit = ChunkedMSSIM(lines, cfg["loss_chunk_lines"]) if cfg["misfit"] == "mssim" else L2Misfit()
    sketch = TuckerSketchConfig(**cfg["sketch"])
    ground = (slice(None), slice(air, None), slice(None))
    block_histories: list = []  # the block history the solver builds for the current shot (for its size)
    if cfg["gradient"] == "block":
        import DeepGPR.solver.sketched as solver_sketched

        class CapturedBlockHistory(solver_sketched.BlockTuckerHistory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                block_histories.append(self)

        solver_sketched.BlockTuckerHistory = CapturedBlockHistory

    if self_test and rank == 0 and cfg["gradient"] == "exact":
        # One shot at the start model: exact gradient against the Tucker one, both against the model error.
        s = shots[0]
        obs = observed[s].to(device)
        init = torch.from_numpy(np.load(out / "eps_init.npy")).to(device)
        grads, datas = {}, {}
        for method in ("exact", "tucker"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            e = init.clone().requires_grad_(True)
            if method == "exact":
                d = exact_gradient(e, sigma, options(s), cfg["exact_segment_steps"], lambda x: misfit(x, obs)[1], cfg["exact_outer_steps"], cfg["exact_offload"], cfg["exact_pin"])
            else:
                d = DeepGPR.compute(eps_r=e, sigma=sigma, wavefield_sketch=sketch, **options(s))[-1]
                d.backward(misfit(d, obs)[1])
            torch.cuda.synchronize()
            grads[method], datas[method] = e.grad[ground].cpu(), d.detach().cpu()
            log(f"self-test {method} gradient shot {s}: {time.perf_counter() - t0:.0f} s, "
                f"GPU peak {torch.cuda.max_memory_allocated() / GIB:.2f} GiB")
            del e, d
            torch.cuda.empty_cache()
        ge, gt = grads["exact"], grads["tucker"]
        target = eps_true_host[ground] - init[ground].cpu()
        log(f"self-test exact vs Tucker: data rel diff {float((datas['tucker'] - datas['exact']).norm() / datas['exact'].norm()):.2e}, "
            f"gradient rel diff {float((gt - ge).norm() / ge.norm()):.3e}, "
            f"cosine {float((gt.double() * ge.double()).sum() / (gt.double().norm() * ge.double().norm())):.6f}; "
            f"corr(-g, true - init): exact {pearson(-ge, target):.4f}, Tucker {pearson(-gt, target):.4f}")
        (out / "gradients").mkdir(exist_ok=True)
        np.save(out / "gradients" / f"selftest_shot{s:02d}_exact.npy", ge.numpy())
        np.save(out / "gradients" / f"selftest_shot{s:02d}_tucker.npy", gt.numpy())
        del obs, init, grads, datas, ge, gt, target
        torch.cuda.empty_cache()

    if self_test and rank == 0 and cfg["misfit"] == "mssim":
        # Chunked MS-SSIM value/gradient against the full evaluation, TV gradient against autograd.
        syn = observed[shots[0]].to(device) * 0.7 + 0.05 * torch.randn_like(observed[shots[0]].to(device)) * observed[shots[0]].abs().max().to(device)
        obs = observed[shots[0]].to(device)
        value, grad = misfit(syn, obs)
        leaf = syn.clone().requires_grad_(True)
        full = misfit.full(leaf, obs)
        (g_full,) = torch.autograd.grad(full, leaf)
        log(f"self-test MS-SSIM: chunked {value:.6f} full {float(full):.6f}, grad rel diff {float((grad - g_full).norm() / g_full.norm()):.2e}")
        m = torch.from_numpy(np.load(out / "eps_init.npy"))[:, air:].to(device).requires_grad_(True)
        tv = DeepGPR.TVRegularization(weight_ep=1.0, weight_sigma=1.0).to(device)
        (g_auto,) = torch.autograd.grad(tv(ep=m, sigma=None), m)
        log(f"self-test TV: grad rel diff {float((tv_gradient(m.detach()) - g_auto).norm() / g_auto.norm()):.2e}")
        del syn, obs, leaf, grad, g_full, m, g_auto
        torch.cuda.empty_cache()

    eps = torch.from_numpy(np.load(out / "eps_init.npy")).to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([eps], lr=cfg["lr"])
    history = []
    ckpt = out / "checkpoint.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        written = dict(config(profile), **state.get("config", {}))  # keys missing in old checkpoints = defaults
        for key in CHECKED_KEYS:
            if written[key] != cfg[key]:
                raise SystemExit(f"{ckpt} was written with {key} {written[key]}, not {cfg[key]}")
        if cfg["gradient"] == "tucker" and written["sketch"]["ranks"] != cfg["sketch"]["ranks"]:
            raise SystemExit(f"{ckpt} was written with Tucker rank {written['sketch']['ranks']}, not {cfg['sketch']['ranks']}")
        with torch.no_grad():
            eps.copy_(state["eps"].to(device))
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        del state
        if rank == 0:
            log(f"resumed after iteration {len(history) - 1}")
    truth_ground = eps_true_host[ground] if rank == 0 else None
    half = (shape[1] - air) // 2
    (out / "snapshots").mkdir(exist_ok=True)
    (out / "slices").mkdir(exist_ok=True)
    (out / "gradients").mkdir(exist_ok=True)

    def model_metrics(model_host: torch.Tensor) -> dict:
        diff = model_host[ground] - truth_ground
        return dict(rmse=float(diff.square().mean().sqrt()), rmse_upper=float(diff[:, :half].square().mean().sqrt()),
                    rmse_lower=float(diff[:, half:].square().mean().sqrt()), corr=pearson(model_host[ground], truth_ground))

    def save_slices(model_host: torch.Tensor, iteration: int) -> None:
        nx, _, nz = shape
        depths = [air + int(f * (shape[1] - air)) for f in (0.15, 0.4, 0.7)]
        np.savez_compressed(out / "slices" / f"it{iteration:04d}.npz", x_slice=model_host[nx // 2].numpy(),
                            z_slice=model_host[:, :, nz // 2].numpy(), depth_index=np.array(depths),
                            depth_slices=np.stack([model_host[:, d].numpy() for d in depths]))

    if rank == 0 and not history:
        init_metrics = model_metrics(eps.detach().cpu())
        log(f"initial model: {init_metrics}")
        save_slices(eps.detach().cpu(), -1)
        save_slices(eps_true_host, -2)  # the true model
    iterations = cfg["iterations"]
    base_lr = cfg["lr"]
    last = iterations if stop_after is None else min(iterations, stop_after)  # stop early, same cosine schedule
    for iteration in range(len(history), last):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_local = l2_local = energy_local = 0.0
        shot_seconds = []
        history_sizes = []  # block Tucker: (stored GiB, of which spilled to disk GiB) per shot
        raw_history_bytes = None
        if cfg["gradient"] == "block":
            # Block-wise rank-adaptive Tucker of the E history (host memory); the adjoint shadow below the
            # surface receivers follows the current model.
            keep = surface_adjoint_keep_until(eps.detach().cpu(), air, cfg["dx"], cfg["dt"], nt, cfg["pml"],
                                              margin=cfg["block_margin_ns"] * 1e-9)
            sketch_it = BlockTuckerConfig(**{k: v for k, v in cfg["sketch"].items() if k != "ranks"},
                                          block=(cfg["block_size"],) * 3, origin=(0, cfg["pml"] + air, 0),
                                          tolerance=cfg["block_tolerance"], floor=cfg["block_floor"], keep_until=keep,
                                          storage_device=str(device) if history_on_gpu else "cpu",
                                          device_chunk_bytes=512 * 2**20 if history_on_gpu else None,
                                          host_limit_bytes=None if host_limit_gib is None else int(host_limit_gib * GIB),
                                          spill_dir=spill_dir, payload_dtype=cfg["block_payload_dtype"])
        else:
            sketch_it = sketch

        def shot_sketch(s: int):
            """Block Tucker: the blocks around the source (near-singular direct wave) get factor x tolerance."""
            if cfg["gradient"] != "block" or not cfg["block_source_factor"]:
                return sketch_it
            x, z = (int(v) + cfg["pml"] for v in sources[s, 0, [0, 2]])
            r, top = cfg["block_source_radius_cells"], cfg["pml"] + air
            region = (x - r, x + r + 1, top, top + cfg["block_source_depth_cells"], z - r, z + r + 1,
                      cfg["block_tolerance"] * cfg["block_source_factor"])
            return dataclasses.replace(sketch_it, tight_regions=(region,))
        for s in shots:
            t_shot = time.perf_counter()
            obs = observed[s].to(device)
            if cfg["gradient"] in ("exact", "binomial"):
                values = []

                def data_gradient(d: torch.Tensor) -> torch.Tensor:
                    v, gd = misfit(d, obs)
                    values.append(v)
                    return gd / n_shots

                if cfg["gradient"] == "binomial":
                    data = binomial_gradient(eps, sigma, options(s), cfg["exact_segment_steps"], data_gradient,
                                             cfg["binomial_checkpoints"])
                else:
                    data = exact_gradient(eps, sigma, options(s), cfg["exact_segment_steps"], data_gradient, cfg["exact_outer_steps"], cfg["exact_offload"], cfg["exact_pin"])
                value = values[0]
            else:
                data = DeepGPR.compute(eps_r=eps, sigma=sigma, wavefield_sketch=shot_sketch(s), **options(s))[-1]
                value, grad = misfit(data, obs)
                data.backward(grad / n_shots)
                del grad
                if block_histories:  # compressed E history of this shot (host + spill file)
                    h = block_histories.pop()
                    rep = h.size_report()
                    history_sizes.append((round(rep.stored_bytes / GIB, 3), round(h.spilled_bytes / GIB, 3)))
                    raw_history_bytes = rep.raw_bytes
                    block_histories.clear()
                    del h, rep
            l2_local += float(0.5 * (data.detach() - obs).square().sum())
            energy_local += float(obs.square().sum())
            loss_local += value / n_shots
            del data, obs
            torch.cuda.synchronize()
            shot_seconds.append(time.perf_counter() - t_shot)
            log(f"rank {rank} it {iteration} shot {s}: {shot_seconds[-1]:.0f} s, GPU peak so far "
                f"{torch.cuda.max_memory_allocated() / GIB:.2f} GiB"
                + (f", history {history_sizes[-1][0]:.2f} GiB (spilled {history_sizes[-1][1]:.2f})" if history_sizes else ""))
        t_fdtd = time.perf_counter() - t0
        with torch.no_grad():
            g = eps.grad
            g[:, :air] = 0
            g_host = g.detach().cpu()
            dist.all_reduce(g_host, op=dist.ReduceOp.SUM)
            g.copy_(g_host.to(device))
            scalars = allreduce_host(torch.tensor([loss_local, l2_local, energy_local], dtype=torch.float64))
            if rank == 0:  # data gradient against the model error, before TV / smoothing / Adam
                previous = eps.detach()[ground].cpu()
                grad_corr = pearson(-g_host[ground], truth_ground - previous)
                if iteration == 0 or (iteration + 1) % cfg["gradient_every"] == 0:
                    np.save(out / "gradients" / f"g_it{iteration:04d}.npy", g_host[ground].numpy())
            del g_host
            g_tv = tv_gradient(eps.detach()[ground])
            g_data_norm = float(g[ground].norm())
            tv_weight = cfg["tv_ratio"] * (g_data_norm + 1e-8) / (float(g_tv.norm()) + 1e-8)
            g[ground] += tv_weight * g_tv
            del g_tv
            t_smooth = time.perf_counter()
            if cfg["grad_smooth_cells"] > 0:  # data + TV gradient, so Adam only sees the smooth part
                g[ground] = smooth_gradient(g[ground], cfg["grad_smooth_cells"])
            smooth_s = time.perf_counter() - t_smooth
            del g  # else this name keeps the old gradient on the GPU through the next iteration's shots
        factor = cfg["lr_final_factor"] + (1 - cfg["lr_final_factor"]) * 0.5 * (1 + math.cos(math.pi * iteration / max(1, iterations - 1)))
        for group in optimizer.param_groups:
            group["lr"] = base_lr * factor
        optimizer.step()
        with torch.no_grad():
            eps.clamp_(*cfg["eps_bounds"])
            eps[:, :air] = 1.0
            host = eps.detach().cpu()
            dist.broadcast(host, src=0)
            eps.copy_(host.to(device))
        peaks = [None] * world  # per rank: (allocated GiB, reserved GiB, seconds per shot)
        dist.all_gather_object(peaks, (round(torch.cuda.max_memory_allocated() / GIB, 2), round(torch.cuda.max_memory_reserved() / GIB, 2), [round(x, 1) for x in shot_seconds]))
        record = dict(iteration=iteration, loss=float(scalars[0]), l2=float(scalars[1] / scalars[2]), lr=base_lr * factor,
                      tv_weight=tv_weight, smooth_s=smooth_s, fdtd_s=t_fdtd, wall_s=time.perf_counter() - t0, peaks=peaks,
                      host_maxrss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2))
        if history_sizes:  # this rank's shots only (all of them with one process)
            record.update(history_gib=history_sizes, raw_history_gib=round(raw_history_bytes / GIB, 1))
        if rank == 0:
            record.update(model_metrics(host))
            record.update(grad_corr=grad_corr, update_corr=pearson(host[ground] - previous, truth_ground - previous))
            del previous
            history.append(record)
            if (iteration + 1) % cfg["slices_every"] == 0 or iteration == iterations - 1 or iteration == 0:
                save_slices(host, iteration)
            if (iteration + 1) % cfg["snapshot_every"] == 0 or iteration == iterations - 1:
                np.save(out / "snapshots" / f"eps_it{iteration + 1:04d}.npy", host[ground].numpy())
            tmp = ckpt.with_suffix(".tmp")
            torch.save(dict(eps=host, optimizer=optimizer.state_dict(), history=history, config=cfg), tmp)
            os.replace(tmp, ckpt)
            (out / "history.json").write_text(json.dumps(history, indent=1))
            log(f"it {iteration:3d} loss {record['loss']:.4f} l2 {record['l2']:.4f} rmse {record['rmse']:.4f} "
                f"(upper {record['rmse_upper']:.4f}, lower {record['rmse_lower']:.4f}, corr {record['corr']:.4f}) "
                f"gcorr {grad_corr:.4f} ucorr {record['update_corr']:.4f} "
                f"lr {record['lr']:.4f} tv {tv_weight:.3g} smooth {smooth_s:.0f} s {record['wall_s']:.0f} s, peaks {peaks}, "
                f"host max RSS {record['host_maxrss_gib']} GiB"
                + (f", history {np.mean([h[0] for h in history_sizes]):.1f} GiB/shot "
                   f"({record['raw_history_gib'] / np.mean([h[0] for h in history_sizes]):.0f}x of {record['raw_history_gib']:.0f} GiB), "
                   f"spilled {sum(h[1] for h in history_sizes):.1f} GiB" if history_sizes else ""))
        else:
            history.append(record)
        del host
    dist.barrier()
    if rank == 0:
        log("DONE")
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--profile", choices=tuple(PROFILES), default="full",
                        help="full: 2 cm / 500 MHz (138.9 M cells); half: same model on 4 cm / 250 MHz; quick: 8 cm test")
    parser.add_argument("--quick", action="store_true", help="same as --profile quick")
    parser.add_argument("--gradient", choices=("tucker", "exact", "binomial", "block"), default=None,
                        help="Tucker sketch (default), exact multi-level checkpointing (host memory), exact binomial "
                             "checkpointing (all on the GPU), or block-wise rank-adaptive Tucker")
    parser.add_argument("--checkpoints", type=int, default=None,
                        help="binomial gradient: checkpoint states on the GPU (default 7; segments of --segment-steps)")
    parser.add_argument("--block-tol", type=float, default=None, help="block Tucker: relative error tolerance per block (default 0.03)")
    parser.add_argument("--host-limit-gib", type=float, default=None,
                        help="block Tucker: host memory for the compressed history; the rest goes to a file in --spill-dir")
    parser.add_argument("--spill-dir", default=None, help="block Tucker: folder (local disk) of the spill file")
    parser.add_argument("--history-on-gpu", action="store_true",
                        help="block Tucker: keep the compressed history on the GPU (512 MiB device buffers), no host memory")
    parser.add_argument("--payload-dtype", choices=("float32", "float16", "bfloat16"), default=None,
                        help="block Tucker: storage precision of the compressed history (default float32)")
    parser.add_argument("--misfit", choices=("mssim", "l2"), default=None, help="default mssim")
    parser.add_argument("--iterations", type=int, default=None, help="override the profile's iteration count (sets the cosine schedule)")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="stop after this many iterations in total, keeping the cosine schedule of --iterations")
    parser.add_argument("--segment-steps", type=int, default=None, help="finest segment length of the exact gradient (recomputed with history)")
    parser.add_argument("--outer-steps", default=None, help="comma-separated outer checkpoint levels of the exact gradient, coarse to fine "
                        "(e.g. 160 or 480,64); each costs one more history-free forward pass and cuts host memory")
    parser.add_argument("--offload", action="store_true", help="exact gradient: keep the segment E history in pinned host memory")
    parser.add_argument("--pin", action="store_true", help="exact gradient: keep the checkpoint states in pinned host memory")
    parser.add_argument("--rank", type=int, default=None, help="Tucker rank (default 16)")
    parser.add_argument("--self-test", action="store_true", help="check the chunked MS-SSIM and the TV gradient first")
    parser.add_argument("--shot-grid", type=int, default=None, help="N x N regular shot grid instead of the 3-4-3 rows")
    parser.add_argument("--run-dir", default=None, help="output folder name next to this script (default run / quick)")
    parser.add_argument("--lr", type=float, default=None, help="initial Adam learning rate (default 0.03, cosine decay to 0.1 x)")
    parser.add_argument("--grad-smooth", type=float, default=None, help="Gaussian sigma (cells) applied to the gradient before Adam (default 0 = off)")
    args = parser.parse_args()
    profile = "quick" if args.quick else args.profile
    if args.command == "prepare":
        prepare(profile, args.run_dir)
    else:
        run(profile, args.self_test, args.shot_grid, args.run_dir, args.lr, args.grad_smooth, args.gradient, args.misfit,
            args.iterations, args.segment_steps, args.rank,
            None if args.outer_steps is None else tuple(int(v) for v in args.outer_steps.split(",") if v), args.offload, args.pin,
            args.block_tol, args.host_limit_gib, args.spill_dir, args.payload_dtype, args.stop_after, args.history_on_gpu,
            args.checkpoints)


if __name__ == "__main__":
    main()
