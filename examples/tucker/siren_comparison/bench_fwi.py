"""Experiment C: SIREN vs Tucker-SIREN (vs dense, direct Tucker) inside DeepGPR FWI.

Cases (identical physics, data and loss for every method):

* ``3d``: the 120 x 120 x 100 OverThrust-derived model of
  ``examples/tucker/benchmark_tucker.py`` (dx 2 cm, dt 30 ps, 500 MHz,
  nt 600, 4 shots, 10-cell CPML, mode 3), uncompressed default history.
* ``2d``: the 2D FWI notebook at its own size (``2DFWImodel.npy``,
  200 x 100, 50 shots, nt 2000).

Wavefield Tucker sketching is OFF in every run (only the model
parameterization differs). Loss ``0.5 sum ||d - d_obs||^2 / sum ||d_obs||^2``
(protocol v3 adds the TV term of the FWI notebooks, see ``PROTOCOLS``).

Per run: pre-fit to the common initial model (time reported separately), Adam
with the learning rate chosen by a short search (cost reported), per-iteration
time split into model generation / FDTD forward / adjoint / network backward /
optimizer step, model errors (RMSE, SSIM for eps and log10 sigma), peak GPU
memory, checkpoints every 10 iterations (runs resume).

Usage::

    python bench_fwi.py search --case 3d          # learning-rate search
    python bench_fwi.py run --case 3d [--only ...]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parent))

import DeepGPR  # noqa: E402
from metrics import model_errors  # noqa: E402
from networks import (  # noqa: E402
    DenseField,
    DirectTuckerField,
    FixedTopField,
    PropertyMapping,
    gaussian_smooth,
    SirenField,
    TuckerSirenField,
    prefit_field,
    prefit_tucker_siren,
    split_parameters,
)
from runner import RESULTS, PhaseTimer, load_results, optimizer_state_bytes, parameter_bytes, run_specs, worker_main  # noqa: E402

LR_CANDIDATES = {
    "dense": (0.003, 0.01, 0.03, 0.1),
    "siren": (3e-6, 1e-5, 3e-5, 1e-4),
    "tucker_siren": (3e-5, 1e-4, 3e-4, 1e-3),
    "direct_tucker": (3e-4, 1e-3, 3e-3, 1e-2),
}
SEARCH_ITERATIONS = 15

# Protocols. v1: the first round (box-filtered start, constant 3D sigma start,
# sigma mapping from 0 S/m, air inverted, joint eps/sigma from iteration 1).
# v2 (after review): start = Gaussian-smoothed true model (sigma 10 cells in
# 3D, 20 in 2D, as in examples/{2.2DFWI,3.3DFWI}.ipynb and the SIREN-IFWI
# notebooks), air fixed at its true values and excluded from every
# parameterization, sigma mapping range from the subsurface (0.5 min .. 1.5 max),
# eps only for the first third of the iterations, then eps and sigma.
# v3 = v2 plus (a) the TV regularisation of examples/{2.2DFWI,3.3DFWI}.ipynb
# for every method: anisotropic TV (DeepGPR.TVRegularization) of eps and of
# sigma over the parameterized region, its gradient rescaled to TV_RATIO times
# the norm of the data gradient and added to it in model space (then
# back-propagated through the parameterization), and (b) in 3D 16 shots on a
# 4 x 4 grid instead of 4 shots on a diagonal.
# v4 = v3 with a hyper-parameter selection that does not use the truth and
# tests the sigma step: held-out validation shots (VALIDATION_SHOTS, not used
# by the search's gradients); stage A chooses the learning rate by the
# validation misfit after SEARCH_ITERATIONS eps-only iterations; stage B
# continues the chosen stage-A state for SEARCH_ITERATIONS joint iterations
# for each sigma step scale in SIGMA_SCALES and chooses the scale by the
# validation misfit. Sigma step scale: sigma learning rate = scale x lr for
# methods with separate eps and sigma parameters; sigma data+TV gradient x
# scale for the SIREN (one network for both properties). The FWI runs use all
# shots. 3D metrics add the zone sigma error and the eps error below the zone.
# v5 = v4 on the larger 3D model (``build_case_large``; 36 shots on a 6 x 6
# grid, 400 receivers); stage B may be skipped with ``run --sigma-scale``.
# v6 = v5 with the dense tuning of round 5: MS-SSIM misfit (``~/MSSIM``), 500
# iterations, cosine learning-rate decay (lr ... lr/10), sigma updated from
# iteration 250; learning rates re-searched per method on held-out shots with
# the MS-SSIM misfit (15 eps-only iterations, constant rate).
PROTOCOLS = ("v1", "v2", "v3", "v4", "v5", "v6")
V4_LIKE = ("v4", "v5", "v6")
LARGE_MODEL = ("v5", "v6")
SIGMA_START = {"v6": 250}
ITERATIONS = {"v1": {"3d": 100, "2d": 150}, "v2": {"3d": 150, "2d": 200}, "v3": {"3d": 150, "2d": 200}, "v4": {"3d": 150, "2d": 200}, "v5": {"3d": 150, "2d": 200}, "v6": {"3d": 500, "2d": 500}}
TV_RATIO = {"2d": (0.1, 0.1), "3d": (0.1, 0.08)}  # (eps, sigma), as in the notebooks
SHOT_GRID_3D = 4
SIGMA_SCALES = (0.0, 0.03, 0.1, 0.3, 1.0)
# v4 reference (``--select oracle``): the same searches, but each method gets
# the candidates with the lowest model error after the search (zone eps RMSE
# for the learning rate, zone log10 sigma RMSE for the sigma scale). It uses
# the true model and is an upper bound of what each parameterization reaches,
# not a usable selection rule.
VALIDATION_SHOTS = {"3d": (1, 7, 8, 14), "2d": tuple(range(2, 50, 5)), "3d_v5": (1, 9, 17, 18, 26, 34)}  # 3D: one per row and column
VALIDATION_SHOTS_QUICK = {"3d": (1,), "2d": (2, 7), "3d_v5": (1, 5, 6)}


def result_dir(quick: bool, protocol: str = "v1") -> Path:
    base = "quick" if quick else "full"
    return RESULTS / (base if protocol == "v1" else f"{base}_{protocol}")


def initial_models(case_name: str, case: Dict[str, Any], protocol: str):
    if protocol == "v1":
        return case["eps_init"], case["sigma_init"]
    width = case.get("smooth_width", 10.0 if case_name == "3d" else 20.0)
    eps0 = gaussian_smooth(case["eps"], width)
    sigma0 = gaussian_smooth(case["sigma"], width)
    air = case.get("air", 0)
    if case_name == "3d" and air:
        eps0[:, :air] = case["eps"][:, :air]
        sigma0[:, :air] = case["sigma"][:, :air]
    return eps0, sigma0


def make_mapping(case: Dict[str, Any], region, protocol: str) -> PropertyMapping:
    if protocol == "v1":
        return PropertyMapping(eps_min=1.0, eps_max=10.0, sigma_min=0.0, sigma_max=float(case["sigma"].max()) * 1.25)
    sub = case["sigma"][region]
    return PropertyMapping(eps_min=1.0, eps_max=10.0, sigma_min=0.5 * float(sub.min()), sigma_max=1.5 * float(sub.max()))


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------


# Round 5 (protocol v5): a larger and deeper 3D model. 4 m x 4 m surface, 2 m
# subsurface plus 0.2 m air, the same OverThrust recipe as the 3D model of
# rounds 1-4 (section in the (depth, z) plane, smooth lateral dip along x,
# sigma = 5e-4 eps). To reach 2 m at an affordable cost the frequency is halved
# and dx doubled (same cells per wavelength as 500 MHz / 2 cm): 250 MHz, dx 4 cm,
# dt 60 ps (same Courant number), a record covering the two-way time to 2 m.
LARGE_3D = dict(surface_m=4.0, depth_m=2.0, air_m=0.2, dx=0.04, dt=6.0e-11, freq=2.5e8, shot_grid=6, receiver_step=5)


def build_case_large(quick: bool, device) -> Dict[str, Any]:
    """The round-5 3D case (see ``LARGE_3D``); ``quick``: dx 8 cm, 125 MHz, 3 x 3 shots."""
    import numpy as np
    import torch.nn.functional as F

    import benchmark_tucker as bt

    cfg = dict(LARGE_3D)
    if quick:
        cfg.update(dx=0.08, dt=1.2e-10, freq=1.25e8, shot_grid=3)
    dx = cfg["dx"]
    nx = nz = int(round(cfg["surface_m"] / dx))
    depth, air = int(round(cfg["depth_m"] / dx)), int(round(cfg["air_m"] / dx))
    section = torch.tensor(np.load(bt.ROOT / "examples" / "OverThrust.npy"), dtype=torch.float32)
    section = F.interpolate(section[None, None], size=(depth, nz), mode="bilinear", align_corners=True)[0, 0]
    rows = torch.arange(depth, dtype=torch.float32)
    eps = torch.empty(nx, depth, nz)
    for ix in range(nx):
        shift = 0.08 * depth * math.sin(math.pi * (ix / (nx - 1) - 0.5))
        source_rows = (rows - shift).clamp(0, depth - 1)
        low = source_rows.floor().long()
        high = (low + 1).clamp(max=depth - 1)
        weight = (source_rows - low.float())[:, None]
        eps[ix] = section[low] * (1 - weight) + section[high] * weight
    eps = torch.cat([torch.ones(nx, air, nz), eps], dim=1)
    sigma = 5e-4 * eps
    sigma[:, :air] = 0.0
    # Record: two-way vertical time to the bottom (slowest column), plus the
    # wavelet delay, plus 25 % for oblique paths to the receivers.
    c = 0.299792458  # m/ns
    two_way = float((2 * torch.sqrt(eps[:, air:, :]) * dx / c).sum(dim=1).max())
    delay = 1.2 / cfg["freq"]
    nt = int(math.ceil((1.25 * two_way * 1e-9 + delay) / cfg["dt"]))
    wavelet = DeepGPR.wavelet.ricker(cfg["freq"], nt, cfg["dt"], delay).reshape(1, nt, 1)
    n = cfg["shot_grid"]
    gx = torch.linspace(nx * 0.1, nx * 0.9, n).round().int()
    gz = torch.linspace(nz * 0.1, nz * 0.9, n).round().int()
    grid = torch.stack(torch.meshgrid(gx, gz, indexing="ij"), dim=-1).reshape(-1, 2)
    source = torch.zeros((n * n, 1, 3), dtype=torch.int32)
    source[:, 0, 0], source[:, 0, 1], source[:, 0, 2] = grid[:, 0], air, grid[:, 1]
    rx = torch.arange(2, nx - 2, cfg["receiver_step"])
    rz = torch.arange(2, nz - 2, cfg["receiver_step"])
    rgrid = torch.stack(torch.meshgrid(rx, rz, indexing="ij"), dim=-1).reshape(-1, 2)
    receivers = torch.zeros((n * n, rgrid.shape[0], 3), dtype=torch.int32)
    receivers[:, :, 0], receivers[:, :, 1], receivers[:, :, 2] = rgrid[:, 0], air, rgrid[:, 1]
    kwargs = dict(device=device, dx=dx, dt=cfg["dt"], source_amplitudes=wavelet.to(device), pmlthick=10, mode=3, source_direction=2, receiver_component=2)
    return dict(
        eps=eps.to(device), sigma=sigma.to(device), eps_init=None, sigma_init=None, air=air, nt=nt, shots=n * n,
        source_location=source.to(device), receiver_location=receivers.to(device), kwargs=kwargs, name="3d4m",
        smooth_width=5.0, zone_depth=depth // 2, two_way_ns=two_way, config=cfg,
    )


def grid_shots_3d(case: Dict[str, Any], quick: bool, device) -> List[torch.Tensor]:
    """v3: sources on a regular grid over the central 60 % of the surface."""
    import benchmark_tucker as bt

    n = SHOT_GRID_3D if not quick else 2
    nx, _, nz = case["eps"].shape
    gx = torch.linspace(nx * 0.2, nx * 0.8, n).round().int()
    gz = torch.linspace(nz * 0.2, nz * 0.8, n).round().int()
    grid = torch.stack(torch.meshgrid(gx, gz, indexing="ij"), dim=-1).reshape(-1, 2)
    source = torch.zeros((n * n, 1, 3), dtype=torch.int32)
    source[:, 0, 0], source[:, 0, 1], source[:, 0, 2] = grid[:, 0], case["air"], grid[:, 1]
    receivers = case["receiver_location"][:1].cpu().repeat(n * n, 1, 1)
    case["source_location"], case["receiver_location"], case["shots"] = source.to(device), receivers.to(device), n * n
    cache = RESULTS / ("quick" if quick else "full") / "cache" / f"observed3d_grid{n}x{n}_{case['nt']}.pt"
    if cache.exists():
        return [t.to(device) for t in torch.load(cache)]
    with torch.no_grad():
        data = [DeepGPR.compute(eps_r=case["eps"], sigma=case["sigma"], save_wavefield_history=False, **bt.shot_kwargs(case, s))[-1].clone() for s in range(n * n)]
    bt.atomic_save([t.cpu() for t in data], cache)
    return data


def split_shots(case_name: str, shots, observed, quick: bool, protocol: str = "v4"):
    """v4 search: (training shots, training data, validation (shot, data) pairs)."""
    key = f"{case_name}_v5" if (protocol in LARGE_MODEL and case_name == "3d") else case_name
    held = set((VALIDATION_SHOTS_QUICK if quick else VALIDATION_SHOTS)[key])
    if case_name == "3d":
        train = [i for i in range(len(shots)) if i not in held]
        return [shots[i] for i in train], [observed[i] for i in train], [(shots[i], observed[i]) for i in sorted(held)]
    options, data = shots[0], observed[0]  # 2D: all shots in one batched call
    n = data.shape[0]
    train = torch.tensor([i for i in range(n) if i not in held], device=data.device)
    val = torch.tensor(sorted(held), device=data.device)

    def subset(index):
        return dict(options, source_location=options["source_location"][index], receiver_location=options["receiver_location"][index])

    return [subset(train)], [data[train]], [(subset(val), data[val])]


def load_case(case_name: str, quick: bool, device, protocol: str = "v1"):
    import benchmark_tucker as bt

    if case_name == "3d" and protocol in LARGE_MODEL:
        case = build_case_large(quick, device)
        cache = RESULTS / ("quick" if quick else "full") / "cache" / f"observed3d4m_{case['shots']}_{case['nt']}_{case['config']['dx']:.3f}.pt"
        if cache.exists():
            observed = [t.to(device) for t in torch.load(cache)]
        else:
            with torch.no_grad():
                observed = [DeepGPR.compute(eps_r=case["eps"], sigma=case["sigma"], save_wavefield_history=False, **bt.shot_kwargs(case, s))[-1].clone() for s in range(case["shots"])]
            bt.atomic_save([t.cpu() for t in observed], cache)
        shots = [bt.shot_kwargs(case, s) for s in range(case["shots"])]
        air = case["air"]
        region = (slice(None), slice(air, None), slice(None))
        zone = (slice(4, -4), slice(air, air + case["zone_depth"]), slice(4, -4))  # top 1 m; "deep" = 1-2 m
    elif case_name == "3d":
        case = bt.build_case(quick, device)
        observed = grid_shots_3d(case, quick, device) if protocol in ("v3", "v4") else bt.observed_data(case, quick)
        shots = [bt.shot_kwargs(case, s) for s in range(case["shots"])]
        air = case["air"]
        region = (slice(None), slice(air, None), slice(None))
        depth = 40 if not quick else 12
        zone = (slice(4, -4), slice(air, air + depth), slice(4, -4))
    else:
        case = bt.build_case_2d(device, quick)
        cache = RESULTS / ("quick" if quick else "full") / "cache" / f"observed2d_{case['nt']}.pt"  # data: same for every protocol
        options = dict(case["kwargs"], source_location=case["source_location"], receiver_location=case["receiver_location"])
        if cache.exists():
            data = torch.load(cache).to(device)
        else:
            with torch.no_grad():
                data = DeepGPR.compute(eps_r=case["eps"], sigma=case["sigma"], save_wavefield_history=False, **options)[-1]
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data.cpu(), cache)
        observed, shots = [data], [options]
        region = (slice(None), slice(None))
        zone = (slice(None), slice(None))
    return case, observed, shots, region, zone


def build_field(spec, case, mapping, device):
    """Pre-fit (or load the cached pre-fit of) one parameterization.

    The pre-fit depends only on (case, method, seed); it is cached so the
    learning-rate search and the main run start from the same state, and its
    originally measured time is reported with every run that uses it.
    """
    quick = bool(spec.get("quick"))
    label = spec["name"].split("|")[2] if "|search" in spec["name"] else spec["name"].split("|")[1]
    cache = result_dir(quick, spec.get("protocol", "v1")) / "prefit" / f"{spec['case']}|{label}|seed{spec.get('seed', 0)}.pt"
    field, info = _build_field(spec, case, mapping, device)
    if cache.exists():
        state = torch.load(cache, map_location=device)
        field.load_state_dict(state["field"])
        info = dict(state["info"], prefit_cached=True)
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(field=field.state_dict(), info=info), cache)
    return field, info


def _build_field(spec, case, mapping, device):
    protocol = spec.get("protocol", "v1")
    eps_init, sigma_init = case["eps_start"], case["sigma_start"]
    air = case.get("air", 0) if (protocol != "v1" and spec["case"] == "3d") else 0
    # v2, v3: the air layer is known and fixed; only the subsurface is parameterized.
    eps_sub, sigma_sub = eps_init[:, air:], sigma_init[:, air:]
    shape = tuple(eps_sub.shape)
    t_eps0 = mapping.eps_inverse(eps_sub)
    t_sig0 = mapping.sigma_inverse(sigma_sub)
    torch.manual_seed(spec.get("seed", 0))
    started = time.perf_counter()
    info: Dict[str, Any] = {}
    method = spec["method"]
    if method == "dense":
        field = DenseField(t_eps0, t_sig0, mapping)
    elif method == "direct_tucker":
        field = DirectTuckerField(t_eps0, t_sig0, mapping, spec["rank"])
    elif method == "siren":
        field = SirenField(shape, mapping, spec.get("hidden", 128), spec.get("layers", 4), device=device).to(device)
        info["prefit_history"] = prefit_field(field, t_eps0, t_sig0, steps=spec.get("prefit_steps", 3000), lr=1e-3)
    else:
        field = TuckerSirenField(shape, mapping, spec["rank"], spec.get("hidden", 32), spec.get("layers", 2), device=device).to(device)
        info["prefit_eps"] = prefit_tucker_siren(field.eps_field, t_eps0, steps=spec.get("prefit_steps", 1500))
        info["prefit_sigma"] = prefit_tucker_siren(field.sigma_field, t_sig0, steps=spec.get("prefit_steps", 1500))
    field = field.to(device)
    if air:
        field = FixedTopField(field, eps_init[:, :air], sigma_init[:, :air]).to(device)
    torch.cuda.synchronize()
    info["prefit_seconds"] = time.perf_counter() - started
    with torch.no_grad():
        eps0, sig0 = field()
        info["prefit_eps_vs_init"] = model_errors(eps0, eps_init)
        info["prefit_sigma_vs_init"] = model_errors(sig0, sigma_init, log=True)
    return field, info


# ---------------------------------------------------------------------------
# FWI loop
# ---------------------------------------------------------------------------


class DataObjective:
    """Data term of the FWI loss (default: the normalised L2 of rounds 1-5).

    ``spec["time_gain"] = p``: residuals weighted by ``(t / t_end)^p`` (both
    data multiplied by the gain). ``spec["loss"] = "mssim"``: the MS-SSIM misfit
    of ``~/MSSIM`` (``Misfit_M_SSIM``, 2D mode: each trace is scaled to its
    maximum over time, Gaussian windows over (time, receiver)); 3D shots with
    a receiver grid are compared line by line (sections along z, one batch entry
    per receiver line along x); the value is averaged over shots (0 = perfect).
    """

    def __init__(self, spec, observed, shots, scale, device):
        import os

        self.kind = spec.get("loss", "l2")
        self.gain_power = float(spec.get("time_gain", 0.0))
        self.scale, self.shots = scale, len(observed)
        self.gain = None
        if self.gain_power:
            nt = observed[0].shape[1]
            t = torch.arange(1, nt + 1, device=device, dtype=observed[0].dtype)
            self.gain = ((t / nt) ** self.gain_power).reshape(1, nt, 1)
            self.scale = sum(float((d * self.gain).square().sum()) for d in observed)
        if self.kind == "mssim":
            for path in (os.environ.get("MSSIM_PARENT"), os.path.expanduser("~"), os.path.expanduser("~/Desktop")):
                if path and os.path.isdir(os.path.join(path, "MSSIM")):
                    sys.path.insert(0, path)
                    break
            from MSSIM import Misfit_M_SSIM

            self.mssim = Misfit_M_SSIM()

    def _lines(self, data, options):
        """(1, T, Rx*Rz) on a receiver grid (z fastest) -> (Rx, T, Rz); 2D batches unchanged."""
        rec = options["receiver_location"][0]
        nx_r, nz_r = int(torch.unique(rec[:, 0]).numel()), int(torch.unique(rec[:, 2]).numel())
        if data.shape[0] != 1 or nx_r * nz_r != data.shape[2] or nx_r == 1 or nz_r == 1:
            return data
        assert int(rec[1, 2]) > int(rec[0, 2]) and int(rec[1, 0]) == int(rec[0, 0]), "receivers must be z-fastest"
        return data[0].reshape(data.shape[1], nx_r, nz_r).permute(1, 0, 2)

    def __call__(self, data, data_obs, options):
        if self.gain is not None:
            data, data_obs = data * self.gain, data_obs * self.gain
        if self.kind == "mssim":
            return self.mssim.forward(self._lines(data_obs, options), self._lines(data, options)) / self.shots
        return 0.5 * (data - data_obs).square().sum() / self.scale


def fwi(spec: Dict[str, Any]) -> Dict[str, Any]:
    import os

    import benchmark_tucker as bt

    device = torch.device("cuda")
    # Exact temporal checkpointing of every shot (benchmark_tucker.checkpointed_shot;
    # gradient cosine 1 - 1e-14 against the stored history, Section 9 of
    # docs/tucker_model_wavefield_compression.md): the same results with ~1/6 of
    # the memory on shared GPUs, at ~1.6x the FDTD time. Off unless set.
    checkpoint_steps = int(os.environ.get("FWI_CHECKPOINT_STEPS", "0"))
    torch.backends.cuda.matmul.allow_tf32 = False
    quick = bool(spec.get("quick"))
    protocol = spec.get("protocol", "v1")
    case, observed, shots, region, zone = load_case(spec["case"], quick, device, protocol)
    case["eps_start"], case["sigma_start"] = initial_models(spec["case"], case, protocol)
    mapping = make_mapping(case, region, protocol)
    validation: List = []
    if spec.get("train_only"):
        shots, observed, validation = split_shots(spec["case"], shots, observed, quick, protocol)
    scale = sum(float(d.square().sum()) for d in observed)
    objective = DataObjective(spec, observed, shots, scale, device)
    field, info = build_field(spec, case, mapping, device)
    sigma_scale = float(spec.get("sigma_scale", 1.0))
    eps_params, sigma_params, shared_params = split_parameters(field)
    if protocol in V4_LIKE:
        groups = [dict(params=eps_params + shared_params, lr=spec["lr"])]
        if sigma_params:
            groups.append(dict(params=sigma_params, lr=spec["lr"] * sigma_scale))
        optimizer = torch.optim.Adam(groups)
    else:
        optimizer = torch.optim.Adam(field.parameters(), lr=spec["lr"])
    sigma_grad_scale = sigma_scale if (protocol in V4_LIKE and shared_params) else 1.0
    # Optional module-B wavefield compression (``spec["wavefield_sketch"]``: TuckerSketchConfig options).
    sketch = bt.sketch_config({"sketch": spec["wavefield_sketch"]}) if spec.get("wavefield_sketch") else None
    base_rates = [group["lr"] for group in optimizer.param_groups]
    if spec.get("init_state"):
        # v4 stage B: continue the chosen stage-A state; keep this spec's rates.
        state = torch.load(spec["init_state"], map_location=device)
        field.load_state_dict(state["field"])
        optimizer.load_state_dict(state["optimizer"])
        for group, rate in zip(optimizer.param_groups, (spec["lr"], spec["lr"] * sigma_scale)):
            group["lr"] = rate
    out_dir = result_dir(quick, protocol) / "fwi_runs" / spec["case"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "initial.pt").exists():
        torch.save({"eps": case["eps_start"].cpu(), "sigma": case["sigma_start"].cpu()}, out_dir / "initial.pt")
    # v2-v4: eps only for the first third of the iterations (examples/2.2DFWI.ipynb).
    eps_only_iterations = spec["iterations"] // 3 if protocol != "v1" else 0
    if protocol != "v1" and spec.get("lr_search"):
        eps_only_iterations = ITERATIONS[protocol][spec["case"]] // 3
    eps_only_iterations = spec.get("eps_only_iterations", eps_only_iterations)
    tv = DeepGPR.TVRegularization(weight_ep=1.0, weight_sigma=1.0).to(device) if protocol in ("v3", "v4", "v5", "v6") else None
    tv_ratio = TV_RATIO[spec["case"]]

    def add_tv(leaf, ratio, conductivity):
        # Notebook rule: grad = g_data + ratio |g_data| / |g_TV| g_TV, over the
        # parameterized region (3D: below the fixed air layer).
        values = leaf.detach()[region].clone().requires_grad_(True)
        penalty = tv(ep=None, sigma=values) if conductivity else tv(ep=values, sigma=None)
        (g_tv,) = torch.autograd.grad(penalty, values)
        g_data = leaf.grad[region]
        weight = ratio * (g_data.norm() + 1e-8) / (g_tv.norm() + 1e-8)
        leaf.grad[region] += weight * g_tv
        return float(weight)
    ckpt_path = out_dir / f"{spec['name']}.ckpt"
    history: List[Dict[str, Any]] = []
    start_iteration = 0
    if ckpt_path.exists() and not spec.get("lr_search"):
        state = torch.load(ckpt_path, map_location=device)
        field.load_state_dict(state["field"])
        optimizer.load_state_dict(state["optimizer"])
        history, info = state["history"], state["info"]
        start_iteration = len(history)

    def evaluate():
        with torch.no_grad():
            eps, sigma = field()
        errors = dict(
            eps=model_errors(eps, case["eps"], region),
            eps_zone=model_errors(eps, case["eps"], zone),
            sigma_log=model_errors(sigma, case["sigma"], region, log=True),
        )
        if protocol in V4_LIKE:
            errors["sigma_log_zone"] = model_errors(sigma, case["sigma"], zone, log=True)
            if spec["case"] == "3d":
                deep = (slice(None), slice(zone[1].stop, None), slice(None))
                errors["eps_deep"] = model_errors(eps, case["eps"], deep)
        return errors

    def misfit(pairs) -> float:
        # The training objective's data term on these shots (L2: normalised by their energy).
        obj = DataObjective(spec, [d for _, d in pairs], [o for o, _ in pairs], sum(float(d.square().sum()) for _, d in pairs), device)
        total = 0.0
        with torch.no_grad():
            eps, sigma = field()
            for options, data_obs in pairs:
                data = DeepGPR.compute(eps_r=eps, sigma=sigma, save_wavefield_history=False, **options)[-1]
                total += float(obj(data, data_obs, options))
        return total

    if start_iteration == 0:
        info["initial_errors"] = evaluate()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    iterations = spec["iterations"]
    for iteration in range(start_iteration, iterations):
        timer = PhaseTimer()
        torch.cuda.synchronize()
        wall0 = time.perf_counter()
        timer.mark("start")
        optimizer.zero_grad(set_to_none=True)
        eps, sigma = field()
        timer.mark("generation")
        eps_leaf = eps.detach().requires_grad_(True)
        sig_leaf = sigma.detach().requires_grad_(True)
        fdtd_forward = adjoint = 0.0
        loss_total = l2_total = 0.0
        for options, data_obs in zip(shots, observed):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            c = torch.cuda.Event(enable_timing=True)
            a.record()
            if checkpoint_steps:
                data = bt.checkpointed_shot(eps_leaf, sig_leaf, options, checkpoint_steps, None)
            elif sketch is not None:  # module B: streaming Tucker compression of the stored history
                data = DeepGPR.compute(eps_r=eps_leaf, sigma=sig_leaf, wavefield_sketch=sketch, **options)[-1]
            else:
                data = DeepGPR.compute(eps_r=eps_leaf, sigma=sig_leaf, **options)[-1]
            loss = objective(data, data_obs, options)
            l2_total += float(0.5 * (data.detach() - data_obs).square().sum() / scale)
            b.record()
            loss.backward()
            c.record()
            torch.cuda.synchronize()
            fdtd_forward += a.elapsed_time(b) / 1e3
            adjoint += b.elapsed_time(c) / 1e3
            loss_total += float(loss)
        timer.mark("fdtd")
        tv_weights = {}
        if tv is not None:
            tv_weights["eps"] = add_tv(eps_leaf, tv_ratio[0], False)
            if iteration >= eps_only_iterations:
                tv_weights["sigma"] = add_tv(sig_leaf, tv_ratio[1], True)
        timer.mark("regularization")
        if iteration < eps_only_iterations:
            # eps only: sigma parameters receive no gradient (separate fields)
            # or only the eps-driven part (the shared SIREN network).
            torch.autograd.backward([eps], [eps_leaf.grad])
        else:
            torch.autograd.backward([eps, sigma], [eps_leaf.grad, sig_leaf.grad * sigma_grad_scale])
        timer.mark("network_backward")
        if spec.get("schedule") == "cosine":  # lr * (0.1 ... 1), keeping the sigma / eps ratio
            factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * iteration / max(1, iterations - 1)))
            for group, base in zip(optimizer.param_groups, base_rates):
                group["lr"] = base * factor
        optimizer.step()
        timer.mark("optimizer")
        phases = timer.durations()
        record = dict(
            iteration=iteration,
            loss=loss_total,
            generation_s=phases["generation"],
            fdtd_forward_s=fdtd_forward,
            adjoint_s=adjoint,
            network_backward_s=phases["network_backward"],
            optimizer_s=phases["optimizer"],
            iteration_s=phases["total"],
            wall_s=time.perf_counter() - wall0,
        )
        if objective.kind != "l2" or objective.gain_power:
            record["l2_misfit"] = l2_total
        if tv is not None:
            record["regularization_s"] = phases["regularization"]
            record["tv_weights"] = tv_weights
        if iteration % 5 == 4 or iteration == iterations - 1:
            record["errors"] = evaluate()
        history.append(record)
        if not spec.get("lr_search") and (iteration % 10 == 9 or iteration == iterations - 1):
            torch.save(dict(field=field.state_dict(), optimizer=optimizer.state_dict(), history=history, info=info), ckpt_path)
    result = dict(
        name=spec["name"],
        info=info,
        parameters=field.parameter_breakdown(),
        weight_bytes=parameter_bytes(field),
        optimizer_state_bytes=optimizer_state_bytes(optimizer),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        history=history,
        resumed_from=start_iteration,
        checkpoint_steps=checkpoint_steps,
        wavefield_sketch=spec.get("wavefield_sketch"),
    )
    if validation:
        result["validation_loss"] = misfit(validation)
        result["training_loss_end"] = misfit(list(zip(shots, observed)))
    if spec.get("save_state"):
        path = Path(spec["save_state"])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(field=field.state_dict(), optimizer=optimizer.state_dict()), path)
    if not spec.get("lr_search"):
        with torch.no_grad():
            eps, sigma = field()
        torch.save({"eps": eps.cpu(), "sigma": sigma.cpu()}, out_dir / f"{spec['name']}_final.pt")
    return result


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


def methods(case: str) -> List[Dict[str, Any]]:
    out = [dict(label="dense", method="dense"), dict(label="siren_4x128", method="siren", hidden=128, layers=4)]
    for rank in (8, 12, 16, 24, 32, 48):
        out.append(dict(label=f"tsiren_2x32_r{rank}", method="tucker_siren", hidden=32, layers=2, rank=rank))
    out.append(dict(label="direct_tucker_r16", method="direct_tucker", rank=16))
    return out


def search_specs(case: str, quick: bool, protocol: str = "v1") -> List[Dict[str, Any]]:
    specs = []
    for m in methods(case):
        for lr in LR_CANDIDATES[m["method"]]:
            spec = dict(m, name=f"{case}|search|{m['label']}|lr{lr:g}", case=case, lr=lr, iterations=SEARCH_ITERATIONS if not quick else 4, lr_search=True, quick=quick, seed=0, protocol=protocol)
            if protocol in V4_LIKE:
                spec.update(train_only=True, save_state=str(search_state(case, quick, protocol, m["label"], lr)))
            if protocol == "v6":
                spec["loss"] = "mssim"
            specs.append(spec)
    return specs


def search_state(case: str, quick: bool, protocol: str, label: str, lr: float) -> Path:
    return result_dir(quick, protocol) / "search_states" / f"{case}|{label}|lr{lr:g}.pt"


def search_sigma_specs(case: str, quick: bool, protocol: str, select: str = "data") -> List[Dict[str, Any]]:
    """v4 stage B: joint eps/sigma iterations from the chosen stage-A state."""
    lrs = chosen_learning_rates(case, quick, protocol, select)
    stage = "search_sigma" if select == "data" else "search_sigma_oracle"
    specs = []
    for m in methods(case):
        if m["label"] not in lrs:
            continue
        lr = lrs[m["label"]]
        for scale in SIGMA_SCALES:
            specs.append(dict(
                m, name=f"{case}|{stage}|{m['label']}|s{scale:g}", case=case, lr=lr, sigma_scale=scale,
                iterations=SEARCH_ITERATIONS if not quick else 4, eps_only_iterations=0, lr_search=True, train_only=True,
                init_state=str(search_state(case, quick, protocol, m["label"], lr)), quick=quick, seed=0, protocol=protocol,
            ))
    return specs


def chosen_sigma_scales(case: str, quick: bool, protocol: str, select: str = "data") -> Dict[str, float]:
    stage = "search_sigma" if select == "data" else "search_sigma_oracle"
    results = load_results(result_dir(quick, protocol) / f"fwi_{stage}.json")
    best: Dict[str, tuple] = {}
    for name, r in results.items():
        if not name.startswith(f"{case}|{stage}|") or "validation_loss" not in r:
            continue
        label, value = name.split("|")[2], r["validation_loss"]
        if select == "oracle":
            value = r["history"][-1]["errors"]["sigma_log_zone"]["rmse"]
        if math.isfinite(value) and (label not in best or value < best[label][0]):
            best[label] = (value, r["spec"]["sigma_scale"])
    return {label: scale for label, (_, scale) in best.items()}


def chosen_learning_rates(case: str, quick: bool, protocol: str = "v1", select: str = "data") -> Dict[str, float]:
    results = load_results(result_dir(quick, protocol) / "fwi_search.json")
    best: Dict[str, tuple] = {}
    for name, r in results.items():
        if not name.startswith(f"{case}|search|") or "error" in r:
            continue
        label = name.split("|")[2]
        losses = [h["loss"] for h in r["history"]]
        # Mean of the last three losses: penalises oscillating learning rates.
        final = sum(losses[-3:]) / len(losses[-3:])
        if protocol in V4_LIKE:  # held-out shots
            final = r.get("validation_loss", float("nan"))
        if select == "oracle":
            final = r["history"][-1]["errors"]["eps_zone"]["rmse"]
        if all(math.isfinite(v) for v in losses + [final]) and (label not in best or final < best[label][0]):
            best[label] = (final, r["spec"]["lr"])
    return {label: lr for label, (_, lr) in best.items()}


def run_specs_for(case: str, quick: bool, seeds=(0, 1, 2), protocol: str = "v1", select: str = "data", sigma_scale: Optional[float] = None) -> List[Dict[str, Any]]:
    lrs = chosen_learning_rates(case, quick, protocol, select)
    scales = chosen_sigma_scales(case, quick, protocol, select) if protocol in V4_LIKE else {}
    if protocol in V4_LIKE and sigma_scale is not None:  # stage B skipped
        scales = {label: sigma_scale for label in lrs}
    suffix = "" if select == "data" else f"|{select}"
    iterations = ITERATIONS[protocol][case] if not quick else 6
    specs = []
    for m in methods(case):
        if m["label"] not in lrs or (protocol in V4_LIKE and m["label"] not in scales):
            continue
        deterministic = m["method"] in ("dense", "direct_tucker")
        for seed in ((0,) if deterministic else seeds):
            spec = dict(m, name=f"{case}|{m['label']}|seed{seed}{suffix}", case=case, lr=lrs[m["label"]], iterations=iterations, seed=seed, quick=quick, protocol=protocol)
            if protocol in V4_LIKE:
                spec["sigma_scale"] = scales[m["label"]]
            if protocol == "v6":
                spec.update(loss="mssim", schedule="cosine", eps_only_iterations=SIGMA_START["v6"] if not quick else iterations // 2)
            specs.append(spec)
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["search", "search_sigma", "run", "tune", "list", "_worker"])
    parser.add_argument("spec", nargs="?")
    parser.add_argument("--case", default="3d", choices=["2d", "3d"])
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--protocol", default="v1", choices=PROTOCOLS)
    parser.add_argument("--select", default="data", choices=["data", "oracle"], help="v4: hyper-parameter selection (oracle: reference only)")
    parser.add_argument("--sigma-scale", type=float, default=None, help="v4/v5 run: fixed sigma step scale instead of stage B")
    # tune: one run of one method with explicit settings (results in fwi_tune.json)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--loss", default="l2", choices=["l2", "mssim"])
    parser.add_argument("--time-gain", type=float, default=0.0)
    parser.add_argument("--schedule", default="constant", choices=["constant", "cosine"])
    parser.add_argument("--tag", default="tune")
    parser.add_argument("--sigma-start", type=int, default=None, help="tune: first iteration that updates sigma (default iterations // 3)")
    parser.add_argument("--sketch-rank", type=int, default=0, help="tune: module-B wavefield compression with this spatial rank (rhs from_e)")
    parser.add_argument("--sketch-segment", type=int, default=32, help="tune: steps per native call of the compressed history")
    args = parser.parse_args()
    if args.command == "_worker":
        worker_main(fwi)
        return
    out_dir = result_dir(args.quick, args.protocol)
    if args.command == "search":
        specs, out, kind = search_specs(args.case, args.quick, args.protocol), out_dir / "fwi_search.json", "fwi-search"
    elif args.command == "tune":
        picked = [m for m in methods(args.case) if args.only and any(t.strip("|") == m["label"] for t in args.only)]
        specs = [dict(m, name=f"{args.case}|{m['label']}|seed0|{args.tag}", case=args.case, lr=args.lr, iterations=args.iterations, seed=0,
                      quick=args.quick, protocol=args.protocol, sigma_scale=1.0 if args.sigma_scale is None else args.sigma_scale,
                      loss=args.loss, time_gain=args.time_gain, schedule=args.schedule,
                      **({} if args.sigma_start is None else {"eps_only_iterations": args.sigma_start}),
                      **({} if not args.sketch_rank else {"wavefield_sketch": dict(ranks=args.sketch_rank, oversampling=8, segment_steps=args.sketch_segment, rhs="from_e", recovery_device="host")}))
                 for m in picked]
        out, kind = out_dir / "fwi_tune.json", "fwi-tune"
        run_specs(Path(__file__), kind, specs, out, force=args.force)
        return
    elif args.command == "search_sigma":
        stage = "search_sigma" if args.select == "data" else "search_sigma_oracle"
        specs, out, kind = search_sigma_specs(args.case, args.quick, args.protocol, args.select), out_dir / f"fwi_{stage}.json", "fwi-search-sigma"
    else:
        name = "fwi.json" if args.select == "data" else f"fwi_{args.select}.json"
        specs, out, kind = run_specs_for(args.case, args.quick, tuple(args.seeds), args.protocol, args.select, args.sigma_scale), out_dir / name, "fwi"
    if args.only:
        specs = [s for s in specs if any(token in s["name"] for token in args.only)]
    if args.command == "list":
        print("\n".join(s["name"] for s in specs))
        return
    run_specs(Path(__file__), kind, specs, out, force=args.force)


if __name__ == "__main__":
    main()
