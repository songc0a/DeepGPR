"""Bitwise A/B harness for the DeepGPR CPU backend.

Runs a matrix of forward+backward configurations and writes a JSON map of
sha256 hashes of every tensor produced (outputs, states, histories and
gradients).  Run it once per candidate library; two libraries are bitwise
equivalent on this matrix when their `results` maps are identical. Metadata
and labels intentionally differ. Every collected tensor must be finite.

Usage: PYTHONPATH=src OMP_NUM_THREADS=N python tests/cpu_index_decoding_cases.py OUT.json LABEL
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import namedtuple
from pathlib import Path

import torch

import DeepGPR


def sha(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(t.flatten().view(torch.uint8).numpy().tobytes()).hexdigest()


def collect(obj, prefix: str, out: dict):
    """Recursively collect tensors from nested structures."""
    if isinstance(obj, torch.Tensor):
        if not torch.isfinite(obj).all():
            raise AssertionError(f"Non-finite tensor: {prefix}")
        out[prefix] = f"{tuple(obj.shape)}|{obj.dtype}|{sha(obj)}"
    elif obj is None:
        out[prefix] = None
    elif isinstance(obj, dict):
        for key in sorted(obj, key=str):
            collect(obj[key], f"{prefix}.{key}", out)
    elif isinstance(obj, (list, tuple)):
        if hasattr(obj, "_fields"):  # namedtuple
            for name, value in zip(obj._fields, obj):
                collect(value, f"{prefix}.{name}", out)
        else:
            for idx, value in enumerate(obj):
                collect(value, f"{prefix}[{idx}]", out)
    elif hasattr(obj, "__dict__"):
        for key in sorted(vars(obj)):
            collect(getattr(obj, key), f"{prefix}.{key}", out)


Case = namedtuple(
    "Case",
    "nx ny nz dx dt nt pml order mode storage sampling shots nsrc label",
)

CASES = [
    Case(48, 56, 1, 0.02, 3e-11, 120, 8, 2, 2, torch.float32, 1, 1, 1, "c1_2d_iso_o2_m2_fp32"),
    Case(52, 44, 1, [0.02, 0.015, 0.02], 3e-11, 140, 6, 4, 3, torch.float32, 1, 2, 1, "c2_2d_aniso_o4_m3_2shot"),
    Case(32, 34, 30, 0.02, 3e-11, 110, 7, 2, 3, torch.float16, 2, 1, 1, "c3_3d_iso_o2_m3_fp16_s2"),
    Case(44, 40, 36, [0.02, 0.012, 0.016], 2.5e-11, 90, 8, 8, 3, torch.float32, 1, 1, 1, "c4_3d_aniso_o8_m3"),
    Case(60, 60, 1, 0.02, 3e-11, 100, 0, 2, 2, torch.float32, 1, 1, 1, "c5_2d_iso_o2_nopml"),
    Case(30, 28, 26, 0.02, 3e-11, 90, 0, 4, 3, torch.bfloat16, 1, 1, 1, "c6_3d_iso_o4_m3_bf16_nopml"),
    Case(56, 48, 1, [0.02, 0.018, 0.02], 3e-11, 130, 8, 2, 3, torch.float32, 1, 4, 1, "c7_2d_aniso_o2_m3_4shot"),
    Case(56, 44, 1, [0.02, 0.017, 0.02], 3e-11, 120, 8, 2, 3, torch.float32, 1, 3, 2, "c8_2d_aniso_o2_m3_3shot_2src"),
]


def build_inputs(case: Case):
    torch.manual_seed(1234)
    x = torch.arange(case.nx, dtype=torch.float32)[:, None, None]
    y = torch.arange(case.ny, dtype=torch.float32)[None, :, None]
    eps_r = torch.full((case.nx, case.ny, case.nz), 4.0)
    eps_r[case.nx // 2:] = 9.0
    eps_r = eps_r * (1.0 + 0.05 * torch.exp(-((x - 0.6 * case.nx) ** 2 + (y - 0.4 * case.ny) ** 2) / (2 * 0.08 * case.nx * 0.08 * case.ny)))
    sigma = 0.002 * torch.exp(-((x - 0.3 * case.nx) ** 2 + (y - 0.7 * case.ny) ** 2) / (2 * 0.1 * case.nx * 0.1 * case.ny))
    sigma = sigma.expand(case.nx, case.ny, case.nz).clone() * torch.ones_like(eps_r)
    eps_r.requires_grad_(True)
    sigma.requires_grad_(True)

    margin = case.pml + 8
    freq = 2.5e8
    peak = 1.0 / freq
    wave = DeepGPR.wavelet.ricker(freq, case.nt, case.dt, peak, device="cpu")
    # One waveform per source; every shot reuses the same source set.
    source_amplitudes = wave.reshape(1, case.nt, 1).repeat(case.nsrc, 1, 1).clone()
    source_amplitudes.requires_grad_(True)

    source_location = torch.tensor(
        [[[margin + 3 * q, margin + 2 * s, 0 if case.nz == 1 else margin + q] for q in range(case.nsrc)]
         for s in range(case.shots)], dtype=torch.int
    )
    nrx = 3
    receiver_location = torch.tensor(
        [[[case.nx - margin - 1, margin + 2 * s + r * 5, 0 if case.nz == 1 else case.nz - margin - 1] for r in range(nrx)]
         for s in range(case.shots)], dtype=torch.int
    )
    # Keep acquisition points valid for the narrow-grid regression case.
    for axis, size in enumerate((case.nx, case.ny, case.nz)):
        source_location[..., axis].remainder_(size)
        receiver_location[..., axis].remainder_(size)
    return eps_r, sigma, source_amplitudes, source_location, receiver_location


def run_case(case: Case, *, extra=None, initial=False) -> dict:
    eps_r, sigma, src_amp, src_loc, rx_loc = build_inputs(case)
    extra = dict(extra or {})
    if extra.get("wavefield_history_region") == "physical":
        eps_r.requires_grad_(False)
        sigma.requires_grad_(False)
    owned = []
    if initial:
        groups = DeepGPR.checkpoint_initial_field(
            device="cpu", dx=case.dx, dt=case.dt, er=eps_r, se=sigma,
            source_amplitudes=src_amp, source_location=src_loc,
            receiver_location=rx_loc, pmlthick=extra.get("pmlthick", case.pml),
            fdtd_order=case.order,
        )
        owned = [(torch.randn_like(t) * 1e-5).requires_grad_()
                 for group in groups for t in group]
        extra.update(E=tuple(owned[:3]), H=tuple(owned[3:6]), PML=tuple(owned[6:]))
    E_saved, E, H, pml_state, receivers = DeepGPR.compute(
        device="cpu",
        dx=case.dx,
        dt=case.dt,
        source_amplitudes=src_amp,
        source_location=src_loc,
        receiver_location=rx_loc,
        eps_r=eps_r,
        sigma=sigma,
        pmlthick=extra.pop("pmlthick", case.pml),
        fdtd_order=case.order,
        mode=case.mode,
        wavefield_storage_dtype=case.storage,
        model_gradient_sampling_interval=case.sampling,
        **extra,
    )
    out: dict = {}
    collect(receivers, "receivers", out)
    collect(E_saved, "E_saved", out)
    collect(E, "E", out)
    collect(H, "H", out)
    collect(pml_state, "pml_state", out)
    node = receivers.grad_fn
    for name in ("R_saved", "E_final"):
        collect(getattr(node, name), name, out)

    # Double precision limits float32 reduction noise in the scalar objective.
    # PyTorch and the native solver may share the same OpenMP runtime.
    loss = receivers.double().square().sum()
    if initial:
        loss = loss + sum(t.double().square().sum() for group in (E, H, pml_state) for t in group)
    loss.backward()
    for name, tensor in (("eps_r", eps_r), ("sigma", sigma), ("source", src_amp)):
        if tensor.requires_grad and tensor.grad is None:
            raise AssertionError(f"Missing {name} gradient")
    collect(eps_r.grad, "grad_eps_r", out)
    collect(sigma.grad, "grad_sigma", out)
    collect(src_amp.grad, "grad_src_amp", out)
    for i, tensor in enumerate(owned):
        if tensor.grad is None:
            raise AssertionError(f"Missing initial-state gradient {i}")
        collect(tensor.grad, f"grad_initial[{i}]", out)
    out["loss"] = f"{loss.item():.17g}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("label")
    parser.add_argument("--library-dir", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    out_path, label = args.out_path, args.label
    if args.library_dir:
        DeepGPR.native.loader.LIB_DIR = args.library_dir.resolve()
    # Explicitly set the native runtime, rather than assuming PyTorch uses a
    # separate OpenMP runtime. This can also change PyTorch's thread count.
    torch.set_num_threads(1)
    library = DeepGPR.get_deepgpr_lib("cpu")
    import ctypes
    if hasattr(library, "omp_set_num_threads"):
        library.omp_set_num_threads.argtypes = [ctypes.c_int]
        library.omp_set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    elif os.environ.get("OMP_NUM_THREADS", "1") != "1":
        raise RuntimeError("A serial library cannot validate multiple OpenMP threads")
    results = {}
    for case in CASES:
        print(f'[{label}] running {case.label} ...', file=sys.stderr, flush=True)
        results[case.label] = run_case(case)
        n_hashes = len(results[case.label])
        print(f"[{label}] {case.label}: {n_hashes} tensors hashed", file=sys.stderr)
    small = CASES[0]._replace(nx=12, ny=15, pml=0, nt=23, shots=2)
    # Exercise cropped histories, disabled faces, and initial-state adjoints.
    for region in ("extended", "physical"):
        for three_d in (False, True):
            case = small._replace(nx=24, ny=28, nz=21 if three_d else 1,
                                  mode=3 if three_d else 2)
            key = f"edge_{region}_{'3d' if three_d else '2d'}"
            extra = dict(pmlthick=[1, 0, 2, 3, 0, 2] if three_d else [1, 0, 2, 3],
                         wavefield_history_region=region, wavefield_rhs_history="stored",
                         source_direction=0 if three_d else 2)
            print(f"[{label}] running {key} ...", file=sys.stderr, flush=True)
            results[key] = run_case(case, extra=extra, initial=True)
    results["edge_narrow_2d"] = run_case(
        small._replace(nx=2, ny=48, nz=1, shots=1), initial=True)
    library_path = Path(DeepGPR.get_deepgpr_library_path("cpu"))
    metadata = dict(platform=platform.platform(), python=platform.python_version(),
                    torch=torch.__version__, omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                    library=str(library_path), library_sha256=hashlib.sha256(library_path.read_bytes()).hexdigest())
    with open(out_path, "w") as fh:
        json.dump({"label": label, "metadata": metadata, "results": results}, fh, indent=1, sort_keys=True)
    if args.reference:
        reference = json.loads(args.reference.read_text())["results"]
        if reference != results:
            differences = [f"{case}/{key}" for case in set(reference) | set(results)
                           for key in set(reference.get(case, {})) | set(results.get(case, {}))
                           if reference.get(case, {}).get(key) != results.get(case, {}).get(key)]
            raise AssertionError(f"A/B differences: {differences}")
    print(f"[{label}] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
