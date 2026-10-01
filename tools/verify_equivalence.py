#!/usr/bin/env python3
"""Compare this DeepGPR tree with a reference release and check the 0.1.0 fixes.

The script imports two copies of the package side by side - the working tree
(``DeepGPR``) and a reference tree (``DeepGPR_reference``, by default the last
pre-refactor commit ``73b2b50``; each uses its *own* native libraries) - and
runs the same forward/adjoint problems through both.

Checks
------
1. **Forward modelling** (receivers, saved history, final E/H/CPML states):
   identical up to float32 rounding (``--forward-rtol``, default 1e-5). The
   only numerical change on the forward path is the CODATA-2018 ε0/µ0 now
   used for the CPML profiles (relative change ~5e-10).
2. **Gradients** (∂/∂εr, ∂/∂σ, ∂/∂source): agree within ``--grad-rtol``
   (default 1e-4). The CPU adjoint now sums in a fixed order instead of with
   OpenMP atomics, so it differs from the old code at rounding level only. On
   CUDA the reference's own run-to-run spread is also reported.
3. **Determinism**: the CPU backward is bitwise identical for every
   ``OMP_NUM_THREADS`` in ``--threads`` (separate processes).
4. **Corrected helpers**: ``legacy=True`` reproduces the reference bitwise;
   the new defaults match independent NumPy implementations of the correct
   FIR design, analytic-signal envelope and isotropic TV.
5. **Native error reporting**: a failing native call raises
   ``NativeLibraryError`` instead of returning silently (CPU and CUDA).

Usage::

    python tools/verify_equivalence.py --reference-src ~/Downloads/DeepGPR-dev/src/DeepGPR
    python tools/verify_equivalence.py                      # reference = git 73b2b50
    python tools/verify_equivalence.py --device cpu --quick

Exit status is 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_REF = "73b2b50"


# --------------------------------------------------------------------------- loading
def _materialise_reference(ref: Optional[str], src: Optional[str], workdir: Path) -> Path:
    """Copy the reference ``DeepGPR`` package into ``workdir/DeepGPR_reference``."""
    target = workdir / "DeepGPR_reference"
    if src:
        shutil.copytree(Path(src).expanduser(), target)
        return target
    archive = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "archive", ref or DEFAULT_REFERENCE_REF, "src/DeepGPR"],
        check=True,
        capture_output=True,
    ).stdout
    extract = workdir / "extract"
    extract.mkdir()
    subprocess.run(["tar", "-x", "-C", str(extract)], input=archive, check=True)
    shutil.move(str(extract / "src" / "DeepGPR"), target)
    return target


def load_candidate() -> Any:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    for name in list(sys.modules):
        if name == "DeepGPR" or name.startswith("DeepGPR."):
            del sys.modules[name]
    return importlib.import_module("DeepGPR")


def load_packages(ref: Optional[str], src: Optional[str], workdir: Path) -> Tuple[Any, Any]:
    reference_dir = _materialise_reference(ref, src, workdir)
    candidate = load_candidate()
    sys.path.insert(0, str(reference_dir.parent))
    reference = importlib.import_module("DeepGPR_reference")
    return reference, candidate


# ------------------------------------------------------------------------- problems
def _locations(shape: Tuple[int, int, int], nstep: int, count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    axes = []
    for size in shape:
        if size > 2:
            axes.append(torch.randint(1, size - 1, (nstep, count), generator=generator))
        else:
            axes.append(torch.zeros((nstep, count), dtype=torch.int64))
    return torch.stack(axes, -1).to(torch.int32)


def make_problem(case: Dict[str, Any], package: Any, device: torch.device) -> Dict[str, Any]:
    """Build identical inputs for ``compute`` (fresh tensors on every call)."""
    generator = torch.Generator().manual_seed(case["seed"])
    shape = tuple(case["shape"])
    shape3 = shape + (1,) * (3 - len(shape))
    eps_r = (1.5 + 5.0 * torch.rand(shape, generator=generator)).to(device)
    sigma = (4.0e-3 * torch.rand(shape, generator=generator)).to(device)
    nt = case["nt"]
    wavelet = package.wavelet.ricker(3.0e8, nt, case["dt"], 4.0e-9, device=device)
    waveforms = wavelet.reshape(1, nt, 1).repeat(case["waveforms"], 1, 1).clone()
    args = dict(
        device=device,
        dx=case["dx"],
        dt=case["dt"],
        source_amplitudes=waveforms.requires_grad_(case["source_grad"]),
        source_location=_locations(shape3, case["nstep"], case["nsr"], case["seed"] + 1).to(device),
        receiver_location=_locations(shape3, case["nstep"], case["nrx"], case["seed"] + 2).to(
            device
        ),
        eps_r=eps_r.requires_grad_(case["material_grad"]),
        sigma=sigma.requires_grad_(case["material_grad"]),
        pmlthick=case["pml"],
        source_direction=case["components"][0],
        receiver_component=case["components"][1],
        model_gradient_sampling_interval=case["interval"],
        wavefield_storage_dtype=case["storage"],
        fdtd_order=case["order"],
        mode=case["mode"],
        save_wavefield_history=case["history"],
    )
    if case["mu_r"]:
        args["mu_r"] = (1.0 + 0.2 * torch.rand(shape, generator=generator)).to(device)
    args.update(case.get("extra", {}))
    return args


def run(package: Any, case: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """Run forward (+ backward when a history exists) and collect every tensor."""
    args = make_problem(case, package, device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if case.get("checkpoint"):
            half = case["nt"] // 2
            source = args.pop("source_amplitudes")
            first = package.compute(source_amplitudes=source[:, :half], **args)
            result = package.compute(
                source_amplitudes=source[:, half:],
                E=tuple(t.clone() for t in first[1]),
                H=tuple(t.clone() for t in first[2]),
                PML=tuple(t.clone() for t in first[3]),
                **args,
            )
            receivers = torch.cat((first[-1], result[-1]), dim=1)
            args["source_amplitudes"] = source
        else:
            result = package.compute(**args)
            receivers = result[-1]

    outputs: Dict[str, torch.Tensor] = {"receivers": receivers.detach().clone()}
    outputs["E_saved"] = result[0].detach().clone()
    for index, tensor in enumerate((*result[1], *result[2], *result[3])):
        outputs[f"state{index}"] = tensor.detach().clone()

    if case["history"] and (case["material_grad"] or case["source_grad"]):
        loss = receivers.square().sum() + 1.0e3 * sum(t.sum() for t in result[1])
        loss.backward()
        if case["material_grad"]:
            outputs["grad_eps_r"] = args["eps_r"].grad.detach().clone()
            outputs["grad_sigma"] = args["sigma"].grad.detach().clone()
        source = args.get("source_amplitudes")
        if case["source_grad"] and source is not None and source.grad is not None:
            outputs["grad_source"] = source.grad.detach().clone()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {key: value.cpu() for key, value in outputs.items()}


# ---------------------------------------------------------------------------- cases
def build_cases(device: torch.device, quick: bool) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    seeds = itertools.count(100)
    orders = (2, 4) if quick else (2, 4, 8)
    storages = ("float32", "float16") if quick else ("float32", "float16", "bfloat16")
    intervals = (1,) if quick else (1, 3)
    for three_d, order, storage, interval in itertools.product(
        (False, True), orders, storages, intervals
    ):
        cases.append(
            dict(
                name=f"{'3d' if three_d else '2d'}-o{order}-{storage}-s{interval}",
                seed=next(seeds),
                shape=(10, 11, 7) if three_d else (20, 24),
                dx=(0.02, 0.018, 0.015) if three_d else 0.02,
                dt=1.5e-11,
                nt=90,
                nstep=2,
                nsr=2,
                nrx=4,
                waveforms=2 if interval == 1 else 1,
                pml=[2, 3, 3, 2, 1, 2] if three_d else 4,
                components=(1, 0) if three_d else (2, 2),
                mode=3 if three_d else 2,
                order=order,
                storage=storage,
                interval=interval,
                history=True,
                material_grad=True,
                source_grad=True,
                mu_r=three_d and order == 2,
            )
        )
    base = dict(
        seed=0,
        shape=(20, 24),
        dx=0.02,
        dt=1.5e-11,
        nt=80,
        nstep=1,
        nsr=1,
        nrx=3,
        waveforms=1,
        pml=3,
        components=(2, 2),
        mode=2,
        order=4,
        storage="float32",
        interval=1,
        history=True,
        material_grad=True,
        source_grad=False,
        mu_r=False,
    )
    cases += [
        {**base, "name": "no-pml", "seed": next(seeds), "pml": 0},
        {**base, "name": "asymmetric-pml", "seed": next(seeds), "pml": [0, 5, 2, 0]},
        {
            **base,
            "name": "forward-only",
            "seed": next(seeds),
            "history": False,
            "material_grad": False,
        },
        {**base, "name": "mode3-2d", "seed": next(seeds), "mode": 3},
        {
            **base,
            "name": "checkpoint-segments",
            "seed": next(seeds),
            "checkpoint": True,
            "source_grad": True,
        },
    ]
    if device.type == "cuda":
        cases += [
            {
                **base,
                "name": "cuda-int8",
                "seed": next(seeds),
                "extra": {"wavefield_compression": "int8"},
            },
            {
                **base,
                "name": "cuda-int8-3d",
                "seed": next(seeds),
                "shape": (12, 12, 8),
                "pml": 2,
                "dx": (0.02, 0.02, 0.02),
                "mode": 3,
                "components": (0, 1),
                "extra": {"wavefield_compression": "int8"},
            },
            {
                **base,
                "name": "cuda-async-offload",
                "seed": next(seeds),
                "extra": {"use_async_offload": True},
            },
            {
                **base,
                "name": "cuda-fp16-legacy",
                "seed": next(seeds),
                "storage": "float16",
                "extra": {"wavefield_conversion_backend": "legacy"},
            },
            {
                **base,
                "name": "cuda-fp16-vec2",
                "seed": next(seeds),
                "storage": "float16",
                "extra": {"wavefield_conversion_backend": "native_vec2"},
            },
        ]
    return cases


# ------------------------------------------------------------------------ comparison
def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    a64, b64 = a.double(), b.double()
    return float((a64 - b64).abs().max() / a64.abs().max().clamp_min(1e-30))


def compare(
    reference: Dict[str, torch.Tensor],
    candidate: Dict[str, torch.Tensor],
    forward_rtol: float,
    grad_rtol: float,
    noise: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[List[str], Dict[str, float]]:
    problems: List[str] = []
    worst = {"forward": 0.0, "gradient": 0.0}
    if reference.keys() != candidate.keys():
        return [f"output keys differ: {sorted(set(reference) ^ set(candidate))}"], worst
    for key, ref in reference.items():
        cand = candidate[key]
        if ref.shape != cand.shape or ref.dtype != cand.dtype:
            problems.append(
                f"{key}: {tuple(ref.shape)}/{ref.dtype} vs {tuple(cand.shape)}/{cand.dtype}"
            )
            continue
        if not ref.is_floating_point():
            if not torch.equal(ref, cand):
                problems.append(f"{key}: integer payload differs")
            continue
        if not torch.isfinite(cand.float()).all():
            problems.append(f"{key}: candidate contains NaN/Inf")
            continue
        rel = _relative(ref.float(), cand.float())
        is_gradient = key.startswith("grad_")
        kind = "gradient" if is_gradient else "forward"
        worst[kind] = max(worst[kind], rel)
        tolerance = grad_rtol if is_gradient else forward_rtol
        if is_gradient and noise is not None:
            tolerance = max(tolerance, 4.0 * _relative(ref.float(), noise[key].float()))
        if key == "E_saved" and ref.dtype == torch.int8:
            tolerance = max(tolerance, 1.0)  # packed INT8 bytes: compared via receivers
            continue
        if rel > tolerance:
            problems.append(f"{key}: max relative difference {rel:.2e} > {tolerance:.1e}")
    return problems, worst


# ------------------------------------------------------------------- helper checks
def _numpy_fir(cutoff: float, fs: float, numtaps: int) -> np.ndarray:
    n = np.arange(numtaps, dtype=np.float64)
    t = n - (numtaps - 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(t == 0, 2 * cutoff / fs, np.sin(2 * np.pi * cutoff / fs * t) / (np.pi * t))
    h *= 0.54 - 0.46 * np.cos(2 * np.pi * n / (numtaps - 1))
    return h / h.sum()


def _numpy_envelope(data: np.ndarray, power: float) -> np.ndarray:
    nt = data.shape[1]
    spectrum = np.fft.fft(data - data.mean(axis=1, keepdims=True), axis=1)
    weights = np.zeros(nt)
    weights[0] = 1.0
    if nt % 2 == 0:
        weights[1 : nt // 2] = 2.0
        weights[nt // 2] = 1.0
    else:
        weights[1 : (nt + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * weights[None, :, None], axis=1)) ** power


def _numpy_isotropic_tv(model: np.ndarray, epsilon: float) -> float:
    squared = np.zeros_like(model, dtype=np.float64)
    for axis in range(model.ndim):
        difference = np.diff(model.astype(np.float64), axis=axis)
        pad = [(0, 0)] * model.ndim
        pad[axis] = (0, 1)
        squared += np.pad(difference, pad) ** 2
    return float(np.sqrt(squared + epsilon).sum())


def check_helpers(reference: Any, candidate: Any, device: torch.device) -> List[str]:
    problems = []
    generator = torch.Generator().manual_seed(7)
    odd = torch.randn((2, 401, 5), generator=generator, dtype=torch.float64)
    even = torch.randn((2, 400, 5), generator=generator, dtype=torch.float64)
    trace = torch.randn(400, generator=generator, dtype=torch.float64)

    def equal(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
        if not torch.equal(a, b):
            problems.append(f"{name}: legacy result differs from reference ({_relative(a, b):.1e})")

    def close(name: str, a: np.ndarray, b: np.ndarray, rtol: float = 1e-10) -> None:
        rel = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))
        if rel > rtol:
            problems.append(f"{name}: differs from the NumPy reference ({rel:.1e})")

    for numtaps in (8, 11, 14, 15):
        equal(
            f"FIR legacy n={numtaps}",
            candidate.design_fir_filter(7e8, 1e10, numtaps, dtype=torch.float64, legacy=True),
            reference.design_fir_filter(7e8, 1e10, numtaps, dtype=torch.float64),
        )
        close(
            f"FIR n={numtaps}",
            candidate.design_fir_filter(7e8, 1e10, numtaps, dtype=torch.float64).numpy(),
            _numpy_fir(7e8, 1e10, numtaps),
        )
    equal(
        "apply_filter legacy",
        candidate.apply_filter(odd, 1e10, 7e8, legacy=True),
        reference.apply_filter(odd, 1e10, 7e8),
    )
    equal(
        "apply_filter legacy 1D",
        candidate.apply_filter(trace, 1e10, 9e8, legacy=True),
        reference.apply_filter(trace, 1e10, 9e8),
    )
    for name, data in (("odd", odd), ("even", even)):
        equal(
            f"hilbert legacy {name}",
            candidate.hilbert_transform(data, p=2, legacy=True),
            reference.hilbert_transform(data, p=2),
        )
        close(
            f"hilbert {name}",
            candidate.hilbert_transform(data, p=2).numpy(),
            _numpy_envelope(data.numpy(), 2),
        )
    equal(
        "hilbert even unchanged",
        candidate.hilbert_transform(even, p=1),
        reference.hilbert_transform(even, p=1),
    )

    model_2d = torch.randn((30, 40), generator=generator, dtype=torch.float64)
    model_3d = torch.randn((12, 13, 14), generator=generator, dtype=torch.float64)
    for model in (model_2d, model_3d):
        equal(
            "TV anisotropic",
            candidate.TVRegularization(1, 0.5)(model, 2 * model),
            reference.TVRegularization(1, 0.5)(model, 2 * model),
        )
        equal(
            "TV legacy isotropic",
            candidate.TVRegularization(1, 0, method="legacy_isotropic")(model),
            reference.TVRegularization(1, 0, method="iso")(model),
        )
        # TVRegularization accumulates in the default dtype (float32), as in
        # 0.0.20, so float64 models are compared at float32 precision.
        close(
            "TV isotropic",
            np.array(float(candidate.TVRegularization(1, 0, method="isotropic")(model))),
            np.array(_numpy_isotropic_tv(model.numpy(), 1e-8)),
            rtol=1e-6,
        )
    return problems


def check_error_reporting(candidate: Any, device: torch.device) -> List[str]:
    from DeepGPR.native import abi, bridge  # candidate package

    lib = candidate.get_deepgpr_lib(device)
    if not bridge.supports_error_reporting(lib):
        return [f"{device.type} library has no error reporting (rebuild the native library)"]
    arguments = {name: 0 for name, _ in abi.BACKWARD_SIGNATURE}
    arguments.update(
        dt=1.0e-12,
        nt=1,
        nshot=1,
        nreceiver=1,
        dx=0.01,
        dy=0.01,
        dz=0.01,
        nx_fields=4,
        ny_fields=4,
        nz_fields=4,
        ndata_source=1,
        nsource=1,
        sampling_interval=1,
        fwi_mode=2,
    )
    if device.type == "cpu":
        arguments.update(nx_fields=1 << 15, ny_fields=1 << 15, nz_fields=1 << 15)
    else:
        # INT8 storage with a 27-voxel block is rejected by the native code
        # before any device memory is touched.
        arguments["storage_type"] = 3 | (3 << 8) | (3 << 14) | (3 << 20)
    try:
        bridge.invoke(lib, "backward", 2, arguments)
    except candidate.NativeLibraryError as exc:
        if bridge.last_native_error(lib):
            return ["error message was not cleared after raising"]
        print(f"         native error surfaced as NativeLibraryError: {exc}")
        return []
    return [f"{device.type}: failing native call returned without an exception"]


# ---------------------------------------------------------------------- determinism
def _worker(output: str, quick: bool) -> None:
    candidate = load_candidate()
    cpu = torch.device("cpu")
    results = {}
    for case in build_cases(cpu, quick)[:4]:
        for key, value in run(candidate, case, cpu).items():
            results[f"{case['name']}/{key}"] = value
    torch.save(results, output)


def check_determinism(threads: List[int], quick: bool, workdir: Path) -> List[str]:
    outputs = []
    for count in threads:
        path = workdir / f"threads_{count}.pt"
        env = {**os.environ, "OMP_NUM_THREADS": str(count)}
        subprocess.run(
            [sys.executable, __file__, "--_worker", str(path)] + (["--quick"] if quick else []),
            check=True,
            env=env,
        )
        outputs.append(torch.load(path))
    problems = []
    for count, result in zip(threads[1:], outputs[1:]):
        different = [k for k in outputs[0] if not torch.equal(outputs[0][k], result[k])]
        if different:
            problems.append(
                f"OMP_NUM_THREADS={count} differs from {threads[0]} in {len(different)} tensors, "
                f"e.g. {different[:3]}"
            )
    return problems


# ----------------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--reference-ref", help=f"git ref of the reference (default {DEFAULT_REFERENCE_REF})"
    )
    parser.add_argument("--reference-src", help="path to a reference src/DeepGPR directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true", help="smaller case matrix")
    parser.add_argument("--forward-rtol", type=float, default=1e-5)
    parser.add_argument("--grad-rtol", type=float, default=1e-4)
    parser.add_argument(
        "--threads", default="1,2,4,8", help="OMP_NUM_THREADS values for the CPU determinism check"
    )
    parser.add_argument("--json", help="write a machine-readable summary to this file")
    parser.add_argument("--_worker", help=argparse.SUPPRESS)
    options = parser.parse_args(argv)

    if options._worker:
        _worker(options._worker, options.quick)
        return 0

    devices = [torch.device("cpu")]
    if options.device in ("auto", "cuda") and torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if options.device == "cuda":
        devices = [d for d in devices if d.type == "cuda"]
        if not devices:
            print("CUDA requested but not available.")
            return 1
    if options.device == "cpu":
        devices = [torch.device("cpu")]
    print(
        f"torch={torch.__version__}  devices={[str(d) for d in devices]}  "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}"
    )

    failures = 0
    summary: Dict[str, Any] = {"cases": []}

    def report(status_ok: bool, label: str, detail: str = "") -> None:
        nonlocal failures
        failures += not status_ok
        print(f"[{'OK  ' if status_ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")

    with tempfile.TemporaryDirectory() as workdir:
        reference, candidate = load_packages(
            options.reference_ref, options.reference_src, Path(workdir)
        )
        print(f"reference: {Path(reference.__file__).parent}")
        print(f"candidate: {Path(candidate.__file__).parent}\n")
        for device in devices:
            for case in build_cases(device, options.quick):
                ref = run(reference, case, device)
                noise = run(reference, case, device) if device.type == "cuda" else None
                cand = run(candidate, case, device)
                problems, worst = compare(ref, cand, options.forward_rtol, options.grad_rtol, noise)
                summary["cases"].append(
                    {"device": device.type, "case": case["name"], "problems": problems, **worst}
                )
                report(
                    not problems,
                    f"{device.type:4s} {case['name']:24s}",
                    f"forward {worst['forward']:.1e}  gradient {worst['gradient']:.1e}",
                )
                for problem in problems:
                    print(f"         - {problem}")
            problems = check_helpers(reference, candidate, device)
            report(not problems, f"{device.type:4s} FIR / envelope / TV fixes")
            for problem in problems:
                print(f"         - {problem}")
            problems = check_error_reporting(candidate, device)
            report(not problems, f"{device.type:4s} native error reporting")
            for problem in problems:
                print(f"         - {problem}")

        threads = [int(value) for value in options.threads.split(",") if value]
        if len(threads) > 1:
            problems = check_determinism(threads, options.quick, Path(workdir))
            report(not problems, f"cpu  bitwise determinism for OMP_NUM_THREADS={threads}")
            for problem in problems:
                print(f"         - {problem}")

    print("\nAll checks passed." if not failures else f"\n{failures} check(s) failed.")
    if options.json:
        summary["failures"] = failures
        Path(options.json).write_text(json.dumps(summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
