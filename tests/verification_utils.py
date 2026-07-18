from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch


C0 = 299_792_458.0
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
RESULTS_DIR = TESTS_DIR / "results"


def file_sha256(path: Path) -> str:
    """Return a SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256() -> str:
    """Return a stable digest of DeepGPR Python, C, and CUDA source files."""
    source_root = REPO_ROOT / "src" / "DeepGPR"
    source_files = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".c", ".cu", ".h"}
    )
    digest = hashlib.sha256()
    for path in source_files:
        relative_path = path.relative_to(source_root).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(4, "little"))
        digest.update(relative_path)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_metadata() -> dict[str, Any]:
    """Return the repository commit and dirty-state flag when Git is available."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        return {"git_commit": commit, "git_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def configure_local_import() -> Path:
    """Place this repository's src directory first on sys.path."""
    package_init = REPO_ROOT / "src" / "DeepGPR" / "__init__.py"
    if not package_init.is_file():
        raise FileNotFoundError(f"Local DeepGPR package not found: {package_init}")
    src_path = str(REPO_ROOT / "src")
    while src_path in sys.path:
        sys.path.remove(src_path)
    sys.path.insert(0, src_path)
    return REPO_ROOT


def assert_local_deepgpr(module: Any, repo_root: Path = REPO_ROOT) -> Path:
    """Assert that DeepGPR was imported from this repository."""
    loaded_path = Path(module.__file__).resolve()
    expected_root = (repo_root / "src" / "DeepGPR").resolve()
    try:
        loaded_path.relative_to(expected_root)
    except ValueError as exc:
        raise RuntimeError(
            f"DeepGPR was loaded from {loaded_path}, expected a path below {expected_root}."
        ) from exc
    return loaded_path


def selected_cuda_device() -> torch.device | None:
    """Return the requested CUDA device when CUDA is available."""
    if not torch.cuda.is_available():
        return None
    value = os.environ.get("DEEPGPR_TEST_CUDA_DEVICE", "cuda:0")
    device = torch.device(value)
    torch.cuda.set_device(device)
    return device


def runtime_metadata(deepgpr: Any, device: torch.device | str) -> dict[str, Any]:
    """Collect reproducibility metadata and validate the native library ABI."""
    device = torch.device(device)
    loaded_package = assert_local_deepgpr(deepgpr)
    library_path = Path(deepgpr.get_deepgpr_library_path(device)).resolve()
    expected_lib_root = (REPO_ROOT / "src" / "DeepGPR" / "lib").resolve()
    try:
        library_path.relative_to(expected_lib_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Native library was loaded from {library_path}, expected {expected_lib_root}."
        ) from exc
    library = deepgpr.get_deepgpr_lib(device)
    abi = int(library.deepgpr_abi_version())
    if abi != 5:
        raise RuntimeError(f"Expected native ABI 5, received ABI {abi}.")

    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "deepgpr_package": str(loaded_package),
        "native_library": str(library_path),
        "native_library_sha256": file_sha256(library_path),
        "native_abi": abi,
        "source_tree_sha256": source_tree_sha256(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    metadata.update(git_metadata())
    if device.type == "cuda":
        metadata.update(
            {
                "cuda_runtime": torch.version.cuda,
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return metadata


def json_ready(value: Any) -> Any:
    """Convert tensors and numeric scalar types to JSON-compatible values."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def record_check(
    checks: list[dict[str, Any]],
    name: str,
    condition: bool,
    **metrics: Any,
) -> None:
    """Record a required check and raise immediately when it fails."""
    passed = bool(condition)
    entry = {"name": name, "status": "PASS" if passed else "FAIL", **metrics}
    checks.append(json_ready(entry))
    print(f"[{entry['status']}] {name}")
    if metrics:
        print(json.dumps(json_ready(metrics), indent=2, sort_keys=True))
    if not passed:
        raise AssertionError(f"Verification check failed: {name}")


def record_skip(checks: list[dict[str, Any]], name: str, reason: str) -> None:
    """Record a check that is not applicable to the current runtime."""
    checks.append({"name": name, "status": "SKIPPED", "reason": reason})
    print(f"[SKIPPED] {name}: {reason}")


def save_report(
    report_name: str,
    checks: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write one machine-readable verification report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    failed = [item for item in checks if item.get("status") == "FAIL"]
    report = {
        "report": report_name,
        "overall_status": "FAIL" if failed else "PASS",
        "metadata": metadata,
        "checks": list(checks),
    }
    if extra:
        report["extra"] = extra
    path = RESULTS_DIR / f"{report_name}.json"
    path.write_text(json.dumps(json_ready(report), indent=2, sort_keys=True) + "\n")
    print(f"Report written to {path}")
    return path


def assert_finite(name: str, *tensors: torch.Tensor) -> None:
    """Raise when any supplied tensor contains NaN or Inf."""
    for index, tensor in enumerate(tensors):
        if tensor is None:
            raise AssertionError(f"{name}[{index}] is None.")
        if not bool(torch.isfinite(tensor).all().item()):
            raise AssertionError(f"{name}[{index}] contains NaN or Inf.")


def relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Return a scale-invariant L2 difference."""
    actual64 = actual.detach().to(device="cpu", dtype=torch.float64)
    reference64 = reference.detach().to(device="cpu", dtype=torch.float64)
    numerator = torch.linalg.vector_norm(actual64 - reference64)
    denominator = torch.linalg.vector_norm(reference64).clamp_min(1.0e-30)
    return float((numerator / denominator).item())


def cosine_similarity(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Return the cosine similarity of two flattened tensors."""
    actual64 = actual.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    reference64 = reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    denominator = (
        torch.linalg.vector_norm(actual64) * torch.linalg.vector_norm(reference64)
    ).clamp_min(1.0e-30)
    return float(torch.dot(actual64, reference64).div(denominator).item())


def max_abs_difference(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Return the maximum absolute elementwise difference."""
    return float(
        (actual.detach().to("cpu") - reference.detach().to("cpu")).abs().max().item()
    )


def normalized_interior_mask(
    shape: Sequence[int], pml: int | Sequence[int], device: torch.device | str
) -> torch.Tensor:
    """Return the material region not occupied by DeepGPR's in-model CPML."""
    ndim = len(shape)
    if isinstance(pml, int):
        values = [pml] * (2 * ndim)
    else:
        values = list(pml)
        if len(values) == 6 and ndim == 2:
            values = values[:4]
    if len(values) != 2 * ndim:
        raise ValueError(f"Expected {2 * ndim} PML values, received {len(values)}.")

    mask = torch.zeros(tuple(shape), dtype=torch.bool, device=device)
    slices = []
    for axis, size in enumerate(shape):
        low, high = int(values[2 * axis]), int(values[2 * axis + 1])
        start = low + 1 if low > 0 else 0
        stop = size - high if high > 0 else size
        slices.append(slice(start, stop))
    mask[tuple(slices)] = True
    return mask


def pml_boundary_mask(
    shape: Sequence[int], pml: int | Sequence[int], device: torch.device | str
) -> torch.Tensor:
    """Return the complement of the physical material region."""
    return ~normalized_interior_mask(shape, pml, device)


def gradient_direction(gradient: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a normalized gradient direction restricted to a mask."""
    direction = torch.where(mask, gradient.detach(), torch.zeros_like(gradient))
    norm = torch.linalg.vector_norm(direction.to(torch.float64)).clamp_min(1.0e-30)
    return direction / norm.to(direction.dtype)


def random_direction(
    shape: Sequence[int], mask: torch.Tensor, seed: int, device: torch.device | str
) -> torch.Tensor:
    """Return a deterministic normalized random direction inside a mask."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    values = values.to(device)
    values = torch.where(mask, values, torch.zeros_like(values))
    norm = torch.linalg.vector_norm(values.to(torch.float64)).clamp_min(1.0e-30)
    return values / norm.to(values.dtype)


def directional_derivative_rows(
    objective: Callable[[torch.Tensor], torch.Tensor],
    base: torch.Tensor,
    direction: torch.Tensor,
    gradient: torch.Tensor,
    steps: Iterable[float],
) -> list[dict[str, float]]:
    """Compare an adjoint directional derivative with central differences."""
    adjoint = float(
        (gradient.detach().to(torch.float64) * direction.to(torch.float64)).sum().item()
    )
    rows: list[dict[str, float]] = []
    for step in steps:
        with torch.no_grad():
            plus = float(objective(base + step * direction).detach().item())
            minus = float(objective(base - step * direction).detach().item())
        finite_difference = (plus - minus) / (2.0 * step)
        relative_error = abs(adjoint - finite_difference) / max(
            abs(adjoint), abs(finite_difference), 1.0e-30
        )
        rows.append(
            {
                "step": float(step),
                "adjoint": adjoint,
                "finite_difference": finite_difference,
                "relative_error": relative_error,
            }
        )
    return rows


def best_relative_error(rows: Sequence[dict[str, float]]) -> float:
    """Return the smallest directional derivative error in a step sweep."""
    return min(float(row["relative_error"]) for row in rows)


def boundary_absmax(gradient: torch.Tensor, boundary_mask: torch.Tensor) -> float:
    """Return the maximum absolute gradient in CPML cells."""
    values = gradient.detach()[boundary_mask]
    return 0.0 if values.numel() == 0 else float(values.abs().max().item())


def signal_rms(signal: torch.Tensor) -> float:
    """Return an RMS value evaluated in float64."""
    values = signal.detach().to(device="cpu", dtype=torch.float64)
    return float(torch.sqrt(torch.mean(values.square())).item())


def peak_sample(signal: torch.Tensor) -> int:
    """Return the index of the largest absolute sample."""
    values = signal.detach().to("cpu").abs().reshape(-1)
    return int(torch.argmax(values).item())


def expected_travel_time(distance: float, relative_permittivity: float) -> float:
    """Return the nondispersive travel time in a nonmagnetic dielectric."""
    return distance * math.sqrt(relative_permittivity) / C0
