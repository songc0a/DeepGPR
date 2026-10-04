# DeepGPR

**Official website / documentation:** [https://songc0a.github.io/DeepGPR/](https://songc0a.github.io/DeepGPR/)

DeepGPR provides a wave propagation module for PyTorch, designed for applications such as Ground Penetrating Radar (GPR) imaging and inversion. Its core concepts are derived from Deepwave. You can use it to perform both forward modeling and backpropagation—thereby enabling the simulation of wave propagation to generate synthetic data—as well as for Full Waveform Inversion (FWI). Furthermore, you can integrate this wave propagation functionality into a larger operational pipeline—incorporating various wavelets, loss functions, and other components—to achieve end-to-end forward and reverse propagation, powered by automatic differentiation and our high-performance operators.

## Features

- 2D and 3D forward modeling of Maxwell's equations with the Finite-Difference Time-Domain (FDTD) method, for single and multiple excitations.
- Gradients of the receiver data with respect to relative permittivity, conductivity, the initial wavefield and the source amplitudes, computed by an exact discrete adjoint.
- Automatic extension of the physical model into a CPML with an independent thickness on every face. Supply only air and the target region; material gradients keep the input model shape.
- CUDA GPU backend and a C/OpenMP CPU backend (selected automatically with `device='cpu'`).
- Spatial FDTD order 2, 4 or 8 (`fdtd_order`), and gradient mode 2 (Ez only) or 3 (Ex, Ey, Ez).
- Large models: checkpointing, DDP, temporal sub-sampling, FP16/BF16 or GPU-native INT8 wavefield histories and asynchronous offload to host memory.

## System requirements

- **OS**: Linux, Windows and macOS for CPU execution; Linux and Windows for CUDA execution
- **Python**: 3.8+
- **Libraries**: `torch`, `numpy`, `matplotlib` (the examples additionally use `scikit-image`)
- **Hardware**: an NVIDIA GPU with sufficient memory for CUDA execution; CPU execution works without a GPU

## Installation

Install a PyTorch build for your platform first ([pytorch.org](https://pytorch.org/get-started/locally/)): a CUDA-enabled build for GPU use, or a CPU build. Then

```bash
pip install DeepGPR                 # runtime
pip install "DeepGPR[examples]"     # + scipy, scikit-image, jupyter for the notebooks
```

For development from a clone:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

The prebuilt native libraries ship in `src/DeepGPR/lib`. To rebuild them see [`docs/BUILDING.md`](docs/BUILDING.md).

## Quick start: a small forward-modeling test

```python
import torch
import matplotlib.pyplot as plt
import DeepGPR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dx = 0.02                 # or [dx, dy, dz], for example [0.02, 0.015, 0.01]
dt = 3e-11
nt = 2000

eps_r = torch.ones(100, 100, 1) * 2
eps_r[50:, :] = 5
sigma = torch.zeros_like(eps_r)
eps_r.requires_grad_()

source_location = torch.tensor([[[10, 10, 0]]], device=device, dtype=torch.int)
receiver_location = torch.tensor([[[10, 90, 0]]], device=device, dtype=torch.int)
freq = 2e8
source_amplitudes = torch.zeros((1, nt, 1), device=device)
source_amplitudes[0, :, 0] = DeepGPR.wavelet.ricker(freq, nt, dt, 1 / freq, device=device)

E_saved, E, H, pml_state, receivers = DeepGPR.compute(
    device=device, dx=dx, dt=dt,
    source_amplitudes=source_amplitudes,
    source_location=source_location,
    receiver_location=receiver_location,
    eps_r=eps_r, sigma=sigma,
    fdtd_order=2,
)

receivers.square().sum().backward()

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
DeepGPR.visualization.plot_trace(receivers[0, :, 0], dt=dt, ax=ax[0], title="Receiver data")
DeepGPR.visualization.plot_model(eps_r.grad, ax=ax[1], title="Gradient", cmap="seismic")
plt.show()
```

![result](Fig/example.png)

More examples are in [`examples/`](examples) (forward modeling, 2D FWI, 3D FWI).

## Source wavelets

Wavelets live in `DeepGPR.wavelet`. Every function returns a one-dimensional
tensor and accepts optional `dtype` and `device` arguments.

```python
ricker = DeepGPR.wavelet.ricker(freq, nt, dt, peak_time, device=device)
gaussian = DeepGPR.wavelet.gaussian(freq, nt, dt, peak_time, device=device)
derivative = DeepGPR.wavelet.gaussian_derivative(freq, nt, dt, peak_time, device=device)
morlet = DeepGPR.wavelet.morlet(freq, nt, dt, peak_time, cycles=3.0, device=device)
burst = DeepGPR.wavelet.sine_burst(freq, nt, dt, peak_time, cycles=3.0, device=device)
```

`DeepGPR.ricker(...)` remains available as a backward-compatible alias.

## Full-waveform inversion results

The figures show the true model, the initial model and the inverted result.

### 2D FWI

| True model | Initial model | Inverted result |
| --- | --- | --- |
| ![2D true model](Fig/2dfwitrue.png) | ![2D initial model](Fig/2dfwiinit.png) | ![2D inverted result](Fig/2dfwipred.png) |

### 3D FWI (central slice)

| Model | Central slice |
| --- | --- |
| True model | ![3D true model central slice](Fig/3dfwitrue.png) |
| Initial model | ![3D initial model central slice](Fig/3dfwiinit.png) |
| Inverted result | ![3D inverted result central slice](Fig/3dfwipred.png) |

## API overview

| Function | Purpose |
| --- | --- |
| `DeepGPR.compute(...)` | FDTD forward modeling with autograd (full reference: [`docs/API.md`](docs/API.md)) |
| `DeepGPR.checkpoint_initial_field(...)` | zero E/H/CPML states for segment-wise (checkpointed) runs |
| `DeepGPR.wavelet.*` | source wavelets |
| `DeepGPR.TVRegularization` | total-variation penalty for εr and σ (`"anisotropic"`, `"isotropic"`) |
| `DeepGPR.apply_filter`, `DeepGPR.hilbert_transform` | low-pass FIR and envelope for multi-scale FWI |
| `DeepGPR.max_stable_time_step(...)` | largest CFL-stable `dt` for a grid, order and material |
| `DeepGPR.estimate_compute_memory(...)` | memory estimate (also printed by `print_parameters=True`) |
| `DeepGPR.decompress_wavefield_history(...)` | decode a packed INT8 history for diagnostics |
| `DeepGPR.visualization.plot_*` | models, radargrams, traces, true/initial/inverted comparison |
| `DeepGPR.configure_logging(level)` | enable DeepGPR log output |

### Logging and errors

DeepGPR logs through the standard `logging` module under the `DeepGPR`
logger and never configures the root logger. Enable console output with

```python
DeepGPR.configure_logging("INFO")    # or "DEBUG" for per-call solver details
```

Errors raised by the package derive from `DeepGPR.DeepGPRError` **and** from the
built-in type used before 0.1.0, e.g. `DeepGPR.CFLConditionError` is also a
`ValueError` and `DeepGPR.NativeLibraryError` is also a `RuntimeError`.
Failures inside the native CPU/CUDA code (allocation, CUDA API or launch
errors) are raised as `NativeLibraryError` with the native message.

## Project structure

```
src/DeepGPR/
├── __init__.py            public API
├── _version.py            package version
├── config/                physical constants, native ABI codes, API defaults
├── native/                library loader, declarative C ABI table, ctypes bridge
├── preprocessing/         input validation, grid/PML normalisation, CFL, wavelets
├── solver/                CPML, field state, history storage, autograd bridge, compute()
├── postprocessing/        FIR filter, envelope, forward-wavefield files
├── inversion/             TV regularisation
├── visualization/         matplotlib helpers
├── utils/                 logging, exceptions, validators, diagnostics
├── lib/                   deepgpr.cu, deepgpr_cpu.c, deepgpr.h and prebuilt binaries
├── common.py, compute2.py, multiscale.py, wavelet.py   compatibility modules
docs/                      API reference, build guide, code audit
examples/                  forward, 2D FWI and 3D FWI notebooks
tests/                     unit tests, verification notebooks, benchmarks
tools/                     verify_equivalence.py
```

Data flow of one `compute` call:
`preprocessing.model_setup.initialization` (validate, CFL, extend into CPML) →
`solver.pml` (CPML coefficients and state) → `solver.autograd.DeepGPR`
(native forward; native adjoint on `backward`) → receiver data, final states
and the saved history.

## Tests and verification

```bash
python -m unittest discover -s tests -p "test_*.py"     # unit tests
python -m unittest tests/test_native_abi.py             # ABI check, no PyTorch needed
python tests/run_notebook.py tests/00_local_backend_and_contracts.ipynb
python tools/verify_equivalence.py --reference-src /path/to/DeepGPR-0.0.20/src/DeepGPR
bash tools/run_local_validation.sh /path/to/DeepGPR-0.0.20   # rebuild CPU+CUDA libs, run everything
```

Run the notebooks `00` to `09` in numeric order, followed by
`99_verification_summary.ipynb`. See [`tests/README.md`](tests/README.md) for
the complete matrix and CUDA options.

**Reproducibility.** The CPU backend gives bitwise identical forward results
and gradients for any `OMP_NUM_THREADS`. On CUDA, 2D Ez-TM runs (the 2D fast
path) gather their adjoint without atomics and are bitwise reproducible; the
general (3D) CUDA adjoint accumulates with atomics and can differ in the last
bits between runs.

**Reproducing results from DeepGPR ≤ 0.0.20.** 0.1.0 corrects the FIR design
for even filter lengths, the envelope for odd trace lengths and the isotropic
TV (see [`CHANGELOG.md`](CHANGELOG.md)). Use `apply_filter(..., legacy=True)`,
`design_fir_filter(..., legacy=True)`, `hilbert_transform(..., legacy=True)`
and `TVRegularization(method="legacy_isotropic")` to reproduce earlier numbers.

## Development

- Code style: PEP 8, enforced with `ruff check src/DeepGPR tools` (configuration in `pyproject.toml`).
- Type hints throughout; `mypy` configuration in `pyproject.toml`.
- Changes are recorded in [`CHANGELOG.md`](CHANGELOG.md); known issues and recommended native work in [`docs/AUDIT.md`](docs/AUDIT.md).
- Changes to `lib/deepgpr.h` must be mirrored in `src/DeepGPR/native/abi.py`; `tests/test_native_abi.py` enforces it.

## Citation

If you find our code useful, please cite:

```bibtex
@article{liu2026fast,
  title     = {Fast ground penetrating radar dual-parameter full waveform inversion method accelerated by hybrid compilation of CUDA kernel function and PyTorch},
  author    = {Liu, Lei and Song, Chao and He, Liangsheng and Wang, Silin and Feng, Xuan and Liu, Cai},
  journal   = {Computers \& Geosciences},
  pages     = {106101},
  year      = {2026},
  publisher = {Elsevier}
}
```

## License

MIT — see [`license`](license).
