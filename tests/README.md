# DeepGPR Tests and Verification

This directory contains the fast numerical unit tests and executable notebook
verification suite for the repository-local DeepGPR implementation. Every test
entry point places `../src` first on `sys.path`, verifies the local package and
native library where applicable, and requires native ABI 5. An installed
DeepGPR package is neither used nor required.

Run the fast terminal suite with:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

`test_discrete_adjoint.py` contains the strict reverse-mode checks for weighted
curl operators, native field updates, all-face CPML state transposes, source
waveform gradients, material Taylor tests, incomplete temporal sampling, and
optional CPU/CUDA parity.

No finite test suite can prove that a numerical program is correct for every
possible input. The suite instead combines independent checks that are
sensitive to different implementation errors:

1. `00_local_backend_and_contracts.ipynb`: local import, native ABI, exported
   backend and wavelets, deterministic smoke run, and rejected invalid inputs.
2. `01_forward_physics.ipynb`: zero response, source linearity, source
   superposition, reciprocity, travel time, and state-continuation equivalence.
3. `02_cpml_absorption.ipynb`: interior transparency, reflected-energy
   reduction, thickness sweep, and zero material gradient in CPML cells.
4. `03_gradient_2d.ipynb`: 2D Ez-TM adjoint directional derivatives for
   relative permittivity and conductivity at orders 2, 4, and 8.
5. `04_gradient_3d.ipynb`: 3D full-vector adjoint directional derivatives for
   all electric polarizations and spatial orders 2, 4, and 8.
6. `05_wavefield_storage.ipynb`: float32, float16, bfloat16, temporal sampling,
   and CUDA asynchronous offload comparisons.
7. `06_cpu_cuda_parity.ipynb`: CPU/CUDA forward and gradient parity. This
   notebook records a skip on machines without CUDA and runs automatically on
   CUDA servers.
8. `07_long_run_stability.ipynb`: long 2D and 3D forward/backward CPML stress
   tests for orders 2, 4, and 8.
9. `08_openmp_parallelism.ipynb`: compares multi-shot forward data and gradients
   from fresh processes using one, two, and four OpenMP threads.
10. `09_anisotropic_grid.ipynb`: verifies unequal constant `dx`, `dy`, and `dz`
    through API contracts, CFL limits, directional updates, CPML coefficients,
    2D and 3D propagation, adjoint gradients, and optional CPU/CUDA parity.
11. `99_verification_summary.ipynb`: validates and summarizes the JSON reports,
    rejecting reports produced by a different ABI, source tree, or native library.

Run the notebooks in numeric order from this directory. Each successful
notebook writes a JSON report to `tests/results`. The summary notebook
requires reports 00 through 09. On a CUDA server, set the target device before
starting Jupyter when GPU zero is not desired:

```bash
export DEEPGPR_TEST_CUDA_DEVICE=cuda:6
```

For release evidence that requires CUDA rather than allowing a documented skip,
also set:

```bash
export DEEPGPR_REQUIRE_CUDA=1
```

The native libraries in `src/DeepGPR/lib` must have been rebuilt from the same
source revision before the suite is used as release evidence. A source change
followed by tests against stale shared libraries is not a valid verification.

## CUDA performance benchmark

`benchmark_cuda.py` reports separate median forward and backward times using the
repository-local package. Its optional power sampler records board power, GPU
utilization, and SM clock through `nvidia-smi`:

```bash
python tests/benchmark_cuda.py --device cuda:0 --sample-power
```

Use identical arguments, power limit, application clocks, PyTorch version, and
native build when comparing revisions. Board power is diagnostic metadata, not
an optimization objective; memory-bound FDTD kernels may reach their best
runtime below the GPU's maximum power limit.
