# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased] — History memory and CUDA adjoint performance

Measurements, protocol and per-item decisions:
[`tests/profiling_results/history_adjoint_gpu_report.md`](tests/profiling_results/history_adjoint_gpu_report.md).

### Added

- `model_gradient_sampling_interval="auto"` selects
  `S = max(1, floor(1 / (4 f_max dt)))`, with `f_max` the highest frequency at
  which the source amplitude spectrum exceeds `3e-3` of its peak (about
  `3 f_peak` for a Ricker wavelet; several waveforms use the smallest `S`).
  `DeepGPR.recommended_sampling_interval` and `DeepGPR.source_max_frequency`
  expose the bound so segmented/checkpointed runs can evaluate it once on the
  full waveform. The default stays `1`.
- `wavefield_rhs_history` (`"auto"`, `"stored"`, `"reconstructed"`). With
  sampling interval 1 the adjoint rebuilds `R^n = (E^(n+1) - ca E^n) / cb` from
  consecutive saved E frames, so only E is stored (plus one internal final
  frame). The default `"auto"` does this for uncompressed `float32` histories:
  history memory halves, the forward no longer runs the R-snapshot kernel, and
  gradients are unchanged (bitwise on CPU). Low-precision and INT8 histories
  keep E+R by default (rebuilding from two rounded frames was measurably less
  accurate in some cases) and can opt in with `"reconstructed"`. New native
  capability probe `deepgpr_supports_rhs_reconstruction` (ABI 6 unchanged);
  older libraries fall back to E+R.
- `wavefield_history_region` (`"extended"` default, `"physical"`). `"physical"`
  stores E/R histories only on the physical model cells (CPML cells never get
  a material gradient), so history memory falls with the cell count and
  gradients are unchanged. INT8 histories widen the box to whole tiles of the
  full-grid tiling so tiles and scales are identical to `"extended"`. New probe
  `deepgpr_supports_physical_history`; saved files record the region.

### Performance

- CUDA full-grid kernels (E/H update and adjoint) take the shot from
  `blockIdx.y` and decode `(i, j, k)` from a 32-bit per-shot index with a
  multiply-high divider instead of four 64-bit `/` and `%` per thread. Grids
  above `2^31 - 1` cells per shot are rejected explicitly.
- CUDA 2D TM fast path: for 2D models with an Ez source and zero Ex/Ey/Hz
  states the solver launches only the `k = 0` layer and updates/transposes
  only Ez, Hx and Hy (probe `deepgpr_supports_tm2d_fast_path`). Receiver data
  is bitwise unchanged; initial-state gradients of Ex/Ey/Hz are zero on this
  path. Other calls use the general kernels.

- CUDA 2D TM adjoint as a gather (AUDIT N-7): on the 2D TM fast path the
  transposed curl and CPML updates no longer scatter with `atomicAdd`; every
  target point gathers its source stencil in a fixed order with CPML
  derivative weights from phi-sized scratch (2D backward about 30 % faster,
  bitwise reproducible). Receiver and source-waveform adjoints add coincident
  points in a fixed order on every path. The general (3D) CUDA adjoint keeps
  the atomic scatter, which measured faster than a gather there;
  `deepgpr_deterministic_adjoint()` stays 0. Gradients differ from the
  previous release at rounding level only.
- CUDA kernel fusion on the 2D TM fast path: each cell's CPML corrections run
  inside `update_e_gpu` / `update_h_gpu` right after its base update (same
  face order and arithmetic; the other field is read only), and float32 /
  float16 / bfloat16 E histories are written by `update_h_gpu`, where E is not
  modified. Forward launches per time step drop from 6 to 3 (2D forward
  11-18 % faster; with INT8 histories 6-10 %). INT8 and `native_vec2`
  histories and the stored-R snapshot keep their own kernels. The general
  (3D) forward keeps separate CPML and snapshot kernels: fused, it was faster
  on small 3D grids but up to 8 % slower on a 140^3 grid. The general scatter
  adjoint applies the CPML transposes inside `adjoint_e_gpu` /
  `adjoint_h_gpu` (6 to 4 launches per reverse step; 3D fp32 backward about
  3.5 % faster, other storage modes within noise). Receiver data, states and
  histories are bitwise unchanged; 2D TM gradients are bitwise unchanged, 3D
  gradients change at atomic-ordering level only. The material-gradient
  kernel stays separate (it reduces over shots per cell in a fixed order and
  runs only every `S` steps).

### Changed

- An explicit `model_gradient_sampling_interval > 1` no longer always raises a
  `RuntimeWarning`; it warns (and states the bound) only above
  `floor(1 / (4 f_max dt))`. Intervals inside the bound are logged at INFO.
- **Default history layout:** float32 histories at sampling interval 1 no
  longer allocate `R_saved`; pass `wavefield_rhs_history="stored"` to restore
  the previous layout. `estimate_compute_memory` gained `reconstruct_rhs` and a
  `final_e_frame` entry; `print_parameters` reports the R-history mode.
- The native `storage_type` argument reserves bits 4-7 for history flags (the
  storage kind now uses bits 0-3).

## [0.1.0] — Structural refactor and numerical fixes

This release reorganises the Python package into a layered, typed and
documented code base and fixes the defects found in the review
([`docs/AUDIT.md`](docs/AUDIT.md)).

**What changes numerically** (all deliberate, see [Numerical fixes](#numerical-fixes)):

| Area | Change | Size |
|---|---|---|
| Forward modelling | none, except ε0/µ0 in the CPML profiles now CODATA 2018 | ~5·10⁻¹⁰ relative in the coefficients; no forward output changed in the verification runs |
| CPU gradients | fixed summation order (deterministic) instead of OpenMP atomics | rounding level (≤ ~7·10⁻⁶ relative on random test problems, same size as the old run-to-run noise) |
| `design_fir_filter` / `apply_filter` | correct taps for an even filter length | filtered data changes when `int(fs/cutoff)` is even |
| `hilbert_transform` | correct analytic signal for an odd trace length | envelopes change for odd `nt`; even `nt` unchanged |
| `TVRegularization(method="isotropic")` | standard isotropic TV | different penalty; `"anisotropic"` (default) unchanged |

`legacy=True` (filters, envelope) and `method="legacy_isotropic"` (TV)
reproduce DeepGPR 0.0.20 bitwise. The refactor itself was first verified to be
bitwise identical to 0.0.20 before any of these fixes were applied — see
[Verification](#verification).

### Architecture

| Before | After | Responsibility |
|---|---|---|
| `__init__.py` (library loading, 81/86-argument `argtypes` lists) | `native/loader.py`, `native/abi.py`, `native/bridge.py` | locate/load/configure the CPU and CUDA libraries; one declarative ABI table; argument marshalling |
| `common.py` (1026 lines, mixed concerns) | `preprocessing/{grid,stability,model_setup}.py`, `solver/{pml,fields}.py`, `inversion/regularization.py`, `utils/tensor_checks.py` | validation & PML extension, CFL, CPML, field state, TV, diagnostics |
| `compute2.py` (1585 lines) | `solver/{modeling,autograd,storage,memory}.py`, `postprocessing/io.py` | `compute()`, autograd bridge, history storage & INT8, memory report, file output |
| `wavelet.py` | `preprocessing/wavelets.py` | source wavelets |
| `multiscale.py` | `postprocessing/filters.py` | FIR low-pass, envelope |
| inline literals | `config/constants.py`, `config/defaults.py` | physical constants, ABI codes, defaults |
| — | `utils/{logger,exceptions,validators,formatting,devices}.py` | logging, errors, validation |
| — | `visualization/plotting.py` | model / radargram / trace plots |

`DeepGPR.common`, `DeepGPR.compute2`, `DeepGPR.multiscale` and `DeepGPR.wavelet`
remain importable as thin compatibility modules, and every name previously
reachable as `DeepGPR.<name>` for the public API is still exported.

### Key refactoring decisions

- **Native changes kept minimal.** The FDTD, CPML and material-gradient
  kernels are unchanged. The native sources changed only for the fixes below:
  error reporting (CPU and CUDA), shared constants in `deepgpr.h`, and the
  deterministic CPU adjoint. The C ABI stays at version 6; new functions are
  additive and probed at run time, so binaries built from older sources still
  load (without the new guarantees).
- **One source of truth for the C ABI.** `native/abi.py` lists every exported
  function once, parameter by parameter, mirroring `deepgpr.h`. `argtypes`
  and the call-argument order are both generated from it, replacing two
  hand-maintained 80+-item positional lists. `tests/test_native_abi.py` parses
  the header and fails on any drift (runs without PyTorch).
- **Named native arguments.** `solver/autograd.py` builds a
  `{c_parameter_name: value}` mapping (`build_forward_arguments`,
  `build_backward_arguments`); a missing or misspelled parameter raises
  instead of silently shifting every following argument.
- **Typed solver state.** The 77 positional arguments of the autograd
  `Function` became a frozen `SolverConfig` dataclass, a `SolverTensors`
  dataclass and the 30 differentiable state tensors.
- **Named tuple for model preparation.** `initialization()` returns a
  `PreparedModel` `NamedTuple`: identical tuple behaviour, readable fields.
- **Constants, not magic numbers.** Physical constants, CPML profile
  parameters, stencil coefficient sums, ABI bit fields, INT8 limits, memory
  margins, Hamming coefficients and API defaults live in `config/`.
- **Duplicate code removed.** Six copy-pasted CPML face blocks, six
  CPML-phi allocation blocks, 30 `.contiguous()` lines, two 60-line `ctypes`
  call sites and the per-face capability checks were replaced by loops/tables.

### Numerical fixes

- **Deterministic CPU adjoint.** The CPU backward scattered the transposed
  curl stencils with `#pragma omp atomic`, so gradients differed in the last
  bits from run to run whenever `OMP_NUM_THREADS > 1`. It now runs in two
  passes: every source point stores its stencil weight per (target field,
  axis) channel, then every target point gathers the transposed stencil in a
  fixed order. No OpenMP atomics remain in `deepgpr_cpu.c`; source injection
  and receiver adjoints run shots in parallel and sources/receivers
  sequentially. Results are **bitwise identical for any thread count**, and
  the backend reports this through `deepgpr_deterministic_adjoint()`. The
  gradient is still the exact discrete transpose (dot-product test below).
  Cost: six field-sized buffers per shot during backward (included in the
  memory estimate as "CPU adjoint stencil weights"). The CUDA adjoint still
  uses `atomicAdd` and is not bitwise reproducible.
- **Native errors reach Python.** CUDA API and launch errors used to be
  printed to stderr while the call returned normally. Both backends now keep a
  per-thread error message (`deepgpr_last_error`, `deepgpr_clear_last_error`,
  `deepgpr_supports_error_reporting`); the bridge clears it before every native
  call and raises `NativeLibraryError` with the message afterwards. The CPU
  backend also reports allocation failures (it used to fall back silently to
  a less accurate R history) and size overflows. INT8 parameter errors in the
  CUDA backend are reported the same way.
- **One set of physical constants.** ε0, µ0 and the 100 S/m PEC threshold are
  defined once in `lib/deepgpr.h` (CODATA 2018) and used by the CPU and CUDA
  backends; `DeepGPR.config.constants` uses the same values (previously
  µ0 = 4π·10⁻⁷ in Python). `tests/test_native_abi.py` fails if they drift.
- **FIR design.** `design_fir_filter` replaced tap `(numtaps-1)//2` with the
  sinc peak `2 fc/fs` also for an even number of taps, where no sample lies
  at the centre. The override is now applied only for odd lengths.
- **Analytic-signal envelope.** For an odd trace length `hilbert_transform`
  did not double bin `nt//2`, which is a positive frequency there. The
  highest positive bin is now doubled (as `scipy.signal.hilbert`). Removing
  the DC bin stays the default (`remove_dc=True`).
- **Isotropic TV.** `method="isotropic"` (or `"iso"`) now computes the
  standard `Σ_cells sqrt(|∇m|² + ε)` with forward differences and a Neumann
  far boundary; the smoothing `epsilon` is a constructor argument. The old
  formula `sqrt(Lx² + Ly² + Lz² + ε)` is available as
  `method="legacy_isotropic"`. Unknown method names now raise `ValueError`
  (they used to select the old formula silently).

**Prebuilt binaries.** The Linux CPU library in `src/DeepGPR/lib` is built
from the 0.1.0 sources. The CUDA (`deepgpr.so`, `deepgpr.dll`) and the Windows /
macOS CPU binaries are still the 0.0.20 builds until the *Build Native
Libraries* workflow runs; they load and compute as before but without error
reporting (and, for the CPU ones, without the deterministic adjoint).
`tools/run_local_validation.sh` rebuilds the CPU and CUDA libraries locally
and runs the full validation.

### Fixed

- **Caller gradients were overwritten.** The native adjoint advances the
  incoming state cotangents in place; the old code passed autograd's buffers
  straight through (`.contiguous()` is a no-op on contiguous tensors), so
  user-supplied `grad_outputs`, retained gradients and hook arguments were
  silently modified. They are now copied first (only cotangents that autograd
  actually supplies; absent ones are allocated as zeros, as before). Returned
  gradients are unchanged; the cost is one transient copy of the E/H/CPML state
  size during backward, small next to the saved wavefield history.
- **Second backward crashed** with `AttributeError: 'NoneType' object has no
  attribute 'contiguous'` (e.g. `retain_graph=True`). It now raises a clear
  `RuntimeError` explaining that the history was released.
- **FDTD-order race.** `set_fdtd_order` sets process-global native state; two
  threads using different orders could run with the wrong stencil. Calls now
  go through an `FdtdOrderGate`: same-order calls stay concurrent, different
  orders are serialised.
- **CUDA device not restored on error.** Stream/device bookkeeping around the
  native call is now a `try/finally` context manager.
- **Library loading was not thread-safe** on first use; it is now locked.
- **`apply_filter` returned NaNs silently** when `cutoff >= fs` (a 0- or
  1-tap FIR divides by zero); it now raises `ValueError`, as it does when the
  filter is longer than the trace.
- **`TVRegularization()(None, None)`** raised `AttributeError`; now `ValueError`.
- **Forward-wavefield file naming race.** Two processes saving in the same
  minute could pick the same filename; names are now reserved atomically
  (same `forward_wavefield_HH-MM[_NN].pt` scheme). Partially written files are
  removed on failure.
- **Invalid requirement shipped in the wheel.** `src/DeepGPR/requirements.txt`
  contained `torch+cu` and was packaged via `MANIFEST.in`; removed.
- Error message of `apply_filter` used full-width punctuation (`。`).

### Changed

- `print()` diagnostics replaced by the `DeepGPR` logger: the waveform
  broadcast "Tips" message (was printed on every call) is `INFO`, the saved
  wavefield path is `INFO`, NaN/Inf reports are `ERROR`. Call
  `DeepGPR.configure_logging("INFO")` to see them. `print_parameters=True`
  still prints its report to stdout, byte-for-byte as before.
- Errors raised by DeepGPR now use a small hierarchy
  (`DeepGPRError`, `NativeLibraryError`, `NativeLibraryNotFoundError`,
  `CFLConditionError`, `NonFiniteTensorError`). Each class also derives from
  the built-in type raised before (`RuntimeError`, `FileNotFoundError`,
  `ValueError`, `FloatingPointError`), so existing `except` clauses still match.
  All messages are unchanged.
- `pmlthick` accepts NumPy integer scalars (`np.int64(10)`), which previously
  raised `TypeError`.
- `model_gradient_sampling_interval=True` is rejected (it was silently
  treated as `1`).
- A warning is logged when conductivity exceeds 100 S/m: the native solvers
  treat such cells as perfect conductors (no update, no gradient).
- `scipy` moved from a required to an optional (`examples`) dependency; it was
  never imported.
- Version is single-sourced from `DeepGPR/_version.py`; `py.typed` marker added.

### Performance

The deterministic CPU adjoint is also faster, because it replaces contended
atomic updates with plain gathers. Forward + backward wall time, prebuilt
0.0.20 library vs 0.1.0 (2-core test machine, 150 time steps, 10-cell CPML):

| Problem | 1 thread | 2 threads |
|---|---|---|
| 2D 320×320, order 2 | 3.81 s → 3.16 s | 2.30 s → 1.70 s |
| 2D 320×320, order 8 | 6.50 s → 5.01 s | 3.79 s → 2.81 s |
| 3D 68³, order 2 | 19.41 s → 13.26 s | 11.91 s → 8.00 s |
| 3D 68³, order 4 | 35.93 s → 20.54 s | 22.17 s → 14.16 s |

The CUDA kernels are unchanged. Python-side overhead was reduced where it was
measurable:

- `debug=True` NaN/Inf validation scans each tensor once (`isfinite`) instead
  of twice (`isnan` + `isinf`), halving device→host synchronisations; the
  detailed breakdown is only computed for failing tensors.
- `checkpoint_initial_field` no longer computes (and discards) the full CPML
  coefficient profiles; it only needs the face geometry.
- The memory estimate uses `dtype.itemsize` instead of allocating three
  temporary tensors per call.
- All validation scalars in `initialization` are still gathered in a single
  device→host transfer (the new PEC check is part of the same `torch.stack`).

### Added

- `DeepGPR.max_stable_time_step(dx, nx, ny, nz, fdtd_order=, min_er_mr=)`.
- `DeepGPR.estimate_compute_memory(...)` (previously private).
- `DeepGPR.visualization`: `plot_model`, `plot_bscan`, `plot_trace`,
  `plot_model_comparison`.
- `DeepGPR.configure_logging()` / `DeepGPR.get_logger()`.
- `DeepGPR.__version__`.
- `tests/test_native_abi.py` (header ↔ ABI table and constants, marshalling,
  order gate, native error reporting, library configuration; no PyTorch
  required).
- `tests/test_signal_and_regularization.py` (FIR, envelope and TV against
  independent NumPy implementations; legacy switches).
- `tests/test_cpu_determinism.py` (bitwise equality for 1, 2 and 5 threads).
- `tools/verify_equivalence.py` (comparison with a reference release: forward
  and gradient tolerances, legacy reproduction, determinism, error reporting).
- `legacy=` keyword for `design_fir_filter`, `apply_filter`,
  `hilbert_transform`; `remove_dc=` for `hilbert_transform`; `epsilon=` and
  `method="legacy_isotropic"` for `TVRegularization`.
- Native functions `deepgpr_supports_error_reporting`, `deepgpr_last_error`,
  `deepgpr_clear_last_error`, `deepgpr_deterministic_adjoint` (ABI 6, additive).
- `.github/workflows/quality.yml` (ruff, ABI test, CPU unit tests, equivalence).
- `requirements.txt`, `requirements-dev.txt`, ruff/mypy/pytest configuration.

### Removed / internal changes

- `from DeepGPR import *` no longer leaks `torch`, `nn`, `F`, `math`, `ctypes`,
  `datetime`, `Path`, `warnings` and `numbers` (they were accidentally part of
  `__all__`).
- The autograd `Function` `DeepGPR.DeepGPR` has a new internal call signature
  (`apply(config, tensors, eps_r, sigma, source, *states)`). It was never a
  documented entry point; use `DeepGPR.compute`.
- Monkey-patching `DeepGPR.compute2.<name>` no longer affects `compute()`;
  patch `DeepGPR.solver.modeling` / `DeepGPR.solver.autograd` (the test-suite
  was updated accordingly).

### Verification

**Numerical fixes** (CPU library built from the 0.1.0 sources):

| Check | Scope | Result |
|---|---|---|
| Thread determinism | 21 native cases (2D/3D, order 2/4/8, fp32/fp16/bf16, coincident sources and receivers), `OMP_NUM_THREADS` = 1, 2, 3, 4, 7, 8; plus 144 full `compute` + backward cases at 1 vs 2 threads | **bitwise identical** for every thread count (the 0.0.20 library differed in 662 of 1,386 buffers between 1 and 4 threads) |
| Exact transpose | adjoint dot-product test `<A x, y> = <x, Aᵀ y>` for the full state/receiver map, 12 configurations (2D/3D, order 2/4/8, all-face and partial CPML) | relative mismatch ≤ 8.3·10⁻⁶ (0.0.20: ≤ 1.2·10⁻⁵), i.e. float32 round-off |
| Agreement with 0.0.20 | same 144 end-to-end cases | forward outputs bitwise identical; gradients within 7·10⁻⁶ relative |
| Error reporting | failing native backward through the Python bridge | `NativeLibraryError` with the native message; message cleared afterwards |

**Refactor** (before the numerical fixes), against commit `73b2b50`:

| Check | Scope | Result |
|---|---|---|
| End-to-end `compute` + adjoint, real CPU library | 155 cases: 2D/3D × order 2/4/8 × fp32/fp16/bf16 history × sampling 1/2 × int/list PML × shared/per-source waveforms, plus `mu_r`, forward-only, checkpoint continuation (2D/3D), 2D mode 3, no/one-sided PML, float64 inputs, 2D waveform arrays, single shot | 10,065 outputs & gradients **bitwise identical**; `print_parameters` report identical |
| Native marshalling (original `ctypes` binding vs ABI table) | 56 cases incl. no-PML, partial PML, all storage types | 3,808 buffers **bitwise identical**; injected argument swaps detected |
| Python numerics | CPML coefficients/descriptors, CPML state allocation & validation, CFL, PML parsing, wavelets, FIR design, INT8 layout/encoding, storage codes, memory estimate, TV, `initialization`, `checkpoint_initial_field` (incl. error types & messages) | 52,910 comparisons identical |
| ABI table vs `deepgpr.h` | all 9 exported functions | identical names, kinds and order |
| `compute()` option resolution (incl. CUDA-only paths) | 25,613 combinations: CPU/CUDA × 2D/3D × history dtype × compression × INT8 block × conversion backend × INT8 reduction × async offload × history on/off, plus argument validation | identical exceptions and messages; the 544 accepted combinations resolve to identical native storage settings |

The end-to-end and numerics checks above used a NumPy implementation of the
PyTorch tensor API. Both package versions used this compatibility layer and
the same native library, with `OMP_NUM_THREADS=1`.

**Real PyTorch and CUDA** (PyTorch 1.11 / CUDA 11.5 on an RTX 4090, CPU and
CUDA libraries rebuilt from the 0.1.0 sources, reference = commit `73b2b50`,
which is byte-identical to the 0.0.21 PyPI release):

| Check | Scope | Result |
|---|---|---|
| `tools/verify_equivalence.py` | CPU and CUDA: 2D/3D × order 2/4/8 × fp32/fp16/bf16 × sampling 1/3, no/asymmetric PML, forward-only, mode 3, checkpoint segments; CUDA INT8 (2D/3D), async offload, fp16 conversion backends; legacy switches; error reporting; CPU determinism for 1/2/4/8 threads | all passed; forward outputs **bitwise identical**, gradients within 1.9·10⁻⁶ (CPU) / 4.7·10⁻⁷ (CUDA) relative |
| Unit tests | 105 tests, `OMP_NUM_THREADS` = 1, 2, 4, 8 | all passed (1 skip: needs two GPUs) |
| Verification notebooks `tests/00`–`09`, `99` | `DEEPGPR_REQUIRE_CUDA=1` | 108 checks passed, CUDA parity evidence present |
| CI simulation | Python 3.12, PyTorch 2.x CPU: unit tests and `verify_equivalence.py --device cpu --quick`; `ruff check src/DeepGPR tools` | all passed |

The isotropic-TV comparisons use a float32 tolerance because
`TVRegularization` accumulates its penalty in float32, as in 0.0.20.
The checkpoint-vs-full-history gradient check in
`test_external_pml.py` scales its absolute tolerance with the gradient
magnitude, because the deterministic CPU adjoint sums in a different (fixed)
order than the OpenMP-atomic 0.0.20 adjoint.

## [0.0.20] and earlier

See the Git history.
