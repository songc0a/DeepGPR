# DeepGPR code audit (0.0.20 → 0.1.0)

This report lists the defects, design flaws and optimisation opportunities
found while reviewing the complete source tree at commit `73b2b50` (PyPI
0.0.20): the Python package (`__init__.py`, `common.py`, `compute2.py`,
`multiscale.py`, `wavelet.py`), the native backends (`lib/deepgpr_cpu.c`,
`lib/deepgpr.cu`, `lib/deepgpr.h`), packaging and CI.

Every item is classified as

- **Fixed** — changed in 0.1.0 without altering any value produced by a
  successful call (error handling, robustness, diagnostics, performance);
- **Fixed (numerical)** — corrected in 0.1.0 *with* a deliberate change of
  results; the previous behaviour stays available where users may need to
  reproduce earlier work (`legacy=True`, `method="legacy_isotropic"`);
- **Recommended** — an improvement left for a later release.

Severity: **High** = wrong results or crashes in plausible use; **Medium** =
wrong behaviour in less common use or silent degradation; **Low** = clarity,
robustness or hygiene.

## Summary

| ID | Severity | Area | Finding | Status |
|----|----------|------|---------|--------|
| P-1 | High | autograd | Backward overwrote caller-owned gradient tensors in place | Fixed |
| P-2 | Medium | autograd | Second backward crashed with `AttributeError` | Fixed |
| P-3 | Medium | native bridge | Race on the process-global FDTD order between threads | Fixed |
| P-4 | Low | native bridge | CUDA current device not restored when the native call raises | Fixed |
| P-5 | Low | loader | First library load not thread-safe | Fixed |
| P-6 | Medium | filters | `apply_filter` returned NaNs for `cutoff >= fs` | Fixed |
| P-7 | Low | regularisation | `TVRegularization()(None, None)` raised `AttributeError` | Fixed |
| P-8 | Low | I/O | Filename race when saving forward wavefields | Fixed |
| P-9 | Medium | packaging | Invalid requirement `torch+cu` shipped inside the package | Fixed |
| P-10 | Low | API | `import *` leaked `torch`, `nn`, `F`, `ctypes`, ... into `DeepGPR` | Fixed |
| P-11 | Low | diagnostics | `print()` in library code; "Tips" message on every FWI iteration | Fixed |
| P-12 | Low | validation | NumPy integer `pmlthick` rejected; `bool` sampling interval accepted | Fixed |
| P-13 | Low | packaging | Unused hard dependency `scipy`; undeclared `scikit-image` in examples | Fixed |
| P-14 | Medium | maintainability | Two hand-written 81/86-argument positional `ctypes` lists | Fixed |
| P-15 | Low | physics hygiene | σ > 100 S/m silently treated as PEC (no update, zero gradient) | Fixed (warning) |
| P-16 | Medium | autograd | Native solver mutates `E`/`H`/`PML` inputs without `mark_dirty` | Documented (behaviour kept) |
| P-17 | Low | filters | Even-length FIR: centre-tap override applied at the wrong index | Fixed (numerical) |
| P-18 | Low | envelope | Odd-length Hilbert transform: bin `nt//2` not doubled | Fixed (numerical) |
| P-19 | Low | regularisation | "Isotropic" TV is not the standard isotropic TV | Fixed (numerical) |
| P-20 | Low | constants | Python and native use different ε₀/µ₀ values | Fixed (numerical) |
| N-1 | Medium | CPU adjoint | Non-deterministic gradients with >1 OpenMP thread | Fixed (numerical) |
| N-2 | High | CUDA | Errors are printed to `stderr` and the call returns normally | Fixed |
| N-3 | Medium | native API | FDTD order passed through process-global state | Mitigated (P-3); ABI change recommended |
| N-4 | Low | CPU forward | `malloc` failure silently degrades fp16/bf16 R-history accuracy | Fixed |
| N-5 | Low | native | PEC threshold `100.0f` duplicated as a literal in 5 places | Fixed |
| N-6 | Perf | CPU kernels | Per-cell 64-bit division/modulo index decode in every update loop | Recommended |
| N-7 | Perf | CPU/CUDA | Atomic scatter in the adjoint curl transpose | Fixed on CPU; CUDA 2D TM path fixed, 3D kept (measured) |
| N-8 | Perf | CUDA | Streams/events/buffers created and destroyed on every call | Recommended |

## Python package

### P-1 Backward overwrote caller gradients (High, Fixed)
`compute2.DeepGPR.backward` called `.contiguous()` on each incoming state
cotangent and passed it to the native adjoint, which advances those arrays in
place. For an already-contiguous tensor `.contiguous()` returns the *same*
object, so the native code overwrote autograd's buffer — which can be a
tensor the user owns (`torch.autograd.backward(outputs, grad_outputs)`), a
retained `.grad`, or an argument seen by a tensor hook. **Fix:** the adjoint
now works on a private copy (`_own_state_gradient`). The returned gradients
are bitwise identical; only the side effect is gone.

### P-2 Second backward crashed (Medium, Fixed)
After the first backward the saved histories are released
(`ctx.E_saved = None`); a second backward (e.g. `retain_graph=True`) failed
with `AttributeError: 'NoneType' object has no attribute 'contiguous'`.
**Fix:** a `RuntimeError` explains that the forward must be re-run.

### P-3 FDTD-order race (Medium, Fixed)
`set_fdtd_order` writes a process-global variable that the solver reads when
it starts (`g_fdtd_order`, CPU and CUDA). Two Python threads using different
orders (e.g. multi-GPU work driven by threads) could interleave the setter and
the solve and silently run the wrong stencil. **Fix:** `native.bridge.FdtdOrderGate`
lets calls with the same order run concurrently and serialises only calls
with different orders. See also N-3.

### P-4 Device not restored on error (Low, Fixed)
`_begin_native_call` / `_end_native_call` were not in a `try/finally`; an
exception in between (e.g. a `ctypes.ArgumentError`) left a different CUDA
device current. Now a context manager (`native_stream_scope`).

### P-5 Library loading not thread-safe (Low, Fixed)
Two threads could load and configure the same library concurrently. A lock
now guards the one-time load.

### P-6 `apply_filter` NaNs for high cut-off (Medium, Fixed)
`numtaps = int(fs / cutoff)`; for `cutoff > fs/2` this is 0 or 1 and
`design_fir_filter` divides by `numtaps - 1 = 0`, producing NaN output
without any error. A filter longer than the trace also failed with an opaque
`reflect`-padding error. Both now raise a descriptive `ValueError`. Valid
inputs are unaffected (the caller's `fs`/`cutoff` objects are passed through
unchanged so tensor-valued arguments keep their exact rounding).

### P-7 TV with no inputs (Low, Fixed)
`TVRegularization.forward(None, None)` dereferenced `None.device`. Now a
`ValueError`.

### P-8 Wavefield filename race (Low, Fixed)
`_save_forward_wavefield` checked `exists()` and then wrote, so concurrent
runs in the same minute could overwrite each other. The file is now reserved
with exclusive creation (`open(path, "xb")`) using the same naming scheme,
and removed again if `torch.save` fails.

### P-9 Invalid requirement file in the wheel (Medium, Fixed)
`src/DeepGPR/requirements.txt` contained the single line `torch+cu` (not a
valid requirement) and was shipped in the sdist/wheel via `MANIFEST.in`.
Removed; root-level `requirements.txt` / `requirements-dev.txt` added.

### P-10 Namespace leakage (Low, Fixed)
`__init__.py` built `__all__` from `globals()` after `from .common import *`
etc., so `from DeepGPR import *` exported `torch`, `nn`, `F`, `math`,
`numbers`, `warnings`, `ctypes`, `datetime` and `Path`. `__all__` is now an
explicit list.

### P-11 `print()` in library code (Low, Fixed)
The "Tips: The number of source waveforms is 1 ..." message was printed on
*every* `compute` call with a shared waveform — i.e. every FWI iteration.
NaN/Inf diagnostics and the saved-wavefield path were also printed. All now go
through the `DeepGPR` logger (`DeepGPR.configure_logging(...)`).
`print_parameters=True` keeps printing its report to stdout unchanged.

### P-12 Validation gaps (Low, Fixed)
`pmlthick=np.int64(10)` raised `TypeError` (only `int` was accepted); it is
now accepted. `model_gradient_sampling_interval=True` was silently treated as
1; it is now rejected like other non-integers.

### P-13 Dependencies (Low, Fixed)
`scipy` was a required dependency but is never imported → optional extra.
The example notebooks import `scikit-image`, which was not declared → added
to the `examples`/dev extras.

### P-14 Positional `ctypes` lists (Medium, Fixed)
The forward/backward signatures (81 and 86 parameters) were written out by
hand twice — once as `argtypes` and once at each call site — with no link to
`deepgpr.h`. A transposition would shift every following argument without
any error. **Fix:** `native/abi.py` describes each exported function once by
parameter *name*; `argtypes` and call order are generated from it; call sites
pass a name→value mapping that must match exactly; `tests/test_native_abi.py`
parses `deepgpr.h` and fails on any difference.

### P-15 Silent PEC threshold (Low, Fixed with a warning)
Both native backends treat cells with σ > 100 S/m as perfect electric
conductors: the E update coefficients are zeroed and no material gradient is
accumulated. Nothing told the user. `initialization` now logs a warning (the
check rides on the existing single device→host transfer). Behaviour unchanged.

### P-16 In-place state mutation without `mark_dirty` (Medium, Documented)
The native solver advances the `E`, `H` and `PML` tensors passed to `compute`
in place, but the autograd function does not call `ctx.mark_dirty`, so
PyTorch's version counters are not bumped. Code that keeps a reference to an
input state and reads it after `compute` sees the advanced values with no
error. Adding `mark_dirty` would change autograd semantics (e.g. raise for
leaf tensors that require grad) and is therefore a behavioural change; the
documented rule "clone states you still need" (README, checkpoint example) is
kept and repeated in the `compute` docstring.

### P-17 Even-length FIR centre tap (Low, Fixed — numerical)
`design_fir_filter` overwrote `sinc[(numtaps-1)//2] = 2 fc/fs` to handle the
0/0 singularity at the centre. For an **even** number of taps there is no
centre sample (no singularity), so a regular tap was replaced by the peak
value, slightly distorting the filter. `apply_filter` produces even lengths
whenever `int(fs/cutoff)` is even. **Fix:** the override is applied only for
odd `numtaps`; `legacy=True` restores the old taps. Verified against an
independent NumPy windowed-sinc implementation.

### P-18 Hilbert envelope details (Low, Fixed — numerical)
`hilbert_transform` doubled bins `1 .. nt//2 - 1` and zeroed `nt//2+1 ..`.
For odd `nt`, bin `nt//2` is a positive frequency that must also be doubled
(a cosine at that frequency had an envelope of 0.5 instead of 1). **Fix:**
bins `1 .. ceil(nt/2) - 1` are doubled, as in `scipy.signal.hilbert`; even
lengths are unchanged. The DC bin is still removed by default, now explicit
as `remove_dc=True`. `legacy=True` restores the old behaviour.

### P-19 "Isotropic" TV (Low, Fixed — numerical)
For `method != "anisotropic"`, `TVRegularization` returned
`sqrt(Lx² + Ly² + Lz² + 1e-8)` where `Lx = Σ|∂x m|` is summed over the whole
model — not the standard isotropic TV. **Fix:** `method="isotropic"` (alias
`"iso"`) computes `Σ_cells sqrt(|∇m|² + ε)` with forward differences and a
zero difference across the far boundary; `ε` is configurable. The old formula
is `method="legacy_isotropic"`; unknown names raise `ValueError`. 4D input is
still treated as a batch of 2D models (documented).

### P-20 Inconsistent ε₀/µ₀ (Low, Fixed — numerical)
Python used µ₀ = 4π·10⁻⁷ and ε₀ = 1/(µ₀c²) (pre-2019 SI) for the CPML
profiles, while both native backends used CODATA-2018 values for the field
updates. **Fix:** `DEEPGPR_EPSILON0`, `DEEPGPR_MU0` (and the PEC threshold)
are defined once in `lib/deepgpr.h` and used by both backends; the Python
constants carry the same CODATA-2018 values and a test compares them with the
header. The relative change (~5·10⁻¹⁰) is below float32 resolution; no
forward output changed in the verification runs.

## Native backends

### N-1 Non-deterministic CPU adjoint (Medium, Fixed — numerical)
`add_staggered_backward_adjoint_cpu` / `add_staggered_forward_adjoint_cpu`
scatter into neighbouring cells with `#pragma omp atomic`; the summation order
depends on thread scheduling. Measured on the shipped `deepgpr_cpu.so`: two
identical backward runs with `OMP_NUM_THREADS=4` differ in `grad_eps_r`,
`grad_sigma`, `grad_source` and all 22 non-empty state cotangents (max relative
difference 5.7·10⁻⁸); with one thread they are bitwise identical. The CUDA
adjoint uses `atomicAdd` (14 call sites) with the same effect.
**Fix:** the transposed stencils are now evaluated in two passes — each
source point writes its weight per (target field, axis) channel, then each
target point gathers in a fixed order (`adjoint_e_step_cpu`,
`adjoint_h_step_cpu`). Source injection and receiver adjoints run shots in
parallel and sources/receivers in index order. There are no OpenMP atomics
left; results are bitwise identical for 1–8 threads, the dot-product test
confirms the exact transpose, and forward + backward is 17–43 % faster.
The gradients differ from 0.0.20 at rounding level only. Memory: six
field-sized buffers per shot during backward. The general CUDA adjoint still
uses `atomicAdd` (`deepgpr_deterministic_adjoint() == 0`); the CUDA 2D TM fast
path gathers without atomics since the unreleased N-7 work.

### N-2 CUDA errors are swallowed (High, Fixed)
`CUDA_CHECK` / `CUDA_CHECK_LAST` print to `std::cerr` and `return` from a
`void` function. Python cannot detect the failure: an out-of-memory
`cudaMalloc` or a failed kernel launch yields a normally returning call with
partially updated or stale outputs (stderr is often invisible in notebooks).
The Python `debug=True` all-zero-field check only catches the most extreme
case. **Fix (additive to ABI 6):** `CUDA_CHECK`/`CUDA_CHECK_LAST` and the INT8
parameter checks store the message in a per-thread buffer
(`deepgpr_last_error`, `deepgpr_clear_last_error`,
`deepgpr_supports_error_reporting`); `forward`/`backward` clear it on entry
and check `cudaGetLastError` once more before returning (no device
synchronisation). The Python bridge raises `NativeLibraryError` with the
message. Asynchronous kernel faults (e.g. an illegal address) still surface at
the next synchronising CUDA call, where PyTorch reports them.

### N-3 Process-global FDTD order (Medium, Mitigated)
`set_fdtd_order` + `g_fdtd_order` is hidden global state. The 0.1.0 bridge
serialises conflicting calls (P-3), but the clean fix is an explicit
`fdtd_order` parameter of `forward`/`backward` (ABI 7).

### N-4 CPU `malloc` failure (Low, Fixed)
`forward` allocates `exact_Eold` for fp16/bf16 histories without checking the
result; on failure the code silently falls back to reconstructing `R` from the
*rounded* stored `E`, i.e. a less accurate gradient with no warning. It now
reports the failure (and size overflows) through the error API.

### N-5 Duplicated PEC threshold (Low, Fixed)
`100.0f` appeared in `build_update_coeffs_cpu`, the CPU gradient kernel and
three CUDA sites. It is now `DEEPGPR_PEC_SIGMA_THRESHOLD_F` in `deepgpr.h`;
the test-suite checks it against `config.constants.PEC_CONDUCTIVITY_THRESHOLD`.

### N-6 CPU index decoding (Performance, Recommended)
Every CPU update loop (`update_e_cpu`, `update_h_cpu`, adjoints, snapshots)
iterates over a flattened index and recovers `(s, i, j, k)` with four 64-bit
`/` and `%` per cell per time step, and recomputes `ue1*dx/dy` per cell.
Nested loops with `collapse(2)` and hoisted ratios typically give a
measurable speed-up on the memory-bound CPU path.

### N-7 Atomic scatter in the adjoint (Performance, Fixed on CPU and CUDA)
Besides determinism (N-1), the atomic scatter of the transposed curl was the
main serialisation point of the CPU adjoint. The CPU gather formulation
removed it (see N-1 for timings).

**CUDA (unreleased):** an experiment that only skipped the four 2D
z-transpose `atomicAdd` per cell cut the 2D backward by 16 % (~18 µs/step),
confirming the atomics as the dominant adjoint cost. On the 2D TM fast path
the adjoint is now a gather: the CPML transposes first store their
derivative weights in phi-sized scratch (before overwriting the auxiliary
cotangent), then one kernel per half step gathers `c(src) * lambda(src)` plus
those weights onto each target point with the radius and coefficient of the
source; `lambda_E <- ce_hist lambda_E` and `lambda_H <- ch_hist lambda_H` are
deferred to the next kernel that writes the field. Receiver and source-waveform
adjoints add coincident points in a fixed order. This path is bitwise
reproducible and 2D backward is 30 % faster. A general 3D gather with the same
structure (several variants: phi-sized scratch, on-the-fly CPML weights,
per-component threads, interior fast path) was correct but 5-25 % slower than
the atomic scatter on the RTX 4090, so the general path keeps the scatter and
`deepgpr_deterministic_adjoint()` stays 0. Measurements:
`tests/profiling_results/history_adjoint_gpu_report.md` §5. The scatter path
now applies each cell's CPML transpose inside `adjoint_e_gpu` /
`adjoint_h_gpu` (§6), which removes two launches per reverse step but keeps
the atomics.

### N-8 Per-call CUDA resources (Performance, Recommended)
The async-offload path creates two streams, four events and staging buffers
on every `forward`/`backward` call and destroys them afterwards. Caching them
per device (or using PyTorch's caching allocator via pre-allocated tensors)
removes this fixed overhead from short FWI iterations.

## Developer-experience improvements made in 0.1.0

- Layered package (`config`, `native`, `preprocessing`, `solver`,
  `postprocessing`, `inversion`, `visualization`, `utils`) with docstrings and
  type hints throughout; `py.typed` marker.
- Frozen `SolverConfig` / `SolverTensors` dataclasses and a `PreparedModel`
  named tuple replace 77 positional arguments and a 16-element tuple.
- Structured logging and an exception hierarchy compatible with the previous
  built-in exception types.
- New helpers: `max_stable_time_step`, public `estimate_compute_memory`,
  `visualization.plot_*`.
- `tests/test_native_abi.py` runs without PyTorch or a GPU;
  `tools/verify_equivalence.py` compares any two revisions bitwise;
  `quality.yml` CI runs ruff, the ABI test, the CPU unit tests and the
  equivalence check.
