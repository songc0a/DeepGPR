# DeepGPR history memory and CUDA adjoint performance report

Date: 2026-10-04 (Asia/Shanghai)

This report records one change per section. Every section states the
correctness evidence and the A/B measurement against the frozen baseline;
changes that failed their gate are recorded as such and not made default.

## 0. Baseline

### Revision, build and environment

| item | value |
|---|---|
| baseline commit | `ccf0e54` (`main`, clean tree) |
| A/B layout | baseline checked out as a detached worktree in `.worktrees/baseline` (ignored by `/.*/`) with its own freshly built `deepgpr.so` / `deepgpr_cpu.so`; every A/B runs the same benchmark script with `--src-root` pointing at either tree |
| GPU | NVIDIA GeForce RTX 4090 24 GiB, driver 575.51.03 (CUDA 12.9 driver API), power limit 450 W |
| toolchain | nvcc 12.4.131, gcc 11.4.0, Nsight Systems 2023.4.4 |
| Python / PyTorch | 3.10.16 / 1.11.0+cu115 (conda env `deepgpr`) |
| CUDA build | `nvcc -std=c++14 -O3 -lineinfo -Xptxas=-v -arch=sm_89 --shared -Xcompiler -fPIC` |
| CPU build | `cc -std=c99 -O3 -fopenmp -fPIC -shared` |
| GPU state before runs | only display clients (Xorg, gnome-shell, browser, QQ; ~1.4 GiB, P8 idle clocks); no compute processes. Recorded in every `nvidia_smi_before.txt`. Nsight Compute counters remain unavailable (`ERR_NVGPUCTRPERM`, unchanged). |

### Test baseline

`python -m unittest discover -s tests -p "test_*.py"` was run four times on
the rebuilt baseline libraries: 105 tests, **all passed** each time, 1 skipped
(`test_cuda_explicit_device_selects_matching_native_runtime`, needs two GPUs).
The INT8 directional-consistency failure and the PyTorch-1.11 `weights_only`
error recorded in `analysis_report.md` / `cuda_native_optimization_report.md`
did not reproduce on this revision, and no run-to-run bitwise flake occurred
in four runs.

### Benchmark protocol

`tests/benchmark_wavefield_compression.py` (extended in this commit with
`--src-root`, `--nz/--gradient-mode` for 3D, `--modes`, `--compute-kwarg`,
`--save-captures` and `--keep-allocator-cache`; history bytes are now read from
the autograd context instead of assuming `2 x E_saved`). CUDA Event timing,
5 warmups + 20 measured repeats, median reported. Every A/B uses at least two
interleaved batches per tree.

| case | grid | other |
|---|---|---|
| 2D | 512 x 384 physical (552 x 424 with the default 20-cell PML), nt = 1200, 4 shots, 64 receivers, order 2, `mode=2`, S = 1 | the existing formal case |
| 3D | 80 x 80 x 80 physical, PML 10 (100^3 cells), nt = 500, 1 shot, 16 receivers, order 2, `mode=3`, S = 1 | fp32 E+R history 11.2 GiB, fits a 24 GiB card |

**Allocation caveat.** The formal protocol calls `torch.cuda.empty_cache()`
around every run, so each run pays `cudaMalloc` for its histories inside the
CUDA-Event window. For the 3D fp32 case this dominates: forward 258 ms with the
formal protocol versus 84 ms with `--keep-allocator-cache` (FDTD alone 32 ms).
Both variants are reported below; the cached variant reflects a steady-state
FWI loop that reuses PyTorch's cached blocks, the formal variant also measures
the allocation cost that a smaller history saves.

### Baseline results (formal protocol, two batches)

Raw files: `history_adjoint_ab/baseline_b1`, `baseline_b2`.

| case | mode | forward ms (b1 / b2) | backward ms | total ms | peak MiB | history MiB |
|---|---|---:|---:|---:|---:|---:|
| 2D fwd-only | fdtd | 66.81 / 63.44 | - | - | 67.6 | 0 |
| 2D fwd-only | fp32 | 115.82 / 115.73 | - | - | 8639.6 | 8571.1 |
| 2D fwd-only | fp16 | 98.85 / 94.89 | - | - | 4353.2 | 4285.5 |
| 2D fwd-only | int8 | 122.07 / 115.90 | - | - | 2344.8 | 2276.7 |
| 2D full | fp32 | 97.93 / 96.20 | 137.44 / 132.59 | 234.93 / 229.27 | 8687.5 | 8571.1 |
| 2D full | fp16 | 98.87 / 98.40 | 134.03 / 133.30 | 232.99 / 232.22 | 4401.1 | 4285.5 |
| 2D full | bf16 | 99.28 / 98.03 | 133.52 / 129.28 | 232.92 / 226.51 | 4401.1 | 4285.5 |
| 2D full | int8 | 120.43 / 119.86 | 134.45 / 132.87 | 255.46 / 253.12 | 2392.6 | 2276.7 |
| 3D fwd-only | fdtd | 32.29 / 32.82 | - | - | 81.0 | 0 |
| 3D fwd-only | fp32 | 258.53 / 257.85 | - | - | 11525.1 | 11444.1 |
| 3D fwd-only | fp16 | 91.44 / 93.03 | - | - | 5805.0 | 5722.0 |
| 3D fwd-only | int8 | 118.49 / 120.16 | - | - | 3121.0 | 3039.8 |
| 3D full | fp32 | 249.41 / 249.22 | 80.29 / 82.06 | 329.40 / 330.66 | 11558.8 | 11444.1 |
| 3D full | fp16 | 91.45 / 96.11 | 70.93 / 75.82 | 162.75 / 171.74 | 5838.7 | 5722.0 |
| 3D full | bf16 | 91.36 / 92.19 | 70.64 / 71.21 | 162.34 / 163.50 | 5838.7 | 5722.0 |
| 3D full | int8 | 118.10 / 123.23 | 80.91 / 82.90 | 199.12 / 206.47 | 3154.7 | 3039.8 |

Same with `--keep-allocator-cache` (raw: `baseline_cached_b1`, `baseline_cached_b2`):

| case | mode | forward ms (b1 / b2) | backward ms | total ms |
|---|---|---:|---:|---:|
| 2D full | fp32 | 86.67 / 83.78 | 131.55 / 131.24 | 217.52 / 215.28 |
| 2D full | fp16 | 85.42 / 83.82 | 127.10 / 126.98 | 212.49 / 211.02 |
| 2D full | int8 | 105.99 / 105.03 | 129.51 / 128.62 | 235.53 / 233.85 |
| 3D full | fp32 | 85.05 / 83.79 | 82.26 / 81.16 | 168.36 / 164.94 |
| 3D full | fp16 | 75.66 / 75.82 | 72.43 / 73.01 | 147.53 / 148.62 |
| 3D full | int8 | 106.19 / 107.50 | 81.42 / 82.03 | 187.81 / 189.71 |

Batch-to-batch medians move by up to ~5 % (thermal/clock state) while the
within-batch standard deviation is about 1 ms, so A/B conclusions below are
drawn only from interleaved batches whose differences agree in sign.

Gradient accuracy of the baseline storage modes against FP32 (relative L2;
identical in both batches): 2D fp16 2.34e-5 / 2.21e-5 (ε / σ), bf16
1.59e-4 / 1.51e-4, int8 4.46e-4 / 3.50e-4; 3D fp16 3.10e-5 / 4.42e-4, bf16
2.57e-4 / 3.89e-3, int8 7.14e-3 / 3.20e-2.

## 1. Automatic gradient sampling interval (`model_gradient_sampling_interval="auto"`)

**Default:** unchanged (`1`). `"auto"` is opt-in.

Implementation (`solver/sampling.py`): `source_max_frequency` takes the
zero-padded (4x) amplitude spectrum of every source waveform and returns the
highest frequency at which it still reaches
`SOURCE_SPECTRUM_CUTOFF_FRACTION = 3e-3` of that waveform's peak (linear
interpolation between bins; largest value over waveforms).
`recommended_sampling_interval` returns `max(1, floor(1 / (4 f_max dt)))`
(1 for an all-zero source). For a Ricker wavelet
`|W(f)|/|W(f_p)| = (f/f_p)^2 exp(1 - (f/f_p)^2)` equals 3.0e-3 at `3 f_p`;
measured `f_max / f_p` = 3.004-3.009 for 100/200/400 MHz and nt = 500/2000.
The `examples/2.2DFWI.ipynb` source (200 MHz, dt = 5e-11) gives `S = 8`, the
benchmark source (400 MHz, dt = 1.5e-11) gives 13.

Warning policy: an explicit `S > 1` above the bound raises one
`RuntimeWarning` that names the bound; an explicit `S` inside the bound and the
automatic choice are logged at INFO. Previously every `S > 1` with a material
gradient warned.

Gradient error against `S = 1` (FP32, CPU, 90 x 60 two-layer model with a
buried anomaly, 3 shots, 90 surface receivers, 200 MHz Ricker, dt = 5e-11,
nt = 700; `.worktrees`-local script, numbers reproducible from the test case
below):

| loss | S=2 | S=5 | S=8 (auto) | S=10 | S=16 | S=20 |
|---|---:|---:|---:|---:|---:|---:|
| MSE ε | 4.5e-6 | 3.7e-5 | 2.4e-4 | 1.8e-4 | 3.0e-2 | 3.4e-2 |
| MSE σ | 3.5e-6 | 2.9e-5 | 1.7e-4 | 1.4e-4 | 2.3e-2 | 2.9e-2 |
| L1 ε | 3.6e-4 | 1.7e-3 | 3.5e-3 | 6.5e-3 | 0.18 | 0.16 |
| L1 σ | 4.5e-4 | 2.7e-3 | 4.9e-3 | 8.8e-3 | 0.13 | 0.17 |

At `S = 8` all of the error lies within six cells of an antenna (receivers
cover the whole surface row): beyond that band the relative error is 4.0e-7
(MSE) and 3.0e-5 (L1). This agrees with the reported 3D `mode=3` observation.
The jump between 10 and 16 matches the bound; the L1 rows show that the
non-smooth misfit's broadband adjoint source is not covered by a bound that
only sees the forward source spectrum. `docs/API.md` §4.4 documents both
tables (the 2.2DFWI numbers are the previously reported CPU measurements).

Tests (`tests/test_gradient_sampling.py`, 8 tests): Ricker `f_max ~= 3 f_p`
(±1 %), the cutoff constant matches the Ricker spectrum at `3 f_p`, `S = 8` for
the 2.2DFWI source, the most conservative interval for two waveforms, zero
source / invalid input handling, warning only above the bound (with the bound
in the message), INFO inside the bound, and the automatic gradient within
**2e-3** relative L2 of `S = 1` on a 48 x 36 two-layer CPU case (measured
4.1e-4, about 5x margin; `S = 16` measured 8.7e-2 on the same case). The
automatic and explicit `S = 8` gradients are bitwise identical.

Full suite after this item: 113 tests, all passed (1 skipped, two-GPU test);
validated on the staged snapshot with the baseline native libraries (item 1 is
Python-only).

## 2. E-only history at sampling interval 1 (`wavefield_rhs_history`)

**Default:** yes for uncompressed float32 histories (`"auto"`); **no** for
float16/bfloat16/INT8 (they keep E+R under `"auto"`, opt in with
`"reconstructed"`). `"stored"` restores the previous layout in every mode.

### Implementation

- Native flag `WAVEFIELD_RHS_FROM_E` (`storage_type` bit 4; the storage kind
  now uses bits 0-3, bits 4-7 are history flags) and capability probe
  `deepgpr_supports_rhs_reconstruction` in both backends (`deepgpr.h`,
  `native/abi.py`). No function signature changed, ABI stays 6. Python only
  sets the flag after the probe returns 1, so an older prebuilt library never
  sees it: `"auto"` falls back to E+R and `"reconstructed"` raises
  `NativeLibraryError`.
- Forward: with the flag, the R snapshot (`save_rhs_snapshot*`,
  `quantize_rhs_int8_snapshot_gpu`), the `exact_Eold` scratch and the async R
  staging buffers are not used. After the last step the final `E^nt` is written
  with the ordinary E snapshot/quantizer into the `R_saved` pointer, which now
  addresses an internal one-frame buffer (`ctx.E_final`, device-resident also
  with async offload). The returned `(Ex, Ey, Ez)` states are never read by the
  adjoint.
- Backward: the material-gradient kernels (CPU, CUDA float/fp16/bf16, CUDA
  INT8) now take frame pointers plus component strides for `E^n` and for a
  second frame (`R^n`, or `E^(n+1)` / the final frame). With the flag they
  rebuild `R = cb != 0 ? (E^(n+1) - ca E^n) / cb : 0` with the expression of
  the forward R snapshot, so float32 R is bit-identical (verified on CPU; on
  CUDA both expressions are contracted to the same FMA). Async offload uses two
  E slots: step `t` copies `E^t` into slot `t % 2` and reads `E^(t+1)` from the
  other slot (or the final frame), i.e. one H2D frame per step instead of E+R.
- Python: `wavefield_rhs_history` option; `SolverConfig.reconstruct_rhs`;
  `estimate_compute_memory(reconstruct_rhs=...)` with a `final_e_frame` entry
  (async: the final frame is device memory); `print_parameters` prints the
  R-history mode.

### Correctness

| check | result |
|---|---|
| CPU float32, mode 2 (2 shots) and mode 3 (asymmetric CPML), E-only vs E+R | receiver data, `E_saved`, ε/σ/source gradients **bitwise identical** (`test_rhs_history.py`) |
| stored `R^n` vs rebuild from `E_saved[t]`, `E_saved[t+1]` / final `Ez` (CPU) | **bitwise equal** for every saved step |
| returned states advanced in place (`mul_(-3).add_(1)`) before backward | gradients bitwise unchanged |
| CUDA float32 E-only vs E+R, mode 2/3, async on/off | data and `E_saved` bitwise; gradients 2-5e-7 relative (`< 2e-6` test bound), same as E+R run-to-run atomic noise |
| benchmark 2D fp32, baseline vs current | data bitwise; ε 1.3e-7, σ 1.3e-7 relative (baseline vs baseline: 2.0e-7 / 1.3e-7) |
| benchmark 3D fp32, baseline vs current | data bitwise; ε 2.2e-7, σ 7.9e-6 relative (baseline vs baseline: 2.8e-7 / 5.0e-6) |
| low-precision defaults (fp16/bf16/int8) | unchanged layout; gradients equal to baseline up to the same atomic noise |

### Low-precision accuracy: E-only vs E+R (relative L2 against float32)

The gate "not worse than baseline E+R" is **not met uniformly**, so `"auto"`
does not enable E-only for these modes:

| case | storage | ε: E+R → E-only | σ: E+R → E-only |
|---|---|---|---|
| 2D benchmark (512x384, 400 MHz, dt 1.5e-11, MSE) | fp16 | 2.34e-5 → 2.24e-5 | 2.21e-5 → 2.24e-5 |
| | bf16 | 1.59e-4 → **2.27e-4** | 1.51e-4 → 1.50e-4 |
| | int8 | 4.46e-4 → 4.70e-4 | 3.50e-4 → 3.43e-4 |
| 3D benchmark (80^3, mode 3, MSE) | fp16 | 3.09e-5 → 2.52e-5 | 4.41e-4 → 4.43e-4 |
| | bf16 | 2.57e-4 → 1.41e-4 | 3.889e-3 → 3.889e-3 |
| | int8 | 7.14e-3 → 6.31e-3 | 3.20e-2 → 3.25e-2 |
| 2D FWI-like (200x120, 200 MHz, dt 5e-11, 10 shots), MSE | fp16 | 1.61e-5 → 1.59e-5 | 1.60e-5 → 1.59e-5 |
| | bf16 | 1.39e-4 → 1.33e-4 | 1.18e-4 → 1.19e-4 |
| | int8 | 3.39e-4 → 3.27e-4 | 3.18e-4 → 3.19e-4 |
| same, L1 loss | fp16 | 1.97e-5 → **2.78e-5** | 1.85e-5 → 1.83e-5 |
| | bf16 | 1.40e-4 → **2.57e-4** | 1.50e-4 → 1.52e-4 |
| | int8 | 4.55e-4 → **6.55e-4** | 5.12e-4 → 5.16e-4 |
| small 2D test case (18x21, nt 60, CPU) | fp16 | 1.38e-4 → **2.08e-4** | 4.31e-5 → 4.30e-5 |
| | bf16 | 1.04e-3 → **2.08e-3** | 2.99e-4 → 3.00e-4 |
| small 3D test case (12x13x11, nt 60, CPU) | fp16 | 5.82e-5 → **1.13e-4** | 4.17e-5 → 4.27e-5 |
| | bf16 | 2.82e-4 → **9.99e-4** | 2.50e-4 → 2.55e-4 |
| same, CUDA | int8 (2D / 3D) | 4.75e-3 / 6.07e-3 → **1.16e-2 / 8.46e-3** | 1.23e-3 / 4.25e-3 → 1.25e-3 / 4.25e-3 |

Cause: the ε integrand is `-eps0 cb / dt (E^(n+1) - E^n)`. A stored `R^n`
carries a rounding error relative to `|E^(n+1) - ca E^n|`, whereas the
difference of two independently rounded E frames carries an error relative to
`|E|`. The σ integrand `-cb/2 (E^(n+1) + E^n)` is unaffected, as observed.
With MSE on realistic cases the two are comparable (the earlier CPU
observation); with an L1 misfit or short, finely sampled runs E-only is up to
3.5x worse in ε. All values stay in the same order of magnitude, so this is
listed as a decision for the maintainer rather than treated as a defect.

### Performance (formal protocol, two interleaved rounds, medians)

Raw: `history_adjoint_ab/item2_{base,cur}_b{1,2}` (and `_cached_`).

| case | metric | baseline (b1 / b2) | E-only (b1 / b2) | change |
|---|---|---:|---:|---:|
| 2D fwd-only fp32 | forward ms | 118.61 / 118.08 | 78.50 / 77.95 | -33.8 % / -34.0 % |
| 2D full fp32 | forward ms | 94.98 / 93.98 | 78.05 / 78.63 | -17.8 % / -16.3 % |
| | backward ms | 131.54 / 127.99 | 124.92 / 124.77 | -5.0 % / -2.5 % |
| | total ms | 225.97 / 222.45 | 203.29 / 203.37 | -10.0 % / -8.6 % |
| | history / peak MiB | 8571 / 8687 | 4289 / 4407 | -50.0 % / -49.3 % |
| 3D fwd-only fp32 | forward ms | 255.78 / 256.03 | 63.55 / 63.16 | -75.2 % / -75.3 % |
| 3D full fp32 | forward ms | 246.37 / 248.63 | 56.84 / 59.56 | -76.9 % / -76.0 % |
| | backward ms | 78.70 / 79.38 | 78.93 / 78.85 | +0.3 % / -0.7 % (noise) |
| | total ms | 324.72 / 327.82 | 135.85 / 138.54 | -58.2 % / -57.7 % |
| | history / peak MiB | 11444 / 11559 | 5733 / 5849 | -49.9 % / -49.4 % |

With `--keep-allocator-cache` (no `cudaMalloc` inside the timed window):
2D full fp32 total 209.58/223.16 → 194.30/205.59 ms (-7.3 %/-7.9 %), forward
-14.3 %/-17.6 %; 3D full fp32 forward 81.57/83.08 → 44.17/44.83 ms
(-45.9 %/-46.0 %), total -23.8 %/-22.7 %. The formal 3D numbers are therefore
mostly allocation savings (11.4 GiB → 5.7 GiB of `cudaMalloc` per call); the
cached numbers are the kernel/bandwidth part. fp16/bf16/int8 (unchanged
default layout) moved by -3.5 % to +2.1 % in the formal batches without a
consistent sign; the second cached round drifted by up to +6 % for every mode
including FDTD-only, i.e. clock/thermal state, not code.

Nsight Systems, 2D fp32 forward, kernel time per 1200-step run
(`history_adjoint_ab/nsys/item2_*_fp32_fwd_cuda_gpu_kern_sum.csv`):

| kernel | baseline ms (avg µs) | E-only ms (avg µs) |
|---|---:|---:|
| `save_rhs_snapshot_gpu<float32>` | 11.12 (9.27) | **removed** |
| `save_e_snapshot_gpu<float32>` | 7.37 (6.14), 1200 calls | 7.59 (6.32), 1201 calls (+ final frame) |
| `update_h_gpu<2>` / `update_e_gpu<2>` | 22.64 / 19.48 | 20.24 / 19.55 |
| `cpml_h_gpu<2>` / `cpml_e_gpu<2>` | 10.97 / 11.00 | 10.58 / 10.35 |
| sum of all kernels | 85.34 | 70.96 |

### Tests

New `tests/test_rhs_history.py` (12 tests: policy, memory estimate, stale
library rejection, CPU bitwise checks, rebuild identity, in-place state
mutation, low-precision/INT8 rebuild with and without async offload, CPU/CUDA
agreement). Changed: `test_numerics.test_cuda_async_memory_estimate_splits_host_and_device_payload`
now expects the host peak to exclude the device-resident final frame
(`saved_gradient_wavefields - final_e_frame`); exact equality is kept. Full
suite: 125 tests, all passed (1 skipped, two-GPU test).
