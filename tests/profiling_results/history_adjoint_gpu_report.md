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

## 3. Physical-region history (`wavefield_history_region`)

**Default:** no (`"extended"` stays the default as requested); `"physical"` is
opt-in.

### Implementation

- Native flag `WAVEFIELD_PHYSICAL_HISTORY` (`storage_type` bit 5) and probe
  `deepgpr_supports_physical_history` (CPU and CUDA). All snapshot kernels
  (E, R, vec2 variants, INT8 quantizers, the E-only final frame) and all
  material-gradient kernels take a `HistoryBox` (origin and extent in compact
  cell coordinates). The gradient kernels now launch over the history box and
  map back to the extended material/gradient index; the CPML skip test is kept
  for the extended box. Staging buffers, `exact_Eold`, async transfers and the
  INT8 scale layout all use the box size.
- INT8: tiles are formed on the box. A box starting at the physical origin
  changes which cells share a tile (and its scale). That changed the 3D
  benchmark INT8 gradient error from 7.1e-3 / 3.2e-2 to **7.6e-2 / 0.37**
  (ε / σ, cosine 0.93), i.e. not the same order of magnitude. A PML sweep
  showed this is the existing INT8 sensitivity to tile alignment, not a defect
  of the physical box: with `pml % 4 == 0` the extended history has the same
  tiling and the same 3.4e-2 / 0.21 error, with other offsets 4e-3 to 7e-3
  (3D 32^3, source/receivers two to five cells below the top face). The
  physical INT8 box is therefore widened to whole tiles of the full-grid
  tiling (origin rounded down, end rounded up and clamped), so every tile and
  scale is identical to the extended history; decoded physical values are
  bitwise equal and gradients unchanged. Non-INT8 storage uses the exact
  physical box.
- Python: option, `SolverConfig.history_region`, `storage.history_box` /
  `history_spatial_shape` (mirror of `history_box_host`), memory estimate
  (`history_region`, `history_shape` entry), `print_parameters` line, saved file
  metadata (`history_region`, `history_origin`, `pmlthick`,
  `uncompressed_shape`; extended histories keep the old format).

### Correctness

| check | result |
|---|---|
| CPU fp32 (E-only and E+R), fp16, bf16 (E-only), 2D two shots / 3D, asymmetric CPML on every face | receiver data and ε/σ gradients **bitwise identical** to `"extended"`; physical `E_saved` equals the cropped extended `E_saved` bitwise |
| CUDA same set, with and without async offload | data and `E_saved` crop bitwise; gradients within atomic noise (< 2e-6 relative bound; measured 1-5e-7) |
| CUDA INT8, 2D/3D, E+R and E-only | decoded physical history equals the extended decode cropped to the box bitwise; gradients within atomic noise |
| benchmark, physical vs baseline | 2D/3D receiver data bitwise for every mode; gradients 1.1-4.3e-7 (ε) and 1.3e-7 / 0.7-1.3e-5 (σ, 2D / 3D) relative, the same as baseline run-to-run noise |
| history memory | 2D fp32 8571 → 3603 MiB (E-only x 512·384/552·424 = 0.420), fp16 4286 → 3600 (0.840), int8 2277 → 1983 (0.871, tile-widened box 520x392); 3D fp32 11444 → 2936 (0.257), fp16 5722 → 2930 (0.512), int8 3040 → 1802 (0.593, box 84^3) |

### Performance (formal protocol, current tree with `wavefield_history_region="physical"` vs baseline, two interleaved rounds)

Raw: `history_adjoint_ab/item3_*`. Changes are relative to the baseline, so
the fp32 rows include item 2.

| case | mode | total ms base → physical (b1; b2) | change |
|---|---|---|---:|
| 2D full | fp32 | 237.09 → 211.55; 237.32 → 214.72 | -10.8 % / -9.5 % |
| 2D full | fp16 | 231.40 → 215.08; 230.37 → 224.11 | -7.1 % / -2.7 % |
| 2D full | bf16 | 231.76 → 222.46; 229.71 → 225.57 | -4.0 % / -1.8 % |
| 2D full | int8 | 252.47 → 240.10; 254.90 → 254.05 | -4.9 % / -0.3 % |
| 3D full | fp32 | 335.18 → 132.70; 339.41 → 133.51 | -60.4 % / -60.7 % |
| 3D full | fp16 | 170.57 → 145.50; 170.38 → 142.80 | -14.7 % / -16.2 % |
| 3D full | bf16 | 169.78 → 146.42; 170.82 → 143.16 | -13.8 % / -16.2 % |
| 3D full | int8 | 208.07 → 167.62; 209.08 → 162.18 | -19.4 % / -22.4 % |

The 3D forward gains (fp16 -27 %, int8 -30 %) come mostly from the smaller
history allocation and fewer snapshot/quantize cells; backward changes are
within ±3 % except 3D INT8 (-3 %/-8 %), consistent with the gradient kernel
covering 84^3 instead of 100^3 tiles.

### Tests

New `tests/test_history_region.py` (7 tests). Full suite: 132 tests, all
passed (1 skipped).

## 4. CUDA: index decode and 2D TM fast path

**Default:** yes (both are internal; the fast path is selected automatically
when it reproduces the general solver).

### 4a. Experiment: are the z-transpose atomics the adjoint cost?

Patch (not committed): in `adjoint_h_gpu` the two z-direction transposes
(`lambda_ey` from Hx, `lambda_ex` from Hy, 4 `atomicAdd` per cell in 2D) were
guarded with `NZ > 2`. Same build otherwise (item-3 tree), 2D 512x384 full
iteration, `--keep-allocator-cache`, two interleaved rounds
(`history_adjoint_ab/exp4a_*`):

| mode | backward ms without guard (b1 / b2) | with guard (b1 / b2) | change |
|---|---:|---:|---:|
| fp32 | 132.59 / 127.31 | 110.70 / 107.23 | -16.5 % / -15.8 % |
| fp16 | 128.67 / 126.02 | 111.55 / 107.13 | -13.3 % / -15.0 % |

About 18 µs per step, matching the ~16 µs/step estimated from the
`adjoint_h`/`adjoint_e` difference: the atomic scatter, not arithmetic, is the
main adjoint cost. (The guard alone changes the λEx/λEy state cotangents, so
it was only a measurement; the fast path below handles them properly.)

### Implementation

- `FastDivmod` (round-up multiply-high division, exact for dividends and
  divisors below 2^31) and `CellGrid`: the E/H update and adjoint kernels use
  `blockIdx.y` as the shot and a 32-bit per-shot flat index; `(i, j, k)` costs
  two `__umulhi` instead of five 64-bit `/`/`%`. The field offset
  `shot * cells + flat` stays 64-bit, so the total over all shots may exceed
  2^31; a single shot above `2^31 - 1` cells now fails with a clear
  `NativeLibraryError` in forward and backward.
- `SOLVER_TM2D` (`storage_type` bit 6) and probe
  `deepgpr_supports_tm2d_fast_path` (CUDA only). Kernels take a `TM2D`
  template parameter: update/adjoint kernels launch `NX * NY` threads per shot
  (k = 0 only) and touch only Ez, Hx, Hy; the z-derivative terms of Hx/Hy are
  literal zeros so the forward arithmetic (and FMA contraction) is that of the
  general kernels; CPML kernels map only the four x/y regions on k = 0
  (`grid.y = 4`) and skip the Ey/Ex/Hz corrections.
- Enabling rule (`modeling._tm2d_fast_path_applies`): CUDA, 2D model,
  `source_direction == 2`, library probe, and — only if the caller passed
  `E`/`H`/`PML` — Ex, Ey, Hz and the eight coupling CPML arrays all zero
  (one `count_nonzero` sync). Otherwise the general path runs (no error). The
  native side rejects the flag for `nz != 1`, a non-Ez source or z PML.
- Backward on the fast path zeroes the cotangents of Ex, Ey, Hz and the eight
  coupled CPML arrays (they are inactive by construction).

### Correctness

| check | result |
|---|---|
| CUDA 2D, orders 2/4/8, PML 0 and `[3, 5, 4, 2]`, two shots, fast path vs baseline | receiver data and final Ez/Hx/Hy **bitwise identical**; ε/σ/source gradients 1.0-2.7e-7 relative (baseline vs baseline noise 1-3e-7) |
| fast path vs general kernels (same build, rule forced off) | receiver data and all six final fields bitwise identical |
| 2D state dot product, orders 2/4/8, PML 0/2 | CPU < 3e-5 and CUDA < 2e-4 (the existing CPU/CUDA thresholds); inactive cotangents exactly 0 |
| CUDA material Taylor test (copy of the CPU test, same thresholds) | passed with and without CPML |
| existing CPU/CUDA consistency (2D, now on the fast path) and 3D CUDA dot products | passed unchanged |
| benchmark data vs baseline | bitwise for every mode, 2D and 3D |

No existing test needed a narrower check: none of them compared Ex/Ey/Hz
cotangents of a 2D CUDA run.

### Performance (formal protocol, current vs item 3 vs baseline, two interleaved rounds)

Raw: `history_adjoint_ab/item4_{cur,prev,base}_b{1,2}`; Nsight:
`history_adjoint_ab/nsys/item4_*`.

| case | mode | metric | item 3 (b1 / b2) | item 4 (b1 / b2) | vs item 3 | vs baseline |
|---|---|---|---:|---:|---:|---:|
| 2D fwd-only | fdtd | forward | 64.08 / 65.30 | 48.37 / 48.22 | -24.5 % / -26.2 % | -24.6 % / -26.5 % |
| 2D full | fp32 | forward | 79.32 / 82.20 | 63.92 / 66.05 | -19.4 % / -19.7 % | -32.4 % / -31.2 % |
| | | backward | 127.48 / 132.00 | 96.40 / 100.65 | -24.4 % / -23.7 % | -28.7 % / -25.0 % |
| | | total | 206.90 / 214.05 | 160.45 / 166.80 | -22.5 % / -22.1 % | -29.9 % / -27.6 % |
| 2D full | fp16 | total | 221.12 / 227.94 | 174.25 / 177.58 | -21.2 % / -22.1 % | -22.2 % / -21.0 % |
| 2D full | bf16 | total | 219.74 / 227.62 | 171.57 / 175.00 | -21.9 % / -23.1 % | -23.5 % / -23.0 % |
| 2D full | int8 | total | 244.64 / 251.97 | 197.05 / 200.98 | -19.5 % / -20.2 % | -20.0 % / -19.7 % |
| 3D fwd-only | fdtd | forward | 32.88 / 33.37 | 32.17 / 32.24 | -2.2 % / -3.4 % | -1.2 % / -1.5 % |
| 3D full | fp32 | total | 138.23 / 141.63 | 140.52 / 142.34 | +1.7 % / +0.5 % | -57.3 % / -57.2 % |
| 3D full | fp16 / bf16 / int8 | total | | | -1.4 % to +1.6 % | -0.7 % to +1.9 % |

3D shows no formal-protocol gain; its fp32 forward was +4.6 %/+3.7 % in the
formal batches. With `--keep-allocator-cache` the same comparison gives
45.16/45.37 → 44.63/44.52 ms forward and 126.91/127.43 → 125.84/126.51 ms total
(-0.8 %/-0.7 %), and Nsight shows the kernels faster (`update_h` 13.79 →
11.95 µs, `update_e` 12.58 → 12.91 µs, kernel sum of the forward −2 ms), so the
formal 3D difference is allocation noise around a 5.7 GiB `cudaMalloc`, not a
kernel regression. The 3D adjoint kernels are unchanged (≈42 µs each, still
atomic-bound).

Nsight, 2D fp32 full iteration, average µs per launch (item 3 → item 4):
`adjoint_h` 49.23 → 33.24, `adjoint_e` 28.72 → 25.24, `update_h` 17.24 → 15.71,
`update_e` 15.08 → 11.63, `cpml_e` 9.10 → 4.26, `cpml_h` 8.43 → 4.50,
`adjoint_cpml_e` 9.35 → 4.37, `adjoint_cpml_h` 8.98 → 4.53.

Full suite: 136 tests, all passed (1 skipped).

## 5. CUDA: adjoint as a gather

**Default:** yes on the 2D TM fast path (2D models with an Ez source and zero
Ex/Ey/Hz, i.e. the common 2D FWI case); **no** for the general (3D) path,
which keeps the atomic scatter because every gather variant measured slower
there. `deepgpr_deterministic_adjoint()` therefore stays 0.

### Implementation (2D TM path)

Per reverse step, after the receiver/source adjoints and the material
gradient:

1. `adjoint_cpml_e_weights_tm2d_gpu` (x/y faces, k = 0): for every Ez CPML
   entry compute the derivative weight
   `(sign upd (RA - 1) lambda_Ez - RF lambda_phi) / spacing` into phi-sized
   scratch, then advance `lambda_phi` (it needs the old value, so the weight
   is stored first).
2. `adjoint_e_gather_tm2d_gpu`: each H point gathers
   `sign * c(src) * lambda_Ez(src)` (only where the forward Ez update ran) plus
   the stored weights, with the stencil radius and coefficient of the source
   (`gather_backward_transpose_gpu`, same form as the CPU
   `gather_*_transpose`). It first applies the deferred
   `lambda_H <- ch_hist lambda_H` of the previous reverse step at the points
   the forward H update touched.
3. `adjoint_cpml_h_weights_tm2d_gpu`: same for the Hy/Hx CPML entries.
4. `adjoint_h_gather_tm2d_gpu`: each Ez point applies `ce_hist` to its own
   cotangent (deferred from step 2, where neighbours still read it; only
   where the forward update ran) and gathers the transposed forward
   differences of `ch_curl lambda_H` plus the magnetic weights.
5. After the loop `scale_magnetic_adjoint_tm2d_gpu` applies the last deferred
   `ch_hist` scaling.

No kernel on this path uses atomics. The receiver adjoint (one thread per
receiver; the first receiver at a location adds all coincident ones in index
order), the source-waveform gradient (one block per source, fixed-order tree
over shots) and the forward source injection (one thread per shot, sources in
order, `__fadd_rn` so the result is bitwise the old `atomicAdd`) are
deterministic on every path. A first version with one thread per shot looping
over the receivers cost 12.5 µs/step (64 receivers, serial global RMW) and was
replaced.

### Correctness

| check | result |
|---|---|
| existing dot-product (3D all faces; CPU and CUDA), Taylor (CPU and CUDA), CPU/CUDA consistency, coincident sources/receivers | pass at the original thresholds |
| new `test_cuda_asymmetric_cpml_dot_products_orders_2_4_8` (3D `[1,2,3,1,2,3]`, `[3,0,1,2,0,2]`; 2D TM `[3,1,0,2]`) | pass (< 2e-4); measured 0 to 7e-6, the same range as the scatter adjoint |
| new `test_cuda_2d_tm_adjoint_is_bitwise_reproducible` (2D mode 2 and 3, two coincident sources and receivers per shot, all gradients incl. states) | bitwise identical repeats |
| benchmark 2D, current b1 vs b2 | data and gradients bitwise identical for every storage mode |
| vs item 4 / vs baseline | data bitwise; 2D gradients 1.1-1.7e-7 relative; 3D (unchanged scatter) 2-9e-7 / 4-10e-6, as base-vs-base noise |
| 2D orders 2/4/8 x PML 0 / `[3,5,4,2]` vs baseline | data bitwise, gradients 1-3e-7 |

### Performance (formal protocol, current vs item 4 vs baseline)

Raw: `history_adjoint_ab/item5h_{cur,prev,base}_b{1,2}`.

| case | mode | backward ms item 4 (b1 / b2) | gather (b1 / b2) | vs item 4 | total vs baseline |
|---|---|---:|---:|---:|---:|
| 2D | fp32 | 96.57 / 95.82 | 67.73 / 67.73 | -29.9 % / -29.3 % | -41.6 % / -41.4 % |
| 2D | fp16 | 95.45 / 96.10 | 67.67 / 67.24 | -29.1 % / -30.0 % | -34.6 % / -33.7 % |
| 2D | bf16 | 93.55 / 94.29 | 65.16 / 65.14 | -30.3 % / -30.9 % | -35.5 % / -34.9 % |
| 2D | int8 | 95.64 / 96.29 | 67.36 / 67.30 | -29.6 % / -30.1 % | -31.1 % / -30.9 % |
| 3D | fp32 / fp16 / bf16 / int8 | (scatter, unchanged) | | -1.6 % to +1.5 % | |

### General-path gather: variants measured and rejected

All variants passed the dot-product, Taylor and determinism tests
(`.worktrees`-local builds; 3D 80^3 mode 3 and 2D 512x384, fp32 full
iteration, `--keep-allocator-cache`, 3 warmups + 10 repeats, backward median):

| variant | 3D backward ms | 2D backward ms |
|---|---:|---:|
| atomic scatter (item 4) | 79.6-82.0 | 95.7-96.1 |
| gather, one thread per cell (6 channels), phi-sized weight scratch | 96.8-97.1 | 76.6-77.0 |
| same + `__launch_bounds__(256, 4)` (64 registers) | 96.2 | 78.5 |
| CPML weights recomputed in the gather (no scratch, no weight kernels) | 111.2 | 80.2 |
| scratch + interior fast path (bitwise equal to the general path) | 106.2 | 76.8 |
| scratch + one thread per (cell, component) | 99.1 | 82.3 |
| recomputed weights + per component + launch bounds | 89.6 | 79.5 |
| same + parallel receiver adjoint | 86.4-89.2 | 70.8-71.4 |
| scratch + parallel receiver adjoint | 96.5 | 68.5 |
| **adopted: 2D TM gather (scratch), 3D scatter** | **78.9** | **67.2** |

Order 4 (3D 80^3, nt 300): scatter 73.2 ms, best gather 77.1 ms (+5 %). A
larger 3D case (140x140x100 + PML 10, nt 300): scatter 242.8 ms, best gather
278.3 ms (+15 %). Nsight (best gather, 3D): the two gather kernels take
82 + 75 µs per step against 42 + 41 + 24 + 25 µs for the scatter adjoint and
its CPML kernels. In 3D each target re-reads 2R source coefficients and
cotangents per channel (six channels), whereas the scatter issues mostly
uncontended `RED` atomics that the 4090 L2 absorbs well; the 3D scratch
version also pays ~20 MB/step of extra DRAM traffic for the weights. Making
the 3D gather competitive would need shared-memory tiling of the source
lines; that is left as a follow-up.

Full suite: 138 tests, all passed (1 skipped).

## 6. CUDA: kernel fusion

**Default:** yes on the 2D TM fast path forward (CPML corrections and the E
snapshot run inside the update kernels) and on the general (scatter) adjoint
(CPML transposes inside `adjoint_e_gpu` / `adjoint_h_gpu`). **No** for the
general (3D) forward, which keeps its separate CPML and snapshot kernels: fused,
it was faster on small 3D grids but slower on the formal 3D case for
low-precision/INT8 histories and on a larger 3D grid for every mode (see
"Variant measured and rejected"). The material-gradient kernel is not fused.

### Implementation

- `cpml_e_cell` / `cpml_h_cell` hold the per-cell body of the former
  `cpml_e_gpu` / `cpml_h_gpu`: faces in the order x0, xm, y0, ym, z0, zm, the
  other field read only. `update_e_gpu<ORDER, 1>` and
  `update_h_gpu<ORDER, 1, SAVE>` call them right after the base update of the
  same cell; the thread owns the cell and the CPML reads only the field the
  kernel does not write, so the result equals the former two-pass order. The
  general path launches `cpml_e_gpu` / `cpml_h_gpu`, now thin wrappers that map
  the six disjoint CPML boxes to cells and call the same functions (a single
  CPML implementation for both paths).
- E history on the 2D TM path: `update_h_gpu<..., SAVE = 1>` stores `E^n`
  (scalar float32 / float16 / bfloat16, resident history or async staging
  slot, extended or physical box, plus the float32 exact copy that low-precision
  E+R histories need). E is not written during the H half step, so this is the
  value the former `save_e_snapshot_gpu`, launched just before `update_h_gpu`,
  stored. INT8 (tile reductions), `native_vec2` (two cells per thread) and the
  stored-R snapshot (needs `E^(n+1)`) keep their kernels.
- Source injection and receiver sampling stay in `inject_sources_and_sample_gpu`
  after `update_e_gpu`, i.e. after the E CPML corrections.
- Scatter adjoint: `adjoint_cpml_e_cell` / `adjoint_cpml_h_cell` run before the
  base transpose of the same cell. They read only the cell's own cotangent
  (which that thread scales afterwards) and add onto the other field with the
  same `atomicAdd`s as before, only in a different order. The 2D TM gather path
  is unchanged: its CPML weights must be complete before neighbours gather them.
- Not fused: `accumulate_material_gradients_*` runs one thread per history cell
  and sums the shots in a fixed loop (deterministic), only every `S` steps; the
  INT8 variant uses tile-mapped threads. The adjoint kernels run one thread per
  (cell, shot), so fusing would need atomics over shots or per-shot gradient
  buffers (losing the 2D TM determinism or adding memory) for a kernel that is
  not launched at every step when `S > 1`.
- Rounding: a first version that kept the field values in registers across the
  base update and the CPML corrections changed which product the compiler
  contracted into an FMA in `ue0 * E + ue1 * dH - ...` (both are legal), and the
  forward differed by up to 3e-4 even without CPML. The final kernels keep the
  memory form of the old expressions; every forward comparison below is
  bitwise.

Kernel launches per time step (forward / reverse), from the Nsight instance
counts:

| path | item 5 | item 6 |
|---|---:|---:|
| 2D TM, fp32 E-only history (`update_h`, `cpml_h`, `update_e`, `cpml_e`, inject, `save_e` → `update_h`+save, `update_e`+CPML, inject) | 6 / 6 | 3 / 6 |
| 3D general, mode 3 fp32 (reverse: receivers, gradient, `adjoint_cpml_e`, `adjoint_e`, `adjoint_cpml_h`, `adjoint_h`) | 8 / 6 | 8 / 4 |

### Correctness

| check | result |
|---|---|
| 114-case dump vs item 5 (2D/3D, orders 2/4/8, fp32/fp16/bf16/int8/fp16 `native_vec2`, resident/async/physical/stored R) | receiver data, all 30 final states and the E/R/final-E histories bitwise; 2D gradients bitwise; 3D gradients ≤ 3.7e-7 relative (atomic order) |
| formal benchmark captures, current vs item 5 and vs baseline, both batches | data bitwise in every mode; 2D gradients bitwise vs item 5 (1.2-1.8e-7 vs baseline, item 5's gather); 3D gradients 3e-7 (ε) / 6e-6 to 1.5e-5 (σ), the same as current b1 vs b2 |
| new `test_cuda_fused_kernels.py` | every saved fp32 frame equals the E field of a run stopped after that many steps (2D/3D, asymmetric CPML, resident/async, extended/physical, S = 1 and 3); fused fp16/bf16 histories equal the `native_vec2` writer bitwise; final fields and data independent of the history writer; a two-segment run reproduces the full run's data and 30 states bitwise |
| existing dot-product (2D/3D, all faces, asymmetric), Taylor (CPU/CUDA), CPU/CUDA consistency, 2D TM determinism and selection tests | pass at the original thresholds; no existing test was changed |

### Performance (formal protocol, current vs item 5 vs baseline, two interleaved rounds)

Raw: `history_adjoint_ab/item6_{cur,prev,base}_b{1,2}`.

| case | mode | metric | item 5 (b1 / b2) | item 6 (b1 / b2) | vs item 5 | vs baseline |
|---|---|---|---:|---:|---:|---:|
| 2D fwd-only | fdtd | forward | 49.76 / 48.02 | 42.12 / 41.99 | -15.4 % / -12.6 % | -34.5 % / -34.4 % |
| 2D fwd-only | fp32 | forward | 64.93 / 63.73 | 53.27 / 52.95 | -18.0 % / -16.9 % | -75.0 % / -74.9 % |
| 2D fwd-only | fp16 | forward | 80.00 / 77.62 | 68.96 / 69.02 | -13.8 % / -11.1 % | -27.8 % / -27.5 % |
| 2D fwd-only | int8 | forward | 105.25 / 100.43 | 94.48 / 94.73 | -10.2 % / -5.7 % | -18.4 % / -18.3 % |
| 2D full | fp32 | forward | 64.58 / 63.71 | 53.90 / 52.76 | -16.5 % / -17.2 % | -74.3 % / -74.9 % |
| | | backward | 69.08 / 67.70 | 68.10 / 68.18 | -1.4 % / +0.7 % | -48.3 % / -48.3 % |
| | | total | 133.53 / 131.58 | 122.00 / 120.86 | -8.6 % / -8.1 % | -64.2 % / -64.6 % |
| 2D full | fp16 | total | 147.65 / 145.22 | 136.99 / 136.62 | -7.2 % / -5.9 % | -38.0 % / -37.9 % |
| 2D full | bf16 | total | 144.99 / 143.53 | 135.09 / 134.40 | -6.8 % / -6.4 % | -38.9 % / -39.0 % |
| 2D full | int8 | total | 174.95 / 167.73 | 162.82 / 162.27 | -6.9 % / -3.3 % | -33.2 % / -34.1 % |
| 3D fwd-only | fdtd / fp32 / fp16 / int8 | forward | (item 5 kernels) | | -2.0 / +0.8, -3.8 / -4.2, -1.1 / +0.7, +3.2 / +1.3 % | |
| 3D full | fp32 / fp16 / bf16 / int8 | forward | (item 5 kernels) | | -5.1 / +1.0, +0.3 / -0.7, -0.0 / -0.6, -1.6 / -1.8 % | |
| 3D full | fp32 | backward | 82.18 / 81.01 | 79.15 / 78.30 | -3.7 % / -3.3 % | -1.6 % / -2.9 % |
| | | total | 142.88 / 140.45 | 136.18 / 138.05 | -4.7 % / -1.7 % | -58.6 % / -58.2 % |
| 3D full | fp16 / bf16 / int8 | backward | | | -1.4 % to +1.0 % (first all-fused batches: -3.3 % to -4.2 %) | |

The 3D forward runs the item 5 kernels (identical register counts and launch
sequence; only the update kernels take two extra by-value structs). Its fp32
forward-only (-3.8 / -4.2 %) and int8 forward-only (+3.2 / +1.3 %) rows agree
in sign across batches, but the int8 forward of the full-iteration case in the
same batches moves the other way (-1.6 / -1.8 %), so these are read as
batch-level noise, not as an effect of this item.

The baseline's 2D fp32 forward is 210-213 ms in these batches against 95-119 ms
in earlier ones: it allocates an 8.4 GiB E+R history inside the timed window
(allocation caveat in §0); the current tree allocates half of that.

Nsight Systems, fp32 full iteration, µs per time step (2D: 2400 steps, 3D:
1000; `history_adjoint_ab/nsys/item6_{prev,final}_{2d,3d}_cuda_gpu_kern_sum.csv`):

| kernel | 2D item 5 | 2D item 6 | 3D item 5 | 3D item 6 |
|---|---:|---:|---:|---:|
| `update_h_gpu` (2D: + CPML + E snapshot) | 15.49 | 19.76 | 13.18 | 11.73 |
| `cpml_h_gpu` | 4.42 | - | 20.70 | 20.71 |
| `update_e_gpu` (2D: + CPML) | 11.34 | 13.95 | 12.46 | 11.54 |
| `cpml_e_gpu` | 4.60 | - | 21.28 | 18.68 |
| `save_e_snapshot_gpu` | 6.10 | - | 21.35 | 19.39 |
| `inject_sources_and_sample_gpu` | 1.79 | 1.76 | 1.72 | 1.64 |
| forward kernels | 43.73 | 35.47 | 90.69 | 83.69 |
| `adjoint_cpml_e_gpu` + `adjoint_e_gpu` | (gather, unchanged) | | 24.65 + 40.38 | 64.10 |
| `adjoint_cpml_h_gpu` + `adjoint_h_gpu` | | | 22.97 + 40.45 | 60.16 |
| kernel sum (ms) | 240.06 | 218.62 | 254.65 | 241.85 |

The 3D forward kernels are the item 5 code; their per-step differences in this
table are run-to-run variation. The fused 3D adjoint kernels take as long as the
pairs they replace (order 2: 48 / 44 registers, against 26-30 for the base
and 52-54 for the CPML kernels before);
the gain is the two launches and the re-read of the cotangent per step.

### Variant measured and rejected: fusion on the general (3D) forward

The first item 6 build fused CPML and the E snapshot on every path. Its formal
A/B (`history_adjoint_ab/item6_allfused_*`) matched the table above in 2D but
not in 3D (forward vs item 5, b1 / b2): fdtd -11.4 % / -12.3 %, fp32 -2.1 % /
+4.5 %, fp16 +14.5 % / +12.2 %, int8 +7.5 % / +5.2 %; 3D full totals fp32
-2.3 % / -2.2 %, fp16/bf16 +4.2 % to +5.2 %, int8 +1.5 % / +1.5 %. Separating
the two forward fusions on the formal 3D case (CPML fusion only, snapshot kernels
kept) gave fp16 100.3 / 101.7 ms against 91.8 / 91.1 ms for item 5, so the CPML
fusion causes it. A size and face sweep (order 4, `item6_size_sweep/`):

| 3D grid (cells incl. CPML), faces | fdtd | fp32 | fp16 | int8 |
|---|---:|---:|---:|---:|
| 80^3, all faces | -13.3 % / -14.4 % | -25.2 % / -24.6 % | -18.5 % / -15.5 % | -3.3 % / -6.2 % |
| 100 x 100 x 80, x/y faces | -12.7 % / -13.5 % | -24.4 % / -25.9 % | -14.1 % / -15.1 % | -4.0 % / -4.1 % |
| 80 x 80 x 100, z faces | -7.3 % / -5.5 % | -21.4 % / -22.2 % | -12.2 % / -12.4 % | -2.3 % / -2.1 % |
| 100^3, all faces | -11.0 % / -9.9 % | -13.1 % / -12.5 % | +11.6 % / +11.9 % | +2.9 % / +3.5 % |
| 140^3, all faces (nt 300) | +6.4 % / +4.1 % | +7.9 % / +2.4 % | -2.9 % / -2.1 % | +3.6 % / +0.6 % |

A large 2D case (1536 x 1152 physical, 4 shots, nt 300) stays faster fused in
every mode (fdtd -1.7 % / -1.9 %, fp32 -6.7 % / -6.2 %, fp16 -5.8 % / -6.0 %,
int8 -1.7 % / -1.3 %). Nsight on the formal 3D case shows the fused update
kernels slowing down with the history traffic while the separate pair does not:
fused `update_e` 26.0 µs (fdtd) / 35.5 (fp32) / 42.7 (int8) / 55.7 (fp16)
against `update_e` + `cpml_e` 30.8 / 31.5 / 36.7 / 42.2 µs. The SASS of the
CPML part is the same in both forms (per face: coefficient loads, reload of
the field, phi load, two stores), and the effect appears only when the 3D
working set no longer fits the 72 MB L2, so without hardware counters
(Nsight Compute is not permitted on this machine) the exact cause stays open.
Because the general path is mostly used for large 3D models, it keeps the
separate kernels.

Also measured and rejected: `__launch_bounds__(256, 5)` / `(256, 6)` on the
fused 3D adjoint kernels (44-48 / 40 registers, no spills) made them 1.2 % /
1.5 % slower in Nsight (`nsys/item6_exp_launch_bounds{5,6}_3d_*`). On a
140^3 3D case the adjoint fusion was neutral for fp32 (+0.4 % / -0.8 %) and
-1.4 % / -1.9 % for fp16 backward (`item6_size_sweep/big3d_*`).

Full suite: 142 tests, all passed (1 skipped).
