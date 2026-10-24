# CPU index decoding: review and Linux validation

Date: 2026-10-13. Baseline: `dd6a67b` (`HEAD` before the N-6 changes).

The final implementation removes per-cell integer index decoding while retaining
bitwise agreement with the baseline in the tested CPU configurations. Review
also found and corrected a performance regression in the original uncommitted
patch: collapsing only shot/x loops leaves threads idle on narrow models.
The final version splits y into contiguous row tiles **only when the current
OpenMP team has more threads than shot/x slices**. Normal grids retain complete
x slices. Material gradients parallelize x/y pairs and sum shots in the original
order. Floating-point expressions, implicit barriers and the C ABI are unchanged.

## Findings from review

- The original nested-loop index arithmetic, history-box offsets and `continue`
  behavior were consistent with the baseline. No numerical discrepancy was
  found in the original patch on this Linux build.
- On a `2 x 256` no-PML model, 100 steps, one shot and eight threads, the original
  patch increased median total time from 2.682 ms to 3.742 ms (+39.5%) over three
  paired rounds. There are too few shot/x slices to occupy the team. Follow-up
  narrow-grid measurements use 1000 steps to reduce fixed-overhead effects.
- Unconditionally collapsing shot/x/y loops removed the large narrow-grid
  penalty, but a screening run lost about 15–16% against the original patch on
  low-order 2D models. Adaptive tiling preserved those normal-grid gains while
  improving the narrow-grid screening time by about 39% against the original
  patch. These screening numbers were used to select the implementation;
  final baseline comparisons are reported separately below.

## Environment and protocol

- Intel Core i7-14700KF, WSL2 x86_64, Debian 13; 28 logical CPUs visible to WSL.
- Python 3.12.14; PyTorch 2.14.1+cpu; GCC 14.2.0 (Debian 14.2.0-19).
- Both baseline and candidate built with
  `-std=c99 -O3 -fopenmp -fPIC -shared -Wall -Wextra -Wno-unused-parameter -Werror`.
  Neither uses `-march=native` or fast-math flags. Both compiled without warnings.
- Baseline and candidate were built in independent directories, without
  replacing the repository's tracked prebuilt libraries. The A/B driver records
  each loaded library's path and SHA-256 hash in its local output.
- Final benchmarks use three paired rounds, alternating execution order. Each
  fresh process warms up once and records five samples; the table reports the
  median of the three per-process medians. Forward and backward are timed
  separately; input construction, assertions and garbage collection are outside
  the timed regions. `OMP_PROC_BIND=true`, `OMP_PLACES=cores`. No other validation
  or compilation jobs ran concurrently with the final benchmarks.
- The four normal final cases sample both dimensions, orders and thread counts.
  The earlier patch and implementation screening used all eight combinations.
  Measurements are local WSL results, not general hardware guarantees.

## Correctness

- Thirteen A/B cases, collectively collecting 614 tensor hashes plus loss/absence entries per run,
  match bitwise against the baseline at **1, 2, 4 and 8 OpenMP threads**. All
  collected tensors are finite. Coverage includes 2D/3D, spacing/order variants,
  multiple shots/sources, fp32/fp16/bf16 histories, temporal sampling, asymmetric
  and disabled PML faces, physical-only history boxes, nonzero initial states,
  and a narrow no-PML model. Outputs include receivers, E/H/PML states, E/R
  histories, material/source gradients and initial-state gradients.
- Separate builds without `-fopenmp` also compile without warnings and pass all
  thirteen cases. Their result maps also equal the OpenMP one-thread maps.
- The final full suite reports **176 tests, zero failures/errors, 45 skip entries**.
  Forty-three skips require CUDA; two are CPU gradient **subtests** whose weak
  joint finite-difference direction cannot resolve the requested 0.2% tolerance.
  Those two inconclusive subtests have the same values on baseline and candidate;
  they are not counted as numerical passes. Do not infer a passed-method count
  by subtracting subtest skips from `testsRun`.
- The CPU determinism regression now includes a two-column no-PML model and
  compares 1/2/5/8 threads. The complete suite was rerun after this change.
- Ruff and `git diff --check` pass. Windows/macOS builds and CUDA execution were
  not rerun in this environment; CUDA source and the ABI were not modified.

## Final performance

| Model | Threads | Order | Baseline total (s) | Final total (s) | Forward reduction | Backward reduction | Total reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2D 96×128, 4 shots, 500 steps | 2 | 2 | 2.348313 | 1.669315 | +40.0% | +19.4% | +28.9% |
| 3D 28×32×24, 1 shot, 100 steps | 2 | 2 | 0.578326 | 0.428327 | +34.7% | +18.7% | +25.9% |
| 2D 96×128, 4 shots, 500 steps | 8 | 8 | 1.883991 | 1.754056 | +18.2% | +1.4% | +6.9% |
| 3D 28×32×24, 1 shot, 100 steps | 8 | 8 | 0.527860 | 0.450747 | +21.2% | +6.1% | +14.6% |
| Narrow 2×256, 1 shot, 1000 steps | 2 | 2 | 0.038362 | 0.036907 | +26.3% | -7.7% | +3.8% |
| Narrow 2×256, 1 shot, 1000 steps | 8 | 2 | 0.019796 | 0.019877 | +11.7% | -2.2% | -0.4% |

The four normal cases reduce total time by 6.9–28.9%. The eight-thread narrow
case is within about 0.4% of baseline total time; its three paired rounds
do not establish a speedup. The original large underutilization penalty is
absent in this longer follow-up. Small phase regressions remain visible in
the table; this change does not promise that every kernel or shape is faster.

Positive percentages mean time reduced: `100 * (1 - candidate / baseline)`.
Forward and backward reductions are shown separately so a phase regression is
not hidden by total-time improvement. The scripts below generate raw samples,
library hashes and paired-round medians for independent verification.

## Reproduction

From the repository root, install CPU PyTorch, NumPy and Matplotlib and use a
Linux C compiler with OpenMP. Build both libraries with the same compiler and
flags; the example keeps them separate from the package's prebuilt libraries:

```sh
mkdir -p validation/n6/baseline-lib validation/n6/candidate-lib
git show dd6a67b:src/DeepGPR/lib/deepgpr_cpu.c > validation/n6/baseline-lib/deepgpr_cpu.c
cp src/DeepGPR/lib/deepgpr.h validation/n6/baseline-lib/deepgpr.h
cp src/DeepGPR/lib/deepgpr_cpu.c src/DeepGPR/lib/deepgpr.h validation/n6/candidate-lib/
cc -std=c99 -O3 -fopenmp -fPIC -shared -Wall -Wextra -Wno-unused-parameter -Werror -o validation/n6/baseline-lib/deepgpr_cpu.so validation/n6/baseline-lib/deepgpr_cpu.c
cc -std=c99 -O3 -fopenmp -fPIC -shared -Wall -Wextra -Wno-unused-parameter -Werror -o validation/n6/candidate-lib/deepgpr_cpu.so validation/n6/candidate-lib/deepgpr_cpu.c
python tests/verify_cpu_index_decoding.py --baseline validation/n6/baseline-lib --candidate validation/n6/candidate-lib --output validation/n6/checks
```

The A/B driver launches `tests/cpu_index_decoding_cases.py` in separate
processes. Each process records its library path/hash and compares the
`results` maps, not metadata or labels. Missing required gradients and non-finite
tensors fail immediately. For serial builds, omit `-fopenmp` and use
`--threads 1`. Run the full unit suite as described in [the test guide](../README.md)
after building the CPU library in the checkout being tested.

Run benchmarks without concurrent CPU-heavy work. These commands reproduce the
final six benchmark configurations:

```sh
OMP_PROC_BIND=true OMP_PLACES=cores python tests/benchmark_cpu_index_decoding.py --baseline validation/n6/baseline-lib --candidate validation/n6/candidate-lib --output validation/n6/bench_low --cases 2d 3d --threads 2 --orders 2
OMP_PROC_BIND=true OMP_PLACES=cores python tests/benchmark_cpu_index_decoding.py --baseline validation/n6/baseline-lib --candidate validation/n6/candidate-lib --output validation/n6/bench_high --cases 2d 3d --threads 8 --orders 8
OMP_PROC_BIND=true OMP_PLACES=cores python tests/benchmark_cpu_index_decoding.py --baseline validation/n6/baseline-lib --candidate validation/n6/candidate-lib --output validation/n6/bench_thin --cases thin
```

The defaults are three paired rounds and five samples per process. Omit the
case/thread/order filters to run the complete matrix. Local builds and raw logs
belong under ignored `validation/` and are not part of this change. The committed
report summarizes the measurements; the drivers produce machine-readable data
for independent verification.
