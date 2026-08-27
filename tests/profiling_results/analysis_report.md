# DeepGPR FDTD-only baseline and profiling report

Date: 2026-08-25 (Asia/Shanghai)

Hardware/software: NVIDIA GeForce RTX 4090 (24 GiB), driver 575.51.03,
CUDA toolkit 12.4, Nsight Systems 2023.4.4, Nsight Compute 2025.2.0,
Python 3.10.16, PyTorch 1.11.0+cu115. The loaded native library was the newly
compiled `/home/ll/下载/DeepGPR-dev/src/DeepGPR/lib/deepgpr.so` (ABI 6,
INT8 capability 1), built for `sm_89` with `-O3 -lineinfo -Xptxas=-v`.

## Implementation and correctness

`save_wavefield_history=False` is independent of
`wavefield_compression="none"`. It returns a zero-length `E_saved` tensor and
sets a native storage-control bit. The CUDA and CPU forward loops use that bit
to skip all E/R history stores, INT8 payload/scale writes, compression launches,
async history buffers, and low-precision exact-snapshot allocation. FDTD E/H
updates, CPML, source injection, receiver recording, every timestep, and every
shot still run normally. Calling a history-dependent backward raises:

`Forward wavefield history is disabled; adjoint/model-gradient backward is unavailable.`

The large-case FDTD-only receiver data matched FP32-history output exactly:
relative L2 = 0 and max absolute error = 0. The same exact result was obtained
in a separate CPU smoke test. Default history-enabled backward behavior remains
unchanged.

## Formal benchmark methodology

All formal timings below are CUDA Event timings without `nsys` or `ncu`
attached. Case: `nx=512`, `ny=384`, `nt=1200`, 4 shots, 64 receivers,
sampling interval 1, warmup 5, repeat 20. The median is the primary comparison.
Peak memory is PyTorch peak allocated memory, reset after warmup. History memory
is the E+R payload used by model-gradient backward.

### A. Forward timing

| mode | forward median ms | mean ms | min ms | std ms | peak MiB | history MiB |
|---|---:|---:|---:|---:|---:|---:|
| fdtd | 57.362946 | 57.104077 | 55.467007 | 0.777594 | 56.53 | 0.00 |
| fp32 | 83.542721 | 89.394931 | 76.436478 | 18.994478 | 7256.53 | 7200.00 |
| fp16 | 81.860096 | 82.151815 | 78.080002 | 2.985454 | 3656.53 | 3600.00 |
| int8 | 100.847614 | 100.427181 | 95.974564 | 3.522424 | 1968.86 | 1912.50 |

The current CUDA Event measurement does **not** reproduce the earlier
approximately 178 ms FP32 forward wall time. The profiler shows no comparable
doubling of FP32 kernel work. Host allocation/synchronization and the prior
build/runtime are possible contributors to the older wall result, but this run
does not isolate them, so that explanation is not claimed as measured fact.

### B. History overhead

With `F = T_fdtd_only = 57.362946 ms` and median timings:

| mode | history/compression overhead ms | overhead vs FDTD-only |
|---|---:|---:|
| FP32 | 26.179775 | 45.639% |
| FP16 | 24.497150 | 42.706% |
| INT8 | 43.484669 | 75.806% |

### C. Full iteration

| mode | forward median ms | backward median ms | total median ms | peak MiB | history MiB |
|---|---:|---:|---:|---:|---:|
| fp32 | 83.393536 | 110.655487 | 194.402298 | 10898.73 | 7200.00 |
| fp16 | 84.672512 | 107.705151 | 192.830978 | 5498.73 | 3600.00 |
| int8 | 104.091648 | 106.763779 | 210.929146 | 2967.32 | 1912.50 |

### D. Gradient accuracy

FP32 is the reference.

| mode | eps relative L2 | eps cosine | sigma relative L2 | sigma cosine |
|---|---:|---:|---:|---:|
| fp32 | 0 | 1.000000000 | 0 | 1.000000000 |
| fp16 | 2.286446761e-05 | 1.000000000 | 2.192302054e-05 | 1.000000000 |
| int8 | 4.549345176e-04 | 0.999999899 | 3.644936951e-04 | 0.999999940 |

## Nsight Systems

Each report contains one warmup and one measured execution. The following
totals are normalized to one 1200-timestep forward; calls are therefore 1200
per listed timestep kernel. Percentages are unchanged by this normalization.

### Forward hotspots

| mode | kernel | total ms/run | calls/run | average us | kernel-time share |
|---|---|---:|---:|---:|---:|
| fdtd | `update_h_gpu<2>` | 16.350 | 1200 | 13.625 | 31.4% |
| fdtd | `update_e_gpu<2>` | 14.750 | 1200 | 12.292 | 28.3% |
| fdtd | `cpml_e_gpu<2>` | 9.300 | 1200 | 7.750 | 17.9% |
| fdtd | `cpml_h_gpu<2>` | 9.031 | 1200 | 7.526 | 17.3% |
| fdtd | source/receiver fused kernel | 2.193 | 1200 | 1.828 | 4.2% |
| fp32 | `save_rhs_snapshot_gpu<float32>` | 8.626 | 1200 | 7.188 | 12.9% |
| fp32 | `save_e_snapshot_gpu<float32>` | 6.285 | 1200 | 5.237 | 9.4% |
| fp16 | `save_rhs_snapshot_gpu<float16>` | 9.016 | 1200 | 7.513 | 13.4% |
| fp16 | `save_e_snapshot_gpu<float16>` | 6.853 | 1200 | 5.711 | 10.2% |
| int8 | `quantize_rhs_int8_snapshot_gpu` | 18.211 | 1200 | 15.176 | 21.3% |
| int8 | `quantize_e_int8_snapshot_gpu` | 16.190 | 1200 | 13.491 | 19.0% |

Summed GPU kernel time per forward was 52.089 ms (fdtd), 66.957 ms (fp32),
67.093 ms (fp16), and 85.366 ms (int8). FP32 and FP16 history kernels cost
14.910 and 15.869 ms/run respectively; INT8 compression costs 34.400 ms/run.
Thus the current FP16 conversion kernels are not materially cheaper than the
FP32 copy/RHS kernels, while INT8's reduction and quantization work clearly
dominates its forward penalty.

The explicit CUDA memcpy reports contain only a few KiB of setup copies. They
do not include loads/stores performed inside kernels and therefore cannot be
used as history-traffic measurements.

### Backward hotspots

The full reports also contain one warmup plus one measured iteration. Per-run
summed backward kernel time (adjoint E/H, adjoint CPML, receiver injection, and
material gradient) was approximately 100.375 ms for FP32 and 99.856 ms for
INT8. The corresponding material-gradient kernels were:

| metric | FP32 | INT8 |
|---|---:|---:|
| kernel | `accumulate_material_gradients_gpu<float32>` | `accumulate_material_gradients_int8_gpu` |
| calls/run | 1200 | 1200 |
| average kernel time | 9.4842 us | 9.2858 us |
| total kernel time/run | 11.3811 ms | 11.1429 ms |
| formal backward median | 110.6555 ms | 106.7638 ms |
| registers/thread (ptxas) | 36 | 40 |
| spill stores / loads (ptxas) | 0 / 0 | 0 / 0 |

INT8's fused decode-gradient kernel is 2.1% faster per launch in the Systems
trace, but only saves about 0.24 ms of summed kernel time over 1200 launches.
That is direct evidence of a small kernel-time improvement, not proof of the
mechanism. Long Scoreboard and DRAM/cache counters were unavailable, so the
claim that reduced history bytes specifically lowered memory stalls remains
unconfirmed.

## Nsight Compute limitation

The installed sections include `SpeedOfLight`, `MemoryWorkloadAnalysis`,
`SchedulerStats`, `WarpStateStats`, `LaunchStats`, and `Occupancy`. A targeted
run selected one middle `update_e_gpu<2>` launch with `--launch-skip 100` and
`--launch-count 1`, but NCU returned `ERR_NVGPUCTRPERM`. The read-only driver
parameter is `RmProfilingAdminOnly: 1`.

Consequently, the following requested hardware-counter results are **未测得**;
no estimates are substituted:

### E. Corresponding forward `update_e_gpu<2>` kernel

| metric | fdtd | fp32 | fp16 | int8 |
|---|---|---|---|---|
| Long Scoreboard | 未测得 | 未测得 | 未测得 | 未测得 |
| Short Scoreboard / Barrier / Wait / throttles | 未测得 | 未测得 | 未测得 | 未测得 |
| DRAM read/write bytes and throughput | 未测得 | 未测得 | 未测得 | 未测得 |
| L1/TEX throughput and hit rate | 未测得 | 未测得 | 未测得 | 未测得 |
| L2 throughput and hit rate | 未测得 | 未测得 | 未测得 | 未测得 |
| achieved/theoretical occupancy | 未测得 | 未测得 | 未测得 | 未测得 |
| SM throughput/utilization | 未测得 | 未测得 | 未测得 | 未测得 |
| registers/thread (ptxas) | 40 | 40 | 40 | 40 |

### F. Backward gradient kernels

| metric | FP32 | INT8 |
|---|---|---|
| Long / Short Scoreboard | 未测得 | 未测得 |
| Barrier / Wait / LG / MIO throttle | 未测得 | 未测得 |
| DRAM read/write bytes and throughput | 未测得 | 未测得 |
| L1/TEX and L2 hit rates | 未测得 | 未测得 |
| achieved/theoretical occupancy | 未测得 | 未测得 |
| SM throughput/utilization | 未测得 | 未测得 |
| registers/thread (ptxas) | 36 | 40 |
| static shared memory (ptxas) | 0 | 16 bytes |
| spills (ptxas) | none | none |

An administrator must either grant non-admin performance-counter access (then
reboot/reload the NVIDIA module as appropriate) or run the profiler with admin
rights. After that change, rerun the already targeted NCU commands; this task
did not modify the system setting. Because no `.ncu-rep` was produced, Source/
SASS counter correlation is also 未测得. `-lineinfo` is present for a future
`ncu-ui` Source-page inspection.

## INT8 bottleneck assessment

The forward bottleneck is the compression pair, not register spilling:

- The two compression kernels consume 34.400 ms/run and about 40.3% of summed
  forward kernel time.
- Each 8x8 tile performs a shared-memory block-maximum reduction with repeated
  barriers, writes/reads a scale, divides, rounds, clamps, and stores INT8.
  Relevant source is `deepgpr.cu` lines 1678-1749 and 1754 onward.
- ptxas reports 30 registers/thread and zero spills for both compression
  kernels. Occupancy and Barrier/Short-Scoreboard counters are 未测得, so a
  barrier/occupancy diagnosis cannot be claimed as hardware-counter evidence.
- The fused INT8 backward uses 40 rather than 36 registers/thread, but has no
  spills. Its scale loads, synchronization, inline decode, and accumulation are
  at `deepgpr.cu` lines 1955-2063.

INT8 still provides the best memory result (1912.50 MiB history and 2967.32 MiB
full peak) with excellent gradient accuracy, but the 34.4 ms compression cost
makes its full iteration 18.10 ms slower than FP16.

## GPU-only checkpoint feasibility estimate (not a checkpoint benchmark)

This is a feasibility estimate only; checkpointing was not implemented.

`F = 57.362946 ms` is the measured cost of one complete history-free forward.
Against the fastest full-history baseline, FP16:

- Removing the FP16 forward history path saves only `Hf = 24.497150 ms`.
- The observed INT8-vs-FP16 backward reduction is only 0.941372 ms, despite a
  much smaller history payload; NCU counters are unavailable to assign it to
  memory stalls.
- Even optimistically combining those savings gives about 25.44 ms, far below
  the 57.36 ms cost of one extra full recomputation. The shortfall is about
  31.92 ms before checkpoint reads/writes, segment history, or scheduling
  overhead.
- Independently, the Systems trace puts only the adjoint E/H and adjoint CPML
  kernels near 87 ms/run. Two FDTD passes already cost about 114.73 ms, leaving
  too little room to beat the 192.83 ms FP16 total even before material-gradient
  and checkpoint overhead.

Conclusion **B: checkpointing is likely mainly a memory optimization; the
probability of beating the current FP16 total runtime is low.** It remains
useful if memory capacity is the binding constraint, but the present data do
not justify implementing it as a speed optimization.

## Tests and limitations

- New CUDA FDTD-only receiver parity test: passed.
- New CUDA no-history backward-error test: passed.
- CPU no-history parity/backward smoke test: passed, exact receiver parity.
- `tests.test_wavefield_compression`: 4 passed, 1 failed. The pre-existing
  small-case INT8 directional-consistency test reported 0.255808 vs threshold
  0.15; large-case gradient accuracy above remains good. The test was not
  removed or relaxed.
- `tests.test_numerics`: 29 passed, 1 environment-compatibility error. PyTorch
  1.11's `torch.load` rejects the test's newer `weights_only` keyword.
- `tests.test_discrete_adjoint`: 12 passed, 1 skipped (one GPU available),
  1 exact-bit equality failure with a 0.09375 absolute / 1.83e-7 relative
  single-element difference in an existing CPU sampling test.
- CUDA compilation, Python bytecode compilation, ABI/path checks, and all
  large benchmarks completed without illegal-memory-access errors.
- The supplied workspace exposes an empty `.git` directory, so `git status`,
  `git diff --check`, and `git diff --stat` cannot operate. No reset, checkout,
  clean, or deletion command was used.

## Recommended next optimization

The single highest-value next speed optimization is to reduce the INT8 forward
compression cost: fuse E/R quantization work where possible and replace the
current full-block shared-memory max reductions/barriers with warp-level
reductions. This directly targets the measured 34.4 ms hotspot. Checkpointing
should not be the next speed project on the present data.
