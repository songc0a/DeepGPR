# DeepGPR CUDA native optimization report

Date: 2026-08-27 (Asia/Shanghai)

## Outcome

The existing optimized INT8 implementation is intact and remains explicitly
selectable. Two changes met the correctness + repeated timing + profiler gate:

1. CUDA FP16 history conversion now auto-selects NVIDIA native scalar
   intrinsics. The legacy manual bit conversion remains available.
2. Default 64-voxel INT8 tiles now auto-select NVIDIA CUB BlockReduce. The
   original shared tree and the two-warp shuffle experiment remain available.

BF16 native scalar and both vec2 paths are retained for reproducible A/B use,
but are not auto-defaults because they did not produce a stable improvement.
No checkpoint/ZFP implementation, INT8 quantization formula, scale, payload,
layout, or fused backward gradient formula was changed.

## Reproducibility

Formal case: RTX 4090, `nx=512`, `ny=384`, `nt=1200`, 4 shots, 64 receivers,
sampling interval 1, CUDA Event timing, 5 warmups and 20 measured repeats.
Profiler values come from Nsight Systems with one warmup and one measured run;
they are not used as formal runtime. The exact loaded library was
`/home/ll/下载/DeepGPR-dev/src/DeepGPR/lib/deepgpr.so`.

Raw formal results:

- `tests/profiling_results/before_native_audit.txt`
- `tests/profiling_results/native_scalar_ab/`
- `tests/profiling_results/native_vec2_ab/`
- `tests/profiling_results/int8_reduction_ab/`
- Nsight reports and SQLite exports: `tests/profiling_results/nsys/`

## Table 1: formal performance and memory

Medians are from the final same-process A/B batch. Peak is the full
forward+backward PyTorch peak for gradient modes; FDTD-only has no backward.

| mode | forward ms | backward ms | total ms | peak MiB | history MiB |
|---|---:|---:|---:|---:|---:|
| fdtd | 55.798784 | - | 55.798784 | 56.53 | 0.00 |
| fp32 | 81.645054 | 104.179169 | 186.052612 | 10898.73 | 7200.00 |
| fp16_legacy | 84.044800 | 101.285889 | 185.515007 | 5498.73 | 3600.00 |
| fp16_native | **82.468353** | **97.995296** | **180.579842** | 5498.73 | 3600.00 |
| fp16_vec2 | 84.585583 | 98.003040 | 182.736389 | 5498.73 | 3600.00 |
| bf16_legacy | 83.153473 | 97.936386 | 181.156349 | 5498.73 | 3600.00 |
| bf16_native | 82.523777 | 98.004528 | 180.567039 | 5498.73 | 3600.00 |
| bf16_vec2 | 84.561409 | 97.991154 | 182.746628 | 5498.73 | 3600.00 |
| int8_current | 101.090000 | 100.581005 | 201.826820 | 2967.32 | 1912.50 |
| int8_cub | **99.167889** | 100.605442 | **199.767555** | 2967.32 | 1912.50 |
| int8_warp | 99.138592 | 100.582878 | 199.870979 | 2967.32 | 1912.50 |

Key same-batch changes:

| Comparison | Forward delta | Total delta | Decision |
|---|---:|---:|---|
| FP16 legacy -> native scalar | -1.88% | -2.66% | native scalar auto-default for CUDA FP16 |
| FP16 native scalar -> vec2 | +2.57% slower | +1.19% slower | scalar preferred |
| BF16 legacy -> native scalar | -0.76% | -0.33% | not stable across the earlier A/B; legacy default |
| BF16 legacy -> vec2 | +1.69% slower | +0.88% slower | legacy preferred |
| INT8 current -> CUB | -1.90% | -1.02% | CUB auto-default for 64 voxels |
| INT8 current -> warp | -1.93% | -0.97% | CUB has slightly better total and simpler ownership |

The independent scalar A/B also measured FP16 total improving 209.972733 ->
203.293182 ms (-3.18%). BF16 changed 204.197380 -> 204.632576 ms (+0.21%
slower) in that run. That sign change, plus the kernel result below, is why the
small BF16 final-total difference was not promoted.

## Table 2: history/compression kernels

Nsight totals are normalized to one 1200-step run. `save_E` and `save_R` each
have 1200 launches. INT8 rows are the corresponding quantizers.

| history backend | save/quantize E ms | save/quantize R ms | total ms | vs reference |
|---|---:|---:|---:|---:|
| FP16 legacy | 7.235811 | 9.268936 | 16.504746 | reference |
| FP16 native scalar | **6.548170** | **8.837893** | **15.386062** | **-6.78%** |
| FP16 native vec2 | 7.053206 | 9.576329 | 16.629535 | +0.76% vs legacy; +8.08% vs scalar |
| BF16 legacy | **6.448575** | 8.799982 | **15.248557** | reference |
| BF16 native scalar | 6.624218 | **8.723442** | 15.347660 | +0.65% |
| BF16 native vec2 | 6.933995 | 9.549076 | 16.483071 | +8.10% |
| INT8 current shared | 16.636904 | 18.241227 | 34.878131 | reference |
| INT8 CUB BlockReduce | **15.204259** | **16.880540** | **32.084799** | **-8.01%** |
| INT8 two-warp shuffle | 15.399287 | 17.613417 | 33.012704 | -5.35% |

The earlier profiler measurement cited in the task was FP32 history 14.91
ms/run and manual FP16 15.87 ms/run. FP16 moved more bytes efficiently, but its
manual exponent/mantissa/subnormal/rounding logic cost enough instructions to
erase that advantage. The new direct trace measures manual FP16 at 16.505 ms
and native scalar at 15.386 ms. Native is therefore close to, but still about
0.48 ms above, the earlier FP32 measurement; a conversion is still not a pure
copy and the RHS kernel also performs arithmetic.

## Table 3: gradient accuracy

FP32 is the reference. Receiver data is independent of history format and
matched across backends. Tiny differences between repeated gradient launches
are expected from the pre-existing atomic accumulation order.

| mode | eps relative L2 | eps cosine | sigma relative L2 | sigma cosine |
|---|---:|---:|---:|---:|
| fp32 | 0 | 1.000000000 | 0 | 1.000000000 |
| fp16_legacy | 2.283402318e-05 | 1.000000000 | 2.189486077e-05 | 1.000000000 |
| fp16_native | 2.284819493e-05 | 1.000000000 | 2.188643521e-05 | 1.000000000 |
| fp16_vec2 | 2.285260052e-05 | 1.000000000 | 2.189621046e-05 | 1.000000000 |
| bf16_legacy | 1.530762966e-04 | 0.999999988 | 1.467170514e-04 | 0.999999989 |
| bf16_native | 1.530601876e-04 | 0.999999988 | 1.467154652e-04 | 0.999999989 |
| bf16_vec2 | 1.530677109e-04 | 0.999999988 | 1.467093098e-04 | 0.999999989 |
| int8_current | 4.549349833e-04 | 0.999999899 | 3.644746030e-04 | 0.999999940 |
| int8_cub | 4.549261648e-04 | 0.999999899 | 3.644928511e-04 | 0.999999940 |
| int8_warp | 4.549298028e-04 | 0.999999899 | 3.644664830e-04 | 0.999999940 |

The dedicated conversion test covers signed zero, normal/small/large values,
FP16 subnormals, infinities, NaNs, rounding boundaries, and deterministic
random values. Legacy, native scalar, and vec2 encodings/decodes are bit-exact
for every non-NaN input tested; signed zero is exact. NaN payload equality is
not required, but every backend decodes the NaN inputs as NaN. Full DeepGPR
history, receiver, and gradient parity uses an unchanged `5e-7` relative
threshold.

INT8 CUB/warp histories (payload and FP32 scales) and receiver data are bit
exact against current. Gradient differences are 1.4e-7 to 2.6e-7 relative,
the same scale as a repeated current-vs-current run (about 2.3e-7), with cosine
at numerical 1.

## Compile and instruction evidence

| kernel/backend | registers/thread | shared memory | spills |
|---|---:|---:|---:|
| scalar save E legacy/native | 36 | 0 | 0 |
| scalar save RHS legacy/native | 32 | 0 | 0 |
| vec2 save E/RHS | 40 | 0 | 0 |
| INT8 current quantize E/R | 30 / 30 | 256 B dynamic at 64 threads | 0 |
| INT8 CUB quantize E/R | 28 / 30 | 16 B static | 0 |
| INT8 warp quantize E/R | 28 / 28 | 16 B static | 0 |
| fused INT8 backward | 40 | 16 B static | 0 |

`cuobjdump --dump-ptx` on the tested `.so` shows
`cvt.rn.f16.f32`, `cvt.f32.f16`, `cvt.rn.bf16.f32`,
`cvt.rn.f16x2.f32`, and `cvt.rn.bf16x2.f32` in the native instantiations.
This proves that the official intrinsic paths reached the corresponding PTX
conversion operations. The Ada SASS dump did not expose a separately named
scalar conversion mnemonic that could be unambiguously assigned after ptxas
lowering, so no stronger SASS instruction claim is made. NCU Source/SASS
counter correlation and occupancy are unmeasured because of
`ERR_NVGPUCTRPERM`.

## Table 4: official primitive audit summary

| candidate | current | official alternative | measured result | recommendation |
|---|---|---|---|---|
| FP16 conversion | manual bits | CUDA FP16 scalar intrinsic | save kernels -6.78%, total -2.66% | default native scalar |
| BF16 conversion | manual bits | CUDA BF16 scalar intrinsic | save kernels +0.65%; total result unstable | retain optional, default legacy |
| pair conversion/store | scalar | CUDA half2/bfloat162 | slower; 40 registers | optional only |
| INT8 max | shared tree | CUB BlockReduce | quantizers -8.01%, total -1.02% | default CUB for 64 voxels |
| INT8 max | shared tree | two-warp shuffle | quantizers -5.35%, total -0.97% | optional only |
| INT8 load/store | scalar gather/store | BlockLoad/Store/vector types | contiguity/boundary preconditions fail for source | no mechanical replacement |
| INT8 prefetch | global-to-register | memcpy_async/pipeline | no reusable shared input tile | not applicable |
| per-call allocation | cudaMalloc/free | reuse / mallocAsync | saves at most about 0.12 ms/call | low priority |
| stream/events | create/destroy in async only | persistent resources | about 0.01-0.04 ms/call opportunity | not a hotspot |
| 1200-step launches | host launch loop | CUDA Graph | real 10-12 ms non-kernel envelope; synthetic replay -43% | next full-solver prototype |
| adjoint atomics | exact transpose scatter | aggregation/gather rewrite | large kernels, but atomic cost unmeasured | audit only; high correctness risk |

The detailed classification, source locations, runtime microbenchmark, async
offload analysis, and build-path audit are in
`tests/profiling_results/cuda_official_api_audit.md`.

## Test status

Command:

```text
python -m unittest tests.test_numerics tests.test_discrete_adjoint \
  tests.test_cuda_conversion_backends tests.test_wavefield_compression -v
```

Result: 56 tests executed; 53 passed, 1 skipped (requires at least two CUDA
devices), and 2 pre-existing failures were preserved without threshold changes:

1. `test_2d_compressed_gradient_directional_consistency`: stable relative
   discrepancy 0.255808 vs existing 0.15 threshold. This test uses the frozen
   lossy INT8 math; all three reduction histories are bit exact, so the new
   reduction does not cause it.
2. `test_compute_saves_forward_wavefield_to_requested_directory`: PyTorch
   1.11 rejects the test's `torch.load(weights_only=...)` keyword before it can
   inspect the saved tensor.

All other numerics, discrete-adjoint, async bounds, conversion, compression,
special-value, backend contract, and full-physics parity tests passed.

## Explicit answers to the 15 requested questions

1. **Is the current optimized INT8 version fully retained?** Yes. Scale,
   rounding/clamp, payload/layout, fused backward, and gradient formula are
   unchanged. `int8_reduction_backend="current"` retains the original tree.
2. **How much faster is native FP16 than the manual version?** Save kernels are
   6.78% faster. Same-batch forward is 1.88% faster and total is 2.66% faster;
   the independent scalar run measured 3.18% total.
3. **How much faster is native BF16?** It is not stably faster. Save kernels are
   0.65% slower. Two formal total comparisons ranged from 0.33% faster to 0.21%
   slower, so legacy remains the default.
4. **Does vector2 add benefit?** No. FP16 vec2 save kernels are 8.08% slower
   than native scalar; BF16 vec2 is 8.10% slower than legacy. It remains
   optional for correctness/instruction experiments.
5. **What is the current INT8 reduction?** One float per thread in dynamic
   shared memory, then a six-level power-of-two max tree with barriers for the
   64-thread tile.
6. **Are CUB BlockReduce or warp reduction faster?** Yes. CUB cuts quantizer
   time 8.01% and total 1.02%; warp cuts quantizer time 5.35% and total 0.97%.
   CUB is the 64-voxel auto-default.
7. **Does `cuda::memcpy_async` fit the current INT8 kernel?** No. The source is
   loaded once into a register; there is no reusable global-to-shared input
   tile to pipeline.
8. **Are cudaMalloc/cudaFree in a hot path?** Yes, one 3 MiB `d_exact_Eold`
   allocation/free occurs in each formal low-precision/INT8 model-gradient
   forward. Larger staging allocations occur only with async offload.
9. **Are mallocAsync or buffer reuse worthwhile?** Mechanically, reuse is best
   (0.00123 vs 0.12055 ms/call; mallocAsync 0.00497 ms), but the end-to-end
   opportunity is ~0.06% and safe concurrent reuse is nontrivial. Not now.
10. **Do stream/event create/destroy have measurable cost?** Yes, roughly
    0.01-0.04 ms/native async call, but only on opt-in async offload and below
    0.2%; PCIe copies dominate.
11. **Is CUDA Graph worthwhile?** Worth a dedicated full-solver prototype, not
    a default change yet. Real traces expose 10-12 ms/call outside native kernel
    duration; a synthetic 9600-node replay saved 5.39 ms but requires persistent
    scratch and robust graph caching.
12. **Are atomics a real hotspot?** Source/receiver atomics are not (about 2%
    combined). Adjoint E/H kernels are large (36.5% of INT8 kernel time), but
    NCU counters are unavailable, so the atomic share is unproven. Their exact
    transpose semantics make them high risk.
13. **What other hand-written operations have official equivalents?** The
    useful findings were FP16/BF16 conversions and INT8 block reduction.
    Allocation/stream lifecycle has official async/pool alternatives but is
    not valuable enough end-to-end today.
14. **What looks hand-written but should not be replaced by a library?** The
    FDTD/CPML stencil, compact padded-field gather, signed INT8 packing/clamp,
    fused decode-gradient, and exact adjoint scatter should stay custom.
    cuBLAS/cuDNN/cuFFT/cuSPARSE/CUTLASS do not match these operations.
15. **What is the single highest-value next optimization?** Build a
    DeepGPR-specific CUDA Graph prototype for a fixed configuration, after
    moving the 3 MiB scratch buffer to a concurrency-safe persistent workspace,
    then repeat the same correctness and 5+20 A/B gate.
