# DeepGPR CUDA official-primitives audit

Date: 2026-08-27 (Asia/Shanghai)

## Scope and evidence

The only baseline for this audit was the workspace state at task start. The
existing block-INT8 payload, scale (`max_abs / 127`), round-to-nearest-even,
clamp, packed layout, and fused backward decode-gradient were not rewritten.
The workspace has an empty `.git` directory and is not a usable Git repository,
so there is no meaningful `git status` or diff baseline.

Hardware/software: NVIDIA GeForce RTX 4090, driver 575.51.03 (driver reports
CUDA 12.9), CUDA toolkit/nvcc 12.4.131, Nsight Systems 2023.4.4, Nsight Compute
2025.2, Python 3.10, and PyTorch 1.11.0+cu115. The tested library was
`/home/ll/下载/DeepGPR-dev/src/DeepGPR/lib/deepgpr.so`, ABI 6, compiled by:

```text
nvcc -std=c++14 -O3 -lineinfo -Xptxas=-v -arch=sm_89 --shared \
  -Xcompiler -fPIC -o src/DeepGPR/lib/deepgpr.so \
  src/DeepGPR/lib/deepgpr.cu
```

Formal timings use CUDA Events with no profiler attached: `nx=512`, `ny=384`,
`nt=1200`, 4 shots, 64 receivers, sampling interval 1, 5 warmups, and 20
measured repetitions. Nsight Systems traces contain one warmup plus one
measured run. Nsight Compute hardware counters were not available:
`ERR_NVGPUCTRPERM` was returned for a targeted INT8 kernel, so occupancy,
cache, throughput, and stall counters are explicitly **not measured**.

## Frozen INT8 implementation

| Item | Workspace implementation |
|---|---|
| E quantizer | `quantize_e_int8_snapshot_gpu`, `deepgpr.cu:2021` |
| RHS quantizer | `quantize_rhs_int8_snapshot_gpu`, `deepgpr.cu:2091` |
| Default tile | 2D `8x8x1`; 3D `4x4x4`; both 64 threads |
| Frozen reduction | one float/thread in 256 B dynamic shared memory, power-of-two tree and a barrier per level, `deepgpr.cu:1966` |
| Scale | unchanged `max_abs / 127`, fallback scale 1 near zero |
| Encode | unchanged `__float2int_rn(value / scale)` and clamp `[-127, 127]` |
| Layout | contiguous signed-INT8 values followed by four-byte-aligned FP32 scales |
| Backward | fused scale load, register decode and gradient accumulation in `accumulate_material_gradients_int8_gpu`, `deepgpr.cu:2285` |

The frozen current backend remains selectable as
`int8_reduction_backend="current"`. CUB and warp implementations change only
the tile maximum; quantization and backward semantics are shared.

## Audit matrix

| Priority | Candidate | Current implementation | Official/authoritative alternative | Measured result | Risk / complexity | Recommendation |
|---|---|---|---|---|---|---|
| P0 | FP16 conversion | manual IEEE bit manipulation at `deepgpr.cu:101-165` | `<cuda_fp16.h>`: `__float2half_rn`, `__half2float` | native scalar save kernels -6.78%; formal total -2.66% in final A/B and -3.18% in the prior independent A/B | low; output is bit-exact for all non-NaNs tested | **Implemented; CUDA FP16 auto-default is native scalar; legacy retained** |
| P2 | BF16 conversion | manual rounding/bit shift at `deepgpr.cu:167-175` and load shift | `<cuda_bf16.h>`: `__float2bfloat16_rn`, `__bfloat162float` | save kernels +0.65% slower; repeated full results changed sign (-0.33% to +0.21%) | low correctness risk, no stable speedup | Implemented as explicit A/B path; **legacy remains auto-default** |
| P2 | FP16/BF16 pair store | scalar conversion/store | `__half2`, `__nv_bfloat162` and pair intrinsics | FP16 vec2 save kernels +0.76% vs legacy and +8.08% vs native scalar; BF16 vec2 +8.10% vs legacy | 40 registers vs 32/36; padded source requires gather and boundary handling | Retain optional correctness path; do not default |
| P0 | INT8 block maximum | 64-thread shared-memory tree | NVIDIA CUB `BlockReduce<float,64>` | quantize E+R -8.01%; formal forward -1.90%, total -1.02% | low; 16 B static shared, no spill | **Implemented; auto-default for 64-voxel tiles** |
| P2 | INT8 block maximum | same | two-warp shuffle reduction | quantize E+R -5.35%; formal forward -1.93%, total -0.97% | custom maintenance burden; no win over CUB total | Retain explicit A/B backend, not default |
| P2 | INT8 scalar loads/stores | each thread gathers padded field value and writes one compact byte | CUB BlockLoad/BlockStore or `float2/4`, `uint2/4` | no isolated hotspot proving packing is limiting; current scalar stores are already coalesced | source is not globally contiguous across padded row/shot boundaries; packing distributed q values adds shuffle/sync | Do not mechanically replace; future experiment only with alignment/boundary proof |
| Not applicable | INT8 shared-tile prefetch | source value goes global-to-register once; shared memory holds reduction state/scales, not a reusable input tile | `cuda::memcpy_async`, `cuda::pipeline`, cooperative-groups async copy | no global-to-shared tile reuse exists | forced tiling would add storage and synchronization | **Not applicable** |
| P2 | exact old-E temporary | one `cudaMalloc/cudaFree` per low-precision/INT8 model-gradient forward; 3 MiB in formal mode-2 case | persistent workspace or `cudaMallocAsync/cudaFreeAsync` | 3 MiB microbenchmark: 0.120548 ms/call current, 0.001232 ms reuse, 0.004967 ms async | global reuse needs device/stream/shape keys and concurrent-autograd lifetime safety | Measurable mechanism cost but only ~0.06% of 200 ms iteration; do not add complexity now |
| P2 | async-offload staging buffers | forward/backward allocate staging buffers per call | pool or persistent per-call workspace | separate 128x96x300 A/B: GPU-resident total/wall 15.030/15.122 ms; async 18.081/18.189 ms; transfers dominate | path is opt-in; pinned histories and PCIe traffic dominate | Keep separate from default baseline; persistent buffers are low priority |
| P2 | stream/event lifecycle | async-offload creates 2 streams and 4 forward / 3 backward events per native call | persistent stream/event set | mechanism: 0.016509 ms create/destroy vs 0.005591 ms reuse; Nsys lifecycle about 0.037 ms/native call and below 0.2% | reuse must preserve PyTorch stream ordering and concurrent calls | Not a current hotspot; no code change |
| P1 | repeated launches | about 15,602 native kernels and 16,016 total launches per full call | CUDA Graph capture/replay | real NVTX-minus-native-kernel envelope is 10.35-12.32 ms/call; 9600-node synthetic prototype 12.484742 ms direct vs 7.094854 ms replay, instantiate 22.860540 ms | full graph needs persistent scratch, stable tensor addresses/config and safe replay across autograd calls | Highest-value next prototype; do not replace default until a full DeepGPR A/B passes |
| P1 / high risk | adjoint stencil atomics | transpose scatter atomics in `add_staggered_*_adjoint`, `deepgpr.cu:759-788` | local/warp/block aggregation or gather-form transpose | `adjoint_h` and `adjoint_e` are 36.5% of INT8 GPU kernel time, but NCU counters unavailable, so atomic fraction/collision is unmeasured | changing update order can alter the exact discrete adjoint and numerical reproducibility | Audit only; require NCU evidence and dot-product/Taylor tests before any experiment |
| Not applicable | source/receiver atomics | source injection, receiver adjoint and source-gradient atomics | warp aggregation when collisions are high | fused source/sample 1.2% and adjoint receiver 0.8% of GPU kernel time; benchmark has one source and mostly unique locations | collision depends on user acquisition geometry | Keep atomics; optional collision-heavy benchmark before revisiting |
| Not applicable | fused source/sample barrier | one block injects sources, barrier, then samples receivers | cooperative-groups barrier | barrier is required when source and receiver coincide; kernel is only 1.2% | no semantic/performance gain demonstrated | Keep current code |
| Not applicable | rounding/clamp/math | INT8 already uses `__float2int_rn`; explicit clamp preserves signed symmetric range; FDTD uses ordinary IEEE division | `__fdividef`, fast math, generic min/max | no math hotspot/correctness allowance; no `exp/sqrt` in CUDA solver | fast math changes physics/gradient numerics and NaN semantics | Keep; do not enable `--use_fast_math` |
| Not applicable | FDTD stencil | direct finite-difference kernels | cuBLAS/cuDNN/cuFFT/cuSPARSE/CUTLASS | computation is not GEMM, convolution API, FFT, or sparse-matrix workload in its current layout | conversion would be a mathematical/layout rewrite | Do not introduce unrelated libraries |
| P0 | build consistency | README manual build formerly omitted an explicit architecture; package build only bundles native files | one documented audited nvcc command | installed `.so` verified at the expected path, sm_89, O3, lineinfo, ptxas verbose | low | README command now matches the audited build; no fast math |

## Runtime allocation and async-offload details

`CudaCallResources` releases buffers, streams, and events at
`deepgpr.cu:262-290`. In the default GPU-resident path, streams/events are null.
The only native runtime allocation in the formal FP16/BF16/INT8 model-gradient
forward is `d_exact_Eold` at `deepgpr.cu:2523`:

```text
elements = components * shots * nx * ny * nz
formal mode-2 bytes = 1 * 4 * 512 * 384 * 1 * 4 = 3,145,728 B
frequency = once per forward; cudaFree occurs before the native call returns
```

The opt-in async path creates resources at `deepgpr.cu:2509-2519` and
`deepgpr.cu:2896-2904`. Nsight on the separate small case measured, across one
warmup and one measured full call, 8 `cudaStreamCreate`, 14 `cudaEventCreate`,
and about 7.01 ms of H2D+D2H GPU copy time. Stream/event lifecycle was tiny
compared with the transfers. It must not be attributed to the default GPU-only
path.

The standalone mechanism benchmark is in
`tests/benchmark_cuda_runtime_primitives.cu`. It is deliberately not an
implementation change: it provides an upper bound before taking on the
concurrency and ownership work of persistent native resources.

## CUDA Graph feasibility

Nsight reports 31,204 native kernel instances across the warmup plus measured
full call (15,602 per call). The NVTX envelope exceeds summed native kernel
duration by 10.35 ms/call for INT8 current, 11.52 ms/call for FP16 legacy, and
12.32 ms/call for FP16 native. This is evidence of a measurable non-kernel
envelope, not proof that every nanosecond is launch latency.

Capture is structurally possible for a fixed shape, `nt`, FDTD order, PML,
storage backend, and stable tensor addresses. Each time-step index is captured
as a distinct node argument. A production graph requires moving `d_exact_Eold`
outside capture, defining cache/lifetime keys, invalidating on configuration or
address changes, and proving PyTorch current/default-stream and concurrent
autograd safety. The synthetic 9600-node result justifies a DeepGPR-specific
prototype; it does not justify changing the default today.

## Build-path audit

- `pyproject.toml` packages `.cu/.h/.so` files but does not compile CUDA.
- README is the only formal CUDA build command found.
- The README command now includes the same `-O3 -lineinfo -Xptxas=-v
  -arch=sm_89` flags used for every formal measurement.
- No build path enables `--use_fast_math`; none was added.
- Change `-arch` for a different target GPU rather than reusing the RTX 4090
  binary.

## Conclusion

Two authoritative substitutions passed both correctness and performance
gates: native scalar FP16 conversion and CUB BlockReduce for the default INT8
tile. BF16 native and both vec2 paths passed correctness but not the stable
performance gate. Allocation and stream/event reuse have measurable
micro-costs but negligible end-to-end value at the current iteration time.
The next justified experiment is a full-solver CUDA Graph prototype; the
adjoint atomic path is larger but carries materially higher discrete-adjoint
risk and lacks hardware-counter evidence on this machine.
