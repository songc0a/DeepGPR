# DeepGPR Tests and Verification

This directory contains the fast numerical unit tests and executable notebook
verification suite for the repository-local DeepGPR implementation. Every test
entry point places `../src` first on `sys.path`, verifies the local package and
native library where applicable, and requires native ABI 6. An installed
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

`checkpoint_example.ipynb` is a separate CUDA example rather than part of the
release-report sequence. It compares full-history autograd with segmented
temporal recomputation, verifies receiver and material-gradient agreement, and
plots measured peak GPU memory (about 20 GiB for the default uncheckpointed
case), runtime, traces, and permittivity gradients.

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

`benchmark_wavefield_compression.py` runs FP32, FP16, and fused block-INT8 on
the same acquisition. Its default case is `512 x 384`, 1200 time steps, four
shots, and 64 receivers. The two FP32 saved histories contain about 7.0 GiB,
which is large enough to expose bandwidth effects while fitting comfortably on
a 24 GiB RTX 4090 with the current solver. It reports saved-history bytes, peak
allocated CUDA memory, wall time, CUDA-event time, host/synchronization delta,
wavefield reconstruction errors, epsilon/conductivity gradient relative L2
errors and cosine similarities, plus payload-rate estimates:

```bash
python tests/benchmark_wavefield_compression.py --device cuda:0
# Reproduce the former smaller case:
python tests/benchmark_wavefield_compression.py --device cuda:0 \
  --nx 384 --ny 256 --nt 800 --shots 4 --receivers 32
```

The optional PyTorch/CUPTI profile is intrusive, so use ordinary repeats for
the headline iteration runtime. It prints forward and backward CUDA time split
into coefficient construction, FDTD/CPML updates, source/receiver work,
history writes, fused history-gradient work, and transfers. It also lists the
top individual kernels and CUDA synchronization API time. Limit the intrusive
profile to one storage mode when desired, and optionally save traces that can
be opened in Perfetto (`https://ui.perfetto.dev`):

```bash
python tests/benchmark_wavefield_compression.py --device cuda:0 \
  --profile-kernels --profile-mode int8 --profile-top-kernels 20 \
  --trace-dir compression_traces --warmup 1 --repeats 5
```

The INT8 backward material-gradient kernel is intentionally fused: history
read, block decode, and gradient accumulation happen inside one kernel. They do
not occupy separable timeline intervals. The script reports that fused kernel's
total time and effective stored/logical payload rates; use Nsight Compute to
explain its internal stalls and throughput.

`benchmark_runtime.py` provides the same deterministic case on CPU or CUDA,
reports peak memory, saves outputs and gradients, and can compare a run with a
saved reference:

```bash
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE python tests/benchmark_runtime.py \
  --device cpu --warmup 2 --repeats 7 --save-result cpu_result.pt
python tests/benchmark_runtime.py --device cuda:0 --nx 384 --ny 256 \
  --nt 800 --shots 4 --receivers 32 --warmup 2 --repeats 7 \
  --torch-profile deepgpr_trace.json --save-result cuda_result.pt
python tests/benchmark_runtime.py --device cuda:0 --reference cuda_reference.pt
```

Use NVIDIA Nsight tools for native timelines and kernel metrics when they are
available on the CUDA host:

```bash
nsys profile --trace=cuda,nvtx,osrt -o deepgpr_compression_nsys \
  --force-overwrite=true \
  python tests/benchmark_wavefield_compression.py --device cuda:0 \
    --warmup 1 --repeats 1 --nvtx
ncu --set full --target-processes all -o deepgpr_ncu \
  python tests/benchmark_runtime.py --device cuda:0 --warmup 0 --repeats 1
```

Nsight Systems is the right first tool for the end-to-end timeline: kernel
ordering, stream overlap/gaps, CUDA API calls, CPU launch time, synchronization,
and waiting. Nsight Compute profiles selected kernels in depth: DRAM/cache
traffic, achieved occupancy, register use, instruction mix, and warp-stall
reasons. It is not the right tool for measuring the full iteration wall time.

For a focused before/after material-gradient comparison, profile each storage
mode separately and filter both gradient kernel variants:

```bash
ncu --target-processes all --kernel-name-base demangled \
  --kernel-name regex:accumulate_material_gradients_int8_gpu.* \
  --launch-count 1 \
  --metrics \
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,launch__occupancy_limit_registers,\
smsp__sass_inst_executed_op_local_ld.sum,smsp__sass_inst_executed_op_local_st.sum \
  -o deepgpr_int8_gradient \
  python tests/benchmark_wavefield_compression.py --device cuda:0 \
    --warmup 0 --repeats 1
```

Run `ncu --query-metrics` first when moving between Nsight Compute versions;
metric spellings occasionally change. Do not call the optimization successful
from compression ratio alone: compare iteration time, DRAM bytes/throughput,
long-scoreboard stalls, occupancy, register count, spills, and gradient cosine.
