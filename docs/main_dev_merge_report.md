# DeepGPR main/dev recovery and merge report

Date: 2026-08-27 (Asia/Shanghai)

## Outcome

The downloaded `dev` snapshot was protected with a complete file-level backup,
attached without modifying its working files to the official `origin/dev`
history, committed, merged with the latest `main`, rebuilt, tested, benchmarked,
fast-forwarded into `main`, and pushed without force.

The validated functional commit pushed to `origin/main` is
`0e55cf68dd210b21c3149b88dbf66b18a0029e5f`. This report is delivered by a
follow-up documentation-only fast-forward commit; that follow-up does not alter
the validated source, binaries, tests, or benchmark results.

## Required recovery and Git facts

1. **Did the original directory have `.git`?** No usable Git metadata existed.
   Before recovery, Git reported `fatal: not a git repository`. The desktop
   environment later exposed an empty, read-only `.git` placeholder, but the
   escalated host view and the backup contained no `.git` directory.
2. **File-level backup:**
   `/home/ll/下载/DeepGPR-dev-before-git-20260827-153548` (152 MiB).
3. **Official remote URL:** `git@github.com:songc0a/DeepGPR.git` for fetch and
   push.
4. **Recovered local `dev` HEAD:**
   `f71ff24c52e7b23b26ebd5ec6512ab2551f8cbbc`.
5. **`origin/dev` at recovery:**
   `f71ff24c52e7b23b26ebd5ec6512ab2551f8cbbc`; it exactly matched recovered
   `dev` HEAD before local changes were committed.
6. **File identity before/after Git recovery:** Exact for project files.
   `diff -qr` was empty after excluding `.git` and the app-injected empty
   `.agents/.codex` placeholders. Git initialization and `reset --mixed` did
   not alter any protected file.
7. **Changes relative to `origin/dev` before the first commit:**
   - Modified: `README.md`, `src/DeepGPR/__init__.py`,
     `src/DeepGPR/compute2.py`, `src/DeepGPR/lib/deepgpr.cu`,
     `src/DeepGPR/lib/deepgpr.h`, `src/DeepGPR/lib/deepgpr.so`,
     `src/DeepGPR/lib/deepgpr_cpu.c`, `src/DeepGPR/lib/deepgpr_cpu.so`,
     `tests/README.md`, `tests/benchmark_wavefield_compression.py`, and
     `tests/test_wavefield_compression.py`.
   - Added: `tests/benchmark_cuda_runtime_primitives.cu`,
     `tests/checkpoint_example.ipynb`,
     `tests/test_cuda_conversion_backends.py`, and the small readable files
     under `tests/profiling_results/`.
   - Deleted: none.
   - `.gitignore` was then modified to exclude raw Nsight/NCU artifacts.
8. **Files committed:** The first explicit 65-file commit contains the source,
   tracked Linux native libraries, retained dev Windows libraries, README/test
   documentation, CUDA conversion and INT8 tests, the checkpoint notebook,
   runtime/benchmark sources, and readable Markdown/TXT/CSV profiling summaries.
   Exact names are available with `git show --name-status f7e42ce`. The later
   validation commit contains `tests/test_numerics.py`, the rebuilt tested
   `src/DeepGPR/lib/deepgpr.so`, and the pre/post merge benchmark summaries.
9. **Profiler/temporary files not committed:** 33 raw captures totaling
   133.79 MiB: 16 `*.nsys-rep`, 16 `*.sqlite`, and one `*.qdstrm` file. The
   ignore rules also cover `*.ncu-rep` and `*.qdrep`. No raw profiler file was
   ever staged. They remain recoverable on disk and in the file-level backup.
10. **Local CUDA optimization commit:**
    `f7e42ce00d02ed12cc75971f7dce29953e7a1f66`, message
    `Optimize CUDA wavefield storage and INT8 reduction`.
11. **Git backup branch:** `backup/dev-before-main-merge-f7e42ce`, retained at
    `f7e42ce00d02ed12cc75971f7dce29953e7a1f66`.
12. **`origin/main` before merge:**
    `de581f7d59eb90868979e95ab0d4539660f832b6`.

## Divergence and conflict resolution

13. **Main-only commits before merge:**
    - `de581f7` Build native DeepGPR libraries [skip ci]
    - `0b438f6` Add persistent cumulative PyPI download statistics
    - `6e92f6b` Build native DeepGPR libraries [skip ci]
    - `f2a1b8e` Refine website navigation and download statistics
    - `d52d0f4` docs: acknowledge [skip ci]
    - `7259e60` Build native DeepGPR libraries [skip ci]
    - `c2e4300` Add DeepGPR website and GitHub Pages deployment
14. **Dev-only commits before merge:**
    - `f7e42ce` Optimize CUDA wavefield storage and INT8 reduction
    - `f71ff24` Build native DeepGPR libraries [skip ci]
    - `0ce9dd9` feat: add fused INT8 wavefield compression
15. **Merge conflicts:** Only the binary files
    `src/DeepGPR/lib/deepgpr.dll` and
    `src/DeepGPR/lib/deepgpr_cpu.dll`. README and all text source files merged
    automatically without conflict markers.
16. **Core conflict resolution:** No whole-file `checkout --ours/--theirs` was
    used. Blob sizes, hashes, and exported capability strings were audited.
    The dev CUDA DLL (7,793,152 B) exports
    `deepgpr_supports_int8_wavefield`; the main DLL (7,267,840 B) does not.
    The existing dev DLLs were therefore explicitly staged, preserving the
    newer INT8-capable binaries rather than regressing to main's older build.
    Linux `deepgpr.so` was rebuilt from merged source and is the binary actually
    validated below. This Linux host cannot rebuild the Windows DLLs; those
    retained DLLs do not expose the later conversion/reduction capability
    symbols and should be refreshed by the Windows native-library build job.
    The merge commit is `4f5973b269eaba52660511485ac16cdb9f65335b`.

## Tests

17. **Before merge:** 56 tests ran: 53 passed, one skipped, one known failure,
    and one environment-compatibility error. The unchanged known failure was
    INT8 directional consistency `0.255808285 > 0.15`. PyTorch 1.11 rejected
    the newer `torch.load(weights_only=...)` test keyword. No threshold was
    changed.
18. **After merge:** The final 65-test run (including main's nine website
    statistics tests) produced 63 passed, one skipped, and only the same known
    INT8 directional failure (`0.255808145 > 0.15`). The test-only
    `torch.load` call now supplies `weights_only=True` only when supported, so
    PyTorch 1.11 passes the same exact tensor assertion. One strict bit-equality
    discrete-adjoint test showed a one-element relative difference of
    `1.22e-7` in the first aggregate run; five focused repeats gave four passes
    and one similarly tiny failure, confirming pre-existing CUDA atomic-order
    nondeterminism. It passed in the final aggregate run, and its threshold was
    not modified.

## Formal RTX 4090 benchmark

Both sides used `nx=512`, `ny=384`, `nt=1200`, four shots, 64 receivers,
sampling interval 1, five warmups, 20 measured repeats, and CUDA Event timing.
The loaded library was
`/home/ll/下载/DeepGPR-dev/src/DeepGPR/lib/deepgpr.so`.

19. **Before merge:**

| mode | forward median ms | backward median ms | total median ms | peak MiB | history MiB |
|---|---:|---:|---:|---:|---:|
| FDTD-only | 55.783119 | - | 55.783119 | 56.53 | 0.00 |
| FP32 | 80.730209 | 104.740143 | 185.439743 | 10898.73 | 7200.00 |
| FP16 default | 81.483265 | 98.089981 | 179.540985 | 5498.73 | 3600.00 |
| INT8 default | 98.510544 | 100.302689 | 198.910469 | 2967.32 | 1912.50 |

20. **After merge:**

| mode | forward median ms | backward median ms | total median ms | peak MiB | total delta |
|---|---:|---:|---:|---:|---:|
| FDTD-only | 55.981567 | - | 55.981567 | 56.53 | +0.36% forward |
| FP32 | 80.592896 | 104.894466 | 185.586174 | 10898.73 | +0.08% |
| FP16 default | 81.679985 | 97.887615 | 179.649025 | 5498.73 | +0.06% |
| INT8 default | 98.578110 | 100.615826 | 199.173119 | 2967.32 | +0.13% |

The noisy FP32 forward-only batch initially reported an 82.808834 ms median
with a 19.59 ms standard deviation. A separate formal 5+20 retry produced
76.753918 ms, only +0.89% versus the pre-merge 76.073471 ms. FP16 and INT8
forward-only deltas were +1.54% and +0.99%. Receiver output for FDTD-only and
FP32 remained exactly equal.

21. **Final FP16 default backend:** NVIDIA native scalar (`cuda_fp16.h`,
    `__float2half_rn`, `__half2float`) for CUDA FP16. Legacy and vec2 remain
    explicit reference/A-B options.
22. **Final BF16 default backend:** Legacy. Native scalar and vec2 remain
    optional/reference implementations.
23. **Final INT8 default backend:** NVIDIA CUB `BlockReduce` for the default
    64-voxel tile. The current/shared tree and warp-shuffle versions remain
    explicit reference/A-B options. Signed INT8 payloads, per-block FP32
    scales, partial tiles, E/R compressed histories, inline fused backward
    decode-gradient, and the no-full-FP32-reconstruction invariant remain
    intact.
24. **Performance regression:** None. Every final full total delta is below
    +0.14%, and every investigated forward-only delta is below +1.55%, well
    inside the 5% gate. Memory payloads are unchanged.

## Fast-forward and push

25. **Fast-forward dev -> main:** Successful with `git merge --ff-only dev`;
    no ordinary merge commit was created on main.
26. **Validated main/dev SHA:** At the functional validation and first push,
    local `main`, local `dev`, and `origin/main` all equaled
    `0e55cf68dd210b21c3149b88dbf66b18a0029e5f`. The report-only delivery commit
    advances both local branches together; its exact SHA is reported by the
    final `git rev-parse main dev` handoff output.
27. **Push status:** Successful, normal non-force push:
    `de581f7..0e55cf6  main -> main`. No force or force-with-lease was used.
28. **Final status at functional push:** Clean on main; raw profiler captures
    are ignored and retained locally. After this document is committed, the
    report-only follow-up is fast-forwarded to main and pushed normally, with
    final status rechecked in the handoff.

The local `dev` branch and `backup/dev-before-main-merge-f7e42ce` branch are
retained. The remote `dev` branch was not modified or deleted.
