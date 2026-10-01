# Building the native backends

The Python package loads prebuilt shared libraries from `src/DeepGPR/lib`. Rebuild them after changing `deepgpr.cu` or `deepgpr_cpu.c` (CI does this automatically, see `.github/workflows/workflow.yaml`).

## CUDA backend

Build the CUDA shared library from the repository root on Linux. `-lineinfo`
preserves source correlation for Nsight; `-Xptxas=-v` prints registers and spill
stores/loads for each kernel.

```bash
nvcc -std=c++14 -O3 -lineinfo -Xptxas=-v -arch=sm_89 --shared -Xcompiler -fPIC \
  -o src/DeepGPR/lib/deepgpr.so src/DeepGPR/lib/deepgpr.cu
```

The command above is the audited RTX 4090 build path. Select the matching
architecture when building for a different GPU; do not add `--use_fast_math`
unless a separate numerical and gradient validation explicitly permits it.

The new INT8 path deliberately keeps native ABI 6 because the forward/backward
C signatures are unchanged. A capability symbol prevents an older ABI-6 CUDA
library from accepting the packed storage code and corrupting memory. External PML also requires the `deepgpr_supports_external_pml` capability on CPU and CUDA, including the INT8 gradient path. Rebuild native libraries on each target platform; older binaries are rejected for PML runs rather than silently dropping physical edge gradients.

## CPU backend

The CPU backend is a plain C shared library and is built with OpenMP by default. Build it into `src/DeepGPR/lib` before running with `device='cpu'`. You can control CPU thread count with `OMP_NUM_THREADS`.

```bash
# Linux
cc -std=c99 -O3 -fopenmp -fPIC -shared -o src/DeepGPR/lib/deepgpr_cpu.so src/DeepGPR/lib/deepgpr_cpu.c

# macOS
brew install libomp
LIBOMP_PREFIX="$(brew --prefix libomp)"
LIBOMP_RUNTIME_NAME="/opt/llvm-openmp/lib/libomp.dylib"
cc -std=c99 -O3 -Xpreprocessor -fopenmp -DDEEPGPR_USE_OPENMP -I"$LIBOMP_PREFIX/include" -L"$LIBOMP_PREFIX/lib" -fPIC -shared -o src/DeepGPR/lib/deepgpr_cpu.dylib src/DeepGPR/lib/deepgpr_cpu.c -lomp
cp "$LIBOMP_PREFIX/lib/libomp.dylib" src/DeepGPR/lib/libomp.dylib
install_name_tool -id "$LIBOMP_RUNTIME_NAME" src/DeepGPR/lib/libomp.dylib
LIBOMP_DEP="$(otool -L src/DeepGPR/lib/deepgpr_cpu.dylib | awk '/libomp\.dylib/ {print $1; exit}')"
install_name_tool -change "$LIBOMP_DEP" "$LIBOMP_RUNTIME_NAME" src/DeepGPR/lib/deepgpr_cpu.dylib
codesign --force --sign - src/DeepGPR/lib/libomp.dylib
codesign --force --sign - src/DeepGPR/lib/deepgpr_cpu.dylib
```

On Windows, build `src\DeepGPR\lib\deepgpr_cpu.dll` with MSVC:

```powershell
cl /LD /O2 /openmp /Fe:src\DeepGPR\lib\deepgpr_cpu.dll src\DeepGPR\lib\deepgpr_cpu.c
```

After rebuilding, run `python -m unittest tests/test_native_abi.py` to confirm the library exports ABI 6 and matches the Python ABI table.
