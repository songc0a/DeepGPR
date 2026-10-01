#!/usr/bin/env bash
# Build the native libraries from this tree and run the complete validation.
#
# Usage (inside the conda environment that has PyTorch/CUDA):
#   bash tools/run_local_validation.sh [REFERENCE_DIR] [--quick]
#
# REFERENCE_DIR is a checkout of the previous release (default: ~/下载/DeepGPR-dev,
# then ~/Downloads/DeepGPR-dev). Reports are written to ./validation/.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE="${1:-}"
QUICK=""
for argument in "$@"; do [[ "$argument" == "--quick" ]] && QUICK="--quick"; done
if [[ -z "$REFERENCE" || "$REFERENCE" == "--quick" ]]; then
    for candidate in "$HOME/下载/DeepGPR-dev" "$HOME/Downloads/DeepGPR-dev"; do
        [[ -d "$candidate" ]] && REFERENCE="$candidate" && break
    done
fi
if [[ ! -d "$REFERENCE/src/DeepGPR" ]]; then
    echo "Reference release not found (expected <dir>/src/DeepGPR): '$REFERENCE'" >&2
    exit 2
fi

LIB="$ROOT/src/DeepGPR/lib"
OUT="$ROOT/validation"
mkdir -p "$OUT"
cd "$ROOT"
status=0

echo "== Environment"
python - <<'PY' | tee "$OUT/environment.txt"
import platform, torch
print("python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
PY

echo "== Building the CPU backend"
cc -std=c99 -O3 -fopenmp -fPIC -shared -o "$LIB/deepgpr_cpu.so" "$LIB/deepgpr_cpu.c" \
    2>&1 | tee "$OUT/build_cpu.log" || status=1

if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "== Building the CUDA backend"
    NVCC="$(command -v nvcc || true)"
    [[ -z "$NVCC" && -n "${CUDA_HOME:-}" ]] && NVCC="$CUDA_HOME/bin/nvcc"
    [[ -z "$NVCC" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/nvcc" ]] && NVCC="$CONDA_PREFIX/bin/nvcc"
    if [[ -z "$NVCC" ]]; then
        echo "nvcc not found: install it (e.g. 'conda install -c nvidia cuda-nvcc') or set CUDA_HOME." \
            | tee "$OUT/build_cuda.log"
        status=1
    else
        ARCH="$(python -c "import torch; m, n = torch.cuda.get_device_capability(0); print(f'sm_{m}{n}')")"
        cp "$LIB/deepgpr.so" "$OUT/deepgpr.so.previous" 2>/dev/null || true
        "$NVCC" -std=c++14 -O3 -lineinfo -arch="$ARCH" --shared -Xcompiler -fPIC \
            -o "$LIB/deepgpr.so" "$LIB/deepgpr.cu" 2>&1 | tee "$OUT/build_cuda.log"
        [[ ${PIPESTATUS[0]} -eq 0 ]] || status=1
    fi
fi

echo "== Unit tests"
python -m unittest discover -s tests -p "test_*.py" -v > "$OUT/unittest.log" 2>&1
unit=$?
tail -n 5 "$OUT/unittest.log"
[[ $unit -eq 0 ]] || status=1

echo "== Comparison with the reference release ($REFERENCE)"
python tools/verify_equivalence.py --reference-src "$REFERENCE/src/DeepGPR" $QUICK \
    --json "$OUT/equivalence.json" 2>&1 | tee "$OUT/equivalence.log"
[[ ${PIPESTATUS[0]} -eq 0 ]] || status=1

echo
if [[ $status -eq 0 ]]; then echo "VALIDATION PASSED  (reports in $OUT)"; else echo "VALIDATION FAILED  (see $OUT)"; fi
exit $status
