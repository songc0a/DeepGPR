#!/usr/bin/env bash
# Sand-pit-like 3D benchmark (examples/subgrid/amr_sandpit3d.py): reference data and forward
# floors for M1-M4, then every method on M1, M2, M3, M4 (M4 = validation), report after each.
# Resumable: finished methods are skipped.  Progress: results/sandpit3d/chain.log
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
export PYTHONPATH=src
OUT=examples/subgrid/results/sandpit3d
LOG=$OUT/chain.log
mkdir -p "$OUT"
S=examples/subgrid/amr_sandpit3d.py
step() {
    $PY examples/subgrid/ensure_gpu.py >> "$OUT/stdout.log" 2>&1
    echo "=== $(date '+%F %T') start: $*" >> "$LOG"
    local t0=$SECONDS
    if "$PY" "$S" "$@" >> "$OUT/stdout.log" 2>&1; then
        echo "=== $(date '+%F %T') done ($((SECONDS - t0)) s): $*" >> "$LOG"
    else
        echo "!!! $(date '+%F %T') FAILED ($((SECONDS - t0)) s): $*" >> "$LOG"
    fi
}
echo "=== $(date '+%F %T') CHAIN START" >> "$LOG"
for m in M1 M2 M3 M4; do step check "$m"; done
for m in M1 M2 M3 M4; do
    step run "$m"
    step report
done
echo "=== $(date '+%F %T') CHAIN DONE" >> "$LOG"
