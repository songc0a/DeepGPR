#!/usr/bin/env bash
# Supplementary runs of the sand-pit-like 3D benchmark (examples/subgrid/amr_sandpit3d.py):
#   coarse_long      uniform 2.25 cm to 3600 s (its converged accuracy)     M1-M4
#   amr_dyn_v3_long  dynamic v3 continued from 2400 s to 10800 s            M1-M4
# then the report and the three-comparison report (claims.md, claim{1,2,3}_*.png) after each step.
# Resumable: finished methods are skipped.  Progress: results/sandpit3d/chain_supp.log
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
export PYTHONPATH=src
OUT=examples/subgrid/results/sandpit3d
LOG=$OUT/chain_supp.log
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
for k in coarse_long amr_dyn_v3_long; do
    for m in M1 M2 M3 M4; do
        step run "$m" --methods "$k"
        step report
        step claims
    done
done
echo "=== $(date '+%F %T') CHAIN DONE" >> "$LOG"
