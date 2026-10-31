#!/usr/bin/env bash
# Round 2 of the equal-budget study (examples/subgrid/amr_budget_study.py):
#   1. reference data of the fresh models M5 / M6;
#   2. every scenario: the methods not run yet (AMR v2 "fine" and its one-shot variant on the
#      round-1 scenarios; all methods on the fresh ones), PE evaluation, figures / GIFs;
#   3. long uniform 1 cm FDTD runs (FDTD1_BUDGET seconds) on the 20 dB scenarios;
#   4. the report after every step.
# Resumable: finished work is skipped when the chain is started again.
#
#   PY=/path/to/python setsid nohup examples/subgrid/run_amr_budget_study2.sh &
#
# Progress: examples/subgrid/results/amr_budget/chain2.log
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
export PYTHONPATH=src
OUT=examples/subgrid/results/amr_budget
LOG=$OUT/chain2.log
mkdir -p "$OUT"
S=examples/subgrid/amr_budget_study.py

step() {
    echo "=== $(date '+%F %T') start: $*" >> "$LOG"
    local t0=$SECONDS
    if "$PY" "$S" "$@" >> "$OUT/stdout2.log" 2>&1; then
        echo "=== $(date '+%F %T') done ($((SECONDS - t0)) s): $*" >> "$LOG"
    else
        echo "!!! $(date '+%F %T') FAILED ($((SECONDS - t0)) s): $* (see $OUT/stdout2.log)" >> "$LOG"
    fi
}

echo "=== $(date '+%F %T') CHAIN START (python: $PY)" >> "$LOG"
step data
for sc in $("$PY" "$S" list 2>/dev/null); do
    step scenario "$sc"
    step report
done
for sc in $("$PY" "$S" list_fdtd1 2>/dev/null); do
    step fdtd1long "$sc"
    step report
done
step report
echo "=== $(date '+%F %T') CHAIN DONE" >> "$LOG"
