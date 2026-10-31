#!/usr/bin/env bash
# Runs the whole equal-budget study (examples/subgrid/amr_budget_study.py) step by step:
# reference data and checks, every scenario, the long run, the evaluator check and the
# report (rebuilt after every scenario, so partial results can be looked at any time).
# Resumable: finished steps are skipped when the chain is started again.
#
#   PY=/path/to/python nohup examples/subgrid/run_amr_budget_study.sh &
#
# Progress: examples/subgrid/results/amr_budget/chain.log
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-python}
export PYTHONPATH=src
OUT=examples/subgrid/results/amr_budget
LOG=$OUT/chain.log
mkdir -p "$OUT"
S=examples/subgrid/amr_budget_study.py

step() {
    echo "=== $(date '+%F %T') start: $*" >> "$LOG"
    local t0=$SECONDS
    if "$PY" "$S" "$@" >> "$OUT/stdout.log" 2>&1; then
        echo "=== $(date '+%F %T') done ($((SECONDS - t0)) s): $*" >> "$LOG"
    else
        echo "!!! $(date '+%F %T') FAILED ($((SECONDS - t0)) s): $* (see $OUT/stdout.log)" >> "$LOG"
    fi
}

echo "=== $(date '+%F %T') CHAIN START (python: $PY)" >> "$LOG"
step data
for sc in $("$PY" "$S" list 2>/dev/null); do
    step scenario "$sc"
    step report
done
step longrun
step verify
step report
echo "=== $(date '+%F %T') CHAIN DONE" >> "$LOG"
