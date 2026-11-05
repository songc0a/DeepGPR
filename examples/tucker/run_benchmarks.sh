#!/usr/bin/env bash
# Run every Tucker benchmark at full scale (120 x 120 x 100) and write the report.
# Each configuration runs in its own process. Needs ~18 GiB of free GPU memory
# for the uncompressed reference. Usage: bash examples/tucker/run_benchmarks.sh [--quick]
set -eu
cd "$(dirname "$0")/../.."
PY=${PY:-python}
export PYTHONPATH=src
"$PY" examples/tucker/benchmark_tucker.py wavefield "$@"
"$PY" examples/tucker/benchmark_tucker.py model "$@"
"$PY" examples/tucker/benchmark_tucker.py fwi "$@"
"$PY" examples/tucker/benchmark_tucker.py report "$@"
