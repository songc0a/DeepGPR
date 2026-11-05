#!/usr/bin/env bash
# Run the Tucker benchmarks at full scale (120 x 120 x 100) on up to four GPUs.
#
#   bash examples/tucker/run_benchmarks.sh GPU_A GPU_B GPU_C [PY]
#
# GPU_A: wavefield group 1 (reference, checkpointing, per-frame ranks)
# GPU_B: wavefield group 2 (E-only, time blocks, long runs) then the model benchmark
# GPU_C: FWI benchmark
# Each configuration runs in its own process; results are merged into
# examples/tucker/results/full/*.json under a file lock, then report.md is written.
set -u
cd "$(dirname "$0")/../.."
A=${1:?GPU_A}; B=${2:?GPU_B}; C=${3:?GPU_C}; PY=${4:-python}
export PYTHONPATH=src OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
LOG=examples/tucker/results/full/logs
mkdir -p "$LOG"
S=examples/tucker/benchmark_tucker.py
(CUDA_VISIBLE_DEVICES=$A "$PY" "$S" wavefield --only none none_ER_stored checkpoint_60 checkpoint_150 \
    tucker_r8 tucker_r12 tucker_r16 tucker_r24 tucker_r16_seg32 checkpoint_150+tucker_r16 > "$LOG/wavefield_A.log" 2>&1; echo "A done $?" >> "$LOG/status") &
(CUDA_VISIBLE_DEVICES=$B "$PY" "$S" wavefield --only tucker_r12_fromE tucker_r16_fromE tucker_r24_fromE tucker_r16_tb8 tucker_r24_tb8 \
    long_none long_checkpoint_150 long_tucker_r16 long_tucker_r16_fromE long_tucker_r16_tb8 long_checkpoint_150+tucker_r16 > "$LOG/wavefield_B.log" 2>&1;
 CUDA_VISIBLE_DEVICES=$B "$PY" "$S" model > "$LOG/model.log" 2>&1; echo "B done $?" >> "$LOG/status") &
(CUDA_VISIBLE_DEVICES=$C "$PY" "$S" fwi > "$LOG/fwi.log" 2>&1; echo "C done $?" >> "$LOG/status") &
wait
"$PY" "$S" report > "$LOG/report.log" 2>&1
echo "ALL DONE" >> "$LOG/status"
