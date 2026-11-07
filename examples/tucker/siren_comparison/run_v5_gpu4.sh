#!/usr/bin/env bash
# Round 5 (protocol v5, 4 x 4 x 2 m model) on GPU 4 only: stage-A learning-rate
# search on held-out shots, then one FWI run (seed 0) per method with sigma
# step 1 (stage B skipped: it chose 1 in every case of round 4). Each step
# waits until GPU 4 is idle and runs twice (the second pass retries failures,
# e.g. OOM when another job takes the GPU; FWI runs resume from checkpoints).
cd ~/DeepGPR-tucker/examples/tucker/siren_comparison
export PYTHONPATH=$HOME/DeepGPR-tucker/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4
PY=~/anaconda3/envs/gprmax/bin/python
L=results/full_v5/logs
mkdir -p $L
st() { echo "$(date "+%F %T") $*" >> $L/status; }
wait_free() {  # GPU 4 with >= 20000 MiB free at two checks one minute apart
  while true; do
    f=$(nvidia-smi -i 4 --query-gpu=memory.free --format=csv,noheader,nounits)
    if [ "$f" -ge 20000 ]; then sleep 60; f=$(nvidia-smi -i 4 --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -ge 20000 ] && return; fi
    sleep 120
  done
}
METHODS=("|dense|" siren_4x128 tsiren_2x32_r16 tsiren_2x32_r24)
step() {  # step LOG ARGS...
  local log=$1; shift
  for pass in 1 2; do
    wait_free; st "$log pass$pass start"
    $PY bench_fwi.py "$@" --protocol v5 >> $L/$log.log 2>&1; st "$log pass$pass exit $?"
  done
}
st "start (waiting for GPU 4)"
step search search --case 3d --only "${METHODS[@]}"
step run run --case 3d --sigma-scale 1 --seeds 0 --only "${METHODS[@]}"
st "ALL DONE"
