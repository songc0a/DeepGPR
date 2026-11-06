#!/usr/bin/env bash
# Round 3 (protocol v3) on hdnl: GPUs 7, 3, 5 (3 and 5 only once they are
# completely free; other jobs share the host). Each GPU runs its own queue.
cd ~/DeepGPR-tucker/examples/tucker/siren_comparison
export PYTHONPATH=$HOME/DeepGPR-tucker/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=~/anaconda3/envs/gprmax/bin/python
L=results/full_v3/logs
mkdir -p $L
st() { echo "$(date "+%F %T") $*" >> $L/status; }
free_mib() { nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits; }
wait_free() {  # GPU $1 idle (>= 23000 MiB free) at two checks one minute apart
  while true; do
    if [ "$(free_mib "$1")" -ge 23000 ]; then sleep 60; [ "$(free_mib "$1")" -ge 23000 ] && return; fi
    sleep 60
  done
}
job() {  # job GPU LOG ARGS...: run, then once more for runs that failed (e.g. OOM)
  local gpu=$1 log=$2; shift 2
  for pass in 1 2; do
    wait_free "$gpu"
    CUDA_VISIBLE_DEVICES=$gpu $PY bench_fwi.py "$@" --protocol v3 >> "$L/$log.log" 2>&1
    st "$log gpu$gpu pass$pass exit $?"
  done
}
CACHE=results/full/cache/observed3d_grid4x4_600.pt

st "start"
# Phase 1: learning-rate searches (and all of 2D on GPU 5).
(job 7 search_3d_a search --case 3d --only "|dense|" siren_4x128 tsiren_2x32_r8) &
(while [ ! -f $CACHE ]; do sleep 30; done
 job 3 search_3d_b search --case 3d --only tsiren_2x32_r12 tsiren_2x32_r16 tsiren_2x32_r24) &
(job 5 search_2d search --case 2d
 job 5 run_2d run --case 2d
 while [ ! -f $CACHE ]; do sleep 30; done
 job 5 search_3d_c search --case 3d --only direct_tucker) &
wait
st "phase 1 done"
# Phase 2: 17 3D runs (about 23 min each) in three queues.
(job 7 run_3d_gpu7 run --case 3d --only "|dense|" siren_4x128 "tsiren_2x32_r8|seed0" direct_tucker) &
(job 3 run_3d_gpu3 run --case 3d --only "tsiren_2x32_r8|seed1" "tsiren_2x32_r8|seed2" tsiren_2x32_r12 "tsiren_2x32_r24|seed2") &
(job 5 run_3d_gpu5 run --case 3d --only tsiren_2x32_r16 "tsiren_2x32_r24|seed0" "tsiren_2x32_r24|seed1") &
wait
st "ALL DONE"
