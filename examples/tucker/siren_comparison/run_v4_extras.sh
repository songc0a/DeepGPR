#!/usr/bin/env bash
# Round-4 extras in idle GPU time (GPUs 3 and 7 only; GPU 5 is left to the
# main 3D queue of schedule_v4.py): 2D oracle reference, stripe check of v3.
cd ~/DeepGPR-tucker/examples/tucker/siren_comparison
export PYTHONPATH=$HOME/DeepGPR-tucker/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=~/anaconda3/envs/gprmax/bin/python
L=results/full_v4/logs
st() { echo "$(date "+%F %T") extras: $*" >> $L/status; }
pick() {  # print a GPU from 3, 7 with >= $1 MiB free at two checks 30 s apart
  while true; do
    for g in 3 7; do
      f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)
      if [ "$f" -ge "$1" ]; then sleep 30; f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -ge "$1" ] && { echo $g; return; }; fi
    done
    sleep 60
  done
}
step() {  # step NEED LOG ARGS...: run twice (second pass retries failures)
  local need=$1 log=$2; shift 2
  for pass in 1 2; do
    g=$(pick $need); st "$log gpu$g pass$pass start"
    CUDA_VISIBLE_DEVICES=$g $PY "$@" >> $L/$log.log 2>&1; st "$log gpu$g pass$pass exit $?"
  done
}
step 13000 2d_oracle_B bench_fwi.py search_sigma --case 2d --protocol v4 --select oracle
step 13000 2d_oracle_runs bench_fwi.py run --case 2d --protocol v4 --select oracle
step 8000 stripe_check_v3 stripe_check.py --protocol v3
st "DONE"
