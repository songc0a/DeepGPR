#!/usr/bin/env bash
# Round-6 follow-up: Tucker-SIREN with higher ranks (r32 on GPU 2, r48 on GPU 5),
# v6 protocol (learning-rate search on held-out shots, then MS-SSIM, 500
# iterations, cosine lr, sigma from 250, exact wavefield history).
cd ~/DeepGPR-tucker/examples/tucker/siren_comparison
export PYTHONPATH=$HOME/DeepGPR-tucker/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=~/anaconda3/envs/gprmax/bin/python
L=results/full_v6/logs
st() { echo "$(date "+%F %T") tsiren-rank: $*" >> $L/status; }
chain() {  # chain GPU LABEL
  local gpu=$1 label=$2
  for step in search run; do
    for pass in 1 2; do
      st "$label $step gpu$gpu pass$pass start"
      if [ $step = search ]; then
        CUDA_VISIBLE_DEVICES=$gpu $PY bench_fwi.py search --case 3d --protocol v6 --only $label >> $L/rank_${label}.log 2>&1
      else
        CUDA_VISIBLE_DEVICES=$gpu $PY bench_fwi.py run --case 3d --protocol v6 --sigma-scale 1 --seeds 0 --only "$label|seed0" >> $L/rank_${label}.log 2>&1
      fi
      st "$label $step gpu$gpu pass$pass exit $?"
    done
  done
}
chain 2 tsiren_2x32_r32 &
chain 5 tsiren_2x32_r48 &
wait
st "DONE"
