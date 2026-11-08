#!/usr/bin/env bash
# Round-6 follow-up: Tucker-SIREN with smaller learning rates than the searched
# 3e-4 (v6 schedule: MS-SSIM, 500 iterations, cosine, sigma from 250).
# r16 on GPU 2, r24 on GPU 5, lr 1e-4 and 3e-5 side by side on each GPU.
# Results: results/full_v6/fwi_tune.json.
cd ~/DeepGPR-tucker/examples/tucker/siren_comparison
export PYTHONPATH=$HOME/DeepGPR-tucker/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=~/anaconda3/envs/gprmax/bin/python
L=results/full_v6/logs
st() { echo "$(date "+%F %T") tsiren-lr: $*" >> $L/status; }
one() {  # one GPU LABEL LR
  local gpu=$1 label=$2 lr=$3
  for pass in 1 2; do
    st "$label lr$lr gpu$gpu pass$pass start"
    CUDA_VISIBLE_DEVICES=$gpu $PY bench_fwi.py tune --case 3d --protocol v6 --only $label --loss mssim --lr $lr --schedule cosine \
      --iterations 500 --sigma-start 250 --tag v6_lr$lr >> $L/tsiren_lr_${label}_$lr.log 2>&1
    st "$label lr$lr gpu$gpu pass$pass exit $?"
  done
}
one 2 tsiren_2x32_r16 1e-4 &
sleep 120; one 2 tsiren_2x32_r16 3e-5 &
one 5 tsiren_2x32_r24 1e-4 &
sleep 120; one 5 tsiren_2x32_r24 3e-5 &
wait
st "DONE"
