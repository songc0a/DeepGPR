#!/usr/bin/env bash
# Round-6 reference: dense with the round-5 tuned learning rate 0.01 under the
# v6 schedule (MS-SSIM, 500 iterations, cosine, sigma from 250), after the main
# round-6 queue; GPU 5, or GPU 4 if it is idle. Results: results/full_v6/fwi_tune.json.
cd ~/DeepGPR-tucker/examples/tucker/siren_comparison
export PYTHONPATH=$HOME/DeepGPR-tucker/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=~/anaconda3/envs/gprmax/bin/python
L=results/full_v6/logs
st() { echo "$(date "+%F %T") reference: $*" >> $L/status; }
free() { nvidia-smi -i $1 --query-gpu=memory.free --format=csv,noheader,nounits; }
pick() { while true; do for g in 4 5; do if [ "$(free $g)" -ge 14000 ]; then sleep 60; [ "$(free $g)" -ge 14000 ] && { echo $g; return; }; fi; done; sleep 120; done; }
# GPU 4 as soon as it is idle; GPU 5 only after the main queue
while true; do
  [ "$(free 4)" -ge 14000 ] && break
  grep -q "ALL DONE" $L/status && break
  sleep 120
done
for pass in 1 2; do
  g=$(pick); st "dense lr0.01 gpu$g pass$pass start"
  CUDA_VISIBLE_DEVICES=$g $PY bench_fwi.py tune --case 3d --protocol v6 --only dense --loss mssim --lr 0.01 --schedule cosine \
    --iterations 500 --sigma-start 250 --tag v6_lr0.01 >> $L/reference_dense_lr0.01.log 2>&1
  st "dense lr0.01 gpu$g pass$pass exit $?"
done
st "DONE"
