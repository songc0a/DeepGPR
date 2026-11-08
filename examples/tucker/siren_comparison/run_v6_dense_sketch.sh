#!/usr/bin/env bash
# Round-6 follow-up: dense with module-B Tucker wavefield compression (rank 32,
# rhs from_e, 32-step segments, host recovery with 8 threads) under the v6
# schedule and the v6 dense learning rate (0.03), GPU 7; compare with the exact
# run 3d|dense|seed0 of results/full_v6/fwi.json. Results: fwi_tune.json.
cd ~/DeepGPR-tucker/examples/tucker/siren_comparison
export PYTHONPATH=$HOME/DeepGPR-tucker/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=~/anaconda3/envs/gprmax/bin/python
L=results/full_v6/logs
st() { echo "$(date "+%F %T") sketch: $*" >> $L/status; }
for pass in 1 2; do
  st "dense sketch r32 gpu7 pass$pass start"
  CUDA_VISIBLE_DEVICES=7 $PY bench_fwi.py tune --case 3d --protocol v6 --only dense --loss mssim --lr 0.03 --schedule cosine \
    --iterations 500 --sigma-start 250 --sketch-rank 32 --sketch-segment 32 --tag v6_sketch_r32 >> $L/dense_sketch_r32.log 2>&1
  st "dense sketch r32 gpu7 pass$pass exit $?"
done
st "DONE"
