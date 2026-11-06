#!/usr/bin/env bash
# $1 = case (2d|3d), $2 = GPU
cd ~/DeepGPR-tucker/examples/tucker/siren_comparison
export PYTHONPATH=$HOME/DeepGPR-tucker/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$2
PY=~/anaconda3/envs/gprmax/bin/python
L=results/full/logs
echo "$(date "+%F %T") $1 search start (gpu $2)" >> $L/fwi_status
$PY bench_fwi.py search --case $1 > $L/fwi_search_$1.log 2>&1
echo "$(date "+%F %T") $1 search exit $?" >> $L/fwi_status
$PY bench_fwi.py run --case $1 --seeds 0 1 2 > $L/fwi_run_$1.log 2>&1
echo "$(date "+%F %T") $1 runs exit $? DONE" >> $L/fwi_status
