#!/usr/bin/env bash
cd ~/DeepGPR-tucker/examples/tucker/siren_comparison
export PYTHONPATH=$HOME/DeepGPR-tucker/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=~/anaconda3/envs/gprmax/bin/python
L=results/full_v2/logs
st() { echo "$(date "+%F %T") $*" >> $L/status; }
st "rerun of failed 3d runs (GPU 3 was taken by another process): r16 gpu7, r24 gpu5, direct gpu3"
(CUDA_VISIBLE_DEVICES=7 $PY bench_fwi.py run --case 3d --protocol v2 --only tsiren_2x32_r16 > $L/rerun_3d_r16.log 2>&1; st "rerun r16 exit $?") &
(CUDA_VISIBLE_DEVICES=5 $PY bench_fwi.py run --case 3d --protocol v2 --only tsiren_2x32_r24 > $L/rerun_3d_r24.log 2>&1; st "rerun r24 exit $?") &
(CUDA_VISIBLE_DEVICES=3 $PY bench_fwi.py run --case 3d --protocol v2 --only direct_tucker > $L/rerun_3d_direct.log 2>&1; st "rerun direct exit $?") &
wait
st "RERUN DONE"
