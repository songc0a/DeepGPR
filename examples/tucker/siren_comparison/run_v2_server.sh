#!/usr/bin/env bash
cd ~/DeepGPR-tucker/examples/tucker/siren_comparison
export PYTHONPATH=$HOME/DeepGPR-tucker/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=~/anaconda3/envs/gprmax/bin/python
L=results/full_v2/logs
st() { echo "$(date "+%F %T") $*" >> $L/status; }
(st "2d search start gpu5"; CUDA_VISIBLE_DEVICES=5 $PY bench_fwi.py search --case 2d --protocol v2 > $L/search_2d.log 2>&1; st "2d search exit $?";
 CUDA_VISIBLE_DEVICES=5 $PY bench_fwi.py run --case 2d --protocol v2 > $L/run_2d.log 2>&1; st "2d runs exit $? DONE") &
(st "3d search start gpu7"; CUDA_VISIBLE_DEVICES=7 $PY bench_fwi.py search --case 3d --protocol v2 > $L/search_3d.log 2>&1; st "3d search exit $?";
 (CUDA_VISIBLE_DEVICES=7 $PY bench_fwi.py run --case 3d --protocol v2 --only "|dense|" siren_4x128 tsiren_2x32_r8 tsiren_2x32_r12 > $L/run_3d_gpu7.log 2>&1; st "3d runs gpu7 exit $? DONE") &
 (CUDA_VISIBLE_DEVICES=3 $PY bench_fwi.py run --case 3d --protocol v2 --only tsiren_2x32_r16 tsiren_2x32_r24 direct_tucker > $L/run_3d_gpu3.log 2>&1; st "3d runs gpu3 exit $? DONE") &
 wait) &
wait
st "ALL DONE"
