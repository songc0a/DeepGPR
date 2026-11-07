#!/usr/bin/env bash
# Round-5 dense tuning (4 x 4 x 2 m model): MS-SSIM misfit (~/MSSIM), 1000
# epochs, Adam lr 0.01 cosine-decayed to 0.001. Run A on GPU 5 now; run B
# (MS-SSIM on t^2.5-gained traces) on GPU 4 when it is idle, else on GPU 5
# after A. Each run is retried once (resumes from its checkpoint).
cd ~/DeepGPR-tucker/examples/tucker/siren_comparison
export PYTHONPATH=$HOME/DeepGPR-tucker/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=~/anaconda3/envs/gprmax/bin/python
L=results/full_v5/logs
st() { echo "$(date "+%F %T") tune: $*" >> $L/status; }
free() { nvidia-smi -i $1 --query-gpu=memory.free --format=csv,noheader,nounits; }
COMMON=(tune --case 3d --protocol v5 --only dense --loss mssim --lr 0.01 --schedule cosine --iterations 1000)
run() {  # run GPU TAG ARGS...
  local gpu=$1 tag=$2; shift 2
  for pass in 1 2; do
    st "$tag gpu$gpu pass$pass start"
    CUDA_VISIBLE_DEVICES=$gpu $PY bench_fwi.py "${COMMON[@]}" --tag $tag "$@" >> $L/tune_$tag.log 2>&1; st "$tag gpu$gpu pass$pass exit $?"
  done
}
(run 5 mssim_lr0.01) &
A=$!
(while true; do  # B: GPU 4 when idle, else GPU 5 after A
   if [ "$(free 4)" -ge 20000 ]; then sleep 60; [ "$(free 4)" -ge 20000 ] && { run 4 mssim_gain2.5_lr0.01 --time-gain 2.5; break; }; fi
   if ! kill -0 $A 2>/dev/null; then run 5 mssim_gain2.5_lr0.01 --time-gain 2.5; break; fi
   sleep 120
 done) &
wait
st "DONE"
