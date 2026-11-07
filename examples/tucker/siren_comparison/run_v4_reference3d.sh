#!/usr/bin/env bash
# Round-4 3D reference (seed 0, truth-selected sigma step; SIREN, Tucker-SIREN
# r24 and r16, whose truth-selected learning rate equals the data-selected one,
# so their stage-B entries are copied, not recomputed). Hard stop at 14:00.
cd ~/DeepGPR-tucker/examples/tucker/siren_comparison
export PYTHONPATH=$HOME/DeepGPR-tucker/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 FWI_CHECKPOINT_STEPS=60
PY=~/anaconda3/envs/gprmax/bin/python
L=results/full_v4/logs
DEADLINE=$(date -d "today 14:00" +%s)
st() { echo "$(date "+%F %T") reference3d: $*" >> $L/status; }
$PY - <<'PYEOF'
import json, sys
sys.path.insert(0, ".")
from runner import load_results, merge_result
import bench_fwi as b
out = b.result_dir(False, "v4")
data = load_results(out / "fwi_search_sigma.json")
oracle_lr, data_lr = b.chosen_learning_rates("3d", False, "v4", "oracle"), b.chosen_learning_rates("3d", False, "v4")
for name, r in data.items():
    label = name.split("|")[2]
    if name.startswith("3d|search_sigma|") and label in ("siren_4x128", "tsiren_2x32_r24", "tsiren_2x32_r16"):
        assert oracle_lr[label] == data_lr[label]
        copy = dict(r, name=name.replace("|search_sigma|", "|search_sigma_oracle|"), copied_from=name)
        merge_result(out / "fwi_search_sigma_oracle.json", copy["name"], copy)
print("oracle scales", b.chosen_sigma_scales("3d", False, "v4", "oracle"))
PYEOF
pick() {  # GPU from 5, 3, 7 with >= $1 MiB free at two checks 30 s apart
  while true; do
    for g in 5 3 7; do
      f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits)
      if [ "$f" -ge "$1" ]; then sleep 30; f=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -ge "$1" ] && { echo $g; return; }; fi
    done
    sleep 60
  done
}
one() {  # one NEED LABEL
  g=$(pick $1); left=$(( DEADLINE - $(date +%s) )); [ $left -le 600 ] && { st "$2 not started (deadline)"; return; }
  st "$2 gpu$g start (timeout ${left}s)"
  CUDA_VISIBLE_DEVICES=$g timeout $left $PY bench_fwi.py run --case 3d --protocol v4 --select oracle --seeds 0 --only "$2|seed0|oracle" >> $L/reference3d_$2.log 2>&1
  st "$2 gpu$g exit $?"
}
one 10000 siren_4x128 &
sleep 240; one 5000 tsiren_2x32_r24 &
sleep 240; one 5000 tsiren_2x32_r16 &
wait
st "DONE"
