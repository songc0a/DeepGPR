#!/usr/bin/env bash
cd "$(dirname "$0")"
export PYTHONPATH=../../../src
PY=${PY:-python}
L=results/full/logs
mkdir -p $L
echo "$(date '+%F %T') profile start" >> $L/local_status
$PY profile_reconstruction.py > $L/profile.log 2>&1
echo "$(date '+%F %T') profile exit $?" >> $L/local_status
$PY bench_representation.py run > $L/representation.log 2>&1
echo "$(date '+%F %T') representation exit $? DONE" >> $L/local_status
