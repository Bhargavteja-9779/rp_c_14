#!/bin/bash
# lane B: ECG real-noise experiment, exact-LOO sensitivity, scaling (all resumable)
cd "$(dirname "$0")/.."
while pgrep -f run_cinc2017.py >/dev/null; do sleep 30; done
python3 experiments/run_cinc2017.py --repeats 0,1 --threads 2 >> results/logs/cinc2017.log 2>&1
OMP_NUM_THREADS=1 python3 experiments/run_exact_loo.py --workers 2 >> results/logs/exact_loo.log 2>&1
while pgrep -f run_ctw.py >/dev/null && [ "$(free -g | awk '/Mem/{print $7}')" -lt 9 ]; do sleep 60; done
python3 experiments/run_scaling.py >> results/logs/scaling.log 2>&1
echo done > results/logs/lane_b.done
