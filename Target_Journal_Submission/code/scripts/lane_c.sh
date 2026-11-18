#!/bin/bash
# lane C: stage-4 exploratory criterion runs (UCR, then ECG once the main ECG run is finished)
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 experiments/run_main.py --workers 2 --threads 1 --sub minirocket_v5 --methods ridge_wide_err,ridge_wide_bal --max-n 3000 >> results/logs/main_v5.log 2>&1
while pgrep -f "run_cinc2017.py" >/dev/null; do sleep 30; done
OMP_NUM_THREADS=2 python3 experiments/run_cinc2017_criteria.py --threads 2 >> results/logs/cinc2017_criteria.log 2>&1
echo done > results/logs/lane_c.done
