#!/bin/bash
# lane A: class-wise/calibration pass, then CTW (all steps resumable; safe to re-launch after a restart)
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while pgrep -f "sub classwise" >/dev/null; do sleep 30; done
python3 experiments/run_main.py --workers 2 --threads 1 --sub classwise --methods ridge_std,ridge_wide,loo_em_bic,conf_learning_cal --max-n 3000 >> results/logs/classwise.log 2>&1
python3 experiments/run_ctw.py --workers 2 >> results/logs/ctw.log 2>&1
echo done > results/logs/lane_a.done
