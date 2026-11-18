#!/bin/bash
# stage-3 passes over all 108 main datasets (resumable)
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 experiments/run_main.py --workers 2 --threads 1 --sub minirocket_v4 --methods backward_ridge,loo_em_inst --max-n 3000 >> results/logs/main_v4.log 2>&1
python3 experiments/run_main.py --workers 2 --threads 1 --sub classwise --methods ridge_std,ridge_wide,loo_em_bic,conf_learning_cal --max-n 3000 >> results/logs/classwise.log 2>&1
echo QUEUE_DONE > results/logs/queue.done
