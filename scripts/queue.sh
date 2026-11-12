#!/bin/bash
# secondary experiments, run one after another (all resumable; memory-heavy transforms retried with fewer workers)
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 experiments/run_other.py --methods quant,nn_ed --seeds 0,1,2 --workers 4 --datasets "$(tail -1 scripts/main108.txt)" > results/logs/other.log 2>&1
python3 experiments/run_uea.py --workers 3 > results/logs/uea.log 2>&1

CORE=ridge_std,ridge_wide,loo_em_bic,loo_em,conf_learning
for tr in hydra multirocket; do
  for w in 3 2 1 1; do
    python3 experiments/run_main.py --transform $tr --methods $CORE --seeds 0,1,2 --workers $w --max-n 3000 >> results/logs/main_$tr.log 2>&1 && break
  done
done
python3 experiments/run_sensitivity.py --workers 4 > results/logs/sensitivity.log 2>&1
python3 experiments/run_deep.py --datasets ECG200,Plane,SyntheticControl,MoteStrain,TwoLeadECG,SonyAIBORobotSurface2 --seeds 0,1,2 --conds sym:0,sym:0.3,asym:0.3 --workers 4 > results/logs/deep.log 2>&1
python3 experiments/run_main.py --workers 1 --threads 4 --sub large --seeds 0,1,2 --methods ridge_std,ridge_wide,loo_em_bic,loo_em,em_sym_T,enn,loo_relabel --datasets ElectricDevices >> results/logs/large.log 2>&1
python3 experiments/run_main.py --workers 4 --threads 1 --sub minirocket_v3 --methods conf_learning_cal --max-n 3000 > results/logs/main_v3.log 2>&1
echo QUEUE_DONE > results/logs/queue.done
