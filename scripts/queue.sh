#!/bin/bash
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 experiments/run_deep.py --datasets ECG200,Plane,SyntheticControl,MoteStrain,TwoLeadECG,SonyAIBORobotSurface2 --seeds 0,1,2 --conds sym:0,sym:0.3,asym:0.3 --workers 4 > results/logs/deep.log 2>&1
for w in 2 1 1; do python3 experiments/run_main.py --transform hydra --methods ridge_std,ridge_wide,loo_em_bic,loo_em,conf_learning --seeds 0,1,2 --workers $w --datasets ArrowHead,BeetleFly,Coffee,Computers,CricketX,CricketZ,DistalPhalanxOutlineCorrect,DistalPhalanxTW,ECG5000,EOGHorizontalSignal,FaceAll,Fish,GunPointAgeSpan,Ham,Herring,InsectEPGRegularTrain,Lightning7,MiddlePhalanxOutlineCorrect,MixedShapesSmallTrain,MoteStrain,Phoneme,PigAirwayPressure,Plane,SemgHandSubjectCh2,SmallKitchenAppliances,StarLightCurves,ToeSegmentation2,Wine,Worms,Yoga >> results/logs/main_hydra.log 2>&1 && break; done
python3 experiments/run_sensitivity.py --workers 4 > results/logs/sensitivity.log 2>&1
echo QUEUE_DONE > results/logs/queue.done
