#!/usr/bin/env python3
"""Reproduce the study end to end.

    python run_all.py --mode quick     # ~2 min: 6 small datasets, 2 seeds, full pipeline check
    python run_all.py --mode full      # complete experiment (many CPU-hours; resumable)
    python run_all.py --mode report    # only re-aggregate existing raw results -> tables, figures, stats

Outputs: results/raw/**.csv (one file per dataset, resumable), results/raw_results.csv, results/metrics.json,
results/statistics.json, tables/*.csv|md, figures/*.png|pdf, results/logs/*.log.
Hardware: CPU only is sufficient; device detection is reported for completeness (the deep baselines use it)."""
import argparse, json, os, platform, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
PY = sys.executable
BIG = "ElectricDevices,Crop,FordA,FordB"
DEEP = "ECG200,Wine,Plane,Trace,SyntheticControl,MoteStrain,TwoLeadECG,SonyAIBORobotSurface2"


def device():
    try:
        import torch
        if torch.cuda.is_available(): return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(): return "mps"
    except Exception:
        pass
    return "cpu"


def sh(cmd, log):
    os.makedirs(os.path.join(ROOT, "results", "logs"), exist_ok=True)
    print("»", cmd, flush=True)
    with open(os.path.join(ROOT, "results", "logs", log), "a") as f:
        rc = subprocess.call(cmd, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        print(f"   FAILED (exit {rc}); see results/logs/{log}")
    return rc


def report(quick=False):
    import warnings; warnings.filterwarnings("ignore")
    from src.evaluation import aggregate as ag, extra
    from src.visualization import plots
    from src.training import runner
    if quick:
        base = os.path.join(ROOT, "results", "quick")
        ag.RAW_SUB, ag.R, ag.T, plots.F = "quick/minirocket", base, os.path.join(base, "tables"), os.path.join(base, "figures")
    df, metrics, stats = ag.run(dev=runner.dev_datasets(runner.load_config()))
    plots.make_all(df, stats)
    if not quick:
        extra.run(df)
    print(f"report written: {len(df)} result rows, {metrics['n_datasets']} datasets, {metrics['n_errors']} failed runs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "full", "report"], default="quick")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    t0 = time.time(); w = a.workers
    print(f"python {platform.python_version()} | {platform.platform()} | cores {os.cpu_count()} | device {device()}")
    if a.mode == "quick":
        sh(f"{PY} -m pytest -q tests", "tests.log")
        sh(f"{PY} experiments/run_main.py --quick --workers {min(w, 2)}", "quick.log")
        report(quick=True)
    elif a.mode == "full":
        sh(f"{PY} -m pytest -q tests", "tests.log")
        sh(f"{PY} scripts/dataset_summary.py", "dataset_summary.log")
        sh(f"{PY} scripts/leakage_audit.py", "leakage_audit.log")
        sh(f"{PY} experiments/run_main.py --workers {w} --max-n 3000", "main_minirocket.log")           # 108 datasets
        sh(f"{PY} experiments/run_main.py --workers 1 --threads {w} --sub large --seeds 0,1,2 --methods ridge_std,ridge_wide,loo_em_bic,loo_em,em_sym_T,enn,loo_relabel --datasets {BIG}", "large.log")  # 4 largest: scalability study
        sh(f"{PY} experiments/run_main.py --workers {w} --max-n 3000 --sub minirocket_v2 --methods loo_em_bic,bic_hard,bic_in_sample,bic_std_grid,bic_pen_0.5,bic_pen_2,bic_iter_5", "main_v2.log")
        core = "ridge_std,ridge_wide,loo_em_bic,loo_em,conf_learning"
        for tr in ("multirocket", "hydra"):                                                                # other transforms
            sh(f"{PY} experiments/run_main.py --transform {tr} --methods {core} --seeds 0,1,2 --workers {w} --max-n 3000", f"main_{tr}.log")
        sh(f"{PY} experiments/run_other.py --methods quant,nn_ed --seeds 0,1,2 --workers {w}", "other.log")  # other families
        sh(f"{PY} experiments/run_sensitivity.py --workers {w}", "sensitivity.log")
        sh(f"{PY} experiments/run_uea.py --workers {w}", "uea.log")
        sh(f"{PY} experiments/run_deep.py --datasets {DEEP} --seeds 0,1,2 --conds sym:0,sym:0.3,asym:0.3 --workers {w}", "deep.log")
        report()
    else:
        report()
    print(f"done in {(time.time() - t0) / 60:.1f} min")
