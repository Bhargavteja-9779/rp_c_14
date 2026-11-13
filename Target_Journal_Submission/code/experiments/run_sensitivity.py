"""Sensitivity of LOO-EM to its hyper-parameters (gate factor, prior strength, EM iterations) and to the
number of random kernels, on a fixed random subset of 30 non-development datasets, seeds 0-2."""
import argparse, os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
NOISE = {"sym": [0.0, 0.2, 0.4], "asym": [0.2, 0.4], "boundary": [0.2]}


def subset():
    import numpy as np, pandas as pd
    from src.training import runner
    cfg = runner.load_config(); dev = set(runner.dev_datasets(cfg))
    size = pd.read_csv(os.path.join(runner.ucr.ROOT, "data", "dataset_summary.csv")).set_index("dataset").n_train
    pool = [d for d in runner.all_datasets() if d not in dev and size[d] < 3000]
    return sorted(str(d) for d in np.random.RandomState(7).choice(pool, 30, replace=False))


def one(args):
    d, transform = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import threadpoolctl; threadpoolctl.threadpool_limits(1)
    from src.training import runner
    cfg = runner.load_config()
    allm = {**cfg["methods"], **cfg["methods_v2"]}
    m = {k: allm[k] for k in ("ridge_std", "ridge_wide", "loo_em_bic", "loo_em", "em_no_gate", "bic_pen_0.5", "bic_pen_2", "bic_iter_5")}
    if transform == "minirocket":
        m.update({"bic_prior_1": ["loo_em_bic", "wide", {"prior_strength": 1}], "bic_prior_50": ["loo_em_bic", "wide", {"prior_strength": 50}],
                  "bic_iter_50": ["loo_em_bic", "wide", {"max_iter": 50}], "bic_pen_4": ["loo_em_bic", "wide", {"penalty": 4.0}]})
    runner.run_dataset(d, cfg, transform=transform, methods=m, seeds=[0, 1, 2], noise=NOISE, n_jobs=1,
                       out_dir=os.path.join(runner.ucr.ROOT, "results", "raw", "sensitivity", transform))
    return d, transform


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    jobs = [(d, t) for t in ("minirocket", "minirocket@1000", "minirocket@2500", "minirocket@5000") for d in subset()]
    from multiprocessing import Pool
    with Pool(a.workers) as p:
        for r in p.imap_unordered(one, jobs): print(*r, flush=True)
