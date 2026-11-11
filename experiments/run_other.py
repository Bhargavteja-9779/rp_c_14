"""Other classifier families trained directly on noisy labels (no cleaning):
QUANT (interval quantiles + extremely randomised trees), 1-NN Euclidean, 1-NN DTW.
Same noise realisations and CSV schema as the main experiment.
usage: python experiments/run_other.py --methods quant,nn_ed [--datasets a,b] [--seeds 0,1,2] [--workers 4]"""
import argparse, os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd


def build(name, seed):
    if name == "quant":
        from aeon.classification.interval_based import QUANTClassifier
        return QUANTClassifier(random_state=seed)
    if name == "nn_ed":
        from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
        return KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="euclidean")
    if name == "nn_dtw":
        from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
        return KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw", distance_params={"window": 0.1})
    raise ValueError(name)


def one(args):
    d, methods, seeds, conds = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch; torch.set_num_threads(1)
    from src.training import runner
    from src.data import ucr
    from src.models import noise as nz
    from src.models.ridge_loo import GramRidge, one_hot_pm
    from src.preprocessing.features import get_features
    from src.evaluation import metrics
    cfg = runner.load_config()
    out_dir = os.path.join(ucr.ROOT, "results", "raw", "other"); os.makedirs(out_dir, exist_ok=True)
    rows = []
    for m in methods:
        out = os.path.join(out_dir, f"{d}__{m}.csv")
        if os.path.exists(out):
            continue
        rows = []
        for seed in seeds:
            f = get_features(d, seed, "minirocket", n_jobs=1)  # cached; gives the split and clean LOO scores
            ytr, yte = f["ytr"], f["yte"]; C = int(max(ytr.max(), yte.max()) + 1)
            loo = None
            for kind, rate in conds:
                if kind == "boundary" and loo is None:
                    g = GramRidge(np.logspace(*cfg["alphas_wide"])).fit(f["Ftr"]); loo = g.select_alpha(one_hot_pm(ytr, C))[1]; del g
                yn = nz.apply_noise(kind, ytr, C, rate, runner.noise_rng(d, seed, kind, rate), clean_scores=loo)
                if len(np.unique(yn)) < len(np.unique(ytr)):
                    continue
                t0 = time.time()
                clf = build(m, seed).fit(f["Xtr"], yn)
                pred = clf.predict(f["Xte"])
                row = dict(dataset=d, transform="raw", seed=seed, noise=kind, rate=rate, method=m, n_train=len(ytr),
                           n_test=len(yte), n_classes=C, true_noise=float(np.mean(yn != ytr)), time_s=time.time() - t0)
                row.update(metrics.classification(yte, pred)); rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="quant,nn_ed")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--conds", default="sym:0,sym:0.2,sym:0.4,asym:0.2,asym:0.4,boundary:0.2")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from src.training import runner
    names = a.datasets.split(",") if a.datasets else runner.all_datasets()
    conds = [(c.split(":")[0], float(c.split(":")[1])) for c in a.conds.split(",")]
    jobs = [(d, a.methods.split(","), [int(s) for s in a.seeds.split(",")], conds) for d in names]
    from multiprocessing import Pool
    with Pool(a.workers) as p:
        for d in p.imap_unordered(one, jobs):
            print(d, flush=True)
