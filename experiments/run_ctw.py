"""CTW (re-implementation, src/models/ctw.py) on the same splits and noise realisations as every other method.
Default datasets: the eight UCR datasets of the CTW paper's own experiment list (main.py of the official code).
usage: python experiments/run_ctw.py [--datasets a,b] [--seeds 0,1,2] [--workers 3] [--epochs 300]"""
import argparse, os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

# The CTW paper's UCR list minus MelbournePedestrian, which has unequal-length series with missing values and is
# therefore not part of the equal-length benchmark on which all other methods were run.
CTW_UCR = "ArrowHead,CBF,FaceFour,OSULeaf,Plane,Symbols,Trace"
CONDS = [("sym", 0.0), ("sym", 0.2), ("sym", 0.4), ("asym", 0.2), ("asym", 0.4), ("boundary", 0.3)]


def one(args):
    try:
        return _one(args)
    except Exception as e:                       # recorded, never silently dropped
        return args[0], args[1], "FAILED " + repr(e)[:200]


def _one(args):
    d, seed, epochs = args
    import torch; torch.set_num_threads(1)
    from src.training import runner
    from src.models import noise as nz, ctw
    from src.models.ridge_loo import GramRidge, one_hot_pm
    from src.preprocessing.features import get_features
    from src.evaluation import metrics
    out_dir = os.path.join(runner.ucr.ROOT, "results", "raw", "ctw"); os.makedirs(out_dir, exist_ok=True)
    todo = [(k, r) for k, r in CONDS if not os.path.exists(os.path.join(out_dir, f"{d}__seed{seed}__{k}{int(100*r)}.csv"))]
    if not todo:
        return d, seed, "skip"
    cfg = runner.load_config()
    f = get_features(d, seed, "minirocket", n_jobs=1)
    ytr, yte = f["ytr"], f["yte"]; C = int(max(ytr.max(), yte.max()) + 1)
    g = GramRidge(np.logspace(*cfg["alphas_wide"])).fit(f["Ftr"])
    loo = g.select_alpha(one_hot_pm(ytr, C))[1]; del g
    zn = lambda X: ((X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-8)).transpose(0, 2, 1).astype(np.float32)
    Xtr, Xte = zn(f["Xtr"]), zn(f["Xte"])
    tot = 0.0
    for kind, rate in todo:
        yn = nz.apply_noise(kind, ytr, C, rate, runner.noise_rng(d, seed, kind, rate), clean_scores=loo)
        if len(np.unique(yn)) < len(np.unique(ytr)):
            continue
        t0 = time.time()
        net = ctw.train_ctw(Xtr, yn, C, seed=seed, epochs=epochs)
        pred = ctw.predict(net, Xte)
        row = dict(dataset=d, transform="raw", seed=seed, noise=kind, rate=rate, method="ctw", n_train=len(ytr),
                   n_test=len(yte), n_classes=C, true_noise=float(np.mean(yn != ytr)), time_s=time.time() - t0)
        row.update(metrics.classification(yte, pred)); tot += row["time_s"]
        pd.DataFrame([row]).to_csv(os.path.join(out_dir, f"{d}__seed{seed}__{kind}{int(100*rate)}.csv"), index=False)
    return d, seed, f"{tot:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=CTW_UCR)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()
    jobs = [(d, int(s), a.epochs) for s in a.seeds.split(",") for d in a.datasets.split(",")]
    from multiprocessing import Pool
    with Pool(a.workers) as p:
        for r in p.imap_unordered(one, jobs):
            print(*r, flush=True)
