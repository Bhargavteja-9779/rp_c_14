"""Sanity check of the CTW re-implementation under the CTW paper's own protocol (stratified 5-fold split of the
pooled data, each fold given on the command line, symmetric noise 30%), to compare with the accuracies reported in the authors' supplementary
material (Table: 30% symmetric noise, CTW column): CBF 0.958, ArrowHead 0.851.  Results -> results/raw/ctw_check.jsonl"""
import json, os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np

REPORTED = {"CBF": 0.958, "ArrowHead": 0.851}


def one(job):
    d, fold = job if isinstance(job, tuple) else (job, 0)
    import torch; torch.set_num_threads(1)
    from sklearn.model_selection import StratifiedKFold
    from src.data import ucr
    from src.models import ctw, noise as nz
    Xa, ya, Xb, yb = ucr.load(d)
    X, y = np.concatenate([Xa, Xb]), np.concatenate([ya, yb])
    tr, te = list(StratifiedKFold(5).split(X, y))[fold]
    C = int(y.max() + 1)
    yn = nz.apply_noise("sym", y[tr], C, 0.3, np.random.RandomState(0))
    zn = lambda Z: ((Z - Z.mean(-1, keepdims=True)) / (Z.std(-1, keepdims=True) + 1e-8)).transpose(0, 2, 1).astype(np.float32)
    t0 = time.time()
    net = ctw.train_ctw(zn(X[tr]), yn, C, seed=0, epochs=300)
    acc = float(np.mean(ctw.predict(net, zn(X[te])) == y[te]))
    return dict(dataset=d, fold=fold, protocol="CTW paper: 5-fold pooled, sym 30%", acc=acc, reported=REPORTED[d], time_s=time.time() - t0)


if __name__ == "__main__":
    from multiprocessing import Pool
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw", "ctw_check.jsonl")
    jobs = [tuple(map(lambda v: int(v) if v.isdigit() else v, j.split(":"))) for j in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(REPORTED)
    with Pool(2) as p:
        for r in p.imap_unordered(one, jobs):
            open(out, "a").write(json.dumps(r) + "\n"); print(r, flush=True)
