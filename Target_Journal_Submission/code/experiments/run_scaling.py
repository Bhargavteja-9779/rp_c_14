"""Scalability: wall-clock time and peak memory against training-set size, for the dual (n <= p) and the
primal (n > p) route.  ElectricDevices (16,637 series of length 96, 7 classes) pooled; for each n a
stratified training sample; fixed held-out test set of 2,000 series; 30 % asymmetric noise.
Each configuration runs in a fresh process so that its peak resident memory is measured in isolation.
usage: python experiments/run_scaling.py"""
import os, sys, time, json, resource, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np

CONFIGS = [(n, k) for k in (2000, 10000) for n in (500, 1000, 2000, 4000, 8000, 14000)]


def one(cfg):
    n, n_kernels = cfg
    import threadpoolctl; threadpoolctl.threadpool_limits(2)
    from aeon.transformations.collection.convolution_based import MiniRocket
    from sklearn.model_selection import train_test_split
    from src.data import ucr
    from src.training import runner
    from src.models.ridge_loo import GramRidge, one_hot_pm
    from src.models import cleaning as cl, noise as nz
    Xa, ya, Xb, yb = ucr.load("ElectricDevices")
    X, y = np.concatenate([Xa, Xb]), np.concatenate([ya, yb])
    rest, test = train_test_split(np.arange(len(y)), test_size=2000, stratify=y, random_state=0)
    tr = rest if n >= len(rest) else train_test_split(rest, train_size=n, stratify=y[rest], random_state=1)[0]
    C = int(y.max() + 1)
    out = dict(n=len(tr), n_kernels=n_kernels)
    t = time.time(); mr = MiniRocket(n_kernels=n_kernels, random_state=0, n_jobs=2)
    Ftr = np.asarray(mr.fit_transform(X[tr]), np.float32); Fte = np.asarray(mr.transform(X[test]), np.float32)
    out["p"] = Ftr.shape[1]; out["t_feat"] = time.time() - t
    t = time.time(); g = GramRidge(np.logspace(-3, 9, 25)).fit(Ftr); out["t_decomp"] = time.time() - t; out["route"] = g.route_
    Zte = g.test_features(Fte)
    yn = nz.apply_noise("asym", y[tr], C, 0.3, np.random.RandomState(0))
    for name, fn in (("ridge_wide", cl.none), ("loo_em_bic", cl.loo_em_bic)):
        t = time.time(); r = fn(g, yn, C)
        Y = one_hot_pm(r["y_clean"], C) if r.get("soft") is None else 2 * r["soft"] - 1
        a, _, _ = g.select_alpha(Y)
        pred = g.predict_features(Zte, Y, a).argmax(1)
        out[f"t_{name}"] = time.time() - t; out[f"acc_{name}"] = float(np.mean(pred == y[test]))
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return out


if __name__ == "__main__":
    from src.data import ucr
    path = os.path.join(ucr.ROOT, "results", "raw", "scaling.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["n_req"], r["n_kernels"]) for r in map(json.loads, open(path))}
    from multiprocessing import Pool
    for cfg in CONFIGS:
        if cfg in done:
            continue
        with Pool(1, maxtasksperchild=1) as p:          # fresh process -> isolated peak RSS
            try:
                r = p.apply(one, (cfg,))
            except Exception as e:
                r = dict(n=cfg[0], n_kernels=cfg[1], error=repr(e)[:200])
        r["n_req"] = cfg[0]
        open(path, "a").write(json.dumps(r) + "\n")
        print(r, flush=True)
