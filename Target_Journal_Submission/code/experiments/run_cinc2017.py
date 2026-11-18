"""Real-world annotation noise: PhysioNet/CinC 2017 (protocol: supplementary/protocol_cinc2017.md).

Train on the original labels (v0), evaluate against the expert-revised labels (v3).  One .npz per (repeat, fold)
with per-record test predictions/probabilities and per-record training suspicion scores; resumable.
usage: python experiments/run_cinc2017.py [--repeats 0,1] [--threads 2]"""
import argparse, os, sys, time, resource, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np

METHODS = ["ridge_std", "ridge_wide", "loo_em_bic", "loo_em", "loo_em_inst", "conf_learning_cal", "conf_learning",
           "kfold_relabel", "enn", "loo_relabel", "loo_remove", "backward_ridge"]


def run_fold(rep, k, X, y0, y3, out_dir, threads):
    out = os.path.join(out_dir, f"rep{rep}_fold{k}.npz")
    if os.path.exists(out):
        return out
    import threadpoolctl
    threadpoolctl.threadpool_limits(threads)
    from sklearn.model_selection import StratifiedKFold
    from aeon.transformations.collection.convolution_based import MiniRocket
    from src.training import runner
    from src.models.ridge_loo import GramRidge
    from src.models import cleaning as cl
    cfg = runner.load_config()
    allm = {**cfg["methods"], **cfg.get("methods_v2", {}), **cfg.get("methods_v3", {}), **cfg.get("methods_v4", {})}
    tr, te = list(StratifiedKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y0)), y0))[k]
    C = 4
    res = dict(test_idx=te, train_idx=tr)
    t0 = time.time()
    mr = MiniRocket(n_kernels=10000, random_state=rep * 10 + k, n_jobs=threads)
    Ftr = np.asarray(mr.fit_transform(X[tr]), np.float32)
    Fte = np.asarray(mr.transform(X[te]), np.float32)
    res["t_feat"] = time.time() - t0
    grids = {"std": np.logspace(*cfg["alphas_std"]), "wide": np.logspace(*cfg["alphas_wide"])}
    t0 = time.time()
    g = GramRidge(grids["wide"]).fit(Ftr)
    res["t_eig"] = time.time() - t0
    K_te = g.test_kernel(Fte)
    gstd = GramRidge(grids["std"])
    for a in ("n_", "mu_", "sd_", "Z_", "s_", "U_", "U2_", "route_"):
        setattr(gstd, a, getattr(g, a))
    gm = {"std": gstd, "wide": g}
    jobs = [(m, allm[m][0], allm[m][1], allm[m][2] or {}, y0[tr]) for m in METHODS]
    jobs.append(("ridge_wide_v3labels", "ridge", "wide", {}, y3[tr]))     # reference: expert labels
    for name, fn, grid, kw, ytrain in jobs:
        t0 = time.time()
        r = cl.METHODS[fn](gm[grid], ytrain, C, seed=rep, **kw)
        F, a, P = cl.final_predict_proba(gm[grid], Fte, K_te, r, C)
        res[f"{name}__time"] = time.time() - t0
        res[f"{name}__pred"] = F.argmax(1).astype(np.int8)
        res[f"{name}__proba"] = P.astype(np.float32)
        res[f"{name}__score"] = np.asarray(r["score"], np.float32)
        res[f"{name}__ytrain_clean"] = np.asarray(r["y_clean"], np.int8)
        res[f"{name}__keep"] = np.asarray(r["keep"], bool)
        res[f"{name}__alpha"] = a
        res[f"{name}__model"] = str(r.get("info", {}).get("model", ""))
        print(f"rep{rep} fold{k} {name:22s} {res[f'{name}__time']:7.1f}s acc_v3={np.mean(F.argmax(1) == y3[te]):.4f}", flush=True)
    # deployment: single-record inference latency (transform + linear model), model size
    a = res["ridge_wide__alpha"]
    W = g.Z_.T @ g.dual_coef(cl.one_hot_pm(y0[tr], C), a)[0] / g.sd_[:, None]   # weights on raw features
    lat = []
    for i in range(50):
        x = X[te[i]:te[i] + 1]
        t0 = time.perf_counter()
        f = np.asarray(mr.transform(x), np.float64)
        _ = (f - g.mu_) @ W
        lat.append(time.perf_counter() - t0)
    res["latency_ms_median"] = 1000 * float(np.median(lat[5:]))
    kern_bytes = int(sum(np.asarray(v).nbytes for v in mr.parameters))
    res["model_bytes"] = int(W.astype(np.float32).nbytes + g.mu_.astype(np.float32).nbytes + kern_bytes)
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    res["n_train"] = len(tr)
    np.savez_compressed(out + ".tmp.npz", **res)
    os.replace(out + ".tmp.npz", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", default="0,1")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    from src.data import cinc2017
    from src.data import ucr
    X, y0, y3, rec, dur = cinc2017.load()
    out_dir = os.path.join(ucr.ROOT, "results", "raw", "cinc2017")
    os.makedirs(out_dir, exist_ok=True)
    for rep in [int(r) for r in a.repeats.split(",")]:
        for k in range(5):
            run_fold(rep, k, X, y0, y3, out_dir, a.threads)
