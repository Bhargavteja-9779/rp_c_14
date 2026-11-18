"""Stage-4 exploratory: penalty-selection criterion on the ECG folds.  Recomputes the identical fold features
(same MiniRocket seeds as run_cinc2017.py) and fits ridge with the LOO error / balanced-error criteria.
Writes results/raw/cinc2017/crit_rep{r}_fold{k}.npz."""
import argparse, os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", default="0,1"); ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    import threadpoolctl; threadpoolctl.threadpool_limits(a.threads)
    from sklearn.model_selection import StratifiedKFold
    from aeon.transformations.collection.convolution_based import MiniRocket
    from src.data import cinc2017, ucr
    from src.training import runner
    from src.models.ridge_loo import GramRidge
    from src.models import cleaning as cl
    X, y0, y3, rec, dur = cinc2017.load()
    cfg = runner.load_config()
    out_dir = os.path.join(ucr.ROOT, "results", "raw", "cinc2017")
    for rep in map(int, a.repeats.split(",")):
        for k in range(5):
            out = os.path.join(out_dir, f"crit_rep{rep}_fold{k}.npz")
            if os.path.exists(out):
                continue
            tr, te = list(StratifiedKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y0)), y0))[k]
            mr = MiniRocket(n_kernels=10000, random_state=rep * 10 + k, n_jobs=a.threads)
            Ftr = np.asarray(mr.fit_transform(X[tr]), np.float32); Fte = np.asarray(mr.transform(X[te]), np.float32)
            g = GramRidge(np.logspace(*cfg["alphas_wide"])).fit(Ftr); K = g.test_kernel(Fte)
            res = dict(test_idx=te)
            for name, fn, lab in (("ridge_wide_err", cl.none_err, y0), ("ridge_wide_bal", cl.none_bal, y0),
                                  ("ridge_wide_bal_v3labels", cl.none_bal, y3)):
                r = fn(g, lab[tr], 4)
                F, al, P = cl.final_predict_proba(g, Fte, K, r, 4)
                res[f"{name}__pred"] = F.argmax(1).astype(np.int8); res[f"{name}__alpha"] = al
                print(rep, k, name, f"alpha={al:.0e}", f"acc_v3={np.mean(F.argmax(1) == y3[te]):.4f}", flush=True)
            np.savez_compressed(out + ".tmp.npz", **res); os.replace(out + ".tmp.npz", out)
