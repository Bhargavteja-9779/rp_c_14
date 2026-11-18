"""Sensitivity of the closed-form LOO to the global standardisation.

The closed form keeps the column scaling computed on all n training series.  The exact LOO pipeline re-computes
the means and standard deviations without series i.  For every dataset with at most 100 training series
(seed 0) we compute the exact LOO operator M_exact(alpha) (n x n, linear in the targets for a fixed alpha) by n
refits, and compare (i) LOO decision values and (ii) LOO-EM test accuracy with the exact and the closed-form
operator under clean, 30 % symmetric and 40 % asymmetric noise.  The MiniRocket transform itself is fitted once
on all training series in both cases (re-fitting it n times is the remaining approximation, stated in the paper).
usage: python experiments/run_exact_loo.py [--workers 2]"""
import argparse, os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd


class ExactLOO:
    """Drop-in for GramRidge inside the EM: LOO values from precomputed exact operators."""
    def __init__(self, base, M):
        self.base, self.M, self.alphas = base, M, base.alphas
        for a in ("n_", "Z_", "mu_", "sd_", "s_", "U_", "U2_"):
            setattr(self, a, getattr(base, a))

    def loo(self, Y, alpha, UtY=None):
        k = int(np.argmin(np.abs(self.alphas - alpha)))
        return self.M[k] @ Y, self.base.loo(Y, alpha)[1], None

    def select_alpha(self, Y, criterion="mse", around=None, width=2):
        best = (np.inf, None, None)
        for k, a in enumerate(self.alphas):
            L = self.M[k] @ Y
            sc = float(np.mean((L - Y) ** 2))
            if sc < best[0] - 1e-15:
                best = (sc, a, L)
        return best[1], best[2], best[0]

    def __getattr__(self, name):
        return getattr(self.base, name)


def exact_operators(Phi, alphas):
    n = len(Phi)
    M = np.zeros((len(alphas), n, n))
    for i in range(n):
        R = np.r_[0:i, i + 1:n]
        mu, sd = Phi[R].mean(0), Phi[R].std(0); sd[sd < 1e-8] = 1.0
        ZR = (Phi[R] - mu) / sd; zi = (Phi[i] - mu) / sd
        s, U = np.linalg.eigh(ZR @ ZR.T); s = np.clip(s, 0, None)
        ki = U.T @ (ZR @ zi)
        m = len(R)
        Cm = np.eye(m) - 1.0 / m
        for k, a in enumerate(alphas):
            row = (U @ (ki / (s + a))) @ Cm + 1.0 / m
            M[k, i, R] = row
    return M


def one(d):
    from src.training import runner
    from src.preprocessing.features import get_features
    from src.models.ridge_loo import GramRidge, one_hot_pm
    from src.models import cleaning as cl, noise as nz
    out_dir = os.path.join(runner.ucr.ROOT, "results", "raw", "exact_loo"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{d}.csv")
    if os.path.exists(out):
        return d
    cfg = runner.load_config()
    f = get_features(d, 0); ytr, yte = f["ytr"], f["yte"]; C = int(max(ytr.max(), yte.max()) + 1)
    alphas = np.logspace(*cfg["alphas_wide"])
    g = GramRidge(alphas).fit(f["Ftr"]); K = g.test_kernel(f["Fte"])
    gx = ExactLOO(g, exact_operators(np.asarray(f["Ftr"], np.float64), alphas))
    loo_clean = g.select_alpha(one_hot_pm(ytr, C))[1]
    rows = []
    for kind, rate in (("sym", 0.0), ("sym", 0.3), ("asym", 0.4)):
        yn = nz.apply_noise(kind, ytr, C, rate, runner.noise_rng(d, 0, kind, rate), clean_scores=loo_clean)
        if len(np.unique(yn)) < len(np.unique(ytr)):
            continue
        Y = one_hot_pm(yn, C)
        a_c, Lc, _ = g.select_alpha(Y); a_x, Lx, _ = gx.select_alpha(Y)
        row = dict(dataset=d, n=len(ytr), cond=f"{kind} {int(100*rate)}%", alpha_closed=a_c, alpha_exact=a_x,
                   corr=float(np.corrcoef(Lc.ravel(), Lx.ravel())[0, 1]), argmax_agree=float(np.mean(Lc.argmax(1) == Lx.argmax(1))),
                   max_abs_diff=float(np.abs(Lc - Lx).max()))
        for tag, gg in (("closed", g), ("exact", gx)):
            r = cl.loo_em_bic(gg, yn, C)
            Yf = one_hot_pm(r["y_clean"], C) if r.get("soft") is None else 2 * r["soft"] - 1
            a = g.select_alpha(Yf)[0] if tag == "closed" else gx.select_alpha(Yf)[0]
            row[f"acc_em_{tag}"] = float(np.mean(g.predict(K, Yf, a).argmax(1) == yte))
            row[f"model_{tag}"] = r["info"].get("model", "")
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2); a = ap.parse_args()
    s = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset_summary.csv"))
    main = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "main108.txt")).read().strip().split("\n")[-1].split(",")
    names = sorted(s[(s.n_train <= 100) & s.dataset.isin(main)].dataset)
    print(len(names), "datasets", flush=True)
    from multiprocessing import Pool
    with Pool(a.workers) as p:
        for d in p.imap_unordered(one, names):
            print(d, flush=True)
