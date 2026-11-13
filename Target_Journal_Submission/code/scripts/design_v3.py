"""Design-stage evaluation of LOO-EM v2 on the 36 datasets completed when v1's failure was observed
(supplementary/design_set_v2.txt), seeds 0-1.  The other datasets are NOT touched here."""
import sys, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
os.environ["OMP_NUM_THREADS"] = "1"
from src.training import runner
from src.preprocessing.features import get_features
from src.models.ridge_loo import GramRidge, one_hot_pm
from src.models import cleaning as cl, noise as nz
cfg = runner.load_config(); names = open("supplementary/design_set_v2.txt").read().split()
conds = [("sym", 0.0), ("sym", 0.2), ("sym", 0.4), ("asym", 0.1), ("asym", 0.2), ("asym", 0.4), ("boundary", 0.2)]
V = {"ridge_wide": (cl.none, {}), "anc": (cl.loo_em, dict(anchor=True)), "anc_fix": (cl.loo_em, dict(anchor=True, fix_T=True)),
     "anc_g1": (cl.loo_em, dict(anchor=True, gate=1.0)), "anc_hard": (cl.loo_em, dict(anchor=True, soft_targets=False)),
     "anc_m5C": (cl.loo_em, dict(anchor=True, prior_strength=1e6))}
OUT = "design_v3"
def one(d):
    out = f"supplementary/{OUT}/{d}.csv"
    if os.path.exists(out): return d
    rows = []
    for seed in (0, 1):
        f = get_features(d, seed); ytr, yte = f["ytr"], f["yte"]; C = int(max(ytr.max(), yte.max()) + 1)
        g = GramRidge(np.logspace(*cfg["alphas_wide"])).fit(f["Ftr"]); K = g.test_kernel(f["Fte"])
        loo = g.select_alpha(one_hot_pm(ytr, C))[1]
        for kind, rate in conds:
            yn = nz.apply_noise(kind, ytr, C, rate, runner.noise_rng(d, seed, kind, rate), clean_scores=loo)
            if len(np.unique(yn)) < len(np.unique(ytr)): continue
            for v, (fn, kw) in V.items():
                r = fn(g, yn, C, **kw); F, a = cl.final_predict(g, f["Fte"], K, r, C)
                rows.append(dict(dataset=d, seed=seed, cond=f"{kind}{rate}", v=v, acc=(F.argmax(1) == yte).mean(), model=r["info"].get("model", "")))
    pd.DataFrame(rows).to_csv(out, index=False); return d

if __name__ == "__main__":
    os.makedirs(f"supplementary/{OUT}", exist_ok=True)
    from multiprocessing import Pool
    with Pool(3) as p:
        for d in p.imap_unordered(one, names): print(d, flush=True)
    import glob
    rows = pd.concat(pd.read_csv(f) for f in glob.glob(f"supplementary/{OUT}/*.csv")).to_dict("records")
    df = pd.DataFrame(rows); df.to_csv(f"supplementary/{OUT}_results.csv", index=False)
    t = df.groupby(["dataset", "cond", "v"]).acc.mean().unstack("v").groupby("cond").mean()
    print((t * 100).round(2).to_string())
    print(df[df.v == "anc"].groupby(["cond", "model"]).size().unstack(fill_value=0).to_string())
