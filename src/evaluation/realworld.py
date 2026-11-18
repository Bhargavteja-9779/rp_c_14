"""Aggregation of the real-world annotation-noise experiment (PhysioNet/CinC 2017).
Writes tables/table_cinc_*.csv|md and merges results into results/metrics.json / statistics.json ('cinc2017')."""
from __future__ import annotations

import glob, json, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.data import ucr
from src.evaluation import aggregate as ag

R = os.path.join(ucr.ROOT, "results")
NAMES = {"ridge_std": "Ridge (default grid)", "ridge_wide": "Ridge (wide grid)", "loo_em_bic": "LOO-EM (BIC, proposed)",
         "loo_em": "LOO-EM v1 (full T)", "loo_em_inst": "LOO-EM + instance model (rejected)",
         "conf_learning_cal": "Confident learning (calibrated)", "conf_learning": "Confident learning (uncalibrated)",
         "kfold_relabel": "5-fold relabel", "enn": "ENN", "loo_relabel": "LOO relabel (1 round)", "loo_remove": "LOO remove",
         "backward_ridge": "Backward-corrected ridge", "ridge_wide_v3labels": "Ridge (wide), trained on expert labels",
         "ridge_wide_err": "Ridge (wide, LOO-error criterion)", "ridge_wide_bal": "Ridge (wide, LOO balanced-error criterion)",
         "ridge_wide_bal_v3labels": "Ridge (wide, balanced-error), expert labels"}
CRIT = ["ridge_wide_err", "ridge_wide_bal", "ridge_wide_bal_v3labels"]
ORDER = list(NAMES)


def f1_challenge(y, p):
    """PhysioNet/CinC 2017 score: mean F1 of N, A, O (class codes 0, 1, 2); the noisy class is not averaged."""
    f = []
    for c in (0, 1, 2):
        tp = np.sum((p == c) & (y == c)); den = np.sum(p == c) + np.sum(y == c)
        f.append(2 * tp / den if den else 0.0)
    return float(np.mean(f))


def _metrics(y, p):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    rec = [float(np.mean(p[y == c] == c)) for c in range(4)]
    return dict(acc=float(np.mean(p == y)), f1_nao=f1_challenge(y, p), bal_acc=float(balanced_accuracy_score(y, p)),
                macro_f1=float(f1_score(y, p, average="macro")), rec_N=rec[0], rec_A=rec[1], rec_O=rec[2], rec_noisy=rec[3])


def mcnemar_exact(ca, cb):
    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    p = 1.0 if b + c == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5)))
    return b, c, p


def load():
    from src.data import cinc2017
    _, y0, y3, rec, dur = cinc2017.load()
    files = sorted(glob.glob(os.path.join(R, "raw", "cinc2017", "rep*_fold*.npz")))
    folds = [(f, np.load(f, allow_pickle=False)) for f in files]
    return y0, y3, folds


def run(write=True):
    y0, y3, folds = load()
    if not folds:
        return None
    reps = sorted({int(os.path.basename(f).split("_")[0][3:]) for f, _ in folds})
    methods = [m for m in ORDER if f"{m}__pred" in folds[0][1].files]
    crit = {}
    for f, _ in folds:
        cf = os.path.join(os.path.dirname(f), "crit_" + os.path.basename(f))
        if os.path.exists(cf):
            crit[f] = np.load(cf)
    has_crit = len(crit) == len(folds)
    if has_crit:
        methods += [m for m in CRIT if f"{m}__pred" in next(iter(crit.values())).files]
    pred = {m: {r: np.full(len(y0), -1) for r in reps} for m in methods}
    det, lab, info = [], [], []
    for f, z in folds:
        r = int(os.path.basename(f).split("_")[0][3:]); te, tr = z["test_idx"], z["train_idx"]
        noisy_tr = (y0[tr] != y3[tr])
        for m in methods:
            if m in CRIT:
                pred[m][r][te] = crit[f][f"{m}__pred"]
                continue
            pred[m][r][te] = z[f"{m}__pred"]
            if m == "ridge_wide_v3labels":
                continue
            s = z[f"{m}__score"].astype(float)
            auroc = roc_auc_score(noisy_tr, s) if np.ptp(s) > 0 else np.nan
            k = int(noisy_tr.sum())
            top = np.argsort(-s)[:k]
            keep = z[f"{m}__keep"]; yc = z[f"{m}__ytrain_clean"]
            det.append(dict(method=m, rep=r, auroc=auroc, prec_at_k=float(noisy_tr[top].mean()),
                            agree_v3_after=float(np.mean(yc[keep] == y3[tr][keep])), agree_v3_before=float(np.mean(y0[tr] == y3[tr])),
                            frac_kept=float(keep.mean()), changed=float(np.mean(yc != y0[tr]))))
            info.append(dict(method=m, rep=r, time_s=float(z[f"{m}__time"]), model=str(z[f"{m}__model"])))
        info.append(dict(method="_fold", rep=r, time_s=float(z["t_feat"]), model="t_feat"))
        info.append(dict(method="_fold", rep=r, time_s=float(z["t_eig"]), model="t_eig"))
        info.append(dict(method="_deploy", rep=r, time_s=float(z["latency_ms_median"]), model=f"{int(z['model_bytes'])}|{float(z['peak_rss_mb']):.0f}"))
    complete = [r for r in reps if all((pred[m][r] >= 0).all() for m in methods)]
    rows = []
    for m in methods:
        for r in complete:
            rows.append(dict(method=m, rep=r, ref="v3", **_metrics(y3, pred[m][r])))
            rows.append(dict(method=m, rep=r, ref="v0", **_metrics(y0, pred[m][r])))
    perf = pd.DataFrame(rows)
    out = dict(n_records=int(len(y0)), n_revised=int(np.sum(y0 != y3)), repetitions=complete,
               n_folds_done=len(folds))
    if not complete:
        return out
    tab = perf[perf.ref == "v3"].groupby("method")[["acc", "f1_nao", "bal_acc", "macro_f1", "rec_N", "rec_A", "rec_O", "rec_noisy"]].mean()
    tab = tab.loc[[m for m in ORDER if m in tab.index]]
    tab0 = perf[perf.ref == "v0"].groupby("method")[["acc", "f1_nao"]].mean().loc[tab.index]
    tab["acc_vs_v0"], tab["f1_nao_vs_v0"] = tab0.acc, tab0.f1_nao
    D = pd.DataFrame(det).groupby("method").mean(numeric_only=True).drop(columns="rep")
    I = pd.DataFrame(info)
    tm = I[~I.method.str.startswith("_")].groupby("method").time_s.mean()
    tab["time_s"] = tm.reindex(tab.index)
    # paired tests: McNemar per repetition, paired bootstrap over records (all repetitions together)
    rng = np.random.RandomState(0)
    B = 10000
    boot_idx = rng.randint(0, len(y3), (B, len(y3)))
    tests = []
    for a_, b_ in [("loo_em_bic", "ridge_wide"), ("ridge_wide", "ridge_std"), ("loo_em_bic", "conf_learning_cal"),
                   ("loo_em_bic", "ridge_std"), ("conf_learning_cal", "ridge_wide"), ("loo_relabel", "ridge_wide"),
                   ("loo_remove", "ridge_wide"), ("enn", "ridge_wide"), ("backward_ridge", "ridge_wide"),
                   ("loo_em", "ridge_wide"), ("loo_em_inst", "ridge_wide"), ("ridge_wide_v3labels", "ridge_wide"),
                   ("ridge_wide_bal", "ridge_wide"), ("ridge_wide_err", "ridge_wide"), ("ridge_wide_bal", "ridge_std"),
                   ("ridge_wide_v3labels", "ridge_std"), ("ridge_wide_bal_v3labels", "ridge_wide_bal")]:
        if a_ not in methods or b_ not in methods:
            continue
        mc = [mcnemar_exact(pred[a_][r] == y3, pred[b_][r] == y3) for r in complete]
        ca = np.stack([pred[a_][r] == y3 for r in complete]).mean(0); cb = np.stack([pred[b_][r] == y3 for r in complete]).mean(0)
        d_acc = (ca - cb)[boot_idx].mean(1)
        def f1diff(idx):
            return np.mean([f1_challenge(y3[idx], pred[a_][r][idx]) - f1_challenge(y3[idx], pred[b_][r][idx]) for r in complete])
        d_f1 = np.array([f1diff(boot_idx[i]) for i in range(0, B, 10)])        # 1,000 resamples for F1 (cost)
        tests.append(dict(candidate=a_, reference=b_, acc_diff=float((ca - cb).mean()), acc_ci_lo=float(np.percentile(d_acc, 2.5)),
                          acc_ci_hi=float(np.percentile(d_acc, 97.5)), f1_diff=float(f1diff(np.arange(len(y3)))),
                          f1_ci_lo=float(np.percentile(d_f1, 2.5)), f1_ci_hi=float(np.percentile(d_f1, 97.5)),
                          mcnemar_b=[m[0] for m in mc], mcnemar_c=[m[1] for m in mc], mcnemar_p=[m[2] for m in mc],
                          mcnemar_p_max=float(max(m[2] for m in mc))))
    T = pd.DataFrame(tests)
    prim = T.candidate.isin(["loo_em_bic", "ridge_wide"]) & T.reference.isin(["ridge_wide", "ridge_std"]) & \
        (((T.candidate == "loo_em_bic") & (T.reference == "ridge_wide")) | ((T.candidate == "ridge_wide") & (T.reference == "ridge_std")))
    from src.statistics.tests import holm
    T["p_holm_primary"] = np.nan
    T.loc[prim, "p_holm_primary"] = holm(T.loc[prim, "mcnemar_p_max"].values)
    dep = I[I.method == "_deploy"]
    out.update(performance=tab.to_dict(), detection=D.to_dict(), tests=T.to_dict(orient="records"),
               models_chosen=I[I.method.isin(["loo_em_bic", "loo_em_inst"])].groupby(["method", "model"]).size().to_dict().__repr__(),
               t_feat_mean=float(I[(I.method == "_fold") & (I.model == "t_feat")].time_s.mean()),
               t_eig_mean=float(I[(I.method == "_fold") & (I.model == "t_eig")].time_s.mean()),
               latency_ms_median=float(dep.time_s.median()),
               model_kb=float(np.median([int(s.split("|")[0]) for s in dep.model]) / 1024),
               peak_rss_mb=float(np.max([float(s.split("|")[1]) for s in dep.model])),
               n_train_mean=float(np.mean([z["n_train"] for _, z in folds])))
    if write:
        ag.write(tab.rename(index=NAMES), "table_cinc_performance")
        ag.write(D.loc[[m for m in ORDER if m in D.index]].rename(index=NAMES), "table_cinc_detection")
        T2 = T.copy(); T2["mcnemar_p"] = T2.mcnemar_p.map(lambda v: ", ".join(f"{x:.4f}" for x in v))
        T2["mcnemar_b"] = T2.mcnemar_b.map(str); T2["mcnemar_c"] = T2.mcnemar_c.map(str)
        ag.write(T2, "stats_cinc", index=False)
    return out


def merge_into_results(out):
    for fn, key in (("metrics.json", "cinc2017"), ("statistics.json", "cinc2017")):
        p = os.path.join(R, fn); d = json.load(open(p)); d[key] = out
        json.dump(d, open(p, "w"), indent=1, default=float)
