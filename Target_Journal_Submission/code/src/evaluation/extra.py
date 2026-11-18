"""Aggregation of the secondary experiments (other transforms, other classifier families, deep
baselines, multivariate UEA, sensitivity).  Every table is written to tables/ and the numbers are
merged into results/metrics.json / statistics.json under the key 'extra'."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from src.evaluation import aggregate as ag
from src.statistics import tests as st


def _load(sub):
    files = sorted(glob.glob(os.path.join(ag.R, "raw", sub, "*.csv")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["cond"] = [ag.cond_label(k, r) for k, r in zip(df.noise, df.rate)]
    return df


def _paired_rows(df, cand, ref, tag):
    s = ag.stats_table(df, cand, ref)
    s.insert(0, "experiment", tag)
    return s


def _paired_rows_metric(df, cand, ref, tag, metric):
    s = ag.stats_table(df, cand, ref, None, metric)
    s.insert(0, "experiment", tag)
    return s


def _wil(a, b):
    r = st.paired(np.asarray(a), np.asarray(b))
    return dict(mean_diff=r["mean_diff"], p=r["p"], wins=r["wins"], losses=r["losses"], ties=r["ties"])


def _restrict(main, other):
    """Rows of the main experiment restricted to the datasets/seeds/conditions of another run."""
    key = other[["dataset", "seed", "cond"]].drop_duplicates()
    return main.merge(key, on=["dataset", "seed", "cond"])


def error_analysis(main_df):
    """Where does LOO-EM (BIC) win / lose against wide-grid ridge, and what predicts it?"""
    pdt = ag.per_dataset(main_df)
    meta = main_df.drop_duplicates("dataset").set_index("dataset")[["n_train", "n_classes"]]
    rows, corr = [], {}
    for c in ("sym 0%", "sym 30%", "asym 40%", "boundary 30%"):
        sub = pdt[c][["ridge_wide", "loo_em_bic", "loo_em"]].dropna()
        d = (sub.loo_em_bic - sub.ridge_wide).sort_values()
        for name, sel in (("largest losses", d.index[:5]), ("largest wins", d.index[::-1][:5])):
            for ds in sel:
                rows.append(dict(cond=c, group=name, dataset=ds, n_train=int(meta.n_train[ds]), n_classes=int(meta.n_classes[ds]),
                                 ridge_wide=sub.ridge_wide[ds], loo_em_bic=sub.loo_em_bic[ds], diff=d[ds]))
        from scipy.stats import spearmanr
        clean = pdt["sym 0%"]["ridge_wide"].reindex(d.index)
        corr[c] = dict(spearman_diff_vs_log_n=float(spearmanr(d, np.log(meta.n_train.reindex(d.index))).statistic),
                       spearman_diff_vs_n_classes=float(spearmanr(d, meta.n_classes.reindex(d.index)).statistic),
                       spearman_diff_vs_clean_accuracy=float(spearmanr(d, clean).statistic),
                       frac_datasets_changed=float((d != 0).mean()))
    t = pd.DataFrame(rows)
    ag.write(t, "table_error_analysis", index=False)
    ag.write(pd.DataFrame(corr).T, "table_error_correlates")
    return dict(extremes=rows, correlates=corr)


def run(main_df):
    out, stats_rows = {}, []
    out["error_analysis"] = error_analysis(main_df)
    d = _load("large")
    if d is not None:
        t = ag.mean_table(d, ["ridge_std", "ridge_wide", "loo_em_bic", "loo_em", "em_sym_T", "enn", "loo_relabel"])
        ag.write(t.rename(index=ag.LABEL), "table_large")
        per = d.groupby(["dataset", "method"]).agg(time_s=("time_s", "mean"), t_eig=("t_eig", "mean"), t_feat=("t_feat", "mean"), n_train=("n_train", "first")).reset_index()
        ag.write(per, "table_large_timing", index=False)
        out["large"] = dict(datasets=sorted(d.dataset.unique()), mean_accuracy=t.to_dict(), timing=per.to_dict(orient="records"))
    core = ["ridge_std", "ridge_wide", "loo_em_bic", "loo_em", "conf_learning"]

    # ---- other random-kernel transforms -------------------------------------------------
    for tr in ("multirocket", "hydra"):
        d = _load(tr)
        if d is None:
            continue
        t = ag.mean_table(d, core)
        ag.write(t.rename(index=ag.LABEL), f"table_transform_{tr}")
        out[f"transform_{tr}"] = dict(n_datasets=int(d.dataset.nunique()), mean_accuracy=t.to_dict())
        for ref in ("ridge_std", "ridge_wide"):
            stats_rows.append(_paired_rows(d, "loo_em_bic", ref, tr))
        stats_rows.append(_paired_rows(d, "ridge_wide", "ridge_std", tr))

    # ---- other classifier families and deep baselines ------------------------------------
    for sub, tag in (("other", "families"), ("deep", "deep")):
        d = _load(sub)
        if d is None:
            continue
        m = _restrict(main_df[main_df.method.isin(["ridge_std", "ridge_wide", "loo_em_bic", "loo_em"])], d)
        both = pd.concat([m, d], ignore_index=True)
        methods = ["ridge_std", "ridge_wide", "loo_em_bic", "loo_em"] + sorted(d.method.unique())
        t = ag.mean_table(both, methods)
        ag.write(t.rename(index={**ag.LABEL, "fcn_ce": "FCN (cross-entropy)", "fcn_gce": "FCN (GCE loss)",
                                 "fcn_coteach": "FCN (Co-teaching, oracle rate)"}), f"table_{tag}")
        out[tag] = dict(n_datasets=int(d.dataset.nunique()), seeds=sorted(map(int, d.seed.unique())),
                        mean_accuracy=t.to_dict(), mean_time_s=d.groupby("method").time_s.mean().to_dict())
        for ref in sorted(d.method.unique()):
            stats_rows.append(_paired_rows(both, "loo_em_bic", ref, tag))
            stats_rows.append(_paired_rows(both, "ridge_wide", ref, tag))

    # ---- multivariate (UEA) ----------------------------------------------------------------
    d = _load("uea")
    if d is not None:
        t = ag.mean_table(d, ["ridge_std", "ridge_wide", "loo_em_bic", "loo_em", "em_sym_T", "conf_learning", "enn"])
        ag.write(t.rename(index=ag.LABEL), "table_uea")
        out["uea"] = dict(n_datasets=int(d.dataset.nunique()), datasets=sorted(d.dataset.unique()), mean_accuracy=t.to_dict())
        for ref in ("ridge_std", "ridge_wide"):
            stats_rows.append(_paired_rows(d, "loo_em_bic", ref, "uea"))
        stats_rows.append(_paired_rows(d, "ridge_wide", "ridge_std", "uea"))

    # ---- sensitivity -------------------------------------------------------------------------
    parts = []
    for f in sorted(glob.glob(os.path.join(ag.R, "raw", "sensitivity", "*"))):
        d = _load(os.path.join("sensitivity", os.path.basename(f)))
        if d is None:
            continue
        t = ag.mean_table(d, sorted(d.method.unique()))
        t.insert(0, "transform", os.path.basename(f))
        parts.append(t.reset_index().rename(columns={"index": "method"}))
    if parts:
        s = pd.concat(parts, ignore_index=True)
        ag.write(s, "table_sensitivity", index=False)
        out["sensitivity"] = s.to_dict(orient="records")

    # ---- CTW re-implementation on the CTW paper's UCR datasets -------------------------------------
    d = _load("ctw")
    if d is not None:
        ok = d.groupby(["dataset", "seed"]).cond.nunique()
        m = _restrict(main_df[main_df.method.isin(["ridge_std", "ridge_wide", "loo_em_bic", "conf_learning_cal"])], d)
        both = pd.concat([m, d], ignore_index=True)
        t = ag.mean_table(both, ["ridge_std", "ridge_wide", "loo_em_bic", "conf_learning_cal", "ctw"])
        ag.write(t.rename(index={**ag.LABEL, "ctw": "CTW (re-implemented)"}), "table_ctw")
        per = both.groupby(["dataset", "cond", "method"]).acc.mean().unstack("method")
        ag.write(per[["ridge_wide", "loo_em_bic", "ctw"]].reset_index(), "table_ctw_per_dataset", index=False)
        out["ctw"] = dict(n_datasets=int(d.dataset.nunique()), datasets=sorted(d.dataset.unique()), seeds=sorted(map(int, d.seed.unique())),
                          mean_accuracy=t.to_dict(), mean_time_s=float(d.time_s.mean()), median_time_s=float(d.time_s.median()))
        for ref in ("ridge_wide", "loo_em_bic"):
            stats_rows.append(_paired_rows(both, "ctw", ref, "ctw"))

    p = os.path.join(ag.R, "raw", "ctw_check.jsonl")
    if os.path.exists(p):
        ck = pd.DataFrame([json.loads(l) for l in open(p)])
        out["ctw_check"] = {d: dict(n_folds=int(len(g)), mean=float(g.acc.mean()), sd=float(g.acc.std(ddof=0)),
                                    reported=float(g.reported.iloc[0])) for d, g in ck.groupby("dataset")}

    # ---- revision baselines/candidate on all datasets (main_df already contains them) -------------------
    if "backward_ridge" in set(main_df.method):
        for cand in ("backward_ridge", "loo_em_inst"):
            if cand in set(main_df.method):
                stats_rows.append(_paired_rows(main_df, cand, "ridge_wide", "main"))
                stats_rows.append(_paired_rows(main_df, cand, "loo_em_bic", "main"))

    # ---- class-wise and calibration pass ----------------------------------------------------------------
    d = _load("classwise")
    if d is not None:
        res = {}
        for met in ("minority_recall", "worst_recall", "ece", "brier"):
            t = ag.mean_table(d, ["ridge_std", "ridge_wide", "loo_em_bic", "conf_learning_cal"], metric=met)
            ag.write(t.rename(index=ag.LABEL), f"table_classwise_{met}")
            res[met] = t.to_dict()
            for cand, ref in (("loo_em_bic", "ridge_wide"), ("ridge_wide", "ridge_std"), ("conf_learning_cal", "ridge_wide")):
                stats_rows.append(_paired_rows_metric(d, cand, ref, f"classwise_{met}", met))
        out["classwise"] = dict(n_datasets=int(d.dataset.nunique()), **res)

    # ---- scaling ------------------------------------------------------------------------------------------
    p = os.path.join(ag.R, "raw", "scaling.jsonl")
    if os.path.exists(p):
        sc = pd.DataFrame([json.loads(l) for l in open(p)])
        ag.write(sc, "table_scaling", index=False)
        out["scaling"] = sc.to_dict(orient="records")

    # ---- exact vs closed-form LOO -------------------------------------------------------------------------
    d = sorted(glob.glob(os.path.join(ag.R, "raw", "exact_loo", "*.csv")))
    if d:
        ex = pd.concat([pd.read_csv(f) for f in d], ignore_index=True)
        g = ex.groupby("cond").agg(n_datasets=("dataset", "nunique"), corr_median=("corr", "median"), corr_min=("corr", "min"),
                                   argmax_agree=("argmax_agree", "mean"), alpha_same=("alpha_closed", lambda v: np.nan),
                                   acc_closed=("acc_em_closed", "mean"), acc_exact=("acc_em_exact", "mean"))
        g["alpha_same"] = ex.assign(same=np.isclose(ex.alpha_closed, ex.alpha_exact)).groupby("cond").same.mean()
        grid = np.logspace(-3, 9, 25)
        step = lambda v: np.argmin(np.abs(np.log(grid) - np.log(v)))
        ex["alpha_steps"] = [abs(step(x) - step(y)) for x, y in zip(ex.alpha_closed, ex.alpha_exact)]
        g["alpha_within1"] = ex.groupby("cond").alpha_steps.apply(lambda v: float((v <= 1).mean()))
        g["alpha_steps_max"] = ex.groupby("cond").alpha_steps.max()
        g["model_same"] = ex.assign(same=ex.model_closed == ex.model_exact).groupby("cond").same.mean()
        g["max_abs_acc_diff"] = ex.assign(dd=(ex.acc_em_closed - ex.acc_em_exact).abs()).groupby("cond").dd.max()
        ag.write(g, "table_exact_loo")
        out["exact_loo"] = dict(per_cond=g.reset_index().to_dict(orient="records"), n_datasets=int(ex.dataset.nunique()),
                                wilcoxon={c: _wil(x.acc_em_exact.values, x.acc_em_closed.values) for c, x in ex.groupby("cond")})

    if stats_rows:
        s = pd.concat(stats_rows, ignore_index=True)
        cols = ["experiment", "candidate", "reference", "cond", "n", "mean_diff", "ci_lo", "ci_hi", "p", "p_holm",
                "effect_rank_biserial", "wins", "ties", "losses"]
        ag.write(s[cols], "stats_secondary", index=False)
    with open(os.path.join(ag.R, "metrics.json")) as f:
        m = json.load(f)
    m["extra"] = out
    with open(os.path.join(ag.R, "metrics.json"), "w") as f:
        json.dump(m, f, indent=1, default=float)
    if stats_rows:
        with open(os.path.join(ag.R, "statistics.json")) as f:
            sj = json.load(f)
        sj["secondary"] = s.to_dict(orient="records")
        with open(os.path.join(ag.R, "statistics.json"), "w") as f:
            json.dump(sj, f, indent=1, default=float)
    return out
