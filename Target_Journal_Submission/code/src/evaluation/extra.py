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
