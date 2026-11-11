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


def run(main_df):
    out, stats_rows = {}, []
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
