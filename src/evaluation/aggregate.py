"""Aggregate raw per-dataset CSVs into results/raw_results.csv, metrics.json,
statistics.json and the manuscript tables (CSV + Markdown).  Nothing here is typed by hand."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from src.data import ucr
from src.statistics import tests as st

R = os.path.join(ucr.ROOT, "results")
T = os.path.join(ucr.ROOT, "tables")
KEY = ["ridge_std", "ridge_wide", "loo_em", "conf_learning", "kfold_relabel", "enn", "loo_relabel", "loo_remove"]
ABL = ["loo_em", "em_no_gate", "em_sym_T", "em_hard", "em_in_sample", "em_no_T", "em_std_grid",
       "loo_relabel", "loo_relabel_it", "loo_remove", "ridge_wide"]
LABEL = {"ridge_std": "Ridge (default grid)", "ridge_wide": "Ridge (wide grid)", "loo_em": "LOO-EM (proposed)",
         "conf_learning": "Confident learning (5-fold)", "kfold_relabel": "5-fold relabel", "enn": "ENN",
         "loo_relabel": "LOO relabel (1 round)", "loo_relabel_it": "LOO relabel (iterated)",
         "loo_remove": "LOO remove", "em_no_gate": "– likelihood gate", "em_sym_T": "– full T (symmetric T)",
         "em_hard": "– soft targets (hard)", "em_in_sample": "– LOO (in-sample scores)",
         "em_no_T": "– noise model (self-training)", "em_std_grid": "– wide grid (default grid)",
         "quant": "QUANT", "nn_ed": "1-NN Euclidean", "nn_dtw": "1-NN DTW"}


def cond_label(kind, rate):
    return f"{kind} {int(round(rate * 100))}%"


def load_raw(sub="raw/minirocket"):
    files = sorted(glob.glob(os.path.join(R, sub, "*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["cond"] = [cond_label(k, r) for k, r in zip(df.noise, df.rate)]
    return df


def per_dataset(df, metric="acc"):
    """dataset x (cond, method) table of seed-averaged scores."""
    return df.groupby(["dataset", "cond", "method"])[metric].mean().unstack(["cond", "method"])


def cond_order(df):
    c = df.drop_duplicates("cond").sort_values(["noise", "rate"])
    order = {"sym": 0, "asym": 1, "boundary": 2}
    return list(c.assign(o=c.noise.map(order)).sort_values(["o", "rate"]).cond)


def mean_table(df, methods, datasets=None, metric="acc"):
    pd_ = per_dataset(df, metric)
    if datasets is not None:
        pd_ = pd_.loc[pd_.index.intersection(datasets)]
    out = pd.DataFrame(index=[m for m in methods if m in df.method.unique()], columns=cond_order(df), dtype=float)
    for c in out.columns:
        sub = pd_[c].dropna(how="all")
        full = sub.dropna(subset=[m for m in out.index if m in sub.columns])  # datasets where condition is defined
        for m in out.index:
            out.loc[m, c] = full[m].mean() if m in full else np.nan
    return out


def stats_table(df, cand, ref, datasets=None, metric="acc"):
    pd_ = per_dataset(df, metric)
    if datasets is not None:
        pd_ = pd_.loc[pd_.index.intersection(datasets)]
    rows = []
    for c in cond_order(df):
        if cand not in pd_[c] or ref not in pd_[c]:
            continue
        sub = pd_[c][[cand, ref]].dropna()
        r = st.paired(sub[cand].values, sub[ref].values)
        r.update(cond=c, candidate=cand, reference=ref)
        rows.append(r)
    out = pd.DataFrame(rows)
    out["p_holm"] = st.holm(out.p.values)
    return out


def write(df, name, floatfmt=".4f", index=True):
    os.makedirs(T, exist_ok=True)
    df.to_csv(os.path.join(T, name + ".csv"), index=index)
    with open(os.path.join(T, name + ".md"), "w") as f:
        f.write(df.to_markdown(floatfmt=floatfmt, index=index) + "\n")


def run(dev=None):
    df = load_raw()
    df.to_csv(os.path.join(R, "raw_results.csv"), index=False)
    n_err = int((df.error.fillna("") != "").sum())
    all_ds = sorted(df.dataset.unique())
    held = [d for d in all_ds if d not in (dev or [])]
    metrics, stats = {"n_datasets": len(all_ds), "n_heldout": len(held), "n_rows": len(df), "n_errors": n_err}, {}

    main = mean_table(df, KEY)
    write(main.rename(index=LABEL), "table_main_accuracy")
    write(mean_table(df, KEY, held).rename(index=LABEL), "table_main_accuracy_heldout")
    write(mean_table(df, KEY, metric="bal_acc").rename(index=LABEL), "table_main_balanced_accuracy")
    write(mean_table(df, KEY, metric="macro_f1").rename(index=LABEL), "table_main_macro_f1")
    write(mean_table(df, ABL).rename(index=LABEL), "table_ablation")
    metrics["mean_accuracy"] = main.to_dict()
    metrics["mean_accuracy_heldout"] = mean_table(df, KEY, held).to_dict()
    metrics["ablation"] = mean_table(df, ABL).to_dict()

    # seed variability: std over seeds of the across-dataset mean
    sv = df[df.method.isin(KEY)].groupby(["cond", "method", "seed"]).acc.mean().groupby(["cond", "method"]).std().unstack("cond")
    write(sv.loc[[m for m in KEY if m in sv.index], cond_order(df)].rename(index=LABEL), "table_seed_std")
    metrics["seed_std"] = sv.to_dict()

    for cand, ref in [("loo_em", "ridge_std"), ("loo_em", "ridge_wide"), ("ridge_wide", "ridge_std"),
                      ("loo_em", "conf_learning"), ("loo_em", "kfold_relabel"), ("loo_em", "enn"),
                      ("loo_em", "em_no_gate"), ("loo_em", "em_sym_T"), ("loo_em", "em_hard"),
                      ("loo_em", "em_in_sample"), ("loo_em", "em_no_T"), ("loo_em", "em_std_grid")]:
        for tag, ds in (("all", None), ("heldout", held)):
            s = stats_table(df, cand, ref, ds)
            stats[f"{cand}__vs__{ref}__{tag}"] = s.to_dict(orient="records")
            if tag == "all" or ref in ("ridge_std", "ridge_wide"):
                write(s[["cond", "n", "mean_diff", "ci_lo", "ci_hi", "p", "p_holm", "effect_rank_biserial",
                         "wins", "ties", "losses"]], f"stats_{cand}_vs_{ref}_{tag}", index=False)

    # detection quality
    det = {}
    for metric in ("det_auroc", "det_f1", "det_prec", "det_rec", "label_acc"):
        t = mean_table(df[df.rate > 0], [m for m in KEY + ["em_in_sample"] if m not in ("ridge_std", "ridge_wide")] + ["ridge_wide"], metric=metric)
        write(t.rename(index=LABEL), f"table_{metric}")
        det[metric] = t.to_dict()
    metrics["detection"] = det

    # efficiency
    e = df.groupby("method").agg(median_time_s=("time_s", "median"), mean_time_s=("time_s", "mean"),
                                 total_time_s=("time_s", "sum"))
    base = df.drop_duplicates(["dataset", "seed"])
    e["time_over_eig_median"] = [float((df[df.method == m].time_s / df[df.method == m].t_eig.clip(lower=1e-3)).median()) for m in e.index]
    write(e.loc[[m for m in KEY + ["em_no_gate"] if m in e.index]].rename(index=LABEL), "table_efficiency")
    metrics["efficiency"] = e.to_dict()
    metrics["base_cost"] = dict(median_t_feat=float(base.t_feat.median()), median_t_eig=float(base.t_eig.median()),
                                total_t_feat=float(base.t_feat.sum()), total_t_eig=float(base.t_eig.sum()))

    # stratification by training-set size
    size = df.drop_duplicates("dataset").set_index("dataset").n_train
    bins = pd.cut(size, [0, 100, 500, 1e9], labels=["n<100", "100<=n<500", "n>=500"], right=False)
    pd_ = per_dataset(df)
    rows = []
    for b in bins.cat.categories:
        ds = bins.index[bins == b]
        for c in cond_order(df):
            sub = pd_.loc[pd_.index.intersection(ds), c][["ridge_std", "ridge_wide", "loo_em", "conf_learning"]].dropna()
            rows.append(dict(size_bin=b, cond=c, n_datasets=len(sub), **sub.mean().to_dict()))
    strat = pd.DataFrame(rows)
    write(strat, "table_by_train_size", index=False)
    metrics["by_train_size"] = strat.to_dict(orient="records")

    # gate / noise-rate estimation diagnostics
    em = df[df.method == "loo_em"]
    diag = em.groupby("cond").agg(accept_rate=("accepted", "mean"), est_noise=("est_noise", "mean"),
                                  true_noise=("true_noise", "mean"), rounds=("rounds", "mean")).loc[cond_order(df)]
    write(diag, "table_em_diagnostics")
    metrics["em_diagnostics"] = diag.to_dict()

    # mean ranks (Friedman / Nemenyi) per condition for the key methods
    ranks = {}
    for c in cond_order(df):
        sub = pd_[c][[m for m in KEY if m in pd_[c]]].dropna()
        mr, p, cd, n = st.friedman_ranks(sub.values)
        ranks[c] = dict(mean_rank=dict(zip(sub.columns, map(float, mr))), friedman_p=p, nemenyi_cd=cd, n=n)
    stats["ranks"] = ranks
    write(pd.DataFrame({c: v["mean_rank"] for c, v in ranks.items()}).rename(index=LABEL), "table_mean_ranks", floatfmt=".2f")

    with open(os.path.join(R, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1, default=float)
    with open(os.path.join(R, "statistics.json"), "w") as f:
        json.dump(stats, f, indent=1, default=float)
    return df, metrics, stats
