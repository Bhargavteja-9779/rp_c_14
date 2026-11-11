"""Publication figures generated from results/raw_results.csv (no hand-entered numbers)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import ucr
from src.evaluation import aggregate as ag

F = os.path.join(ucr.ROOT, "figures")
COL = {"ridge_std": "#7f7f7f", "ridge_wide": "#1f77b4", "loo_em": "#ff9896", "loo_em_bic": "#d62728", "conf_learning": "#2ca02c",
       "kfold_relabel": "#9467bd", "enn": "#8c564b", "loo_relabel": "#e377c2", "loo_remove": "#17becf"}
MK = {"ridge_std": "o", "ridge_wide": "s", "loo_em": "d", "loo_em_bic": "D", "conf_learning": "^", "kfold_relabel": "v", "enn": "P",
      "loo_relabel": "X", "loo_remove": "*"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
KIND = {"sym": "Symmetric noise", "asym": "Asymmetric (pair-flip) noise", "boundary": "Boundary (instance-dependent) noise"}


def save(fig, name):
    os.makedirs(F, exist_ok=True)
    fig.savefig(os.path.join(F, name + ".png"))
    fig.savefig(os.path.join(F, name + ".pdf"))
    plt.close(fig)


def _complete(pdt, cond, methods):
    return pdt[cond][[m for m in methods if m in pdt[cond]]].dropna()


def fig_accuracy_curves(df, methods=("ridge_std", "ridge_wide", "loo_em_bic", "loo_em", "conf_learning", "enn"), name="fig_accuracy_vs_noise", metric="acc"):
    pdt = ag.per_dataset(df, metric)
    clean = _complete(pdt, "sym 0%", methods)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, kind in zip(axes, ("sym", "asym", "boundary")):
        rates = sorted(df[df.noise == kind].rate.unique())
        for m in methods:
            xs, ys, es = [], [], []
            for r in ([0.0] if kind != "sym" else []) + rates:
                c = ag.cond_label("sym" if r == 0 else kind, r)
                sub = _complete(pdt, c, methods)
                if m not in sub: continue
                xs.append(100 * r); ys.append(sub[m].mean()); es.append(1.96 * sub[m].std(ddof=1) / np.sqrt(len(sub)))
            ax.errorbar(xs, ys, yerr=None, marker=MK[m], color=COL[m], lw=1.5, ms=5, label=ag.LABEL[m])
        ax.set_title(KIND[kind]); ax.set_xlabel("Training-label noise rate (%)"); ax.grid(alpha=.3)
    axes[0].set_ylabel("Mean test accuracy over datasets")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    save(fig, name)


def fig_scatter(df, conds=("sym 40%", "asym 40%", "boundary 30%"), a="loo_em_bic", b="ridge_wide", name="fig_scatter_em_vs_ridge"):
    pdt = ag.per_dataset(df)
    size = df.drop_duplicates("dataset").set_index("dataset").n_train
    fig, axes = plt.subplots(1, len(conds), figsize=(3.4 * len(conds), 3.4))
    for ax, c in zip(np.atleast_1d(axes), conds):
        sub = _complete(pdt, c, [a, b])
        s = np.log10(size.loc[sub.index])
        sc = ax.scatter(sub[b], sub[a], c=s, cmap="viridis", s=14, alpha=.85, vmin=1.2, vmax=3.9)
        ax.plot([0, 1], [0, 1], "k--", lw=.8)
        w, l = int((sub[a] > sub[b]).sum()), int((sub[a] < sub[b]).sum())
        ax.set_title(f"{c}  (wins {w} / losses {l})", fontsize=9)
        ax.set_xlabel(f"{ag.LABEL[b]} accuracy"); ax.set_ylabel(f"{ag.LABEL[a]} accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=axes, shrink=.8, pad=.02); cb.set_label("log10(training-set size)")
    save(fig, name)


def fig_alpha(df, name="fig_selected_alpha"):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    sub = df[(df.noise == "sym")]
    for ax, m in zip(axes, ("ridge_std", "ridge_wide")):
        d = [np.log10(sub[(sub.method == m) & (sub.rate == r)].alpha.dropna()) for r in sorted(sub.rate.unique())]
        ax.violinplot(d, showmedians=True)
        ax.set_xticks(range(1, len(d) + 1)); ax.set_xticklabels([f"{int(100*r)}%" for r in sorted(sub.rate.unique())])
        ax.set_title(ag.LABEL[m]); ax.set_xlabel("Symmetric noise rate"); ax.set_ylabel("log10 selected penalty")
        if m == "ridge_std": ax.axhline(3, color="r", ls=":", lw=1); ax.text(0.6, 3.1, "upper end of default grid", color="r", fontsize=7)
    save(fig, name)


def fig_bars(table, name, ylabel, ref=None):
    """table: methods x conditions DataFrame."""
    fig, ax = plt.subplots(figsize=(10, 3.4))
    k = len(table.index); w = .8 / k
    for i, m in enumerate(table.index):
        ax.bar(np.arange(len(table.columns)) + i * w, table.loc[m].values, w, label=ag.LABEL.get(m, m))
    ax.set_xticks(np.arange(len(table.columns)) + .4 - w / 2); ax.set_xticklabels(table.columns, rotation=30, ha="right")
    ax.set_ylabel(ylabel); ax.axhline(0, color="k", lw=.6); ax.legend(frameon=False, fontsize=7, ncol=2); ax.grid(axis="y", alpha=.3)
    save(fig, name)


def fig_ablation(df):
    t = ag.mean_table(df, ag.ABL)
    d = (t - t.loc["loo_em_bic"]).drop(index="loo_em_bic") * 100
    fig_bars(d, "fig_ablation", "Accuracy change vs LOO-EM (BIC), points")


def fig_detection(df):
    t = ag.mean_table(df[df.rate > 0], ["loo_em_bic", "bic_in_sample", "conf_learning", "kfold_relabel", "enn", "loo_relabel"], metric="det_auroc")
    fig_bars(t, "fig_detection_auroc", "Label-error detection AUROC")
    fig, ax = plt.subplots(figsize=(4, 3.2))
    em = df[(df.method == "em_no_gate") & (df.rate > 0)]
    for kind, c in (("sym", "#1f77b4"), ("asym", "#d62728"), ("boundary", "#2ca02c")):
        s = em[em.noise == kind]
        ax.scatter(s.true_noise, s.est_noise, s=3, alpha=.15, color=c)
        g = s.groupby("rate").est_noise.median()
        ax.plot(g.index, g.values, "o-", color=c, label=f"{kind} (median)")
    ax.plot([0, .45], [0, .45], "k--", lw=.8); ax.set_xlabel("True noise rate"); ax.set_ylabel("Estimated noise rate (1 − mean diag T)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig_noise_rate_estimation")


def fig_runtime(df):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    one = df[(df.noise == "asym") & (df.rate == 0.3)]
    base = one.drop_duplicates(["dataset", "seed"])
    ax.scatter(base.n_train, base.t_eig, s=6, color="k", alpha=.4, label="Shared eigendecomposition")
    for m in ("loo_em_bic", "conf_learning", "enn"):
        s = one[one.method == m]
        ax.scatter(s.n_train, s.time_s, s=6, alpha=.4, color=COL[m], label=ag.LABEL[m])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Training-set size n"); ax.set_ylabel("Wall-clock time (s), 1 CPU thread")
    ax.legend(frameon=False, fontsize=7); ax.grid(alpha=.3)
    save(fig, "fig_runtime")


def fig_gain_vs_size(df, cond="asym 40%"):
    pdt = ag.per_dataset(df)
    size = df.drop_duplicates("dataset").set_index("dataset").n_train
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, c in zip(axes, (cond, "sym 0%")):
        sub = _complete(pdt, c, ["loo_em_bic", "ridge_wide"])
        d = (sub.loo_em_bic - sub.ridge_wide) * 100
        ax.scatter(size.loc[sub.index], d, s=10, color="#d62728", alpha=.7)
        ax.axhline(0, color="k", lw=.7); ax.set_xscale("log"); ax.set_title(c)
        ax.set_xlabel("Training-set size n"); ax.set_ylabel("LOO-EM (BIC) − Ridge (wide), points"); ax.grid(alpha=.3)
    save(fig, "fig_gain_vs_train_size")


def fig_ranks(stats):
    conds = ["sym 0%", "sym 20%", "sym 40%", "asym 20%", "asym 40%", "boundary 20%"]
    conds = [c for c in conds if c in stats["ranks"]]
    fig, axes = plt.subplots(1, len(conds), figsize=(2.3 * len(conds), 3), sharey=True)
    for ax, c in zip(axes, conds):
        r = stats["ranks"][c]; items = sorted(r["mean_rank"].items(), key=lambda kv: kv[1])
        ax.barh([ag.LABEL[k] for k, _ in items][::-1], [v for _, v in items][::-1], color=[COL[k] for k, _ in items][::-1])
        ax.set_title(f"{c}\nCD={r['nemenyi_cd']:.2f}", fontsize=8); ax.set_xlabel("Mean rank (lower is better)")
        ax.tick_params(axis="y", labelsize=7)
    save(fig, "fig_mean_ranks")


def fig_method_diagram():
    fig, ax = plt.subplots(figsize=(10, 2.6)); ax.axis("off")
    boxes = [("Time series +\nnoisy labels", .01), ("Random conv. kernels\n(MiniRocket), frozen", .18),
             ("ONE eigendecomposition\nof the centred Gram matrix\n(label-independent)", .37),
             ("LOO-EM loop (matrix products only)\nE: q ∝ p_LOO(z|x)·T[z, y]\nM: ridge on soft targets, β, T", .59),
             ("BIC choice: none /\nsymmetric / full T →\nfinal ridge", .84)]
    for (t, x), wd in zip(boxes, (.14, .16, .19, .22, .15)):
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, .25), wd, .5, boxstyle="round,pad=0.01", fc="#eef3fb", ec="#1f3b73"))
        ax.text(x + wd / 2, .5, t, ha="center", va="center", fontsize=7.5)
    for x0, x1 in ((.15, .18), (.34, .37), (.56, .59), (.81, .84)):
        ax.annotate("", xy=(x1, .5), xytext=(x0, .5), arrowprops=dict(arrowstyle="->", color="#1f3b73"))
    ax.annotate("", xy=(.62, .78), xytext=(.78, .78), arrowprops=dict(arrowstyle="->", color="#d62728", connectionstyle="arc3,rad=0.5"))
    ax.text(.70, .93, "repeat until posteriors converge", ha="center", fontsize=7, color="#d62728")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig_method_overview")


def fig_dataset_distribution():
    d = pd.read_csv(os.path.join(ucr.ROOT, "data", "dataset_summary.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8))
    axes[0].hist(np.log10(d.n_train), 20, color="#1f77b4"); axes[0].set_xlabel("log10(training-set size)")
    axes[1].hist(d.n_classes, 20, color="#2ca02c"); axes[1].set_xlabel("Number of classes")
    axes[2].hist(np.log10(d.length), 20, color="#9467bd"); axes[2].set_xlabel("log10(series length)")
    for a in axes: a.set_ylabel("Datasets")
    save(fig, "fig_dataset_distribution")


def make_all(df, stats):
    fig_method_diagram(); fig_dataset_distribution()
    fig_accuracy_curves(df); fig_scatter(df); fig_alpha(df); fig_ablation(df); fig_detection(df)
    fig_runtime(df); fig_gain_vs_size(df); fig_ranks(stats)
