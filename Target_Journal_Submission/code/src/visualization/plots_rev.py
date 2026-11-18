"""Figures added in the revision.  Every number shown comes from results/ files."""
from __future__ import annotations

import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from src.data import ucr
from src.visualization import plots as P

F = P.F
INK, BLUE, RED, GREEN, GREY = "#1f3b73", "#eef3fb", "#d62728", "#2ca02c", "#7f7f7f"


def _box(ax, x, y, w, h, text, fc=BLUE, ec=INK, fs=7.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)


def _arrow(ax, x0, y0, x1, y1, color=INK, style="->", rad=0.0, lw=1.1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))


def fig_method_overview():
    """LOO-EM: shared label-independent stage, the EM loop (classifier M-step, noise-model M-step, E-step),
    BIC selection and outputs."""
    fig, ax = plt.subplots(figsize=(10.5, 4.9)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(-0.04, 1)
    ax.text(0.01, 0.98, "(a) computed once, independent of the labels", fontsize=8, weight="bold", color=INK, va="top")
    _box(ax, 0.01, 0.72, 0.15, 0.17, "Training series\nx₁ … xₙ\n+ observed labels ỹ")
    _box(ax, 0.20, 0.72, 0.17, 0.17, "MiniRocket transform\n(frozen random kernels)\n→ standardised Z")
    _box(ax, 0.41, 0.72, 0.20, 0.17, "One eigendecomposition\nZZᵀ = U diag(s) Uᵀ  (n ≤ p)\nor ZᵀZ (n > p)")
    _box(ax, 0.65, 0.72, 0.20, 0.17, "LOO operator for each α\nM(α) = D⁻¹(H − diag h)\n(label-independent, cached)")
    for x0, x1 in ((0.16, 0.20), (0.37, 0.41), (0.61, 0.65)):
        _arrow(ax, x0, 0.805, x1, 0.805)
    ax.text(0.01, 0.63, "(b) EM loop — each step is a matrix product on the cached quantities", fontsize=8, weight="bold", color=INK, va="top")
    _box(ax, 0.02, 0.22, 0.21, 0.30, "M-step: classifier\nridge on soft targets 2Q − 1\nα by LOO squared error\nF_loo = M(α)(2Q − 1)", fc="#fff4e5", ec="#b35900")
    _box(ax, 0.29, 0.22, 0.21, 0.30, "M-step: noise model\nmaximise LOO likelihood\nΣᵢ log Σ_z pᵢ(z) T[z, ỹᵢ]\nover β (temperature) and T", fc="#fff4e5", ec="#b35900")
    _box(ax, 0.56, 0.22, 0.18, 0.30, "E-step\nqᵢ(z) ∝ pᵢ(z) T[z, ỹᵢ]\npᵢ = softmax(β F_loo,i)", fc="#fff4e5", ec="#b35900")
    _arrow(ax, 0.23, 0.37, 0.29, 0.37, color="#b35900"); _arrow(ax, 0.50, 0.37, 0.56, 0.37, color="#b35900")
    _arrow(ax, 0.65, 0.22, 0.125, 0.22, color="#b35900", rad=-0.3)
    ax.text(0.39, -0.02, "repeat until max |ΔQ| < 10⁻³ (at most 20 iterations)", ha="center", fontsize=7, color="#b35900")
    _arrow(ax, 0.75, 0.72, 0.75, 0.66, color=GREY, lw=0.9)
    ax.text(0.76, 0.665, "M(α) used in every M-step", fontsize=6.3, color=GREY, va="bottom")
    ax.text(0.785, 0.55, "(c) model choice and outputs", fontsize=8, weight="bold", color=INK, va="top")
    _box(ax, 0.79, 0.27, 0.19, 0.22, "BIC on the LOO likelihood\nnone (k = 0)\nsymmetric T (k = 1)\nfull T (k = C(C−1))", fc="#eaf6ea", ec=GREEN, fs=6.8)
    _box(ax, 0.79, 0.00, 0.19, 0.21, "Outputs\n• final ridge on Q\n• calibrated probabilities\n• ranked audit list 1 − qᵢ(ỹᵢ)", fc="#eaf6ea", ec=GREEN, fs=6.8)
    _arrow(ax, 0.74, 0.37, 0.79, 0.38, color=GREEN); _arrow(ax, 0.885, 0.27, 0.885, 0.21, color=GREEN)
    P.save(fig, "fig_method_overview")


def fig_deployment(m):
    """End-to-end use: training with label audit, and on-line inference, annotated with measured numbers."""
    c = m.get("cinc2017", {})
    lat = c.get("latency_ms_median", float("nan")); kb = c.get("model_kb", float("nan")); rss = c.get("peak_rss_mb", float("nan"))
    tfe, tei = c.get("t_feat_mean", float("nan")), c.get("t_eig_mean", float("nan"))
    tem = c.get("performance", {}).get("time_s", {}).get("loo_em_bic", float("nan"))
    ntr = c.get("n_train_mean", float("nan"))
    fig, ax = plt.subplots(figsize=(10.5, 4.6)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02)
    ax.text(0.01, 1.0, f"(a) Training and periodic re-training (measured on {ntr:,.0f} ECG recordings, 2 CPU threads)", fontsize=8, weight="bold", color=INK, va="top")
    _box(ax, 0.01, 0.62, 0.15, 0.27, "Archive of recordings\nwith operational labels\n(technicians,\noutsourced, rules)")
    _box(ax, 0.20, 0.62, 0.14, 0.27, f"MiniRocket\nfeatures\n{tfe:.0f} s")
    _box(ax, 0.38, 0.62, 0.16, 0.27, f"Eigendecomposition\n{tei:.0f} s\n+ LOO-EM\n{tem:.0f} s")
    _box(ax, 0.59, 0.73, 0.17, 0.16, "Linear classifier\n(weights + scaling)", fc="#eaf6ea", ec=GREEN)
    _box(ax, 0.59, 0.50, 0.17, 0.17, "Ranked audit list\n(most suspicious\nlabels first)", fc="#fff4e5", ec="#b35900")
    _box(ax, 0.81, 0.50, 0.18, 0.17, "Expert review of\nthe top of the list\n→ corrected labels", fc="#fff4e5", ec="#b35900")
    for x0, x1, y in ((0.16, 0.20, 0.755), (0.34, 0.38, 0.755)):
        _arrow(ax, x0, y, x1, y)
    _arrow(ax, 0.54, 0.80, 0.59, 0.81); _arrow(ax, 0.54, 0.68, 0.59, 0.60)
    _arrow(ax, 0.76, 0.585, 0.81, 0.585, color="#b35900")
    _arrow(ax, 0.90, 0.50, 0.085, 0.62, color="#b35900", rad=-0.18)
    ax.text(0.47, 0.375, "corrected labels flow back into the archive; transform and decomposition need not be recomputed",
            fontsize=6.8, color="#b35900", ha="center")
    ax.text(0.01, 0.33, "(b) On-line inference (per recording, 1 CPU thread, using the classifier from (a))", fontsize=8, weight="bold", color=INK, va="top")
    _box(ax, 0.01, 0.03, 0.15, 0.22, "Sensor stream\n(30-s window)")
    _box(ax, 0.20, 0.03, 0.30, 0.22, f"MiniRocket transform + linear model\nmedian latency {lat:.0f} ms per recording")
    _box(ax, 0.54, 0.03, 0.22, 0.22, f"Model size {kb:,.0f} kB\npeak training memory {rss/1024:.1f} GB")
    _box(ax, 0.80, 0.03, 0.19, 0.22, "Class + calibrated\nconfidence → alarm /\nhuman follow-up", fc="#eaf6ea", ec=GREEN)
    _arrow(ax, 0.16, 0.14, 0.20, 0.14); _arrow(ax, 0.50, 0.14, 0.54, 0.14); _arrow(ax, 0.76, 0.14, 0.80, 0.14)
    P.save(fig, "fig_deployment")


def fig_scaling():
    p = os.path.join(ucr.ROOT, "results", "raw", "scaling.jsonl")
    if not os.path.exists(p):
        return None
    d = pd.DataFrame([json.loads(l) for l in open(p)])
    if "error" in d:
        d = d[d.error.isna()]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for k, mk, col in ((2000, "o", "#1f77b4"), (10000, "s", "#ff7f0e")):
        s = d[d.n_kernels == k].sort_values("n")
        if s.empty:
            continue
        lab = f"{int(s.p.iloc[0]):,} features"
        axes[0].plot(s.n, s.t_decomp, mk + "-", color=col, label=f"decomposition, {lab}")
        axes[0].plot(s.n, s.t_loo_em_bic, mk + "--", color=col, alpha=.6, label=f"LOO-EM, {lab}")
        axes[1].plot(s.n, s.peak_rss_mb / 1024, mk + "-", color=col, label=lab)
        pr = s[s.route == "primal"]
        axes[1].plot(pr.n, pr.peak_rss_mb / 1024, mk, ms=10, mfc="none", mec="k", mew=0.8, label="primal route" if k == 10000 else None)
        axes[2].plot(s.n, 100 * s.acc_ridge_wide, mk + "-", color=col, label=f"ridge, {lab}")
        axes[2].plot(s.n, 100 * s.acc_loo_em_bic, mk + "--", color=col, label=f"LOO-EM, {lab}")
    axes[0].set_ylabel("Seconds"); axes[1].set_ylabel("Peak resident memory (GB)"); axes[2].set_ylabel("Test accuracy (%)")
    for a in axes:
        a.set_xscale("log"); a.set_xlabel("Training-set size n"); a.grid(alpha=.3)
    axes[0].set_yscale("log")
    for a in axes:
        a.legend(frameon=False, fontsize=6.3)
    P.save(fig, "fig_scaling")
    return d


def fig_cinc(m):
    """ROC of label-error detection against the documented expert revisions, and confusion matrices."""
    from sklearn.metrics import roc_curve, confusion_matrix
    from src.data import cinc2017
    files = sorted(glob.glob(os.path.join(ucr.ROOT, "results", "raw", "cinc2017", "rep0_fold*.npz")))
    if not files:
        return
    _, y0, y3, _, _ = cinc2017.load()
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1, 1, 1]))
    show = [("loo_em_bic", "LOO-EM (BIC)", RED), ("conf_learning_cal", "Confident learning (cal.)", GREEN),
            ("enn", "ENN", "#8c564b"), ("loo_relabel", "LOO margin (1 round)", "#e377c2")]
    zs = [np.load(f) for f in files]
    for key, lab, col in show:
        s = np.concatenate([z[f"{key}__score"] for z in zs]); t = np.concatenate([(y0 != y3)[z["train_idx"]] for z in zs])
        fpr, tpr, _ = roc_curve(t, s); axes[0].plot(fpr, tpr, color=col, lw=1.3, label=lab)
    axes[0].plot([0, 1], [0, 1], "k:", lw=.8); axes[0].set_xlabel("False-positive rate"); axes[0].set_ylabel("True-positive rate")
    axes[0].set_title("Finding the expert-revised labels", fontsize=8.5); axes[0].legend(frameon=False, fontsize=6.3, loc="lower right")
    labs = ["N", "A", "O", "~"]
    pred = {k: np.full(len(y0), -1) for k in ("ridge_wide", "loo_em_bic", "ridge_wide_v3labels")}
    for z in zs:
        for k in pred:
            pred[k][z["test_idx"]] = z[f"{k}__pred"]
    for ax, (k, t) in zip(axes[1:], (("ridge_wide", "Ridge (wide), v0 labels"), ("loo_em_bic", "LOO-EM, v0 labels"),
                                     ("ridge_wide_v3labels", "Ridge (wide), expert labels"))):
        cm = confusion_matrix(y3, pred[k], labels=range(4)).astype(float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{100*cmn[i,j]:.0f}", ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > .55 else "black")
        ax.set_xticks(range(4)); ax.set_xticklabels(labs); ax.set_yticks(range(4)); ax.set_yticklabels(labs)
        ax.set_xlabel("Predicted"); ax.set_title(t, fontsize=8.5)
    axes[1].set_ylabel("Expert-revised label (v3)")
    P.save(fig, "fig_cinc2017")


def fig_classwise():
    d = glob.glob(os.path.join(ucr.ROOT, "results", "raw", "classwise", "*.csv"))
    if not d:
        return None
    from src.evaluation import aggregate as ag
    df = pd.concat([pd.read_csv(f) for f in d], ignore_index=True)
    df["cond"] = [ag.cond_label(k, r) for k, r in zip(df.noise, df.rate)]
    meths = [("ridge_std", "Ridge (default grid)"), ("ridge_wide", "Ridge (wide grid)"), ("loo_em_bic", "LOO-EM (BIC)"), ("conf_learning_cal", "Confident learning (cal.)")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for ax, (met, yl) in zip(axes, (("minority_recall", "Recall of smallest class (%)"), ("worst_recall", "Worst-class recall (%)"), ("ece", "Expected calibration error (%)"))):
        for kind, ls in (("sym", "-"), ("asym", "--")):
            for mth, lab in meths:
                s = df[(df.method == mth) & ((df.noise == kind) | (df.rate == 0))].groupby(["rate", "dataset"])[met].mean().groupby("rate").mean()
                ax.plot(100 * s.index, 100 * s.values, ls, marker=P.MK.get(mth, "o"), ms=3.5, color=P.COL.get(mth, "k"),
                        label=f"{lab}" if kind == "sym" else None)
        ax.set_xlabel("Noise rate (%)  (solid: symmetric, dashed: asymmetric)"); ax.set_ylabel(yl); ax.grid(alpha=.3)
    axes[0].legend(frameon=False, fontsize=6.5)
    P.save(fig, "fig_classwise")
    return df


def make_all(metrics):
    fig_method_overview(); fig_deployment(metrics); fig_scaling(); fig_cinc(metrics); fig_classwise()
