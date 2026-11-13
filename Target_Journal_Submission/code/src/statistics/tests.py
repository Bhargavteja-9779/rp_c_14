"""Statistical comparison of methods across datasets.

Unit of analysis: the dataset (per-dataset score = mean over resample seeds), the
standard design for multi-dataset classifier comparison (Demsar, 2006).  Paired
two-sided Wilcoxon signed-rank tests, Holm correction within a family, matched-
pairs rank-biserial effect size, and a percentile bootstrap CI (datasets
resampled) for the mean paired difference."""
from __future__ import annotations

import numpy as np
from scipy import stats


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def bootstrap_ci(d, n_boot=10000, seed=0, level=0.95):
    d = np.asarray(d, float)
    rng = np.random.RandomState(seed)
    means = d[rng.randint(0, len(d), (n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def paired(a, b):
    """Compare a (candidate) against b (reference) over paired datasets."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if len(d) == 0 or np.all(d == 0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    lo, hi = bootstrap_ci(d) if len(d) else (np.nan, np.nan)
    return dict(n=int(len(d)), mean_diff=float(d.mean()) if len(d) else np.nan,
                median_diff=float(np.median(d)) if len(d) else np.nan, ci_lo=lo, ci_hi=hi,
                p=p, effect_rank_biserial=rank_biserial(d),
                wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()))


def friedman_ranks(table):
    """table: datasets x methods array (higher is better). Returns mean ranks, Friedman p,
    and the Nemenyi critical difference at alpha=0.05."""
    t = np.asarray(table, float)
    t = t[np.isfinite(t).all(1)]
    ranks = np.vstack([stats.rankdata(-row) for row in t])
    n, k = t.shape
    p = float(stats.friedmanchisquare(*t.T).pvalue) if k > 2 else np.nan
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
           11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426}.get(k, np.nan)
    cd = q05 * np.sqrt(k * (k + 1) / (6.0 * n))
    return ranks.mean(0), p, float(cd), n
