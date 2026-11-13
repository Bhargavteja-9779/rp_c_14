"""Label-cleaning strategies operating on a fitted :class:`GramRidge`.

Every function takes the label-independent ``GramRidge`` object ``g`` and the
(noisy) integer labels ``y`` and returns a dict with

    y_clean   cleaned labels (same length as y)
    keep      boolean mask of training samples retained for the final fit
    flagged   boolean mask of samples judged mislabelled
    score     label-quality score (higher = more likely mislabelled)
    info      method-specific diagnostics
"""
from __future__ import annotations

import numpy as np

from src.models.ridge_loo import GramRidge, one_hot_pm


def margins(F: np.ndarray, y: np.ndarray):
    """Margin of the given label against the strongest rival class, and that rival."""
    n = len(y)
    own = F[np.arange(n), y]
    G = F.copy()
    G[np.arange(n), y] = -np.inf
    return own - G.max(1), G.argmax(1)


def none(g: GramRidge, y, C, **kw):
    n = len(y)
    return dict(y_clean=y.copy(), keep=np.ones(n, bool), flagged=np.zeros(n, bool),
                score=np.zeros(n), info={})


def loo_clean(g: GramRidge, y, C, max_iter: int = 10, tau: float = 0.0,
              action: str = "relabel", reselect: bool = True, in_sample: bool = False,
              min_class: int = 1, **kw):
    """LOO-Clean (proposed).

    Repeat: select alpha by LOO on the current labels; compute closed-form LOO
    decision values; every sample whose current label loses to another class by
    more than ``tau`` is relabelled to the LOO-predicted class.  Because the hat
    matrix is label-independent, each round is a matrix product.

    action     'relabel' (default) or 'remove' (drop flagged samples, one round)
    reselect   re-select alpha every round (False: keep the first-round alpha)
    in_sample  ablation: use in-sample fitted values instead of LOO values
    """
    y0 = y.copy()
    y_cur = y.copy()
    alpha = None
    first_score = None
    n_rounds = 0
    history = []
    for it in range(max_iter):
        Y = one_hot_pm(y_cur, C)
        if alpha is None or reselect:
            alpha, F, _ = g.select_alpha(Y)
            if in_sample:
                F = g.loo(Y, alpha)[1]
        else:
            loo, fit, _ = g.loo(Y, alpha)
            F = fit if in_sample else loo
        m, rival = margins(F, y_cur)
        if first_score is None:
            first_score = -m
        flip = m < -tau
        if min_class > 0:  # never empty a class
            for c in range(C):
                idx = np.flatnonzero(flip & (y_cur == c))
                room = int((y_cur == c).sum()) - min_class
                if len(idx) > room:
                    order = idx[np.argsort(m[idx])]
                    flip[order[max(room, 0):]] = False
        n_rounds += 1
        history.append(int(flip.sum()))
        if not flip.any():
            break
        if action == "remove":
            keep = ~flip
            return dict(y_clean=y0, keep=keep, flagged=flip, score=first_score,
                        info=dict(rounds=1, history=history))
        y_cur = y_cur.copy()
        y_cur[flip] = rival[flip]
    return dict(y_clean=y_cur, keep=np.ones(len(y), bool), flagged=y_cur != y0,
                score=first_score, info=dict(rounds=n_rounds, history=history))


def _softmax(F):
    F = F - F.max(1, keepdims=True)
    e = np.exp(F)
    return e / e.sum(1, keepdims=True)


def _fit_beta(F, lik_fn, lo=-4.0, hi=6.0):
    """1-D maximum-likelihood fit of the inverse temperature (log-scale search)."""
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda t: -lik_fn(_softmax(np.exp(t) * F)), bounds=(lo, hi),
                        method="bounded", options=dict(xatol=1e-3))
    return float(np.exp(r.x))


def loo_em(g: GramRidge, y, C, max_iter: int = 20, tol: float = 1e-3,
           transition: str = "full", prior_strength: float | None = None,
           soft_targets: bool = True, in_sample: bool = False, reselect: bool = True,
           use_noise_model: bool = True, gate: float | None = None,
           anchor: bool = False, anchor_frac: float = 0.03, fix_T: bool = False,
           local_search: bool = False, **kw):
    """LOO-EM (proposed): expectation-maximisation for class-conditional label
    noise in which the classifier's predictive distribution is always the
    closed-form leave-one-out one.

    Latent true label z_i, observed label y_i, noise transition T[z, y].
    E-step  q_i(z)  ~  p_loo(z | x_i ; Q_{-i}) * T[z, y_i]
    M-step  ridge refit on the soft targets 2Q-1 (a matrix product, because the
            hat matrix is label-independent), inverse temperature beta and T by
            maximum observed-data likelihood  sum_i log sum_z p_loo,i(z) T[z, y_i].

    transition      'full' (C x C, shrunk to a symmetric prior), 'sym' (one rate)
    prior_strength  pseudo-count of the symmetric prior per class row (default C)
    soft_targets    final/intermediate ridge targets are the soft posteriors
    in_sample       ablation: in-sample fitted values instead of LOO values
    use_noise_model ablation: T fixed to identity-free 'trust the model' rule,
                    i.e. q_i = p_loo,i (self-training on LOO predictions)
    """
    n = len(y)
    onehot = np.zeros((n, C))
    onehot[np.arange(n), y] = 1.0
    Q = onehot.copy()
    m0 = C if prior_strength is None else prior_strength
    eps = 0.1
    T = np.full((C, C), eps / max(C - 1, 1))
    np.fill_diagonal(T, 1 - eps)
    alpha = None
    first_score = None
    beta = 1.0
    hist = []
    for it in range(max_iter):
        Y = 2 * Q - 1 if soft_targets else one_hot_pm(Q.argmax(1), C)
        if alpha is None or reselect:
            alpha, F, _ = g.select_alpha(Y, around=alpha if (local_search and alpha is not None) else None)
            if in_sample:
                F = g.loo(Y, alpha)[1]
        else:
            loo, fit, _ = g.loo(Y, alpha)
            F = fit if in_sample else loo
        if first_score is None:
            first_score = -margins(F, y)[0]
            b0 = _fit_beta(F, lambda P: np.sum(np.log(P[np.arange(n), y] + 1e-12)))
            P0 = _softmax(b0 * F)  # calibrated LOO estimate of the NOISY-label posterior
            ll_clean = float(np.sum(np.log(P0[np.arange(n), y] + 1e-12)))
            T_anchor = None
            if anchor:  # anchor points (Patrini et al., 2017): the most confident samples of a class reveal its row of T
                m_a = max(1, int(round(anchor_frac * n)))
                T_anchor = np.vstack([P0[np.argsort(-P0[:, c])[:m_a]].mean(0) for c in range(C)])
                T_anchor = np.clip(T_anchor, 1e-6, None)
                T_anchor /= T_anchor.sum(1, keepdims=True)
                T = T_anchor.copy()
        if not use_noise_model:
            beta = _fit_beta(F, lambda P: np.sum(np.log(P[np.arange(n), y] + 1e-12)))
            Qn = _softmax(beta * F)
        else:
            for _ in range(5):  # alternate beta | T
                Ty = T[:, y].T  # n x C : T[z, y_i]
                beta = _fit_beta(F, lambda P: np.sum(np.log((P * Ty).sum(1) + 1e-12)))
                P = _softmax(beta * F)
                R = P * Ty
                R /= R.sum(1, keepdims=True) + 1e-12
                N = R.T @ onehot  # expected counts N[z, k]
                if fix_T:
                    continue
                if T_anchor is not None:
                    T = (N + m0 * T_anchor) / (N.sum(1, keepdims=True) + m0)
                    continue
                if transition == "sym":
                    e = 1.0 - (np.trace(N) + m0 * 0.9) / (n + m0)
                    e = float(np.clip(e, 1e-4, 1 - 1.0 / C))
                    T = np.full((C, C), e / max(C - 1, 1))
                    np.fill_diagonal(T, 1 - e)
                else:
                    e = float(np.clip(1.0 - np.trace(N) / n, 1e-4, 1 - 1.0 / C))
                    prior = np.full((C, C), e / max(C - 1, 1))
                    np.fill_diagonal(prior, 1 - e)
                    T = (N + m0 * prior) / (N.sum(1, keepdims=True) + m0)
            P = _softmax(beta * F)
            Qn = P * T[:, y].T
            Qn /= Qn.sum(1, keepdims=True) + 1e-12
        ll_noise = float(np.sum(np.log((P * T[:, y].T).sum(1) + 1e-12))) if use_noise_model else 0.0
        delta = float(np.abs(Qn - Q).max())
        Q = Qn
        hist.append(delta)
        if delta < tol:
            break
    accepted = True
    if gate is not None and use_noise_model and ll_noise - ll_clean <= gate * 0.5 * np.log(n):
        accepted = False  # the noise model does not explain the labels better
        Q = onehot.copy()
    y_new = Q.argmax(1)
    for c in np.unique(y):  # never empty a class
        if not (y_new == c).any():
            j = np.flatnonzero(y == c)
            y_new[j[np.argmax(Q[j, c])]] = c
    return dict(y_clean=y_new, keep=np.ones(n, bool), flagged=y_new != y,
                score=1.0 - Q[np.arange(n), y] if use_noise_model else first_score,
                soft=Q if soft_targets else None,
                info=dict(rounds=len(hist), beta=beta, accepted=accepted, ll_noise=ll_noise, ll_clean=ll_clean, noise_rate=float(1 - np.trace(T) / C),
                          T=T, alpha=alpha))


def loo_em_bic(g: GramRidge, y, C, penalty: float = 1.0, **kw):
    """LOO-EM v2: choose between {no noise, symmetric T, full T} by BIC on the
    leave-one-out observed-label log-likelihood, counting 0, 1 and C(C-1) free
    noise parameters respectively.  (v1 gated the full model with a one-parameter
    penalty, which let a C x C matrix absorb ordinary model error on many-class
    data; see supplementary/protocol.md.)"""
    n = len(y)
    half_log_n = 0.5 * np.log(n) * penalty
    cands = []
    for name, k in (("sym", 1), ("full", C * (C - 1))):
        r = loo_em(g, y, C, transition=name, gate=None, local_search=True, **kw)
        cands.append((r["info"]["ll_noise"] - k * half_log_n, name, r))
    ll_clean = cands[0][2]["info"]["ll_clean"]
    best = max(cands, key=lambda t: t[0])
    if best[0] <= ll_clean:
        r = none(g, y, C)
        r["score"] = cands[0][2]["score"]
        r["info"] = dict(model="none", accepted=False, noise_rate=0.0,
                         rounds=cands[0][2]["info"]["rounds"] + cands[1][2]["info"]["rounds"])
        return r
    r = best[2]
    r["info"].update(model=best[1], accepted=True,
                     rounds=cands[0][2]["info"]["rounds"] + cands[1][2]["info"]["rounds"])
    return r


def kfold_scores(g: GramRidge, y, C, k: int = 5, seed: int = 0):
    """Out-of-fold ridge decision values via explicit k-fold refits (the costly
    route that closed-form LOO replaces)."""
    from sklearn.model_selection import StratifiedKFold, KFold
    n = len(y)
    Zraw = g.Z_ * g.sd_ + g.mu_
    F = np.zeros((n, C))
    counts = np.bincount(y, minlength=C)
    k_eff = int(max(2, min(k, n // 2)))
    if counts[counts > 0].min() >= k_eff:
        splits = StratifiedKFold(k_eff, shuffle=True, random_state=seed).split(np.zeros(n), y)
    else:
        splits = KFold(k_eff, shuffle=True, random_state=seed).split(np.zeros(n))
    for tr, te in splits:
        gi = GramRidge(g.alphas, g.dtype).fit(Zraw[tr])
        Y = one_hot_pm(y[tr], C)
        a, _, _ = gi.select_alpha(Y)
        F[te] = gi.predict(gi.test_kernel(Zraw[te]), Y, a)
    return F


def confident_learning(g: GramRidge, y, C, k: int = 5, seed: int = 0, calibrate: bool = False, **kw):
    """Confident learning (Northcutt et al., 2021) with k-fold out-of-sample
    probabilities (softmax of ridge decision values); flagged samples are removed."""
    from cleanlab.filter import find_label_issues
    F = kfold_scores(g, y, C, k, seed)
    if calibrate:  # temperature fitted on the out-of-fold scores (maximum likelihood of the given labels)
        n = len(y)
        beta = _fit_beta(F, lambda P: np.sum(np.log(P[np.arange(n), y] + 1e-12)))
        P = _softmax(beta * F)
    else:
        P = _softmax(F)
    present = np.unique(y)
    try:
        if len(present) < C:  # cleanlab needs every class present
            raise ValueError
        issues = find_label_issues(labels=y, pred_probs=P, filter_by="prune_by_noise_rate")
    except Exception:
        issues = P.argmax(1) != y
    keep = ~issues
    for c in present:  # never empty a class
        if not (keep & (y == c)).any():
            keep[np.flatnonzero(y == c)[0]] = True
    m, _ = margins(F, y)
    return dict(y_clean=y.copy(), keep=keep, flagged=~keep, score=-m, info={})


def kfold_relabel(g: GramRidge, y, C, k: int = 5, seed: int = 0, **kw):
    """One-shot relabelling from k-fold out-of-fold predictions (CV analogue of
    a single LOO-Clean round)."""
    F = kfold_scores(g, y, C, k, seed)
    m, rival = margins(F, y)
    y2 = y.copy()
    flip = m < 0
    y2[flip] = rival[flip]
    for c in np.unique(y):
        if not (y2 == c).any():
            y2 = y.copy()
            break
    return dict(y_clean=y2, keep=np.ones(len(y), bool), flagged=y2 != y, score=-m, info={})


def enn(g: GramRidge, y, C, k: int = 3, **kw):
    """Edited nearest neighbours (Wilson, 1972) in the standardised feature space:
    remove samples whose label disagrees with the majority of their k neighbours."""
    n = len(y)
    sq = np.sum(g.Z_ ** 2, 1)
    D = sq[:, None] + sq[None, :] - 2 * (g.Z_ @ g.Z_.T)
    np.fill_diagonal(D, np.inf)
    k = min(k, n - 1)
    nn = np.argpartition(D, k - 1, axis=1)[:, :k]
    votes = np.zeros((n, C))
    for j in range(k):
        np.add.at(votes, (np.arange(n), y[nn[:, j]]), 1)
    agree = votes[np.arange(n), y] / k
    keep = votes.argmax(1) == y
    keep |= agree >= 0.5
    for c in np.unique(y):
        if not (keep & (y == c)).any():
            keep[np.flatnonzero(y == c)[0]] = True
    return dict(y_clean=y.copy(), keep=keep, flagged=~keep, score=1 - agree, info={})


def final_predict(g: GramRidge, Fte, K_te, res, C):
    """Fit the final ridge on the cleaned training set; return test decision values
    and the selected alpha.  ``K_te`` is the test kernel of the full model ``g``;
    if samples were removed the model is refitted on the retained subset."""
    y, keep = res["y_clean"], res["keep"]
    if keep.all():
        Y = one_hot_pm(y, C) if res.get("soft") is None else 2 * res["soft"] - 1
        a, _, _ = g.select_alpha(Y)
        return g.predict(K_te, Y, a), a
    idx = np.flatnonzero(keep)
    gs = g.subset(idx)
    Y = one_hot_pm(y[idx], C)
    a, _, _ = gs.select_alpha(Y)
    return gs.predict(gs.test_kernel(Fte), Y, a), a


METHODS = {
    "ridge": none,
    "loo_em": loo_em,
    "loo_em_bic": loo_em_bic,
    "loo_relabel": loo_clean,
    "confident_learning": confident_learning,
    "kfold_relabel": kfold_relabel,
    "enn": enn,
}
