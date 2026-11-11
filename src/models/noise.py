"""Synthetic label-noise models (applied to training labels only)."""
from __future__ import annotations

import numpy as np


def symmetric(y, n_classes, rate, rng):
    """Flip exactly round(rate*n) labels uniformly to one of the other classes."""
    y = y.copy()
    n = len(y)
    idx = rng.choice(n, int(round(rate * n)), replace=False)
    for i in idx:
        y[i] = rng.choice([c for c in range(n_classes) if c != y[i]])
    return y


def asymmetric(y, n_classes, rate, rng):
    """Pair flip: a fraction ``rate`` of every class c is relabelled (c+1) mod C."""
    y0 = y
    y = y.copy()
    for c in range(n_classes):
        members = np.flatnonzero(y0 == c)
        k = int(round(rate * len(members)))
        if k:
            y[rng.choice(members, k, replace=False)] = (c + 1) % n_classes
    return y


def boundary(y, n_classes, rate, rng, clean_scores):
    """Instance-dependent noise: the round(rate*n) samples with the smallest clean
    LOO margin are relabelled to their strongest competing class.  ``clean_scores``
    are LOO decision values of a ridge model fitted on the clean labels."""
    y = y.copy()
    n = len(y)
    s = clean_scores.copy()
    true = s[np.arange(n), y].copy()
    s[np.arange(n), y] = -np.inf
    rival = s.argmax(1)
    margin = true - s.max(1) + 1e-9 * rng.standard_normal(n)
    idx = np.argsort(margin)[: int(round(rate * n))]
    y[idx] = rival[idx]
    return y


def apply_noise(kind, y, n_classes, rate, rng, clean_scores=None):
    if rate <= 0:
        return y.copy()
    if kind == "sym":
        return symmetric(y, n_classes, rate, rng)
    if kind == "asym":
        return asymmetric(y, n_classes, rate, rng)
    if kind == "boundary":
        return boundary(y, n_classes, rate, rng, clean_scores)
    raise ValueError(kind)
