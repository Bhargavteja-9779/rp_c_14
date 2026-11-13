"""UCR/UEA archive access (downloaded on demand through aeon) and resampling."""
from __future__ import annotations

import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "raw", "aeon")


def load(name: str):
    """Return X_train, y_train, X_test, y_test (labels encoded 0..C-1, float32 X)."""
    from aeon.datasets import load_classification

    os.makedirs(RAW, exist_ok=True)
    Xtr, ytr = load_classification(name, split="train", extract_path=RAW)
    Xte, yte = load_classification(name, split="test", extract_path=RAW)
    classes = np.unique(np.concatenate([ytr, yte]))
    lut = {c: i for i, c in enumerate(classes)}
    ytr = np.array([lut[c] for c in ytr])
    yte = np.array([lut[c] for c in yte])
    return np.asarray(Xtr, np.float32), ytr, np.asarray(Xte, np.float32), yte


def resample(Xtr, ytr, Xte, yte, seed: int):
    """Stratified resample keeping the original train/test sizes and class counts
    (seed 0 returns the archive's default split), as in the UCR bake-off protocol."""
    if seed == 0:
        return Xtr, ytr, Xte, yte
    X = np.concatenate([Xtr, Xte])
    y = np.concatenate([ytr, yte])
    rng = np.random.RandomState(seed)
    tr_idx = []
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        rng.shuffle(members)
        tr_idx.extend(members[: int((ytr == c).sum())])
    tr = np.zeros(len(y), bool)
    tr[tr_idx] = True
    return X[tr], y[tr], X[~tr], y[~tr]
