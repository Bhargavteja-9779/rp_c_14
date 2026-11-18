"""PhysioNet/CinC Challenge 2017 single-lead ECG with original (v0) and expert-revised (v3) labels.

Preprocessing (fixed in supplementary/protocol_cinc2017.md): z-normalise, decimate 300 -> 150 Hz, centre-crop or
tile to 30 s (4,500 samples).  Cached as data/cache/cinc2017.npz."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from src.data import ucr

RAW = os.path.join(ucr.ROOT, "data", "raw", "cinc2017")
CACHE = os.path.join(ucr.ROOT, "data", "cache", "cinc2017.npz")
CLASSES = ["N", "A", "O", "~"]
URL = "https://physionet.org/files/challenge-2017/1.0.0/"
LENGTH = 4500


def _download():
    import urllib.request, zipfile
    os.makedirs(RAW, exist_ok=True)
    for v in range(4):
        f = os.path.join(RAW, f"REFERENCE-v{v}.csv")
        if not os.path.exists(f):
            urllib.request.urlretrieve(URL + f"REFERENCE-v{v}.csv", f)
    z = os.path.join(RAW, "training2017.zip")
    if not os.path.exists(os.path.join(RAW, "extracted", "training2017")):
        if not os.path.exists(z):
            urllib.request.urlretrieve(URL + "training2017.zip", z)
        with zipfile.ZipFile(z) as zz:
            zz.extractall(os.path.join(RAW, "extracted"))


def _fix_length(x):
    if len(x) >= LENGTH:
        s = (len(x) - LENGTH) // 2
        return x[s:s + LENGTH]
    return np.tile(x, int(np.ceil(LENGTH / len(x))))[:LENGTH]


def load():
    """Return X (n, 1, 4500) float32, y_v0, y_v3 (int, CLASSES order), record names, original lengths (s)."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=False)
        return z["X"], z["y0"], z["y3"], z["rec"], z["dur"]
    from scipy.io import loadmat
    from scipy.signal import decimate
    _download()
    lab = {v: pd.read_csv(os.path.join(RAW, f"REFERENCE-v{v}.csv"), header=None, names=["rec", "y"]).set_index("rec").y
           for v in (0, 3)}
    recs = list(lab[0].index)
    X = np.zeros((len(recs), 1, LENGTH), np.float32)
    dur = np.zeros(len(recs))
    for i, r in enumerate(recs):
        x = loadmat(os.path.join(RAW, "extracted", "training2017", r + ".mat"))["val"].ravel().astype(np.float64)
        dur[i] = len(x) / 300.0
        x = (x - x.mean()) / (x.std() + 1e-8)
        x = decimate(x, 2, zero_phase=True)
        X[i, 0] = _fix_length(x)
    enc = {c: k for k, c in enumerate(CLASSES)}
    y0 = np.array([enc[lab[0][r]] for r in recs])
    y3 = np.array([enc[lab[3][r]] for r in recs])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, X=X, y0=y0, y3=y3, rec=np.array(recs), dur=dur)
    return X, y0, y3, np.array(recs), dur
