"""Random-convolutional-kernel feature transforms with on-disk caching."""
from __future__ import annotations

import os
import numpy as np

from src.data import ucr

CACHE = os.path.join(ucr.ROOT, "data", "cache")


def _transformer(kind: str, seed: int, n_jobs: int):
    if kind == "minirocket":
        from aeon.transformations.collection.convolution_based import MiniRocket
        return MiniRocket(n_kernels=10000, random_state=seed, n_jobs=n_jobs)
    if kind == "multirocket":
        from aeon.transformations.collection.convolution_based import MultiRocket
        return MultiRocket(n_kernels=6250, random_state=seed, n_jobs=n_jobs)
    if kind == "rocket":
        from aeon.transformations.collection.convolution_based import Rocket
        return Rocket(n_kernels=10000, random_state=seed, n_jobs=n_jobs)
    if kind == "hydra":
        from aeon.transformations.collection.convolution_based import HydraTransformer
        return HydraTransformer(random_state=seed, n_jobs=n_jobs)
    raise ValueError(kind)


def get_features(dataset: str, seed: int, kind: str = "minirocket", n_jobs: int = 4, cache: bool = True):
    """Return dict(Ftr, ytr, Fte, yte, Xtr, Xte) for a (dataset, resample-seed, transform)."""
    path = os.path.join(CACHE, kind, f"{dataset}_{seed}.npz")
    Xtr, ytr, Xte, yte = ucr.resample(*ucr.load(dataset), seed)
    if cache and os.path.exists(path):
        z = np.load(path)
        Ftr, Fte = z["Ftr"], z["Fte"]
    else:
        Xa, Xb = Xtr, Xte
        min_len = 9 if kind != "hydra" else 15  # shortest length the transforms accept
        if Xa.shape[-1] < min_len:  # pad very short series by edge replication
            pad = ((0, 0), (0, 0), (0, min_len - Xa.shape[-1]))
            Xa, Xb = np.pad(Xa, pad, mode="edge"), np.pad(Xb, pad, mode="edge")
        t = _transformer(kind, seed, n_jobs)
        Ftr = np.asarray(t.fit_transform(Xa), dtype=np.float32)
        Fte = np.asarray(t.transform(Xb), dtype=np.float32)
        if kind == "hydra":  # Hydra's own sparse scaling (Dempster et al., 2023)
            Ftr, Fte = np.sqrt(np.clip(Ftr, 0, None)), np.sqrt(np.clip(Fte, 0, None))
        if cache:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, Ftr=Ftr, Fte=Fte)
    return dict(Ftr=Ftr, ytr=ytr, Fte=Fte, yte=yte, Xtr=Xtr, Xte=Xte)
