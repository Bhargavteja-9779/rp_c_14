"""Experiment runner: one CSV per (transform, dataset) with one row per
(seed, noise model, noise rate, method).  Resumable and deterministic."""
from __future__ import annotations

import os
import time
import zlib
import warnings

import numpy as np
import pandas as pd
import yaml

from src.data import ucr
from src.evaluation import metrics
from src.models import cleaning as cl
from src.models import noise as noise_mod
from src.models.ridge_loo import GramRidge, one_hot_pm
from src.preprocessing.features import get_features

warnings.filterwarnings("ignore")


def load_config(path=None):
    path = path or os.path.join(ucr.ROOT, "config", "default.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def all_datasets():
    from aeon.datasets.tsc_datasets import univariate_equal_length
    return sorted(str(d) for d in univariate_equal_length)


def dev_datasets(cfg):
    rng = np.random.RandomState(cfg["dev_seed"])
    return sorted(str(d) for d in rng.choice(all_datasets(), cfg["n_dev"], replace=False))


def noise_rng(dataset, seed, kind, rate):
    key = f"{dataset}|{seed}|{kind}|{rate:.3f}".encode()
    return np.random.RandomState(zlib.crc32(key) % (2 ** 31))


def run_dataset(dataset, cfg, transform=None, methods=None, seeds=None, out_dir=None,
                noise=None, n_jobs=1, force=False):
    transform = transform or cfg["transform"]
    methods = methods or cfg["methods"]
    seeds = cfg["seeds"] if seeds is None else seeds
    noise = noise or cfg["noise"]
    out_dir = out_dir or os.path.join(ucr.ROOT, "results", "raw", transform)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{dataset}.csv")
    if os.path.exists(out) and not force:
        return out
    grids = {"std": np.logspace(*cfg["alphas_std"]), "wide": np.logspace(*cfg["alphas_wide"])}
    rows = []
    for seed in seeds:
        t0 = time.time()
        f = get_features(dataset, seed, transform, n_jobs=n_jobs)
        t_feat = time.time() - t0
        ytr, yte = f["ytr"], f["yte"]
        C = int(max(ytr.max(), yte.max()) + 1)
        t0 = time.time()
        g = GramRidge(grids["wide"]).fit(f["Ftr"])
        t_eig = time.time() - t0
        K_te = g.test_kernel(f["Fte"])
        gstd = GramRidge(grids["std"])
        for k in ("n_", "mu_", "sd_", "Z_", "s_", "U_", "U2_"):  # share the decomposition
            setattr(gstd, k, getattr(g, k))
        gm = {"std": gstd, "wide": g}
        Yc = one_hot_pm(ytr, C)
        a_c, loo_clean_scores, _ = g.select_alpha(Yc)
        for kind, rates in noise.items():
            for rate in rates:
                yn = noise_mod.apply_noise(kind, ytr, C, rate, noise_rng(dataset, seed, kind, rate),
                                           clean_scores=loo_clean_scores)
                if len(np.unique(yn)) < len(np.unique(ytr)):
                    continue  # a class vanished (tiny datasets): condition undefined
                is_noisy = yn != ytr
                for name, (fn, grid, kw) in methods.items():
                    gg = gm[grid]
                    t0 = time.time()
                    try:
                        res = cl.METHODS[fn](gg, yn, C, seed=seed, **(kw or {}))
                        F, alpha = cl.final_predict(gg, f["Fte"], K_te, res, C)
                        err = ""
                    except Exception as e:  # recorded, never silently dropped
                        res, F, alpha, err = None, None, np.nan, repr(e)[:200]
                    dt = time.time() - t0
                    row = dict(dataset=dataset, transform=transform, seed=seed, noise=kind, rate=rate,
                               method=name, n_train=len(ytr), n_test=len(yte), n_classes=C,
                               true_noise=float(is_noisy.mean()), alpha=alpha, time_s=dt,
                               t_feat=t_feat, t_eig=t_eig, error=err)
                    if res is not None:
                        row.update(metrics.classification(yte, F.argmax(1)))
                        row.update(metrics.detection(is_noisy, res["flagged"], res["score"],
                                                     res["y_clean"], ytr))
                        info = res.get("info", {})
                        row.update(est_noise=info.get("noise_rate", np.nan),
                                   rounds=info.get("rounds", np.nan),
                                   accepted=info.get("accepted", np.nan), model=info.get("model", ""))
                    rows.append(row)
        del g, gstd, K_te, f
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
