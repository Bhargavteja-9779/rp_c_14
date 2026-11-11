"""Classification and label-error-detection metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def classification(y_true, y_pred):
    return dict(acc=float(np.mean(y_true == y_pred)),
                bal_acc=float(balanced_accuracy_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)))


def detection(is_noisy, flagged, score, y_clean_after, y_true):
    """Label-error detection quality; NaN where undefined (no noisy / all noisy)."""
    out = dict(det_auroc=np.nan, det_prec=np.nan, det_rec=np.nan, det_f1=np.nan,
               label_acc=float(np.mean(y_clean_after == y_true)), n_flagged=int(flagged.sum()))
    if 0 < is_noisy.sum() < len(is_noisy):
        if np.ptp(score) > 0:
            out["det_auroc"] = float(roc_auc_score(is_noisy, score))
        tp = float((flagged & is_noisy).sum())
        out["det_prec"] = tp / flagged.sum() if flagged.sum() else np.nan
        out["det_rec"] = tp / is_noisy.sum()
        p, r = out["det_prec"], out["det_rec"]
        out["det_f1"] = 2 * p * r / (p + r) if (p == p and p + r > 0) else 0.0
    return out
