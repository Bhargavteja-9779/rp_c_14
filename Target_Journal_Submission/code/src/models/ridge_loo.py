"""Dual (Gram) ridge classifier with closed-form leave-one-out (LOO) predictions.

The classifier reproduces scikit-learn's ``RidgeClassifierCV`` (one-vs-rest +/-1
targets, unpenalised intercept, LOO selection of the ridge penalty) but exposes
the LOO decision values themselves and makes every label-dependent quantity a
cheap function of a single eigendecomposition of the centred Gram matrix.

Key identity.  With centred features Z (n x p), K = Z Z^T = U diag(s) U^T and
ridge penalty a, the hat matrix of the ridge fit with unpenalised intercept is

    H(a) = U diag(s / (s + a)) U^T + (1/n) 1 1^T ,

and the LOO prediction for sample i is

    f_loo_i = (H y - h_ii y_i) / (1 - h_ii).

H(a) does not depend on the labels, so re-computing all LOO predictions after
the labels are edited costs one n x n by n x C matrix product.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

DEFAULT_ALPHAS = np.logspace(-3, 3, 10)


def one_hot_pm(y: np.ndarray, n_classes: int) -> np.ndarray:
    """+1 / -1 one-vs-rest target matrix (n x C)."""
    Y = -np.ones((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


class GramRidge:
    """Label-independent part of a ridge fit: scaling, Gram eigendecomposition."""

    def __init__(self, alphas=DEFAULT_ALPHAS, dtype=np.float64):
        self.alphas = np.asarray(alphas, dtype=float)
        self.dtype = dtype

    # ------------------------------------------------------------------ fit
    def fit(self, Phi: np.ndarray) -> "GramRidge":
        Phi = np.asarray(Phi, dtype=self.dtype)
        self.n_ = Phi.shape[0]
        self.mu_ = Phi.mean(0)
        sd = Phi.std(0)
        sd[sd < 1e-8] = 1.0
        self.sd_ = sd
        Z = (Phi - self.mu_) / self.sd_
        self.Z_ = Z
        K = Z @ Z.T
        s, U = linalg.eigh(K, overwrite_a=True, check_finite=False)
        s = np.clip(s, 0.0, None)
        self.s_, self.U_ = s, U
        self.U2_ = U * U
        return self

    def test_kernel(self, Phi_te: np.ndarray) -> np.ndarray:
        Zt = (np.asarray(Phi_te, dtype=self.dtype) - self.mu_) / self.sd_
        return Zt @ self.Z_.T

    # ------------------------------------------------------- label-dependent
    def hat_diag(self, alpha: float) -> np.ndarray:
        cache = self.__dict__.setdefault("_hcache", {})  # label-independent, so cached per penalty
        key = float(alpha)
        if key not in cache:
            cache[key] = self.U2_ @ (self.s_ / (self.s_ + alpha)) + 1.0 / self.n_
        return cache[key]

    def loo(self, Y: np.ndarray, alpha: float, UtY: np.ndarray | None = None):
        """Return (LOO decision values, in-sample fitted values, hat diagonal)."""
        if UtY is None:
            UtY = self.U_.T @ Y
        d = self.s_ / (self.s_ + alpha)
        fit = self.U_ @ (d[:, None] * UtY) + Y.mean(0, keepdims=True)
        h = self.hat_diag(alpha)
        loo = (fit - h[:, None] * Y) / (1.0 - h[:, None])
        return loo, fit, h

    def _score(self, Y, UtY, a, criterion):
        loo, _, _ = self.loo(Y, a, UtY)
        if criterion == "mse":
            return float(np.mean((loo - Y) ** 2)), loo
        return float(np.mean(loo.argmax(1) != Y.argmax(1))), loo

    def select_alpha(self, Y: np.ndarray, criterion: str = "mse", around: float | None = None, width: int = 2):
        """LOO model selection over the alpha grid (as RidgeClassifierCV).

        ``around``: warm start.  Only grid points within ``width`` steps of the previous
        optimum are scored; the window is extended while the best point lies on its edge.
        This equals the exhaustive search whenever the criterion is unimodal on the grid."""
        UtY = self.U_.T @ Y
        if around is None:
            best = (np.inf, None, None)
            for a in self.alphas:
                score, loo = self._score(Y, UtY, a, criterion)
                if score < best[0] - 1e-15:
                    best = (score, a, loo)
            return best[1], best[2], best[0]
        m = len(self.alphas)
        c = int(np.argmin(np.abs(np.log(self.alphas) - np.log(around))))
        lo, hi = max(0, c - width), min(m - 1, c + width)
        done = {}
        while True:
            for k in range(lo, hi + 1):
                if k not in done:
                    done[k] = self._score(Y, UtY, self.alphas[k], criterion)
            kb = min(done, key=lambda k: (done[k][0], k))
            if kb == lo and lo > 0:
                lo -= 1
            elif kb == hi and hi < m - 1:
                hi += 1
            else:
                return self.alphas[kb], done[kb][1], done[kb][0]

    def dual_coef(self, Y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        ybar = Y.mean(0, keepdims=True)
        A = self.U_ @ ((self.U_.T @ (Y - ybar)) / (self.s_ + alpha)[:, None])
        return A, ybar

    def predict(self, K_te: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
        A, ybar = self.dual_coef(Y, alpha)
        return K_te @ A + ybar

    def subset(self, idx: np.ndarray) -> "GramRidge":
        """New GramRidge restricted to training rows ``idx`` (re-centred, refit)."""
        g = GramRidge(self.alphas, self.dtype)
        Zraw = self.Z_[idx] * self.sd_ + self.mu_
        g.fit(Zraw)
        return g
