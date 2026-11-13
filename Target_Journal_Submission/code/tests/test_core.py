"""Unit tests: closed-form LOO correctness, sklearn equivalence, noise models, cleaning invariants."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from src.models.ridge_loo import GramRidge, one_hot_pm
from src.models import cleaning as cl, noise


def make(n=60, p=300, C=3, seed=0, sep=1.5):
    rng = np.random.RandomState(seed)
    y = np.arange(n) % C
    centers = rng.randn(C, p) * sep / np.sqrt(p) * 6
    X = centers[y] + rng.randn(n, p)
    Xt = centers[y] + rng.randn(n, p)
    return X, y, Xt, y.copy()


def test_loo_matches_brute_force():
    X, y, _, _ = make(n=30, p=80)
    g = GramRidge().fit(X)
    Y = one_hot_pm(y, 3)
    for a in (0.1, 10.0, 1e3):
        loo, _, _ = g.loo(Y, a)
        for i in (0, 7, 29):
            m = np.ones(30, bool); m[i] = False
            Z = g.Z_  # scaling is fixed; only the intercept and weights are refitted
            Zc = Z[m] - Z[m].mean(0)
            Yc = Y[m] - Y[m].mean(0)
            W = np.linalg.solve(Zc.T @ Zc + a * np.eye(Z.shape[1]), Zc.T @ Yc)
            pred = (Z[i] - Z[m].mean(0)) @ W + Y[m].mean(0)
            assert np.allclose(pred, loo[i], atol=1e-6)


def test_matches_sklearn_ridgeclassifiercv():
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.preprocessing import StandardScaler
    X, y, Xt, yt = make(n=80, p=400, sep=0.6)
    g = GramRidge().fit(X)
    Y = one_hot_pm(y, 3)
    a, _, _ = g.select_alpha(Y)
    pred = g.predict(g.test_kernel(Xt), Y, a).argmax(1)
    sc = StandardScaler().fit(X)
    sk = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(sc.transform(X), y)
    assert np.isclose(sk.alpha_, a)
    assert np.mean(sk.predict(sc.transform(Xt)) == pred) == 1.0


def test_relabelling_does_not_change_hat_matrix():
    X, y, _, _ = make()
    g = GramRidge().fit(X)
    h1 = g.hat_diag(5.0)
    y2 = y.copy(); y2[:10] = (y2[:10] + 1) % 3
    g.loo(one_hot_pm(y2, 3), 5.0)
    assert np.allclose(h1, g.hat_diag(5.0))


@pytest.mark.parametrize("kind,rate", [("sym", 0.2), ("asym", 0.3)])
def test_noise_rate_exact(kind, rate):
    y = np.arange(300) % 3
    yn = noise.apply_noise(kind, y, 3, rate, np.random.RandomState(0))
    assert abs(np.mean(yn != y) - rate) < 1e-9
    if kind == "asym":
        assert np.all(yn[yn != y] == (y[yn != y] + 1) % 3)


def test_boundary_noise_targets_low_margin():
    X, y, _, _ = make(sep=0.5)
    g = GramRidge().fit(X)
    _, loo, _ = g.select_alpha(one_hot_pm(y, 3))
    yn = noise.apply_noise("boundary", y, 3, 0.2, np.random.RandomState(0), clean_scores=loo)
    m = cl.margins(loo, y)[0]
    assert np.mean(yn != y) == pytest.approx(0.2)
    assert m[yn != y].max() <= m[yn == y].min() + 1e-6


def test_loo_em_recovers_asymmetric_noise():
    X, y, Xt, yt = make(n=300, p=400, sep=1.2)
    g = GramRidge(np.logspace(-3, 9, 25)).fit(X)
    yn = noise.apply_noise("asym", y, 3, 0.3, np.random.RandomState(1))
    res = cl.loo_em(g, yn, 3, gate=1.0)
    assert np.mean(res["y_clean"] == y) > np.mean(yn == y) + 0.15
    K = g.test_kernel(Xt)
    acc_em = np.mean(cl.final_predict(g, Xt, K, res, 3)[0].argmax(1) == yt)
    acc_plain = np.mean(cl.final_predict(g, Xt, K, cl.none(g, yn, 3), 3)[0].argmax(1) == yt)
    assert acc_em >= acc_plain


def test_methods_never_empty_a_class_and_shapes():
    X, y, Xt, _ = make(n=45, p=100, sep=0.3)
    g = GramRidge().fit(X)
    yn = noise.apply_noise("sym", y, 3, 0.4, np.random.RandomState(2))
    K = g.test_kernel(Xt)
    for name, fn in cl.METHODS.items():
        res = fn(g, yn, 3, seed=0)
        kept = res["y_clean"][res["keep"]]
        assert set(np.unique(kept)) == set(np.unique(yn)), name
        F, a = cl.final_predict(g, Xt, K, res, 3)
        assert F.shape == (45, 3) and np.isfinite(F).all(), name


def test_symmetric_noise_expected_solution_is_scaled():
    """Proposition 1: E[noisy targets] is an affine map of the clean targets, so the
    expected ridge decision values keep their arg-max under symmetric noise."""
    X, y, Xt, _ = make(n=40, p=60)
    g = GramRidge().fit(X)
    C, rho = 3, 0.3
    Y = one_hot_pm(y, C)
    EY = (1 - rho * C / (C - 1)) * Y + (rho * C / (C - 1)) * (2.0 / C - 1)
    K = g.test_kernel(Xt)
    f, fe = g.predict(K, Y, 1.0), g.predict(K, EY, 1.0)
    assert np.allclose(fe, (1 - rho * C / (C - 1)) * f + (rho * C / (C - 1)) * (2.0 / C - 1), atol=1e-8)
