import math
import numpy as np
from numpy.random import PCG64, default_rng


class SyntheticRegressionData:
    """
    Simple container for one synthetic regression data set.
    Holds train/test splits and the ground‑truth weight vector.
    """

    def __init__(self, X_train, y_train, X_test, y_test, w_true):
        n_train, d = X_train.shape
        n_test, d2 = X_test.shape
        assert d == d2, "train/test feature dims disagree"
        assert y_train.shape == (n_train,)
        assert y_test.shape == (n_test,)
        assert w_true.shape == (d,)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.w_true = w_true


# ------------------------------------------------------------------ #
#   Public
# ------------------------------------------------------------------ #

def generate_synthetic_regression(
    n,
    d,
    noise_sigma=0.01,
    train_ratio=0.8,
    feature_corr=None,
    seed=None,
):
    """
    Return SyntheticRegressionData where  y = X w_true + ε.
    If feature_corr is given (0 <= ρ < 1), columns are equi‑correlated
    with off‑diagonal correlation ρ.
    """
    rng = default_rng(PCG64(seed))

    w_true = rng.standard_normal(d)

    if feature_corr is None:
        X = rng.standard_normal((n, d))
    else:
        X = _build_correlated_features(n, d, feature_corr, rng)

    noise = rng.normal(0.0, noise_sigma, size=n)
    y = X @ w_true + noise

    n_train = math.floor(train_ratio * n)
    indices = rng.permutation(n)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return SyntheticRegressionData(X_train, y_train, X_test, y_test, w_true)


# ------------------------------------------------------------------ #
#   Helpers
# ------------------------------------------------------------------ #
def _build_correlated_features(n, d, rho, rng):
    """
    Equi‑correlated Gaussian design.

    X_ij = sqrt(1‑rho) * Z0_ij + sqrt(rho) * z_common_i
    ⇒  Var[X_ij] = 1,  Cov[X_ik, X_il] = rho  (k ≠ l)
    """
    if not (0.0 <= rho < 1.0):
        raise ValueError("feature_corr must be in [0, 1).")

    Z0 = rng.standard_normal((n, d))          # independent noise
    z_common = rng.standard_normal((n, 1))    # shared across features

    return np.sqrt(1.0 - rho) * Z0 + np.sqrt(rho) * z_common

