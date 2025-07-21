"""
Classical regression baselines + simple timing helpers.
Only external dependency: scikit‑learn.
"""

import time
import numpy as np
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    SGDRegressor,
)
from sklearn.metrics import r2_score, mean_squared_error


# ------------------------------------------------------------------ #
#   Timing decorator
# ------------------------------------------------------------------ #

def _timed(func):
    def wrapper(*args, **kw):
        t0 = time.perf_counter()
        out = func(*args, **kw)
        return out, time.perf_counter() - t0
    return wrapper


# ------------------------------------------------------------------ #
#   Train helpers
# ------------------------------------------------------------------ #

@_timed
def train_ols(X, y):
    """Ordinary Least Squares (closed‑form)."""
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    return model


@_timed
def train_ridge(X, y, alpha=1.0):
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, y)
    return model


@_timed
def train_lasso(X, y, alpha=0.001, max_iter=10_000):
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter)
    model.fit(X, y)
    return model


@_timed
def train_sgd(X, y, lr=0.01, n_iter=1_000):
    model = SGDRegressor(
        penalty="l2",
        alpha=0.0,
        eta0=lr,
        max_iter=n_iter,
        fit_intercept=False,
    )
    model.fit(X, y)
    return model


# ------------------------------------------------------------------ #
#   Evaluation helper
# ------------------------------------------------------------------ #

def evaluate(model, X_test, y_test):
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time = time.perf_counter() - t0

    return {
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "predict_time": pred_time,
    }

