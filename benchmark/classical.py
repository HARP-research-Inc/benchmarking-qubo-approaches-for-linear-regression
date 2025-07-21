"""
Pure functions for classical benchmark sweeps.
No CLI here—main.py owns argument‑parsing.
"""

from itertools import product

from data.data_generator import (
    generate_synthetic_regression,
)
from models import classical as M
from .result_logger import ResultLogger


MODEL_FUNCS = {
    "ols":   M.train_ols,
    "ridge": M.train_ridge,
    "lasso": M.train_lasso,
    "sgd":   M.train_sgd,
}


def run_classical_grid(
    dims,
    noise,
    corr,
    seed,
    models,
    outfile,
):
    """
    Runs (model, d) grid.  Writes CSV via ResultLogger, returns path.
    """
    logger = ResultLogger(outfile)

    for d, model_key in product(dims, models):
        n = 10 * d
        data = generate_synthetic_regression(
            n=n, d=d, noise_sigma=noise,
            feature_corr=corr if corr > 0 else None,
            seed=seed,
        )

        train_fn = MODEL_FUNCS[model_key]
        model, train_time = train_fn(data.X_train, data.y_train)
        metrics = M.evaluate(model, data.X_test, data.y_test)

        logger.add(
            model=model_key,
            d=d,
            n=n,
            noise=noise,
            corr=corr,
            seed=seed,
            train_time=round(train_time, 6),
            predict_time=round(metrics["predict_time"], 6),
            r2=round(metrics["r2"], 6),
            mse=round(metrics["mse"], 6),
        )

        print(
            f"{model_key:5}  d={d:4}  "
            f"train={train_time:.4f}s  R2={metrics['r2']:.4f}"
        )

    logger.flush()
    return outfile

