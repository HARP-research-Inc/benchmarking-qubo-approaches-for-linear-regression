from data.data_generator import (
    generate_synthetic_regression,
)
from models.box_naive import (
    solve_box_naive_amplify,
)
from .result_logger import ResultLogger


def run_box_amplify_grid(
    dims,
    noise,
    corr,
    seed,
    max_iter,
    num_solves,
    timeout_ms,
    outfile,
):
    """
    Adds rows: mode='box-naive', d, n, iterations, encode_time, anneal_time,
               total_time (no network), wall_time (incl. network), error
    """
    logger = ResultLogger(outfile)

    for d in dims:
        n = 10 * d
        data = generate_synthetic_regression(
            n=n,
            d=d,
            noise_sigma=noise,
            feature_corr=corr if corr > 0 else None,
            seed=seed,
        )

        res = solve_box_naive_amplify(
            A=data.X_train.T @ data.X_train,
            b=data.X_train.T @ data.y_train,
            max_iter=max_iter,
            num_solves=num_solves,
            timeout_ms=timeout_ms,
            seed=seed,
        )

        logger.add(
            mode="box-naive",
            d=d,
            n=n,
            iterations=res["iterations"],
            encode_time=round(res["encode_time"], 4),
            anneal_time=round(res["anneal_time"], 4),
            total_time=round(res["total_time"], 4),   # network‑free
            wall_time=round(res["wall_time"], 4),     # includes network
            error=f"{res['error']:.2e}",
        )

        print(
            f"box-naive d={d:3}  iters={res['iterations']:3}  "
            f"total={res['total_time']:.2f}s  wall={res['wall_time']:.2f}s  "
            f"err={res['error']:.2e}"
        )

    logger.flush()
    return outfile

