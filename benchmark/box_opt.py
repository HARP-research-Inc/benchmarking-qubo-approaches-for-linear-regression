# benchmark/box_opt.py
from data.data_generator import generate_synthetic_regression
from models.box_opt import solve_box_opt_amplify
from benchmark.result_logger import ResultLogger


def run_box_opt_grid(
    dims,
    noise,
    corr,
    seed,
    max_iter,
    num_solves,
    timeout_ms,
    outfile,
):
    logger = ResultLogger(outfile)

    for d in dims:
        n = 10 * d
        data = generate_synthetic_regression(
            n=n, d=d, noise_sigma=noise,
            feature_corr=corr if corr > 0 else None,
            seed=seed,
        )

        res = solve_box_opt_amplify(
            A=data.X_train.T @ data.X_train,
            b=data.X_train.T @ data.y_train,
            max_iter=max_iter,
            num_solves=num_solves,
            timeout_ms=timeout_ms,
            seed=seed,
        )

        logger.add(
            mode="box-opt",
            d=d,
            n=n,
            iterations=res["iterations"],
            encode_time=round(res["encode_time"], 4),
            anneal_time=round(res["anneal_time"], 4),
            total_time=round(res["total_time"], 4),
            wall_time=round(res["wall_time"], 4),
            error=f"{res['error']:.2e}",
        )

        print(
            f"box-opt  d={d:3}  iters={res['iterations']:3}  "
            f"total={res['total_time']:.2f}s  wall={res['wall_time']:.2f}s  "
            f"err={res['error']:.2e}"
        )

    logger.flush()
    return outfile

