# benchmark/potok.py
from itertools import product

from data.data_generator   import generate_synthetic_regression
from models.potok          import solve_linreg_potok_amplify
from benchmark.result_logger import ResultLogger


def _default_p_vector(K):
    """
    Build the canonical precision vector  (¼, ½, …, K·¼) used in the
    Date–Potok paper, e.g.  K=4  ->  (0.25, 0.5, 0.75, 1.0).
    Change this helper if you prefer a different rule.
    """
    step = 1.0 / 4.0
    return tuple(step * (i + 1) for i in range(K))


def run_potok_grid(
    dims,
    noise,
    corr,
    seed,
    precision_bits,          # ← list[int]  e.g. [2, 3, 4]
    num_solves,
    timeout_ms,
    outfile,
):
    """
    For every d in `dims` and every K in `precision_bits`[48;55;214;1870;2996t
    run the Potok QUBO solver with a K‑binary precision vector.
    """
    logger = ResultLogger(outfile)

    for d, K in product(dims, precision_bits):
        n = 10 * d
        data = generate_synthetic_regression(
            n=n,
            d=d,
            noise_sigma=noise,
            feature_corr=corr if corr > 0 else None,
            seed=seed,
        )

        P_vec = _default_p_vector(K)

        res = solve_linreg_potok_amplify(
            data.X_train,               # already has bias column
            data.y_train,
            P=P_vec,
            num_solves=num_solves,
            timeout_ms=timeout_ms,
            seed=seed,
        )

        logger.add(
            mode="potok",
            d=d,
            n=n,
            K=K,                        # ← new csv column
            iterations=res["iterations"],
            encode_time=round(res["encode_time"], 4),
            anneal_time=round(res["anneal_time"], 4),
            total_time=round(res["total_time"], 4),
            wall_time=round(res["wall_time"], 4),
            error=f"{res['error']:.2e}",
        )

        print(
            f"potok  d={d:3}  K={K}  "
            f"iters={res['iterations']:2}  total={res['total_time']:.2f}s  "
            f"err={res['error']:.2e}"
        )

    logger.flush()
    return outfile

