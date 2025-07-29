import argparse
from benchmark.classical import run_classical_grid
from benchmark.box_naive import run_box_amplify_grid
from benchmark.box_opt   import run_box_opt_grid
from benchmark.potok     import run_potok_grid


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark linear‑regression approaches")
    p.add_argument("--dims",  type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--models", nargs="+", default=["ols", "ridge", "lasso", "sgd"])
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--corr",  type=float, default=0.0)
    p.add_argument("--seed",  type=int,   default=1)
    p.add_argument("--out",   default="results/bench.csv")
    p.add_argument(
        "--mode",
        choices=["classical", "box-naive", "box-opt", "potok"],
        default="classical",
    )
    # new: list of K values
    p.add_argument("--prec_bits", type=int, nargs="+", default=[4])
    p.add_argument("--max_iter",   type=int, default=40)
    p.add_argument("--num_solves", type=int, default=1)
    p.add_argument("--timeout_ms", type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "classical":
        run_classical_grid(
            dims=args.dims,
            noise=args.noise,
            corr=args.corr,
            seed=args.seed,
            models=args.models,
            outfile=args.out,
        )

    elif args.mode == "box-naive":
        run_box_amplify_grid(
            dims=args.dims,
            noise=args.noise,
            corr=args.corr,
            seed=args.seed,
            max_iter=args.max_iter,
            num_solves=args.num_solves,
            timeout_ms=args.timeout_ms,
            outfile=args.out,
        )

    elif args.mode == "box-opt":
        run_box_opt_grid(
            dims=args.dims,
            noise=args.noise,
            corr=args.corr,
            seed=args.seed,
            max_iter=args.max_iter,
            num_solves=args.num_solves,
            timeout_ms=args.timeout_ms,
            outfile=args.out,
        )

    elif args.mode == "potok":
        run_potok_grid(
            dims=args.dims,
            noise=args.noise,
            corr=args.corr,
            seed=args.seed,
            precision_bits=args.prec_bits,     # ← pass list
            num_solves=args.num_solves,
            timeout_ms=args.timeout_ms,
            outfile=args.out,
        )
    else:
        raise NotImplementedError(args.mode)


if __name__ == "__main__":
    main()

