import argparse
from benchmark.classical import run_classical_grid

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark linear‑regression approaches"
    )
    p.add_argument("--mode", choices=["classical"], default="classical")
    p.add_argument("--dims", type=int, nargs="+", default=[10, 100, 500])
    p.add_argument("--models", nargs="+",
                   default=["ols", "ridge", "lasso", "sgd"])
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--corr", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="results/classical_bench.csv")
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
    else:
        raise NotImplementedError(f"mode {args.mode}")


if __name__ == "__main__":
    main()

