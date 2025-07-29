# Benchmarking QUBO Approaches for Linear Regression  
> Classical baselines • Box‑constrained QUBO (naïve & optimised) on Fixstars Amplify • “Potok” precision‑vector QUBO

---

## 1  | Overview  

This project compares several ways to fit a **small‑dimensional linear‑regression** model:

| Tag        | Idea                                                         | Backend                |
|------------|-------------------------------------------------------------|------------------------|
| `classical`| OLS / Ridge / Lasso / SGD                                   | scikit‑learn           |
| `box-naive`| Shrinking‑box QUBO, rebuild full model every iteration      | Fixstars Amplify       |
| `box-opt`  | Same algorithm but pre‑build quadratic terms → fast encode  | Fixstars Amplify       |
| `potok`    | Date & Potok (2021) precision‑vector QUBO                   | Fixstars Amplify       |

Each run records **encode, anneal, wall time, iterations, error** in a CSV so approaches can be compared side‑by‑side.

---

## 2  | Install & setup  

```bash
git clone https://github.com/colfarl/benchmarking-qubo-approaches-for-linear-regression.git
cd benchmarking-qubo-approaches-for-linear-regression

python -m venv .venv
source .venv/bin/activate
pip install -e .

# Fixstars Amplify (free community tier)
export AE_KEY="YOUR_FIXSTARS_API_TOKEN"
(or create .env file)

## Example Runs
```
python main.py --mode classical \
               --dims 8 16 32        \
               --out results/classical.csv

python main.py --mode box-opt        \
               --dims 8 16 32        \
               --max_iter 30         \
               --timeout_ms 500      \
               --out results/box_opt.csv

python main.py --mode potok          \
               --dims 8 16           \
               --precisions 2 3 4    \
               --out results/potok.csv
```

## References
P. Date & T. Potok, Adiabatic Quantum Linear Regression, Sci. Rep. 11, 21905 (2021).
Fixstars Amplify

