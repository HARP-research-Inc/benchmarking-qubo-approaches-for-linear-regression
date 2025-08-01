# models/potok.py  ── drop‑in module, no typing / __future__

import time
import numpy as np
from amplify import VariableGenerator, Model, FixstarsClient, solve, set_seed
from datetime import timedelta
import os
from dotenv import load_dotenv  
from models.common_amplify import safe_solve

load_dotenv()


# ----------------------------------------------------------------------
#  helpers
# ----------------------------------------------------------------------
def _build_primitives(d, P_arr):
    """Return  (bin_vars  (d×K),   w_syms length‑d)"""
    K = len(P_arr)
    gen = VariableGenerator()

    # One binary matrix of shape (d,K)
    bins = gen.array("Binary", (d, K))          # shape tuple, no extra name

    # Symbolic real weights   w_j = Σ_k  P_k · bin_{j,k}
    w_syms = []
    for j in range(d):
        wj = 0
        for k, p_k in enumerate(P_arr):
            wj += p_k * bins[j][k]
        w_syms.append(wj)

    return bins, np.array(w_syms, dtype=object)


# ----------------------------------------------------------------------
#  public solver
# ----------------------------------------------------------------------
def solve_linreg_potok_amplify(
    X, y,
    P=(0.25, 0.5, 0.75, 1.0),
    num_solves=1,
    timeout_ms=1000,
    seed=0,
):
    """
    Date‑&‑Potok (2021) QUBO formulation solved on Fixstars Amplify.
    Fits y ≈ X w,  w encoded via precision vector P.

    Returns a dict with timings & ‖w_est – w_exact‖.
    """
    N, d_plus1 = X.shape                         # bias already in X
    P_arr = np.array(P, dtype=float)
    bins, w_syms = _build_primitives(d_plus1, P_arr)

    # ------------ Build QUBO once (symbolic) ---------------------------
    XtX = X.T @ X
    Xty = X.T @ y

    obj = 0
    for i in range(d_plus1):
        for j in range(d_plus1):
            obj += 0.5 * XtX[i, j] * w_syms[i] * w_syms[j]
    for i in range(d_plus1):
        obj += -Xty[i] * w_syms[i]

    model = Model(obj)                           # already quadratic

    # ------------ solve -------------------------------------------------
    client = FixstarsClient()
    key = os.getenv("AE_KEY")
    if not key:
        raise RuntimeError("AE_KEY not set")
    client.token = key
    client.parameters.timeout = timedelta(milliseconds=timeout_ms)
    set_seed(seed)

    t0_enc = time.perf_counter()
    # (nothing to encode per‑iteration; compiled once)
    encode_time = time.perf_counter() - t0_enc

    t_start = time.perf_counter()
    result = safe_solve(model, client, num_solves=num_solves)
    t_end   = time.perf_counter()
    if not result:
        raise RuntimeError("Amplify returned no solutions")

    anneal_time = result.execution_time.total_seconds()
    wall_time   = (t_end - t_start)
    network_time = wall_time - anneal_time

    sol = result.best
    w_est = np.array([wj.evaluate(sol.values) for wj in w_syms], dtype=float)
    w_exact = np.linalg.lstsq(X, y, rcond=None)[0]
    err = np.linalg.norm(w_est - w_exact)

    return {
        "iterations"   : 1,
        "encode_time"  : encode_time,
        "anneal_time"  : anneal_time,
        "total_time"   : encode_time + anneal_time,
        "wall_time"    : wall_time + encode_time,
        "network_time" : network_time,
        "error"        : err,
    }


# ----------------------------------------------------------------------
# smoke‑test ------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 40
    X_raw = rng.uniform(-1, 1, size=(n, 1))
    X = np.hstack([np.ones((n, 1)), X_raw])       # prepend bias
    true_w = np.array([0.5, 0.75])
    y = X @ true_w + 0.05 * rng.standard_normal(n)

    res = solve_linreg_potok_amplify(X, y)
    print(res)

