# box_opt_sparse.py

import time
import numpy as np
from amplify import VariableGenerator, Model, FixstarsClient, set_seed
from datetime import timedelta
import os

from .sparse_box import cache_upper_triangle_coo
from models.common_amplify import safe_solve  # your helper
from dotenv import load_dotenv

load_dotenv()

def _build_amplify_primitives_sparse(A_csr):
    """
    One-time construction of Amplify variables and sparse quadratic template.
    """
    d = A_csr.shape[0]
    gen = VariableGenerator()
    q1 = gen.array("Binary", d)
    q2 = gen.array("Binary", d)
    dvec = np.array([-2 * q1[i] + q2[i] for i in range(d)], dtype=object)

    I, J, V = cache_upper_triangle_coo(A_csr)

    quad_poly = 0
    # sum_{i<=j} A_ij * d_i * d_j with 2x for i<j; 0.5 factor applied after.
    for i, j, v in zip(I, J, V):
        term = v * dvec[i] * dvec[j]
        quad_poly += term if i == j else 2 * term

    quad_blk = 0.5 * quad_poly
    return q1, q2, dvec, quad_blk


def solve_box_opt_amplify_sparse(
    A_csr,  # scipy.sparse.csr_matrix
    b,      # dense (d,)
    beta=0.2,
    epsilon=1e-6,
    max_iter=50,
    num_solves=1,
    timeout_ms=1000,
    seed=0,
    ae_key_env="AE_KEY",
):
    d = len(b)
    q1, q2, dvec, quad_blk = _build_amplify_primitives_sparse(A_csr)

    c = np.zeros(d)
    L = 1.0
    best_E = np.inf

    encode_time = 0.0
    anneal_time = 0.0
    wall_time = 0.0

    client = FixstarsClient()
    key = os.getenv("AE_KEY")
    if not key:
        raise RuntimeError(f"{ae_key_env} not found in environment")
    client.token = key
    client.parameters.timeout = timedelta(milliseconds=timeout_ms)
    set_seed(seed)

    # Precompute norms for a decent tolerance baseline
    A_inf = np.linalg.norm(A_csr.toarray(), ord=np.inf) if d <= 1024 else None

    for it in range(1, max_iter + 1):
        # Sparse matvec (dominant per-iter term in opt): O(nnz)
        coeff = A_csr @ c - b

        t0 = time.perf_counter()
        # mask small coefficients to avoid needless Poly ops
        if A_inf is None:
            tol = 1e-12 * (np.linalg.norm(c, ord=np.inf) + np.linalg.norm(b, ord=np.inf))
        else:
            tol = 1e-12 * (A_inf * np.linalg.norm(c, ord=np.inf) + np.linalg.norm(b, ord=np.inf))

        nz_idx = np.flatnonzero(np.abs(coeff) > tol)

        # Build linear block: sum_i coeff[i] * dvec[i]
        lin_blk = 0
        for i in nz_idx:
            lin_blk += coeff[i] * dvec[i]

        poly = L * lin_blk + (L * L) * quad_blk
        model = Model(poly)
        encode_time += time.perf_counter() - t0

        # Solve
        w_start = time.perf_counter()
        result = safe_solve(model, client, num_solves=num_solves)
        w_end = time.perf_counter()
        wall_time += (w_end - w_start)

        if not result:
            raise RuntimeError("Amplify returned no solutions")
        anneal_time += result.execution_time.total_seconds()
        sol = result.best

        # Decode
        q1_sol = q1.evaluate(sol.values)
        q2_sol = q2.evaluate(sol.values)
        w_new = c + L * (-2 * q1_sol + q2_sol)

        # True energy (for accept/contract)
        E_true = 0.5 * w_new @ (A_csr @ w_new) - b @ w_new

        if E_true < best_E:
            c = w_new
            best_E = E_true
        else:
            L *= beta

        if L < epsilon:
            break

    exact = np.linalg.solve(A_csr.toarray(), b)  # for error reporting only
    err = np.linalg.norm(c - exact)
    network_time = wall_time - anneal_time

    return {
        "iterations": it,
        "encode_time": encode_time,
        "anneal_time": anneal_time,
        "total_time": encode_time + anneal_time,
        "wall_time": wall_time + encode_time,
        "network_time": network_time,
        "error": err,
    }

