import time
import numpy as np
from amplify import VariableGenerator, Model, FixstarsClient, solve, set_seed
from datetime import timedelta
from dotenv import load_dotenv
import os
from models.common_amplify import safe_solve

load_dotenv()


def _build_amplify_primitives(A):
    """
    One-time construction of Amplify variables and the quadratic template.

    Returns:
        q1, q2   : amplify variable arrays (length d)
        dvec     : numpy object array of Poly (length d), each element is (-2*q1[i] + q2[i])
        quad_blk : Poly for 0.5 * d^T A d (independent of c and L, so reused)
    """
    d = A.shape[0]
    gen = VariableGenerator()
    q1 = gen.array("Binary", d)
    q2 = gen.array("Binary", d)

    dvec = np.array([-2 * q1[i] + q2[i] for i in range(d)], dtype=object)

    # Build quadratic block once (O(d^2))
    quad_poly = 0
    for i in range(d):
        di = dvec[i]
        Ai = A[i]
        # j>=i to avoid duplicating terms; Amplify Poly handles duplicates fine though
        for j in range(i, d):
            term = Ai[j] * di * dvec[j]
            quad_poly += term if i == j else 2 * term  # because we're summing over all i,j in 0.5 * ...
    quad_blk = 0.5 * quad_poly
    return q1, q2, dvec, quad_blk


def solve_box_opt_amplify(
    A,
    b,
    beta=0.2,
    epsilon=1e-6,
    max_iter=50,
    num_solves=1,
    timeout_ms=1000,
    seed=0,
):
    """
    Optimized box algorithm using only Amplify (no PyQUBO).
    Encoding cost per iteration is O(d), quadratic part is prebuilt.

    Returns:
        dict(iterations, encode_time, anneal_time, total_time, wall_time,
             network_time, error)
    """
    d = len(b)
    q1, q2, dvec, quad_blk = _build_amplify_primitives(A)

    # State
    c = np.zeros(d)
    L = 1.0
    best_E = np.inf

    encode_time = 0.0   # building lin_block + assembling final poly each iter
    anneal_time = 0.0   # reported GPU/solver exec time
    wall_time   = 0.0

    # Fixstars client
    client = FixstarsClient()
    key = os.getenv("AE_KEY")
    if key:
        client.token = key
    else:
        raise RuntimeError("AE_KEY not found in environment")
    client.parameters.timeout = timedelta(milliseconds=timeout_ms)
    set_seed(seed)

    for it in range(1, max_iter + 1):
        # Linear coefficients depend on c: coeff = A c - b
        coeff = A @ c - b
        t0 = time.perf_counter()

        # mask-out exact zeros so we don’t generate useless Poly terms
        tol = 1e-12 * (np.linalg.norm(A, ord=np.inf) * np.linalg.norm(c, ord=np.inf) + np.linalg.norm(b, ord=np.inf))
        nz = np.abs(coeff) > tol

        lin_blk = np.dot(coeff[nz], dvec[nz])      # same as (coeff[nz] * dvec[nz]).sum()

        # final polynomial  (constant term is irrelevant to argmin)
        poly  = L * lin_blk + (L * L) * quad_blk
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
        E_val = sol.objective

        # Decode w
        q1_sol = q1.evaluate(sol.values)
        q2_sol = q2.evaluate(sol.values)
        w_new  = c + L * (-2 * q1_sol + q2_sol)
        E_true = 0.5 * w_new @ (A @ w_new) - b @ w_new

        if E_true < best_E:
            c = w_new
            best_E = E_true 
        else:
            L *= beta

        if L < epsilon:
            break

    exact = np.linalg.solve(A, b)
    err   = np.linalg.norm(c - exact)
    network_time = wall_time - anneal_time

    return {
        "iterations": it,
        "encode_time": encode_time,
        "anneal_time": anneal_time,
        "total_time": encode_time + anneal_time,   # network-free
        "wall_time": wall_time + encode_time,      # encode + network
        "network_time": network_time,
        "error": err,
    }

