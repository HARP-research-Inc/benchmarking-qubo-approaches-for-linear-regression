import time
import numpy as np
from amplify import VariableGenerator, FixstarsClient, solve, set_seed
from datetime import timedelta
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# ------------------------------------------------------------------ #
#   Public solver
# ------------------------------------------------------------------ #

def solve_box_naive_amplify(
    A,
    b,
    beta=0.5,
    epsilon=1e-6,
    max_iter=50,
    num_solves=1,
    timeout_ms=1000,
    seed=0,
):
    d = len(b)
    c = np.zeros(d)
    L = 1.0 
    best_E = np.inf

    encode_time = 0.0          # CPU build only
    anneal_time = 0.0          # GPU execution only
    wall_time = 0.0

    client = FixstarsClient()
    key = os.getenv('AE_KEY')
    if key is not None:
        client.token = key
    else:
        raise Exception('no api key in environment')

    client.parameters.timeout = timedelta(milliseconds=timeout_ms)
    set_seed(seed)

    for it in range(1, max_iter + 1):
        gen = VariableGenerator()
        q1 = gen.array("Binary", d)
        q2 = gen.array("Binary", d)
        w = c + L * (-2 * q1 + q2)
        energy = 0.5 * (w @ (A @ w)) - b @ w

        # ---------------- compile (CPU) ----------------
        t0 = time.perf_counter()
        model = energy
        encode_time += time.perf_counter() - t0

        # --------------- solve (GPU) -------------------
        wall_start = time.perf_counter()
        result = solve(model, client, num_solves=num_solves)
        wall_end = time.perf_counter()
        wall_time += wall_end - wall_start
        if not result:
            raise RuntimeError("Amplify returned no solutions")

        anneal_time += result.execution_time.total_seconds()  # GPU time only
        sol = result.best
        energy_E = sol.objective
        # ----------------------------------------------

        q1_sol = q1.evaluate(sol.values)
        q2_sol = q2.evaluate(sol.values)
        w_new = c + L * (-2 * q1_sol + q2_sol)

        if energy_E < best_E:
            c, best_E = w_new, energy_E
        else:
            L *= beta

        if L < epsilon:
            break

    exact = np.linalg.solve(A, b)
    err = np.linalg.norm(c - exact)

    return {
        "iterations": it,
        "encode_time": encode_time,              # seconds
        "anneal_time": anneal_time,              # seconds, GPU only
        "wall_time":   wall_time + encode_time,
        "total_time": encode_time + anneal_time, # network excluded
        "error": err,
    }

