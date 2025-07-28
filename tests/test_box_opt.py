# tests/test_box_opt.py
import os
import unittest
import numpy as np
from models.box_naive import solve_box_naive_amplify
from models.box_opt import solve_box_opt_amplify

from data.data_generator import generate_synthetic_regression

class TestBoxOptimizedAmplify(unittest.TestCase):
    def setUp(self):
        self.data = generate_synthetic_regression(n=80, d=4, noise_sigma=0.01, seed=123)
        self.A = self.data.X_train.T @ self.data.X_train
        self.b = self.data.X_train.T @ self.data.y_train

    def test_opt_vs_naive_encode(self):
        naive = solve_box_naive_amplify(
            self.A, self.b,
            max_iter=15, num_solves=1, timeout_ms=20, seed=1
        )
        opt = solve_box_opt_amplify(
            self.A, self.b,
            max_iter=15, num_solves=1, timeout_ms=20, seed=1
        )

        # same error scale
        self.assertLess(opt["error"], 1e-1)
        self.assertLess(naive["error"], 1e-1)

        # optimized should not be slower on encode
        self.assertLessEqual(opt["encode_time"], naive["encode_time"] + 1e-1)

        # iterations should be identical or lower (same algorithm)
        self.assertLessEqual(opt["iterations"], naive["iterations"])


if __name__ == "__main__":
    unittest.main()

