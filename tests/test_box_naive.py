import unittest
import numpy as np

from data.data_generator import (
    generate_synthetic_regression,
)


from models.box_naive import (
    solve_box_naive_amplify,
)


class TestBoxNaiveAmplify(unittest.TestCase):
    def test_converges_small(self):
        data = generate_synthetic_regression(n=30, d=3, noise_sigma=0.01, seed=11)
        A = data.X_train.T @ data.X_train
        b = data.X_train.T @ data.y_train

        res = solve_box_naive_amplify(A, b, max_iter=40, seed=11, timeout_ms=12)
        print(res)
        self.assertLess(res["error"], 1e-2)
        self.assertLessEqual(res["iterations"], 40)

if __name__ == "__main__":
    unittest.main()
