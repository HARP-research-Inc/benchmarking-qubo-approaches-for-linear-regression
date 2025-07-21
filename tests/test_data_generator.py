import unittest
import numpy as np

from data.data_generator import (
    generate_synthetic_regression,
)

SEED = 42


class TestDataGenerator(unittest.TestCase):
    def test_shapes_and_split(self):
        data = generate_synthetic_regression(n=100, d=5, seed=SEED)
        self.assertEqual(data.X_train.shape[1], 5)
        self.assertEqual(data.X_test.shape[1], 5)
        self.assertEqual(len(data.y_train) + len(data.y_test), 100)
        self.assertEqual(len(data.y_train), 80)  # default 80–20 split

    def test_reproducibility(self):
        d1 = generate_synthetic_regression(50, 3, seed=SEED)
        d2 = generate_synthetic_regression(50, 3, seed=SEED)
        self.assertTrue(np.allclose(d1.X_train, d2.X_train))
        self.assertTrue(np.allclose(d1.y_train, d2.y_train))
        self.assertTrue(np.allclose(d1.w_true, d2.w_true))

    def test_feature_correlation(self):
        rho = 0.9
        data = generate_synthetic_regression(
            n=200, d=4, feature_corr=rho, seed=SEED
        )
        corr = np.corrcoef(data.X_train, rowvar=False)
        off_diag = corr[np.triu_indices(4, k=1)]
        # allow sampling noise ±0.05
        self.assertAlmostEqual(off_diag.mean(), rho, places=1)


if __name__ == "__main__":
    unittest.main()
 
