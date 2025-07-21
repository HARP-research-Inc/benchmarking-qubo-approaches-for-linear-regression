import unittest
from numpy.linalg import norm

from data.data_generator import (
    generate_synthetic_regression,
)
from models import classical as C

SEED = 7


class TestClassicalModels(unittest.TestCase):
    def setUp(self):
        self.data = generate_synthetic_regression(
            n=200, d=10, noise_sigma=0.01, seed=SEED
        )

    def _sanity(self, res, attr_name="r2", thresh=0.9):
        self.assertGreaterEqual(res[attr_name], thresh)
    
    #Ignores errors to avoid complaints about returning a tuple because of decorator
    def test_ols(self):
        model, _ = C.train_ols(self.data.X_train, self.data.y_train) # pyright: ignore[reportGeneralTypeIssues]
        res = C.evaluate(model, self.data.X_test, self.data.y_test)
        self._sanity(res)

    def test_ridge(self):
        model, _ = C.train_ridge(self.data.X_train, self.data.y_train, alpha=1.0) # pyright: ignore[reportGeneralTypeIssues]
        res = C.evaluate(model, self.data.X_test, self.data.y_test)
        self._sanity(res)

    def test_lasso(self):
        model, _ = C.train_lasso(self.data.X_train, self.data.y_train, alpha=0.001) # pyright: ignore[reportGeneralTypeIssues]
        res = C.evaluate(model, self.data.X_test, self.data.y_test)
        self._sanity(res, thresh=0.85)  # lasso may be slightly worse

    def test_sgd(self):
        model, _ = C.train_sgd(self.data.X_train, self.data.y_train, lr=0.01) # pyright: ignore[reportGeneralTypeIssues]
        res = C.evaluate(model, self.data.X_test, self.data.y_test)
        self._sanity(res, thresh=0.85)


if __name__ == "__main__":
    unittest.main()

