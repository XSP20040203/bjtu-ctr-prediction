import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from rank_calibration import objective, sample_pairs


class RankingContracts(unittest.TestCase):
    def test_analytic_gradient(self):
        rng = np.random.default_rng(28)
        x, z, beta = rng.normal(size=(31, 5)), rng.normal(size=31), rng.normal(size=5)
        _, analytic = objective(beta, x, z, .5, .3)
        numeric = approx_derivative(lambda b: objective(b, x, z, .5, .3)[0], beta).ravel()
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-7)

    def test_pair_classes_and_determinism(self):
        y = np.array([0, 1, 0, 0, 1, 1, 0])
        p, n = sample_pairs(y, 42, 100)
        np.testing.assert_array_equal(y[p], np.ones(100))
        np.testing.assert_array_equal(y[n], np.zeros(100))
        p2, n2 = sample_pairs(y, 42, 100)
        np.testing.assert_array_equal(p, p2)
        np.testing.assert_array_equal(n, n2)

    def test_masked_and_single_class_outcomes_rejected(self):
        for y in [[0, 1, -1], [0, 0], [1, 1], [0, .2, 1]]:
            with self.assertRaises(ValueError):
                sample_pairs(y, 42, 10)


if __name__ == "__main__":
    unittest.main()
