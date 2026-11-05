import unittest
import numpy as np
from rich_group_residual import aggregate, prediction


class ZeroModel:
    def predict(self, values, raw_score):
        if not raw_score:
            raise AssertionError("Residual must use raw tree score")
        return np.zeros(len(values))


class ResidualTests(unittest.TestCase):
    def test_group_aggregate_and_permutation(self):
        p = np.array([.1, .2, .3, .4])
        g = np.array([1, 0, 1, 0])
        mass, any_probability, x = aggregate(p, g, 2)
        np.testing.assert_allclose(mass, [.6, .4])
        np.testing.assert_allclose(any_probability, [.52, .37])
        mass2, probability2, x2 = aggregate(p[::-1], g[::-1], 2)
        np.testing.assert_allclose(mass, mass2)
        np.testing.assert_allclose(any_probability, probability2)
        np.testing.assert_allclose(x, x2)

    def test_zero_residual_preserves_base_and_never_reads_held_labels(self):
        data = {"x": np.zeros((2, 3)), "keys": np.array([[20191116,1], [20191116,2]]),
                "any": np.array([.2, .5]), "mass": np.array([.25, .7]), "count": np.array([-1., -1.])}
        model = ZeroModel()
        for objective, key in [("binary", "any"), ("poisson", "mass")]:
            before = prediction(model, data, objective)
            np.testing.assert_allclose(before, data[key])
            data["count"] = np.array([999., 999.])
            np.testing.assert_array_equal(before, prediction(model, data, objective))
            np.testing.assert_array_equal(before[::-1], prediction(model, data, objective, np.array([1,0])))


if __name__ == "__main__":
    unittest.main()
