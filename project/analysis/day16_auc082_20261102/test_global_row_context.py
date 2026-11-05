import unittest
import numpy as np
import pandas as pd
from global_row_context import selected_training, histories, HISTORY


class GlobalRowContracts(unittest.TestCase):
    def test_sampling_population_and_held_boundary(self):
        y = np.array([1]+[0]*12+[1, 0])
        dates = np.array([20191115]*13+[20191116]*2)
        rows, weights = selected_training(y, dates)
        self.assertEqual(int(y[rows].sum()), 1)
        self.assertTrue(np.all(dates[rows] < 20191116))
        self.assertAlmostEqual(float(weights[y[rows] == 0].sum()), 12.)
        self.assertAlmostEqual(float(weights[y[rows] == 1].sum()), 1.)
        altered = y.copy()
        altered[-2:] = [-99, 37]
        second, second_weights = selected_training(altered, dates)
        np.testing.assert_array_equal(rows, second)
        np.testing.assert_array_equal(weights, second_weights)

    def test_candidate_matches_and_unknown_histories(self):
        frame = pd.DataFrame({name: ["-1", "3^7^7"] for name in HISTORY})
        for name in ["task_id", "adv_prim_id", "spread_app_id"]:
            frame[name] = [3, 7]
        result = histories(frame)
        np.testing.assert_array_equal(result[0], np.zeros(12))
        np.testing.assert_array_equal(result[1, 1::2], np.ones(6))
        np.testing.assert_allclose(result[1, ::2], np.log(3))
        with self.assertRaises(ValueError):
            histories(frame.assign(label=[0, 1]))


if __name__ == "__main__":
    unittest.main()
