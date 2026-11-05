import unittest
import numpy as np
import pandas as pd
from group_score_calibration import group_features


class ContextContracts(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({"user_id": [1,1,2], "task_id": [3,4,3],
                                 "slot_id": [5,5,6], "spread_app_id": [7,8,7]})
        self.p = np.array([.01,.2,.03])

    def test_row_order_equivariant_and_group_mean(self):
        features, names = group_features(self.raw, self.p)
        order = np.array([2,0,1])
        shuffled, shuffled_names = group_features(self.raw.iloc[order], self.p[order])
        np.testing.assert_allclose(features[order], shuffled)
        self.assertEqual(names, shuffled_names)
        self.assertAlmostEqual(features[0,names.index("user_log_mass")], np.log(.21))

    def test_outcomes_rejected(self):
        with self.assertRaises(ValueError):
            group_features(self.raw.assign(label=[0,1,0]), self.p)


if __name__ == "__main__":
    unittest.main()
