import unittest
import numpy as np
from rich_userday_activity import past_outcomes, correct


class ActivityTests(unittest.TestCase):
    def test_past_outcomes_exclude_same_day_and_held(self):
        keys = np.array([[20191113, 1], [20191113, 2], [20191114, 1], [20191115, 1]])
        sizes = np.array([10, 20, 30, 40])
        target = np.array([1., 0., 2., -1.])
        x, names = past_outcomes(keys, sizes, target, 20191115)
        self.assertEqual(x[0, names.index("history_all_log_clicks")], 0)
        self.assertAlmostEqual(x[2, names.index("history_all_log_clicks")], np.log1p(1), places=6)
        self.assertAlmostEqual(x[3, names.index("history_all_log_clicks")], np.log1p(3), places=6)
        changed = target.copy()
        changed[2] = 100
        other, _ = past_outcomes(keys, sizes, changed, 20191115)
        np.testing.assert_array_equal(x[:3], other[:3])
        self.assertNotEqual(x[3, 1], other[3, 1])
        with self.assertRaises(ValueError):
            past_outcomes(keys, sizes, np.array([1., 0., 2., 3.]), 20191115)

    def test_hazard_correction_matches_group_any_target(self):
        base = np.array([.1, .2, .1, .3])
        groups = np.array([0, 0, 1, 1])
        target = np.array([.5, .6])
        p = correct(base, target, groups, "binary", 1.)
        for group in [0, 1]:
            self.assertAlmostEqual(1-np.prod(1-p[groups == group]), target[group], places=12)
        np.testing.assert_array_equal(correct(base, target, groups, "binary", 0), base)


if __name__ == "__main__":
    unittest.main()
