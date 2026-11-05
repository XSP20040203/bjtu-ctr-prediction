import unittest
import numpy as np
from daily_pair_rank import pairs, correct


class DailyPairContracts(unittest.TestCase):
    def test_sampled_pairs_are_same_date_and_fit_only(self):
        keys = np.array([[20191114, 1], [20191114, 2], [20191115, 1], [20191115, 2], [20191116, 1]])
        sizes = np.array([10, 20, 10, 20, 10])
        counts = np.array([3., 0., 2., 0., -1.])
        chosen = np.arange(4)
        result = pairs(keys, sizes, counts, chosen, 1.)
        np.testing.assert_array_equal(keys[result.a, 0], keys[result.b, 0])
        self.assertTrue(np.all(result.a != result.b))
        self.assertTrue(np.all(counts[result.a] > 0))
        with self.assertRaises(ValueError):
            pairs(keys, sizes, counts, np.arange(5), 1.)

    def test_zero_correction_and_within_group_order(self):
        base = np.array([.01, .03, .02, .05])
        groups = np.array([0, 0, 1, 1])
        sizes = np.array([2, 2])
        score = np.array([-.5, .4])
        np.testing.assert_array_equal(correct(base, score, groups, sizes, 0.), base)
        actual = correct(base, score, groups, sizes, .5)
        self.assertLess(actual[0], actual[1])
        self.assertLess(actual[2], actual[3])


if __name__ == "__main__":
    unittest.main()
