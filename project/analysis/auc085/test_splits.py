import importlib
import unittest

import numpy as np


class SplitTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("split_utils")
        except ModuleNotFoundError:
            self.fail("Audited split utilities are pending")

    def test_random_split_is_disjoint_stratified_and_covers_all_rows(self):
        labels = np.array([0] * 80 + [1] * 20)
        dates = np.tile([20191115, 20191116], 50)
        train, valid = self.module.make_split(labels, dates, "random20", 42)
        self.assertEqual(len(valid), 20)
        self.assertEqual(int(labels[valid].sum()), 4)
        self.assertEqual(len(np.intersect1d(train, valid)), 0)
        np.testing.assert_array_equal(np.sort(np.r_[train, valid]), np.arange(100))

    def test_temporal_split_excludes_later_rows(self):
        dates = np.array([20191114, 20191115, 20191116, 20191117])
        train, valid = self.module.make_split(np.array([0, 1, 0, 1]), dates, "time16", 42)
        np.testing.assert_array_equal(train, [0, 1])
        np.testing.assert_array_equal(valid, [2])


if __name__ == "__main__":
    unittest.main()
