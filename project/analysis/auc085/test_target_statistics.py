import importlib
import unittest

import numpy as np


class TargetStatisticsTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("target_statistics")
        except ModuleNotFoundError:
            self.fail("Cross-fitted target statistics are pending")
        self.keys = np.array([10, 10, 20, 20, 30, 30])
        self.labels = np.array([1, 0, 0, 1, 0, 0])
        self.train = np.array([0, 1, 2, 3])
        self.folds = np.array([0, 1, 0, 1, -1, -1])

    def test_fold_labels_excluded_and_unseen_keys_use_training_prior(self):
        encoded, _ = self.module.oof_target_statistics(self.keys, self.labels, self.train, self.folds, alpha=1.)
        np.testing.assert_allclose(encoded[:, 0], [1, 1, 1, 1, 0, 0])
        np.testing.assert_allclose(encoded[:, 1], [.25, .75, .75, .25, .5, .5])
        changed = self.labels.copy()
        changed[0] = 0
        other, _ = self.module.oof_target_statistics(self.keys, changed, self.train, self.folds, alpha=1.)
        np.testing.assert_allclose(encoded[[0, 2]], other[[0, 2]])

    def test_validation_labels_cannot_change_any_feature(self):
        before, _ = self.module.oof_target_statistics(self.keys, self.labels, self.train, self.folds, alpha=1.)
        changed = self.labels.copy()
        changed[4:] = 1
        after, _ = self.module.oof_target_statistics(self.keys, changed, self.train, self.folds, alpha=1.)
        np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
