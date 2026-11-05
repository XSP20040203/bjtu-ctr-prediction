"""Check complete negative participation, temporal boundaries and reproducibility."""
import unittest
import numpy as np
from full_negative_pair_core_v1 import design


class CompleteNegativePairTests(unittest.TestCase):
    def test_imbalanced_all_rows_and_all_negatives(self):
        dates = np.r_[np.full(301, 20191113), np.full(511, 20191114), np.full(17, 20191115)]
        labels = np.r_[1, np.zeros(300), np.ones(7), np.zeros(504), np.full(17, -1)].astype('int32')
        order, groups, pairs = design(dates, labels, 20191115, negative_block=31, positive_draws=3)
        np.testing.assert_array_equal(np.sort(order), np.arange(812))
        np.testing.assert_array_equal(np.unique(pairs[:, 1]), np.flatnonzero(labels[order] == 0))
        np.testing.assert_array_equal(np.unique(pairs[:, 0]), np.flatnonzero(labels[order] == 1))
        self.assertTrue(np.all(np.diff(groups) >= 0))
        np.testing.assert_array_equal(groups[pairs[:, 0]], groups[pairs[:, 1]])
        np.testing.assert_array_equal(dates[order[pairs[:, 0]]], dates[order[pairs[:, 1]]])
        for first, second in zip((order, groups, pairs), design(dates, labels, 20191115, negative_block=31, positive_draws=3)):
            np.testing.assert_array_equal(first, second)

    def test_temporal_contamination_and_one_class_date_rejected(self):
        with self.assertRaisesRegex(ValueError, 'masked'):
            design(np.array([20191114, 20191114, 20191115]), np.array([1, 0, 1]), 20191115)
        with self.assertRaisesRegex(ValueError, 'both classes'):
            design(np.array([20191113, 20191114, 20191114]), np.array([0, 0, 1]), 20191115)


if __name__ == '__main__':
    unittest.main()
