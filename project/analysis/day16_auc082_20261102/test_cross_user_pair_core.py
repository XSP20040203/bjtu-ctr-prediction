"""Verify cross-user pair boundaries, row coverage and deterministic randomization."""
import unittest
import numpy as np
from cross_user_pair_core import design


class CrossUserPairTests(unittest.TestCase):
    def test_pairs_and_full_prior_coverage(self):
        dates = np.repeat([20191113, 20191114, 20191115], 40)
        labels = np.tile(np.r_[np.ones(8), np.zeros(32)], 3).astype('int32')
        labels[dates >= 20191115] = -1
        order, groups, pairs = design(dates, labels, 20191115, block_size=16, draws=7)
        np.testing.assert_array_equal(np.sort(order), np.flatnonzero(dates < 20191115))
        self.assertTrue(np.all(labels[order[pairs[:, 0]]] == 1))
        self.assertTrue(np.all(labels[order[pairs[:, 1]]] == 0))
        np.testing.assert_array_equal(groups[pairs[:, 0]], groups[pairs[:, 1]])
        np.testing.assert_array_equal(dates[order[pairs[:, 0]]], dates[order[pairs[:, 1]]])
        self.assertTrue(np.all(np.diff(groups) >= 0))
        self.assertLessEqual(np.bincount(groups).max(), 16)
        for first, second in zip((order, groups, pairs), design(dates, labels, 20191115, block_size=16, draws=7)):
            np.testing.assert_array_equal(first, second)

    def test_future_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, 'masked'):
            design(np.array([20191114, 20191114, 20191115]), np.array([1, 0, 1]), 20191115)

    def test_zero_click_users_are_retained(self):
        # Users2..101 have no clicks; all their rows remain eligible negative examples.
        dates = np.full(101, 20191114)
        labels = np.r_[1, np.zeros(100)].astype('int32')
        order, _, pairs = design(dates, labels, 20191115, block_size=128, draws=500)
        self.assertEqual(len(order), 101)
        self.assertGreater(len(np.unique(order[pairs[:, 1]])), 90)


if __name__ == '__main__':
    unittest.main()
