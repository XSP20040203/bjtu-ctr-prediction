"""Test temporal exclusion, unseen keys, repeated queries and independent aggregation."""
import unittest
import numpy as np
from full_negative_mask_core_v1 import masks


class MaskTests(unittest.TestCase):
    def test_prior_only_masks_and_duplicate_queries(self):
        dates = np.array([20191113, 20191114, 20191114, 20191115, 20191115, 20191115])
        users, tasks = np.array([1, 1, 2, 1, 2, 9]), np.array([4, 4, 7, 4, 8, 99])
        training, labels, query = np.arange(3), np.array([0, 1, 0]), np.array([5, 3, 4, 5])
        primary = masks(dates, users, tasks, training, labels, query, 20191115)
        other = masks(dates, users, tasks, training, labels, query, 20191115, True)
        for name in primary:
            np.testing.assert_array_equal(primary[name], other[name])
        np.testing.assert_array_equal(primary['cold_task'], [True, False, True, True])
        np.testing.assert_array_equal(primary['user_no_past_ad_click'], [True, False, True, True])
        # Changing a current-date user's ID cannot change another query's past clicks.
        changed = users.copy()
        changed[5] = 1
        value = masks(dates, changed, tasks, training, labels, query, 20191115)
        self.assertFalse(value['user_no_past_ad_click'][0])
        np.testing.assert_array_equal(value['cold_task'], primary['cold_task'])

    def test_future_training_and_partial_training_rejected(self):
        dates, users, tasks = np.array([20191114, 20191114, 20191115]), np.arange(3), np.arange(3)
        for training, labels in [(np.array([0, 2]), np.array([1, 0])), (np.array([0]), np.array([1]))]:
            with self.assertRaisesRegex(ValueError, 'strictly earlier'):
                masks(dates, users, tasks, training, labels, np.array([2]), 20191115)


if __name__ == '__main__':
    unittest.main()
