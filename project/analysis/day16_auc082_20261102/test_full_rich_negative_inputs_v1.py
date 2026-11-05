"""Exercise population coverage and duplicate/query-order-safe feature joining."""
import unittest
import numpy as np
import pandas as pd
from full_rich_negative_inputs_v1 import compose, validate_alignment


class PopulationCompositionTests(unittest.TestCase):
    def test_complete_population_and_misalignment(self):
        left = dict(rows=np.array([2, 4, 7, 8]), train=np.array([0, 1, 2]),
                    held=np.array([3]), labels=np.array([0, 1, 0]))
        validate_alignment(left, {k: v.copy() for k, v in left.items()})
        right = {k: v.copy() for k, v in left.items()}
        right['rows'] = right['rows'][[1, 0, 2, 3]]
        with self.assertRaises(ValueError):
            validate_alignment(left, right)
        missing = dict(left, train=np.array([0, 1]), labels=np.array([0, 1]))
        with self.assertRaises(ValueError):
            validate_alignment(missing, missing)
        overlap = dict(left, held=np.array([2, 3]))
        with self.assertRaises(ValueError):
            validate_alignment(overlap, overlap)

    def test_join_preserves_query_order_duplicates_and_categories(self):
        left = np.array([[1, .2], [3, .4], [7, .8]], dtype='float32')
        right = np.array([[.1, 4], [.3, 6], [.5, 8]], dtype='float32')
        positions = np.array([2, 0, 2, 1])
        names, categories = ['raw_category', 'left', 'right', 'history_token'], ['raw_category', 'history_token']
        primary = compose(left, right, positions, names, categories)
        independent = compose(left, right, positions, names, categories, True)
        pd.testing.assert_frame_equal(primary, independent)
        np.testing.assert_array_equal(primary.to_numpy(), np.c_[left, right][positions])
        self.assertEqual(primary.raw_category.dtype, np.dtype('int64'))
        right[0, 1] = 2.5
        for mode in (False, True):
            with self.assertRaises(ValueError):
                compose(left, right, positions, names, categories, mode)
        right[0, 1] = 2
        left[0, 1] = np.nan
        with self.assertRaises(ValueError):
            compose(left, right, positions, names, categories)


if __name__ == '__main__':
    unittest.main()
