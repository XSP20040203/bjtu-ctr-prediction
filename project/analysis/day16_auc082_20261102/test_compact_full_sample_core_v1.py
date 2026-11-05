import unittest
import numpy as np
from compact_full_sample_core_v1 import compact_frame, select_indices
from cross_user_pair_core import design


class CompactTests(unittest.TestCase):
    def test_categories_and_tie_break(self):
        names = ['cat', 'a', 'b', 'c']
        result = select_indices(names, ['cat'], [0, 2, 2, 1], 1)
        np.testing.assert_array_equal(result, [0, 1])
        with self.assertRaises(ValueError):
            select_indices(names, ['missing'], [0, 2, 2, 1])

    def test_block_and_column_replay(self):
        matrix = np.arange(30, dtype='float32').reshape(5, 6)
        rows = np.array([4, 1, 3])
        quality = np.array([[30, 31], [40, 41], [50, 51]], dtype='float32')
        auxiliary = np.array([[60], [70], [80]], dtype='float32')
        names = ['f0', 'f1', 'q0', 'q1', 'aud']
        expected = np.array([[25, 31, 60], [7, 41, 70], [19, 51, 80]], dtype='float32')
        first, cats = compact_frame(matrix, rows, [1, 5], quality, auxiliary,
                                    [0, 3, 4], names, ['f0'])
        second, other = compact_frame(matrix, rows, [1, 5], quality, auxiliary,
                                      [0, 3, 4], names, ['f0'], independent=True)
        np.testing.assert_array_equal(first.to_numpy(), expected)
        np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())
        self.assertEqual(cats, other)
        self.assertEqual(str(first.f0.dtype), 'int64')

    def test_all_source_rows_kept_and_dates_isolated(self):
        dates = np.array([13, 13, 13, 14, 14, 14, 15])
        y = np.array([0, 0, 1, 0, 1, 0, -1])
        order, groups, pairs = design(dates, y, 15, block_size=3, draws=4)
        np.testing.assert_array_equal(np.sort(order), np.arange(6))
        self.assertTrue(np.all(y[order[pairs[:, 0]]] == 1))
        self.assertTrue(np.all(y[order[pairs[:, 1]]] == 0))
        np.testing.assert_array_equal(groups[pairs[:, 0]], groups[pairs[:, 1]])
        with self.assertRaises(ValueError):
            design(dates, np.where(y == -1, 1, y), 15)


if __name__ == '__main__':
    unittest.main()
