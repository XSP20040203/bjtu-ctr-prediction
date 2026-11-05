"""Temporal leakage, full-negative coverage, and row-order checks."""
import unittest
import numpy as np
import pandas as pd
from full_negative_context_core_v1 import population, positions
from global_row_context import CAT
from prepare_multiscale_stats import build_multiscale_statistics


class PopulationTests(unittest.TestCase):
    def test_full_population_and_held_labels_cannot_change_features(self):
        dates = np.array([20191112, 20191113, 20191112, 20191114, 20191115, 20191113, 20191115])
        labels = np.array([0, 1, 0, 0, 1, 0, 0])
        rows, train, held, safe = population(dates, labels, 20191115)
        np.testing.assert_array_equal(rows[train], [0, 1, 2, 3, 5])
        self.assertEqual(int((safe[rows[train]] == 0).sum()), 4)
        changed = labels.copy()
        changed[dates >= 20191115] = 2000
        other = population(dates, changed, 20191115)
        for original, altered in zip((rows, train, held, safe), other):
            np.testing.assert_array_equal(original, altered)
        frame = pd.DataFrame({name: np.arange(len(dates)) % 2 for name in CAT})
        frame['pt_d'], frame['label'] = dates, safe
        first, names = build_multiscale_statistics(frame, 20191115)
        frame['label'] = other[-1]
        order = np.array([6, 4, 2, 0, 5, 3, 1])
        second, other_names = build_multiscale_statistics(frame.iloc[order], 20191115)
        self.assertEqual(names, other_names)
        np.testing.assert_array_equal(first[order], second)
        with self.assertRaises(ValueError):
            population(dates, np.full(len(dates), -1), 20191115)

    def test_context_join_preserves_duplicates_and_query_order(self):
        keys = np.array([[20191115, 12], [20191114, 11], [20191115, 11]])
        query = keys[[2, 0, 2, 1]]
        for independent in (False, True):
            np.testing.assert_array_equal(positions(keys, query, independent), [2, 0, 2, 1])
            with self.assertRaises(ValueError):
                positions(keys, np.array([[20191116, 11]]), independent)
            with self.assertRaises(ValueError):
                positions(keys[[0, 0]], query, independent)


if __name__ == '__main__':
    unittest.main()
