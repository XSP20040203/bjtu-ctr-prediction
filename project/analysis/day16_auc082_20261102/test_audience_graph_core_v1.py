"""Check chronological graph isolation, sparse replay, and original query order."""
import unittest
import numpy as np
import pandas as pd
from audience_graph_core_v1 import build_day, NAMES


class GraphFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.ads = pd.DataFrame(dict(
            pt_d=[10, 10, 10, 10, 11, 11, 11, 11, 12],
            user_id=[1, 1, 2, 3, 1, 2, 2, 3, 999],
            task_id=[1, 2, 1, 3, 3, 2, 4, 4, 999]))
        self.y = np.array([1, 0, 0, 1, -1, -1, -1, -1, -1])
        self.query = np.array([7, 4, 6, 5])

    def test_sparse_independent_replay_and_original_order(self):
        primary = build_day(self.ads, self.y, 11, self.query)
        replay = build_day(self.ads, self.y, 11, self.query, True)
        np.testing.assert_array_equal(primary, replay)
        reordered = build_day(self.ads, self.y, 11, self.query[::-1])
        np.testing.assert_array_equal(primary[::-1], reordered)
        self.assertEqual(primary.shape, (4, len(NAMES)))
        self.assertGreater(float(np.abs(primary).sum()), 0)

    def test_future_exposure_rows_are_ignored(self):
        original = build_day(self.ads, self.y, 11, self.query)
        changed = self.ads.copy()
        changed.loc[8, ['user_id', 'task_id']] = [1, 1]
        np.testing.assert_array_equal(original, build_day(changed, self.y, 11, self.query))

    def test_target_or_future_labels_rejected(self):
        for position in [4, 8]:
            y = self.y.copy()
            y[position] = 1
            with self.assertRaisesRegex(ValueError, 'must be masked'):
                build_day(self.ads, y, 11, self.query)


if __name__ == '__main__':
    unittest.main()
