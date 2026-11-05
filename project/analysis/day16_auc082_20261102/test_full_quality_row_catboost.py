import unittest
import numpy as np
from full_quality_row_catboost import full_frame, quality_rows, recency_weights, selection


class FullQualityBoundaries(unittest.TestCase):
    def test_quality_alignment_keeps_repeated_users_and_input_order(self):
        keys = np.array([[20191114, 7], [20191115, 7], [20191115, 9]])
        quality = np.arange(3*337).reshape(3, 337).astype('float32')
        query = np.array([[20191115, 9], [20191114, 7], [20191115, 9]])
        np.testing.assert_array_equal(quality_rows(keys, quality, query), quality[[2, 0, 2]])
        with self.assertRaises(ValueError):
            quality_rows(keys, quality, [[20191116, 7]])
        with self.assertRaises(ValueError):
            quality_rows(keys[[0, 0]], quality[:2], keys[:1])

    def test_replacement_retains_exact_categorical_values_and_original_columns(self):
        names = ['category']+['rich_daily__quality__'+str(i) for i in range(64)]+['numeric']
        matrix = np.zeros((3, 66), dtype='float32')
        matrix[:, 0] = [1516217, -1, 1524772]
        matrix[:, -1] = [1., 2., 3.]
        quality = np.arange(3*337).reshape(3, 337).astype('float32')
        frame, categories, keep = full_frame(matrix, np.array([2, 0, 2]), names, [0], quality)
        self.assertEqual(categories, ['f0'])
        self.assertEqual(keep, [0, 65])
        self.assertEqual(str(frame.f0.dtype), 'int64')
        np.testing.assert_array_equal(frame.f0, matrix[[2, 0, 2], 0])
        np.testing.assert_array_equal(frame.f1, matrix[[2, 0, 2], -1])
        np.testing.assert_array_equal(frame.iloc[:, 2:], quality)

    def test_recency_weights_follow_calendar_days_and_reject_target_dates(self):
        dates = np.array([20191030, 20191031, 20191101])
        result = recency_weights(dates, 20191102, 2.)
        self.assertAlmostEqual(float(result.mean()), 1., places=6)
        self.assertAlmostEqual(float(result[-1]/result[0]), 2., places=6)
        with self.assertRaises(ValueError):
            recency_weights(np.array([20191102]), 20191102, 2.)
        with self.assertRaises(ValueError):
            recency_weights(np.array([20191103]), 20191102, None)

    def test_held_and_future_target_mutations_cannot_change_training(self):
        dates = np.repeat(np.arange(20191111, 20191118), 30)
        labels = (np.arange(len(dates)) % 7 == 0).astype('int32')
        for day in [15, 16]:
            changed = labels.copy()
            changed[dates >= 20191100+day] = 1-changed[dates >= 20191100+day]
            for before, after in zip(selection(day, labels, dates), selection(day, changed, dates)):
                np.testing.assert_array_equal(before, after)


if __name__ == '__main__':
    unittest.main()
