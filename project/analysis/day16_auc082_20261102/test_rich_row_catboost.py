import unittest
import numpy as np
from rich_row_catboost import selection, frame_from_matrix, frame_hash, array_hash


class RichCatboostBoundaries(unittest.TestCase):
    def test_query_and_future_labels_cannot_change_training(self):
        dates = np.repeat(np.arange(20191111, 20191118), 30)
        labels = (np.arange(len(dates)) % 7 == 0).astype('int32')
        for day in [15, 16]:
            before = selection(day, labels, dates)
            changed = labels.copy()
            changed[dates >= 20191100+day] = 1-changed[dates >= 20191100+day]
            after = selection(day, changed, dates)
            for a, b in zip(before, after):
                np.testing.assert_array_equal(a, b)
            self.assertTrue(np.all(dates[before[0]] < 20191100+day))

    def test_category_integers_and_row_order_are_preserved(self):
        x = np.array([[1516217, .2], [-1, .3], [1524772, .4]], dtype='float32')
        frame = frame_from_matrix(x, np.array([2, 0, 1]), [0])
        self.assertEqual(str(frame.f0.dtype), 'int64')
        np.testing.assert_array_equal(frame.to_numpy(dtype='float32'), x[[2, 0, 1]])
        self.assertEqual(frame_hash(frame), array_hash(x[[2, 0, 1]]))

    def test_noninteger_category_is_rejected(self):
        x = np.array([[1.25, .2]], dtype='float32')
        with self.assertRaises(ValueError):
            frame_from_matrix(x, np.array([0]), [0])


if __name__ == '__main__':
    unittest.main()
