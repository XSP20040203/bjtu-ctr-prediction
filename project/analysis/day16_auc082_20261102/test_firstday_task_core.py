import unittest
import numpy as np
import pandas as pd
from firstday_task_core import firstday_design


class FirstDayTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame(dict(
            pt_d=[20191111, 20191112, 20191112, 20191112, 20191113, 20191113, 20191114],
            task_id=[1, 1, 2, 2, 2, 3, 99]))

    def test_launches_only_and_pair_label_alignment(self):
        labels = np.array([0, 1, 0, 1, 0, 0])
        order, groups, pairs = firstday_design(self.raw, np.arange(6), labels, 20191114)
        self.assertEqual(set(order), {2, 3, 5})
        self.assertTrue(np.all(labels[order[pairs[:, 0]]] == 1))
        self.assertTrue(np.all(labels[order[pairs[:, 1]]] == 0))
        self.assertTrue(np.all(groups[pairs[:, 0]] == groups[pairs[:, 1]]))

    def test_future_task_ids_do_not_change_training(self):
        labels = np.array([0, 1, 0, 1, 0, 0])
        changed = self.raw.copy()
        changed.loc[changed.pt_d.ge(20191114), 'task_id'] = 2
        first = firstday_design(self.raw, np.arange(6), labels, 20191114)
        second = firstday_design(changed, np.arange(6), labels, 20191114)
        for x, y in zip(first, second):
            np.testing.assert_array_equal(x, y)

    def test_future_rows_and_label_columns_rejected(self):
        with self.assertRaisesRegex(ValueError, 'strictly earlier'):
            firstday_design(self.raw, np.arange(7), np.zeros(7), 20191114)
        with self.assertRaisesRegex(ValueError, 'unlabeled'):
            firstday_design(self.raw.assign(label=0), np.arange(6), np.zeros(6), 20191114)


if __name__ == '__main__':
    unittest.main()
