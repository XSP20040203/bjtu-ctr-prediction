import unittest
import numpy as np
import pandas as pd
from full_negative_prior_offset_core_v1 import build, FIELDS


class PriorOffsetTests(unittest.TestCase):
    def inputs(self):
        raw = pd.DataFrame({
            'pt_d':[20191111]*5+[20191112]*4+[20191113]*4,
            'user_id':[1,1,2,3,3,1,2,3,4,1,2,4,9],
            'spread_app_id':[10,10,20,30,30,20,20,10,40,10,40,40,99],
            'slot_id':[1,2,1,2,2,1,2,2,9,1,9,9,99]})
        labels = np.array([1,0,0,1,0,0,1,0,1,-1,-1,-1,-1])
        return raw[FIELDS], labels

    def test_independent_aggregation_and_permutation(self):
        raw, labels = self.inputs()
        first, audit = build(raw, labels, 20191113)
        second, other = build(raw, labels, 20191113, True)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(audit, other)
        order = np.array([12,0,8,1,3,10,2,4,11,7,9,6,5])
        shuffled, _ = build(raw.iloc[order].reset_index(drop=True), labels[order], 20191113, True)
        np.testing.assert_array_equal(shuffled[np.argsort(order)], first)
        self.assertTrue(all(a['max_label_date'] is None or a['max_label_date'] < a['query_date'] for a in audit))

    def test_temporal_boundary_unseen_keys_and_duplicates(self):
        raw, labels = self.inputs()
        first, _ = build(raw, labels, 20191113)
        changed = labels.copy()
        changed[raw.pt_d.eq(20191112)] = 1 - changed[raw.pt_d.eq(20191112)]
        second, _ = build(raw, changed, 20191113)
        np.testing.assert_array_equal(first[raw.pt_d.le(20191112)], second[raw.pt_d.le(20191112)])
        appended = pd.concat([raw, raw.iloc[[-1]]], ignore_index=True)
        duplicated, _ = build(appended, np.r_[labels,-1], 20191113, True)
        np.testing.assert_array_equal(duplicated[:-1], first)
        self.assertEqual(duplicated[-1], first[-1])
        exposed = labels.copy(); exposed[-1] = 1
        with self.assertRaises(ValueError):
            build(raw, exposed, 20191113)
        future = raw.copy(); future.loc[0,'pt_d'] = 20191114
        with self.assertRaises(ValueError):
            build(future, labels, 20191113)


if __name__ == '__main__':
    unittest.main()
