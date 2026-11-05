import unittest
import numpy as np
import pandas as pd
from unordered_snapshot_core_v1 import CHANNELS, METRICS, build


def sample():
    frame = pd.DataFrame(dict(pt_d=[14,14,14,14,15], user_id=[1,1,1,2,1],
                              task_id=[7,8,9,7,99], adv_prim_id=[70,80,90,70,99],
                              spread_app_id=[700,800,900,700,999]))
    for field, target in CHANNELS:
        if target == 'task_id':
            frame[field] = ['1','1','1^7','1','1^7^99']
        elif target == 'adv_prim_id':
            frame[field] = ['10','10','10^70','10','10^70^99']
        else:
            frame[field] = ['100','100','100^700','100','100^700^999']
    return frame


class SnapshotTests(unittest.TestCase):
    def test_unique_transition_without_order(self):
        frame = sample()
        frame['label'] = [0,1,0,1,1]
        values, names = build(frame)
        self.assertEqual(values[0,names.index('unordered__ad_click_list_v001__single_added_unique_exposure')],1)
        self.assertEqual(values[1,names.index('unordered__ad_click_list_v001__candidate_absent_added_elsewhere')],0)
        self.assertAlmostEqual(values[0,1],np.log1p(2),places=6)
        self.assertAlmostEqual(values[0,3],.5)
        self.assertTrue(np.all(values[3:] == 0))
        frame['label'] = 1-frame.label
        np.testing.assert_array_equal(build(frame)[0],values)

    def test_permutation_day_boundary_and_independent_replay(self):
        frame = sample()
        expected, names = build(frame)
        actual, other = build(frame, independent=True)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(names,other)
        permutation = np.array([4,2,0,3,1])
        shuffled, _ = build(frame.iloc[permutation])
        np.testing.assert_array_equal(shuffled[np.argsort(permutation)],expected)
        only14, _ = build(frame[frame.pt_d == 14])
        np.testing.assert_array_equal(only14,expected[:4])

    def test_equivalent_sets_and_duplicate_exposures(self):
        frame = sample().iloc[:3].copy()
        frame.loc[1,'task_id'] = 7
        frame.loc[1,'ad_click_list_v001'] = '1^1'
        primary, names = build(frame)
        secondary, _ = build(frame, independent=True)
        np.testing.assert_array_equal(primary,secondary)
        self.assertEqual(primary[0,names.index('unordered__ad_click_list_v001__single_added_unique_exposure')],0)
        self.assertEqual(primary[0,names.index('unordered__ad_click_list_v001__inverse_matching_exposures')],.5)
        self.assertEqual(len(names),len(CHANNELS)*len(METRICS))


if __name__ == '__main__':
    unittest.main()
