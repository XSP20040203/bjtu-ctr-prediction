"""Daily input union matches are independent of ad labels and row ordering."""
import unittest
import numpy as np
import pandas as pd
from prepare_snapshot_features import build_features


class SnapshotFeaturesTests(unittest.TestCase):
    def frame(self):
        result=pd.DataFrame({'user_id':[2,2,2,2], 'pt_d':[20200131,20200201,20200201,20200202],
             'task_id':[10,20,10,30], 'adv_prim_id':[10,20,10,30], 'spread_app_id':[10,20,10,30]})
        for action in ['click','close']:
            for index in [1,2,3]:result[f'ad_{action}_list_v00{index}']=['10','10','10^20','30']
        return result

    def test_same_day_other_snapshot_can_reveal_new_input_token(self):
        output,names=build_features(self.frame())
        index=names.index('snapshot_click_1__candidate_new')
        self.assertEqual(output[1,index],1.)
        self.assertEqual(output[2,index],0.)
        other=names.index('snapshot_click_1__candidate_in_other_snapshot')
        self.assertEqual(output[1,other],1.)

    def test_later_day_cannot_change_earlier_inputs(self):
        frame=self.frame();before,_=build_features(frame)
        for f in [c for c in frame if 'list' in c]:frame.loc[3,f]='10^20^30^40'
        after,_=build_features(frame)
        np.testing.assert_array_equal(before[:3],after[:3])

    def test_permutation_preserves_corresponding_rows_and_extra_labels_ignored(self):
        frame=self.frame();before,_=build_features(frame)
        changed=frame.iloc[[2,0,3,1]].copy();changed['label']=[1,1,0,0]
        after,_=build_features(changed)
        np.testing.assert_array_equal(before[[2,0,3,1]],after)


if __name__=='__main__':unittest.main()
