import unittest
import numpy as np
import pandas as pd
from cold_task_core import cold_mask,identity_columns


class ColdTaskTests(unittest.TestCase):
    def test_future_keys_do_not_make_a_task_known(self):
        history=pd.DataFrame(dict(pt_d=[20191114,20191115,20191116],task_id=[1,2,3]))
        query=pd.DataFrame(dict(pt_d=[20191115]*4,task_id=[1,2,3,4]))
        np.testing.assert_array_equal(cold_mask(history,query,20191115),[False,True,True,True])
        changed=history.copy()
        changed.loc[changed.pt_d.ge(20191115),'task_id']=999
        np.testing.assert_array_equal(cold_mask(history,query,20191115),cold_mask(changed,query,20191115))

    def test_ad_labels_rejected(self):
        history=pd.DataFrame(dict(pt_d=[20191114],task_id=[1]))
        with self.assertRaisesRegex(ValueError,'unlabeled'):
            cold_mask(history.assign(label=1),history,20191115)

    def test_feature_mask_preserves_parent_advertiser_and_app(self):
        names=['task_id','adv_id','adv_prim_id','spread_app_id','original__hist_click_task_match']
        names += [f'generic_{i}' for i in range(675)]
        names += [f'rich_daily__quality__{i}' for i in range(64)]
        indices,masked=identity_columns(names)
        self.assertEqual(indices,[0,1,4])
        self.assertNotIn('adv_prim_id',masked)
        self.assertNotIn('spread_app_id',masked)


if __name__=='__main__':
    unittest.main()
