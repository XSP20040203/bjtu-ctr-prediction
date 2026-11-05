import unittest
import numpy as np
import pandas as pd
from task_audience_core import FIELDS,build


class TaskAudienceTests(unittest.TestCase):
    def fixture(self):
        raw=pd.DataFrame({name:np.ones(7,dtype='int64') for name in FIELDS})
        raw.pt_d=[20191112,20191112,20191113,20191113,20191114,20191114,20191115]
        raw.task_id=[2,2,2,3,3,3,99]
        raw.user_id=[1,2,1,2,1,2,9]
        labels=np.array([1,0,0,1,-1,-1,-1])
        keys=np.array([[20191112,1],[20191112,2],[20191113,1],
                       [20191113,2],[20191114,1],[20191114,2]])
        quality=np.arange(12,dtype='float32').reshape(6,2)/7
        return raw,labels,keys,quality

    def test_sparse_and_bincount_match_and_full_rows_align(self):
        raw,y,keys,q=self.fixture()
        a,ca,ma=build(raw,y,keys,q,20191114)
        b,cb,mb=build(raw,y,keys,q,20191114,'sparse')
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(ca,cb)
        for k in ma:np.testing.assert_array_equal(ma[k],mb[k])
        np.testing.assert_array_equal(ma['rows'],[4,5])
        np.testing.assert_array_equal(ma['counts'],[1,0,1,-1])
        np.testing.assert_array_equal(ma['row_groups'],[0,0])

    def test_future_advertisements_do_not_affect_features(self):
        raw,y,keys,q=self.fixture()
        first=build(raw,y,keys,q,20191114)[0]
        raw.loc[raw.pt_d.gt(20191114),'user_id']=12345
        raw.loc[raw.pt_d.gt(20191114),'age']=5678
        second=build(raw,y,keys,q,20191114)[0]
        pd.testing.assert_frame_equal(first,second)

    def test_current_labels_rejected_and_later_labels_do_not_change_prior_features(self):
        raw,y,keys,q=self.fixture()
        bad=y.copy();bad[4]=1
        with self.assertRaisesRegex(ValueError,'must be masked'):
            build(raw,bad,keys,q,20191114)
        before=build(raw,y,keys,q,20191114)[0]
        changed=y.copy();changed[2:4]=1-changed[2:4]
        after=build(raw,changed,keys,q,20191114)[0]
        pd.testing.assert_frame_equal(before.iloc[:3],after.iloc[:3])
        self.assertFalse(np.array_equal(before.iloc[3].to_numpy(),after.iloc[3].to_numpy()))


if __name__=='__main__':unittest.main()
