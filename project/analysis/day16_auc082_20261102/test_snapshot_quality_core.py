import unittest
import numpy as np
import pandas as pd
from snapshot_quality_core import AD_FIELDS,NEWS_FIELDS,build
from news_quality_features import leave_user_out


def examples():
    source=pd.DataFrame({name:np.ones(6,dtype='int64') for name in NEWS_FIELDS})
    source['e_et']=[20191114]*4+[20191115]*2
    source['u_userId']=[1,1,2,2,1,2]
    source['u_newsCatInterestsST']=['16^3','16^3','16^3','16^3','16^3','16^3']
    source['i_docId']=['a','b','a','b','a','b']
    source['label']=[1,-1,1,-1,1,1]
    source['pro']=[50,0,80,0,90,90]
    ads=source.rename(columns={'e_et':'pt_d','u_userId':'user_id'}).iloc[[0,2]][AD_FIELDS].copy()
    return ads,source[NEWS_FIELDS]


class SnapshotQualityTests(unittest.TestCase):
    def test_future_exclusion_and_alignment(self):
        ads,news=examples()
        before,names=build(ads,news,20191114)
        changed=news.copy()
        changed.loc[changed.e_et.eq(20191115),['label','pro']]=[-1,0]
        after,_=build(ads,changed,20191114)
        np.testing.assert_array_equal(before,after)
        reverse,_=build(ads.iloc[::-1],news,20191114)
        np.testing.assert_array_equal(before[::-1],reverse)
        self.assertEqual(len(names),240)

    def test_unknown_context_and_label_rejection(self):
        ads,news=examples()
        ads.loc[:,'user_id']=99
        values,names=build(ads,news,20191114)
        self.assertTrue(np.all(values[:,[i for i,n in enumerate(names) if n.endswith('__missing')]]==1))
        self.assertTrue(np.all(values[:,[i for i,n in enumerate(names) if not n.endswith('__missing')]]==0))
        with self.assertRaisesRegex(ValueError,'declared'):
            build(ads.assign(label=1),news,20191114)

    def test_query_user_excluded_from_expectation(self):
        _,news=examples()
        news=news.iloc[:4].copy()
        users=np.array([0,0,1,1])
        target=np.column_stack([news.label.eq(1).astype(float)]*3)
        before,_=leave_user_out(news,target,users,['i_docId'],5.)
        target[:2]=1-target[:2]
        after,_=leave_user_out(news,target,users,['i_docId'],5.)
        np.testing.assert_allclose(before[:2],after[:2],rtol=0,atol=1e-15)


if __name__=='__main__':
    unittest.main()
