"""Only source dates in the declared window affect each user/day aggregate."""
import unittest
import numpy as np
import pandas as pd
from prepare_rich_source import build_rich_source_features,NUMERIC,SEQUENCES


class RichSourceTests(unittest.TestCase):
    def source(self):
        data=pd.DataFrame({'u_userId':[2,2,2],'e_et':[20200131,20200201,20200202],
              'label':[1,-1,1],'cillabel':[-1,-1,1],'pro':[100,0,50],
              'i_docId':['a','b','c'],'i_s_sourceId':['s','s','t'],'i_cat':[10,20,10]})
        for name in NUMERIC:data[name]=[1,2,3]
        for name in SEQUENCES:data[name]=['10^20','20','10']
        return data

    def test_future_source_does_not_affect_today(self):
        request=pd.DataFrame({'user_id':[2],'pt_d':[20200201]});source=self.source()
        a=build_rich_source_features(source,request)
        source.loc[2,'label']=-1;source.loc[2,'pro']=100
        b=build_rich_source_features(source,request)
        np.testing.assert_array_equal(a,b)

    def test_today_and_past_have_separate_click_progress(self):
        request=pd.DataFrame({'user_id':[2],'pt_d':[20200201]})
        a=build_rich_source_features(self.source(),request)
        self.assertEqual(float(a['rich_source__today__clicks'].iloc[0]),0)
        self.assertEqual(float(a['rich_source__past3__click_progress_mean'].iloc[0]),100)
        self.assertEqual(float(a['rich_source__today__click_progress_mean'].iloc[0]),0)

    def test_missing_user_has_finite_features(self):
        request=pd.DataFrame({'user_id':[3],'pt_d':[20200201]})
        a=build_rich_source_features(self.source(),request)
        self.assertTrue(np.isfinite(a.to_numpy()).all())
        self.assertEqual(float(a['rich_source__today__source_missing'].iloc[0]),1)


if __name__=='__main__':unittest.main()
