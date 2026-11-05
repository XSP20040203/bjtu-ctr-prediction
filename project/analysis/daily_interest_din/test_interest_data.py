"""Date-boundary invariance for ad statistics and provided source snapshots."""
import unittest
import numpy as np
import pandas as pd
from interest_data import build_past_statistics,build_topic_store,request_keys


class InterestDataTests(unittest.TestCase):
    def ads(self):
        return pd.DataFrame({"user_id":[2,2,2,2],"pt_d":[20200130,20200131,20200201,20200202],"label":[1,0,1,1]})

    def feeds(self):
        return pd.DataFrame({"u_userId":[2,2,2],"e_et":[20200131,20200201,20200202],"i_cat":[10,20,20],
                             "label":[1,1,-1],"pro":[100,50,0],"u_newsCatDislike":["0","10","0"]})

    def test_future_and_validation_ad_labels_cannot_affect_features(self):
        a=self.ads();before=build_past_statistics(a,20200201,[(["user_id"],2.)])
        a.loc[a.pt_d>=20200201,"label"]=0
        after=build_past_statistics(a,20200201,[(["user_id"],2.)])
        np.testing.assert_array_equal(before,after)
        self.assertTrue(np.all(before[-1]==0))

    def test_current_label_excluded_but_past_label_can_affect_later_features(self):
        a=self.ads();before=build_past_statistics(a,20200201,[(["user_id"],2.)]);a.loc[1,"label"]=1
        after=build_past_statistics(a,20200201,[(["user_id"],2.)])
        np.testing.assert_array_equal(before[:2],after[:2]);self.assertFalse(np.array_equal(before[2],after[2]))

    def test_future_source_cannot_affect_request_but_same_day_source_can(self):
        request=pd.DataFrame({"user_id":[2],"pt_d":[20200201]});v={"10":2,"20":3};a=self.feeds()
        before=build_topic_store(a,request,v);a.loc[a.e_et==20200202,"label"]=1;a.loc[a.e_et==20200202,"pro"]=100
        after=build_topic_store(a,request,v)
        np.testing.assert_array_equal(before["topic_weights"],after["topic_weights"])
        a.loc[a.e_et==20200201,"label"]=-1;a.loc[a.e_et==20200201,"pro"]=0
        current=build_topic_store(a,request,v)
        self.assertFalse(np.array_equal(before["topic_weights"],current["topic_weights"]))

    def test_calendar_history_missing_and_zero_negative_marker(self):
        request=pd.DataFrame({"user_id":[2,3],"pt_d":[20200201,20200201]})
        data=build_topic_store(self.feeds(),request,{"10":2,"20":3},history_days=1)
        pos=np.searchsorted(data["keys"],request_keys([2,3],[20200201,20200201]))
        self.assertEqual(data["topic_weights"][pos[0],0,1,2],1.)
        self.assertEqual(data["topic_mass"][pos[0],0,3],0.)
        self.assertEqual(data["interest_context"][pos[1],1,-2],1.)
        self.assertEqual(data["topic_weights"][pos[0],1,3,2],1.)
        np.testing.assert_array_equal(data["topic_weights"][0],0)

    def test_source_row_permutation_preserves_interest(self):
        request=pd.DataFrame({"user_id":[2],"pt_d":[20200201]});a=self.feeds()
        x=build_topic_store(a,request,{"10":2,"20":3});y=build_topic_store(a.iloc[::-1],request,{"10":2,"20":3})
        np.testing.assert_array_equal(x["topic_weights"],y["topic_weights"])


if __name__=="__main__":unittest.main()
