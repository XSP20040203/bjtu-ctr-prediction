import unittest
import numpy as np
import pandas as pd
from topic_context_residual import topic_features


class TopicContextTests(unittest.TestCase):
    def test_vocabulary_future_boundary_and_unknown_bucket(self):
        keys=np.array([[20191115,1],[20191116,1]])
        feeds=pd.DataFrame({'e_et':[20191115,20191116,20191117],'u_userId':[1,1,1],
                            'i_cat':[2,9,100],'label':[1,-1,1],'pro':[50,0,99]})
        values,names,topics=topic_features(feeds,keys)
        self.assertEqual(topics,[2])
        self.assertEqual(values[1,names.index('news_topic_exposure_unknown')],1.)
        self.assertEqual(values[0,names.index('news_topic_click_2')],1.)
        changed=feeds.copy()
        changed.loc[2,'i_cat']=200
        np.testing.assert_array_equal(topic_features(changed,keys,topics)[0],values)
        changed.loc[1,'label']=1
        np.testing.assert_array_equal(topic_features(changed,keys,topics)[0][0],values[0])

    def test_reject_additional_ad_label_field(self):
        feeds=pd.DataFrame({'e_et':[20191115],'u_userId':[1],'i_cat':[2],'label':[1],'pro':[50],'ad_label':[1]})
        with self.assertRaises(ValueError):
            topic_features(feeds,np.array([[20191115,1]]))


if __name__=="__main__":
    unittest.main()
