import unittest
import numpy as np
import pandas as pd
from feed_behavior_userday import feed_features,READ,CATEGORIES


class FeedBehaviorTests(unittest.TestCase):
    def sample(self):
        frame=pd.DataFrame({c:[1,1,99] for c in READ})
        frame['u_userId']=[3,3,3]
        frame['e_et']=[20191115,20191116,20191117]
        frame['label']=[1,-1,1]
        frame['i_docId']=['a','b','c']
        frame['i_s_sourceId']=['s','t','u']
        return frame

    def test_schema_excludes_held_category_and_future_rows(self):
        frame=self.sample()
        frame.loc[1,'e_ch']=20
        keys=np.array([[20191115,3],[20191116,3]])
        values,names,schema=feed_features(frame,keys)
        self.assertEqual(schema['e_ch'],[1])
        self.assertEqual(values[1,names.index('e_ch_None_share')],1)
        other=frame.copy()
        other.loc[2,CATEGORIES]=777
        other.loc[2,'pro']=999999
        np.testing.assert_array_equal(feed_features(other,keys,schema)[0],values)

    def test_declared_source_fields_and_source_labels(self):
        frame=self.sample()
        keys=np.array([[20191115,3],[20191116,3]])
        with self.assertRaises(ValueError):
            feed_features(frame.assign(advertisement_label=1),keys)
        frame.loc[0,'label']=0
        with self.assertRaises(ValueError):
            feed_features(frame,keys)


if __name__=="__main__":
    unittest.main()
