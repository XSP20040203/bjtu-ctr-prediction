import unittest
import numpy as np
from task_audience_row_core import aligned,select_columns


class AudienceRowTests(unittest.TestCase):
    def test_alignment_preserves_query_order_and_duplicates(self):
        keys=np.array([[20191114,1],[20191115,2]])
        values=np.array([[1,2],[3,4]])
        query=keys[[1,0,1]]
        np.testing.assert_array_equal(aligned(keys,values,query,20191115),values[[1,0,1]])

    def test_future_or_unknown_keys_rejected(self):
        keys=np.array([[20191114,1]])
        for query in [np.array([[20191115,1]]),np.array([[20191114,9]])]:
            with self.assertRaises(ValueError):aligned(keys,np.ones((1,2)),query,20191114)

    def test_categorical_features_excluded_and_ties_stable(self):
        self.assertEqual(select_columns(['cat','n1','n2','n3'],['cat'],[999,4,4,2],2),['n1','n2'])


if __name__=='__main__':unittest.main()
