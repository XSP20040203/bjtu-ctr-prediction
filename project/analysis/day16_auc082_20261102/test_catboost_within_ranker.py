"""Check complete-query Pool boundaries before full GPU ranking fits."""
import unittest
import numpy as np
from catboost_within_ranker import query_ids


class QueryChecks(unittest.TestCase):
    def test_complete_ordered_queries(self):
        np.testing.assert_array_equal(query_ids([3,2],[0,1,0,1,0]),[0,0,0,1,1])

    def test_partial_query_rejected(self):
        with self.assertRaises(ValueError):
            query_ids([3,2],[0,1,0,1])

    def test_unobserved_target_rejected(self):
        with self.assertRaises(ValueError):
            query_ids([3,2],[0,1,0,1,-1])

    def test_single_class_or_fractional_query_rejected(self):
        for sizes,target in [([3,2],[0,1,0,0,0]),([2.5,2.5],[0,1,0,1,0]),([1,2],[1,1,0])]:
            with self.assertRaises(ValueError):
                query_ids(sizes,target)


if __name__=='__main__':
    unittest.main()
