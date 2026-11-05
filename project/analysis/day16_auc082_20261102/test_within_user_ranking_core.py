import unittest
import numpy as np
from within_user_ranking_core import ranking_order,within_prediction


class RankingBoundaries(unittest.TestCase):
    def setUp(self):
        self.dates=np.array([20191113]*4+[20191114]*4+[20191115]*2)
        self.users=np.array([1,1,2,2,1,1,2,2,1,1])
        self.labels=np.array([0,1,0,0,1,0,1,1,-1,-1])
        self.rows=np.arange(10)

    def test_complete_prior_mixed_queries_only(self):
        order,sizes,target,keys=ranking_order(self.rows,self.dates,self.users,self.labels,20191115)
        np.testing.assert_array_equal(order,[0,1,4,5])
        np.testing.assert_array_equal(sizes,[2,2])
        np.testing.assert_array_equal(target,[0,1,1,0])
        np.testing.assert_array_equal(keys,[[20191113,1],[20191114,1]])

    def test_unmasked_held_targets_are_rejected(self):
        labels=self.labels.copy();labels[-1]=1
        with self.assertRaises(ValueError):
            ranking_order(self.rows,self.dates,self.users,labels,20191115)

    def test_query_order_is_rebuilt_from_date_user_and_original_row(self):
        rows=self.rows[::-1]
        order,sizes,target,keys=ranking_order(rows,self.dates,self.users,self.labels,20191115)
        np.testing.assert_array_equal(rows[order],[0,1,4,5])
        np.testing.assert_array_equal(sizes,[2,2])

    def test_within_user_formula_independent_pandas_replay(self):
        base=np.array([.01,.03,.02,.08,.015])
        margin=np.array([2.,1.,4.,7.,-1.])
        groups=np.array([0,0,1,1,2])
        for mode in ['add_within','replace_within']:
            for weight in [0.,.1,.5,1.]:
                np.testing.assert_allclose(within_prediction(base,margin,groups,weight,mode),
                    within_prediction(base,margin,groups,weight,mode,True),rtol=0,atol=2e-16)


if __name__=='__main__':
    unittest.main()
