"""Boundary and independent dictionary checks for the new export."""
import unittest
import numpy as np
import pandas as pd
from scipy.special import logit
import export_best_v2 as e


class ExportChecks(unittest.TestCase):
    def test_prior_dictionary_unseen_categories(self):
        vocabulary=np.array([0,7,100000])
        values=np.array([-1,0,1,7,99999,100000,100001])
        np.testing.assert_array_equal(e.dense_codes(values,vocabulary),[-1,0,-1,1,-1,2,-1])
        np.testing.assert_array_equal(e.dense_codes(values,vocabulary),pd.Index(vocabulary).get_indexer(values))

    def test_independent_user_and_task_compositions(self):
        raw=pd.DataFrame(dict(user_id=[1,1,2,2,3],task_id=[4,5,4,4,5],slot_id=[0]*5,spread_app_id=[1]*5))
        anchor=np.array([.01,.03,.02,.08,.015])
        current=anchor*np.array([.8,1.2,1.1,.9,1.05])
        daily=anchor*np.array([1.2,.9,.7,1.1,1.3])
        for group,between,within in [('user',0.,1.5),('task',1.5,1.)]:
            record=dict(formula='group_transfer',group=group,between=between,within=within)
            a=e.transfer(raw,current,daily,anchor,record)
            b=e.transfer(raw,current,daily,anchor,record,True)
            np.testing.assert_allclose(a,b,rtol=0,atol=2e-16)

    def test_empty_dictionary_rejects_all_values(self):
        np.testing.assert_array_equal(e.dense_codes(np.array([-1,0,9]),np.array([],dtype='int64')),[-1,-1,-1])

    def test_zero_dense_weight_preserves_anchor(self):
        anchor=np.array([.01,.2,.03,.08])
        actual=e.normalized(anchor,np.array([9.,1.,-2.,4.]),0.)
        np.testing.assert_allclose(actual,anchor,rtol=0,atol=2e-16)


if __name__=='__main__':
    unittest.main()
