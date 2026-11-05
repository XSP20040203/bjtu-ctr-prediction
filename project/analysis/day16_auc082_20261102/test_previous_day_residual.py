import unittest
import numpy as np
import pandas as pd
from previous_day_residual import posterior_error,make_statistics,apply_model


class ErrorTransferTests(unittest.TestCase):
    def test_shrinkage_and_no_error_when_observed_equals_expected(self):
        np.testing.assert_array_equal(posterior_error(np.array([1.,4.]),np.array([1.,4.]),1.),[0.,0.])
        self.assertLess(abs(posterior_error(np.array([0.]),np.array([10.]),10.)[0]),
                        abs(posterior_error(np.array([0.]),np.array([10.]),1.)[0]))

    def test_no_future_labels_in_fitting_and_unseen_groups_zero(self):
        train=pd.DataFrame({'pt_d':[20191115,20191115],'label':[1,0],'user_id':[1,2],
                            'task_id':[1,1],'spread_app_id':[1,1],'slot_id':[1,1],'adv_prim_id':[1,1]})
        stats=make_statistics(train,np.array([.5,.5]))
        query=train.iloc[:1].copy();query['user_id']=99
        np.testing.assert_array_equal(apply_model(query,stats,'user',1.),[0.])
        train.loc[0,'pt_d']=20191116
        with self.assertRaises(ValueError):
            make_statistics(train,np.array([.5,.5]))


if __name__=='__main__':
    unittest.main()
