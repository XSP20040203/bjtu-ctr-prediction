import unittest
import numpy as np
import pandas as pd
from transition_posterior_core_v1 import FIELDS,task_bins,fit_rates,predict


class TransitionPosteriorTests(unittest.TestCase):
    def test_same_date_canonical_states_and_exposure_counts(self):
        frame = pd.DataFrame([[14,1,9,'1^2'],[14,1,9,'2^1'],[14,1,4,'1^2^9'],
                              [14,2,9,'1^2'],[14,2,4,'1^2^9'],[15,2,9,'1^2']],columns=FIELDS)
        expected = np.array([2,2,0,4,0,0],dtype='int8')
        np.testing.assert_array_equal(task_bins(frame),expected)
        np.testing.assert_array_equal(task_bins(frame,True),expected)
        permutation = np.array([4,1,5,0,3,2])
        np.testing.assert_array_equal(task_bins(frame.iloc[permutation])[np.argsort(permutation)],expected)

    def test_strict_prior_boundary_and_probability_replay(self):
        bins = np.array([0,2,4,0,2,4])
        dates = np.array([13,14,14,15,15,16])
        labels = np.array([0,1,1,-1,-1,-1])
        record = fit_rates(bins,dates,labels,15,20,False)
        self.assertEqual(record,fit_rates(bins,dates,labels,15,20,False,True))
        self.assertEqual(record['max_training_date'],14)
        with self.assertRaises(ValueError):
            fit_rates(bins,dates,np.array([0,1,1,0,-1,-1]),15,20,False)
        base = np.array([.01,.02,.03,.04,.05,.06])
        primary = predict(base,bins,record,2,.5)
        np.testing.assert_array_equal(primary[bins<2],base[bins<2])
        self.assertTrue(np.all(primary>=base-1e-15))
        np.testing.assert_allclose(primary,predict(base,bins,record,2,.5,True),atol=1e-14,rtol=0)


if __name__ == '__main__':
    unittest.main()
