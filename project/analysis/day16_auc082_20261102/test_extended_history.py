import unittest
import numpy as np
from extended_history_residual import enhanced_history


class HistoryTests(unittest.TestCase):
    def test_query_before_insert_and_no_future_leak(self):
        keys=np.array([[20191113,1],[20191114,1],[20191115,1],[20191116,1]])
        sizes=np.array([10,20,40,50])
        counts=np.array([1.,2.,3.,-1.])
        x,names=enhanced_history(keys,sizes,counts,20191116)
        i=names.index("previous_log_k")
        np.testing.assert_allclose(x[:,i],np.log1p([0,1,2,3]))
        changed=counts.copy()
        changed[2]=100
        after,_=enhanced_history(keys,sizes,changed,20191116)
        np.testing.assert_array_equal(x[:3],after[:3])
        self.assertNotEqual(x[3,i],after[3,i])
        with self.assertRaises(ValueError):
            enhanced_history(keys,sizes,np.array([1.,2.,3.,9.]),20191116)

    def test_decay_obeys_calendar_gap(self):
        keys=np.array([[20191110,1],[20191115,1]])
        x,names=enhanced_history(keys,np.array([16,32]),np.array([1.,-1.]),20191115)
        self.assertAlmostEqual(x[1,names.index("decay1_log_n")],np.log1p(1.))
        self.assertEqual(x[1,names.index("window3_log_k")],0.)


if __name__=="__main__":
    unittest.main()
