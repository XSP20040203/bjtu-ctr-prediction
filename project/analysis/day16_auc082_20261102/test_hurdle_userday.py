import unittest
import numpy as np
from hurdle_userday import expected_clicks, severity_rows


class HurdleTests(unittest.TestCase):
    def test_total_expectation_and_severity_bounds(self):
        any_probability=np.array([.1,.4,.8])
        actual=expected_clicks(any_probability,np.array([0.,2.,4.]),np.ones(3),np.ones(3),1.)
        np.testing.assert_allclose(actual,[.1,1.2,4.])
        self.assertTrue(np.all(actual>=any_probability))
        base_mass=np.array([.12,.6,2.])
        base_any=np.array([.1,.4,.8])
        np.testing.assert_allclose(expected_clicks(base_any,np.ones(3)*99,base_mass,base_any,0.),base_mass)

    def test_no_zero_or_future_day_can_train_severity(self):
        keys=np.array([[20191113,1],[20191114,1],[20191115,1],[20191116,1]])
        counts=np.array([0.,2.,1.,-1.])
        np.testing.assert_array_equal(severity_rows(keys,counts,20191115),[1])
        np.testing.assert_array_equal(severity_rows(keys,counts,20191116),[1,2])


if __name__=="__main__":
    unittest.main()
