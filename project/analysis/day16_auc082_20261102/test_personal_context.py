import unittest
import numpy as np
from personal_context_residual import personal_changes


class PersonalContextTests(unittest.TestCase):
    def test_deltas_query_only_earlier_user_days(self):
        keys=np.array([[20191113,1],[20191113,2],[20191114,1],[20191115,1]])
        x=np.array([[2.,4.],[90.,100.],[4.,8.],[9.,13.]],dtype="float32")
        result=personal_changes(keys,x)
        np.testing.assert_array_equal(result[:2],np.zeros((2,6)))
        np.testing.assert_allclose(result[2,:4],[2,4,2,4])
        np.testing.assert_allclose(result[3,:4],[5,5,6,7])
        altered=x.copy()
        altered[3]=[90.,120.]
        np.testing.assert_array_equal(personal_changes(keys,altered)[:3],result[:3])

    def test_reject_duplicate_user_day(self):
        with self.assertRaises(ValueError):
            personal_changes(np.array([[20191115,1],[20191115,1]]),np.ones((2,3)))


if __name__=="__main__":
    unittest.main()
