import unittest
import numpy as np
from wide_score_calibration import auxiliary_correction, design


class AuxiliaryContracts(unittest.TestCase):
    def test_exact_group_event_implies_zero_correction(self):
        p = np.array([.01,.02,.04,.1])
        users = np.array([1,1,2,2])
        q = np.array([1-(1-.01)*(1-.02)]*2+[1-(1-.04)*(1-.1)]*2)
        np.testing.assert_allclose(auxiliary_correction(p,q,users),0.,atol=1e-12)
        order = np.array([3,1,0,2])
        np.testing.assert_allclose(auxiliary_correction(p[order],q[order],users[order]),0.,atol=1e-12)

    def test_zero_model_correction_preserves_base_score(self):
        f = np.array([[2.,3.],[4.,5.]])
        z = np.array([-3.,-2.])
        x = design(f,z,f.mean(0),np.ones(2))
        np.testing.assert_array_equal(x@np.array([0.,1.,0.,0.]),z)


if __name__ == "__main__":
    unittest.main()
