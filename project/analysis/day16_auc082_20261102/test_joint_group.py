import unittest
import numpy as np
from joint_group_calibration import fit,predict_correction


class JointCalibrationTests(unittest.TestCase):
    def test_loss_decreases_and_analytic_single_group_solution(self):
        y=np.array([1.,0.,1.,0.]);base=np.full(4,.1);codes=[np.zeros(4,dtype=int)]
        weights,history=fit(y,base,codes,1.)
        self.assertTrue(np.all(np.diff(history)<=1e-8))
        b=weights[0][0]
        self.assertAlmostEqual(.4*np.exp(b)-2+b,0.,places=5)
        np.testing.assert_allclose(fit(y,base,codes,1.)[0][0],weights[0],atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
