import unittest
import numpy as np
from context_scale_rank import ScaleObjective,row_scores


class ScaleRankTests(unittest.TestCase):
    def test_finite_difference_gradients_and_diagonal_hessian(self):
        obj=ScaleObjective(np.array([0,1,0]),np.array([2,2,1]),np.array([.3,-.8,1.]),
                           np.array([-.5,.2,.7]),np.array([.1,-.5,1.]),3,.2)
        x=np.array([.1,-.1,.2,.3,-.2,.1])
        grad,hess=obj(x,None)
        for i in range(len(x)):
            d=np.zeros(len(x));d[i]=1e-5
            self.assertAlmostEqual(grad[i],(obj.value(x+d)-obj.value(x-d))/2e-5,places=6)
            self.assertAlmostEqual(hess[i],(obj(x+d,None)[0][i]-obj(x-d,None)[0][i])/2e-5,places=6)

    def test_zero_correction_preserves_base_and_positive_within_group_order(self):
        data={'keys':np.array([[20191115,1],[20191115,2]]),'groups':np.array([0,0,1]),
              'base':np.array([.01,.1,.3])}
        rows,scores=row_scores(data,np.arange(2),np.zeros(4))
        np.testing.assert_allclose(1/(1+np.exp(-scores)),data['base'])
        _,adjusted=row_scores(data,np.arange(2),np.array([1.,1.,-99.,-99.]))
        self.assertGreater(adjusted[1],adjusted[0])


if __name__=='__main__':
    unittest.main()
