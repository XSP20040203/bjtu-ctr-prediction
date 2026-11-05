import unittest
import numpy as np
from pairwise_group_boost import PairObjective, make_objective


class PairTests(unittest.TestCase):
    def test_gradient_and_diagonal_hessian(self):
        obj = PairObjective(np.array([0,0,1]), np.array([1,2,2]), np.array([.5,-.2,.8]), 3, .5)
        x = np.array([.1,-.3,.2])
        grad, hess = obj(x, None)
        eps = 1e-4
        for i in range(3):
            shift = np.eye(3)[i]*eps
            numeric_grad = (obj.value(x+shift)-obj.value(x-shift))/(2*eps)
            numeric_hess = (obj.value(x+shift)-2*obj.value(x)+obj.value(x-shift))/eps**2
            self.assertAlmostEqual(grad[i], numeric_grad, places=6)
            self.assertAlmostEqual(hess[i], numeric_hess, places=6)

    def test_sampling_restricted_to_train_groups(self):
        data = {"keys":np.arange(6).reshape(3,2), "groups":np.array([0,0,1,1,2,2]),
                "y":np.array([1,0,0,1,0,1]), "base":np.array([.1,.2,.3,.4,.5,.6])}
        a = make_objective(data,np.array([0,1]),1.,42)
        changed = dict(data, y=np.array([1,0,0,1,1,0]))
        b = make_objective(changed,np.array([0,1]),1.,42)
        np.testing.assert_array_equal(a.a,b.a)
        np.testing.assert_array_equal(a.b,b.b)
        np.testing.assert_array_equal(a.margin,b.margin)
        self.assertTrue(np.all(a.a != a.b))
        self.assertTrue(np.all(a.a < 2))
        self.assertTrue(np.all(a.b < 2))


if __name__ == "__main__":
    unittest.main()
