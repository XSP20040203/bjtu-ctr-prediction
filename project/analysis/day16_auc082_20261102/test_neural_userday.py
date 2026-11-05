import unittest
import numpy as np
import torch
from neural_userday import preprocess,transform,DailyNet,loss_value


class NeuralDailyTests(unittest.TestCase):
    def test_normalization_and_new_users_respect_cutoff(self):
        keys=np.array([[20191113,1],[20191114,2],[20191115,3],[20191116,4]])
        x=np.array([[0.,2.],[2.,4.],[99.,99.],[999.,999.]],dtype="float32")
        norm=preprocess(x,keys,20191115)
        np.testing.assert_array_equal(norm['center'],[1.,3.])
        values,users=transform(x,keys,norm)
        np.testing.assert_array_equal(users,[1,2,0,0])
        self.assertTrue(np.isfinite(values).all())

    def test_two_response_heads_receive_finite_gradients(self):
        torch.manual_seed(42)
        model=DailyNet(4,3,4)
        out=model(torch.ones(3,4),torch.tensor([0,1,2]))
        self.assertEqual(out.shape,(3,2))
        loss=loss_value(out,torch.tensor([0.,1.,3.]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.network[-1].weight.grad).all())
        self.assertTrue((model.network[-1].weight.grad.abs().sum(1)>0).all())


if __name__=="__main__":
    unittest.main()
