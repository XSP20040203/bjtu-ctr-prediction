import copy
import unittest
import torch
from sam_step import sam_step


class SamStepContracts(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_cuda_zero_radius_preserves_dropout_rng_and_update(self):
        torch.manual_seed(71)
        torch.cuda.manual_seed_all(71)
        model = torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.3),torch.nn.Linear(4,1)).cuda()
        expected = copy.deepcopy(model)
        x,y = torch.randn(7,3,device="cuda"),torch.randn(7,1,device="cuda")
        actual_opt = torch.optim.AdamW(model.parameters(),lr=.01)
        expected_opt = torch.optim.AdamW(expected.parameters(),lr=.01)
        state = torch.cuda.get_rng_state_all()
        torch.nn.functional.mse_loss(expected(x),y).backward()
        expected_opt.step()
        after = torch.cuda.get_rng_state_all()
        torch.cuda.set_rng_state_all(state)
        sam_step(model,actual_opt,lambda:torch.nn.functional.mse_loss(model(x),y),0.)
        for actual,target in zip(model.parameters(),expected.parameters()):
            torch.testing.assert_close(actual,target,rtol=0,atol=0)
        for actual,target in zip(torch.cuda.get_rng_state_all(),after):
            torch.testing.assert_close(actual,target,rtol=0,atol=0)

    def test_zero_radius_matches_one_pass_with_dropout_and_rng(self):
        torch.manual_seed(41)
        model = torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Dropout(.4),torch.nn.Linear(5,1)).double()
        expected = copy.deepcopy(model)
        x,y = torch.randn(9,3,dtype=torch.float64),torch.randn(9,1,dtype=torch.float64)
        opt = torch.optim.AdamW(model.parameters(),lr=.01)
        ordinary = torch.optim.AdamW(expected.parameters(),lr=.01)
        state = torch.get_rng_state()
        ordinary.zero_grad()
        torch.nn.functional.mse_loss(expected(x),y).backward()
        ordinary.step()
        next_rng = torch.get_rng_state()
        torch.set_rng_state(state)
        sam_step(model,opt,lambda:torch.nn.functional.mse_loss(model(x),y),0.)
        for actual,target in zip(model.parameters(),expected.parameters()):
            torch.testing.assert_close(actual,target,rtol=0,atol=0)
        torch.testing.assert_close(torch.get_rng_state(),next_rng,rtol=0,atol=0)

    def test_second_gradient_is_applied_at_original_parameters(self):
        model = torch.nn.Linear(1,1,bias=False).double()
        with torch.no_grad(): model.weight.fill_(2.)
        optimizer = torch.optim.SGD(model.parameters(),lr=.1)
        info = sam_step(model,optimizer,lambda:(model.weight**2).sum()/2,.25)
        self.assertAlmostEqual(float(model.weight),2.-.1*2.25,places=12)
        self.assertAlmostEqual(info["original_loss"],2.)
        self.assertAlmostEqual(info["perturbed_loss"],2.25**2/2)

    def test_failed_second_pass_restores_original_parameters(self):
        model = torch.nn.Linear(1,1,bias=False).double()
        initial = model.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(),lr=.1)
        calls = []
        def loss():
            calls.append(1)
            if len(calls)==2: raise RuntimeError("intentional")
            return (model.weight**2).sum()
        with self.assertRaises(RuntimeError): sam_step(model,optimizer,loss,.1)
        torch.testing.assert_close(model.weight,initial,rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()
