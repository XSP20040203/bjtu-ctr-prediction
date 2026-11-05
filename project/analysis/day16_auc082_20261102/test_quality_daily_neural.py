import io
import unittest
import numpy as np
import pandas as pd
import torch
from quality_daily_neural import (DailyResidualNet, normalization, encode_categories,
    loss_value, independent_correction)
from rich_userday_activity import correct


class NeuralBoundaries(unittest.TestCase):
    def test_held_values_do_not_change_fitted_normalization_or_vocabulary(self):
        keys=np.array([[20191113,1],[20191114,2],[20191115,3]])
        frame=pd.DataFrame({'value':[1.,3.,99.],'category':[7,8,9]})
        columns,before=normalization(frame,keys,['category'],20191115)
        frame.loc[2]=[1e10,999]
        names,after=normalization(frame,keys,['category'],20191115)
        self.assertEqual(columns,names)
        for name in before:
            np.testing.assert_array_equal(before[name],after[name])

    def test_unknown_held_categories_use_padding_zero(self):
        frame=pd.DataFrame({'value':[1.,3.,9.],'category':[7,8,12345]})
        keys=np.array([[20191113,1],[20191114,2],[20191115,3]])
        _,normal=normalization(frame,keys,['category'],20191115)
        np.testing.assert_array_equal(encode_categories(frame,['category'],normal)[:,0],[1,2,0])

    def test_zero_head_returns_baseline_exactly_once(self):
        torch.set_num_threads(2)
        model=DailyResidualNet(3,[4,5])
        numeric=torch.randn(6,3)
        categories=torch.tensor([[1,2]]*6)
        baseline=torch.randn(6,2)
        for training in [True,False]:
            model.train(training)
            self.assertTrue(torch.equal(model(numeric,categories,baseline),baseline))

    def test_masked_or_nonfinite_targets_are_rejected(self):
        for target in [torch.tensor([-1.,0.]),torch.tensor([float('nan'),1.])]:
            with self.assertRaises(ValueError):
                loss_value(torch.zeros(2,2),target)
        self.assertTrue(torch.isfinite(loss_value(torch.zeros(2,2),torch.tensor([0.,3.]))))

    def test_trained_checkpoint_reloads_exactly(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        model=DailyResidualNet(3,[4])
        numeric=torch.randn(8,3)
        categories=torch.tensor([[1],[2]]*4)
        baseline=torch.randn(8,2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False)
        loss=loss_value(model(numeric,categories,baseline),torch.tensor([0.,2.]*4))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()
        model.eval()
        expected=model(numeric,categories,baseline).detach()
        buffer=io.BytesIO()
        torch.save(model.state_dict(),buffer)
        buffer.seek(0)
        restored=DailyResidualNet(3,[4]).eval()
        restored.load_state_dict(torch.load(buffer,weights_only=True))
        self.assertTrue(torch.equal(expected,restored(numeric,categories,baseline)))
        self.assertLessEqual(float(torch.max(torch.abs(expected-baseline))),4.)

    def test_independent_mass_correction_including_clipping_and_missing_groups(self):
        base=np.array([1e-10,.01,.2,.5])
        groups=np.array([0,0,2,2])
        for objective,response in [('binary',np.array([0.,.5,1.])),('poisson',np.array([0.,1.,1000.]))]:
            for power in [.0625,.125,.5]:
                actual=independent_correction(base,groups,response,objective,power)
                np.testing.assert_allclose(actual,correct(base,response,groups,objective,power),rtol=0,atol=2e-16)


if __name__=='__main__':
    unittest.main()
