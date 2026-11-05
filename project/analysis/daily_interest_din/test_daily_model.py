"""Behavior checks for masked daily interest and candidate-dependent readout."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from daily_model import DailyInterestDCN


class DailyModelTests(unittest.TestCase):
    def model(self, mode="din", time="both"):
        torch.manual_seed(42)
        fields = ["user_id", "spread_app_id", "app_second_class", "adv_prim_id"]
        return DailyInterestDCN(fields, [], dict(zip(fields, fields)), {field: 8 for field in fields},
              n_topics=8, context_dim=6, interest_mode=mode, time_mode=time,
              numeric_dim=2, embedding_dim=2, hidden_dims=[4, 2], dropout=0,
              unknown_dropout=0, cross_layers=1, group_norm_caps={"user_id": .2})

    def batch(self):
        categories = torch.tensor([[2,2,2,2],[3,3,3,3]])
        sequences = torch.zeros(2,0,5,dtype=torch.long)
        numeric = torch.zeros(2,2)
        weights = torch.zeros(2,2,4,8)
        weights[:,:,:,2:4] = .5
        context = torch.ones(2,2,6)
        return categories,sequences,numeric,weights,context

    def test_zero_initial_residual_and_same_initial_backbone(self):
        control, model = self.model("control"), self.model("din")
        for key,value in control.state_dict().items():
            if key.startswith(("encoder.","cross.","deep.","output.")):
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0,atol=0)
        torch.testing.assert_close(control(*self.batch()),model(*self.batch()),rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA RNG check")
    def test_interest_initialization_preserves_backbone_cuda_random_stream(self):
        torch.cuda.manual_seed_all(42)
        self.model("control")
        control = torch.rand(8, device="cuda").cpu()
        torch.cuda.manual_seed_all(42)
        self.model("din")
        candidate = torch.rand(8, device="cuda").cpu()
        torch.testing.assert_close(control, candidate, rtol=0, atol=0)

    def test_today_inputs_cannot_affect_past_only_readout(self):
        model = self.model(time="past")
        categories,_,_,weights,context = self.batch()
        before = model.interest_readout(categories,weights,context)
        weights[:,1] = torch.rand_like(weights[:,1])
        context[:,1] = 1000
        after = model.interest_readout(categories,weights,context)
        for a,b in zip(before,after):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_empty_topic_channels_are_finite_and_zero(self):
        model = self.model()
        categories,_,_,weights,context = self.batch()
        pooled,query,_ = model.interest_readout(categories,weights*0,context*0)
        self.assertTrue(torch.isfinite(pooled).all())
        torch.testing.assert_close(pooled,torch.zeros_like(pooled))

    def test_candidate_can_change_topic_weighting(self):
        model = self.model()
        with torch.no_grad():
            model.activation[0].weight.normal_()
            model.activation[-1].weight.normal_()
        categories,_,_,weights,context=self.batch()
        weights[1]=weights[0]
        pooled,query,_=model.interest_readout(categories,weights,context)
        self.assertFalse(torch.allclose(query[0],query[1]))
        self.assertFalse(torch.allclose(pooled[0],pooled[1]))

    def test_head_can_start_learning_and_state_roundtrips(self):
        first=self.model().eval()
        first(*self.batch()).sum().backward()
        self.assertGreater(float(first.interest_head[-1].weight.grad.abs().sum()),0)
        with torch.no_grad():first.interest_head[-1].weight.fill_(.1)
        second=self.model().eval();second.load_state_dict(first.state_dict())
        torch.testing.assert_close(first(*self.batch()),second(*self.batch()),rtol=0,atol=0)


if __name__ == "__main__":unittest.main()
