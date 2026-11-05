"""Checks for the matched initialization and safe interest pooling."""
import unittest

from features import read, cache_dir
from model import build
import torch


class InterestTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(3)
        self.meta = read(cache_dir(20191115) / "manifest.json")

    def models(self):
        torch.manual_seed(42)
        control = build(self.meta, "control", .0084)
        torch.manual_seed(42)
        attention = build(self.meta, "attention", .0084)
        return control, attention

    def inputs(self):
        cats = torch.full((4, 27), 2, dtype=torch.long)
        seqs = torch.zeros((4, 12, 5), dtype=torch.long)
        seqs[1, :, 0] = 2
        seqs[2, :, :3] = torch.tensor([2, 3, 4])
        seqs[3, :, :] = torch.tensor([2, 3, 4, 5, 6])
        return cats, seqs, torch.zeros(4, 166)

    def test_backbone_and_initial_predictions_match(self):
        control, attention = self.models()
        for name, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, attention.state_dict()[name]), name)
        control.eval()
        attention.eval()
        torch.testing.assert_close(control(*self.inputs()), attention(*self.inputs()), rtol=0, atol=1e-6)

    def test_dropout_stream_matches(self):
        control, attention = self.models()
        control.train()
        attention.train()
        torch.manual_seed(100)
        a = control(*self.inputs())
        state = torch.random.get_rng_state().clone()
        torch.manual_seed(100)
        b = attention(*self.inputs())
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)

    def test_padding_and_singleton_invariance_after_learning(self):
        control, attention = self.models()
        control.eval()
        attention.eval()
        for scorer in attention.encoder.interest_scores.values():
            torch.nn.init.normal_(scorer[-1].weight)
        cats, seqs, numeric = self.inputs()
        a = control(cats[:2], seqs[:2], numeric[:2])
        b = attention(cats[:2], seqs[:2], numeric[:2])
        self.assertTrue(torch.isfinite(b).all())
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)

    def test_padding_order_does_not_change_pooling(self):
        _, attention = self.models()
        attention.eval()
        for scorer in attention.encoder.interest_scores.values():
            torch.nn.init.normal_(scorer[-1].weight)
        cats, seqs, numeric = self.inputs()
        torch.testing.assert_close(attention(cats, seqs, numeric),
                                   attention(cats, seqs.flip(-1), numeric), rtol=0, atol=1e-6)

    def test_attention_score_receives_gradient(self):
        _, attention = self.models()
        attention.eval()
        attention(*self.inputs()).sum().backward()
        gradients = [scorer[-1].weight.grad.norm().item()
                     for scorer in attention.encoder.interest_scores.values()]
        self.assertGreater(sum(gradients), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_dropout_stream_matches(self):
        control, attention = [model.cuda().train() for model in self.models()]
        inputs = tuple(value.cuda() for value in self.inputs())
        torch.cuda.manual_seed_all(100)
        a = control(*inputs)
        state = torch.cuda.get_rng_state().clone()
        torch.cuda.manual_seed_all(100)
        b = attention(*inputs)
        self.assertTrue(torch.equal(state, torch.cuda.get_rng_state()))
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
