"""Behavioral tests for target attention, padding safety and source gating."""
import importlib
import unittest

import torch


class V2Tests(unittest.TestCase):
    def setUp(self):
        try:
            self.v2 = importlib.import_module("v2_models")
        except ModuleNotFoundError:
            self.fail("V2 model implementation is pending")
        self.cats = ["task_id", "adv_prim_id", "spread_app_id"]
        self.seqs = ["ad_click_list_v001", "ad_click_list_v002", "ad_click_list_v003"]
        self.groups = dict(zip(self.cats + self.seqs, ["task", "advertiser", "app"] * 2))
        self.sizes = {"task": 12, "advertiser": 10, "app": 9}

    def test_attention_depends_on_query_and_ignores_padding(self):
        attention = self.v2.TargetAttention(2, 2)
        with torch.no_grad():
            attention.score[0].weight.zero_()
            attention.score[0].bias.zero_()
            attention.score[0].weight[0, 6] = 1
            attention.score[0].weight[1, 7] = 1
            attention.score[2].weight.fill_(1)
            attention.score[2].bias.zero_()
        history = torch.tensor([[[1., 0.], [0., 1.], [99., 99.]]])
        mask = torch.tensor([[True, True, False]])
        first = attention(history, torch.tensor([[1., 0.]]), mask)
        second = attention(history, torch.tensor([[0., 1.]]), mask)
        torch.testing.assert_close(first, torch.tensor([[.7310586, .2689414]]))
        torch.testing.assert_close(second, torch.tensor([[.2689414, .7310586]]))
        empty = attention(history, torch.tensor([[1., 0.]]), torch.zeros_like(mask))
        torch.testing.assert_close(empty, torch.zeros(1, 2))

    def test_gate_can_suppress_or_pass_source_representation(self):
        model = self.v2.OptimizedCTRModel(self.cats, self.seqs, self.groups, self.sizes, attention=False, source_gate=True, dropout=0, unknown_dropout=0)
        model.eval()
        with torch.no_grad():
            for module in model.target_network.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.zero_()
                    module.bias.zero_()
            for module in model.source_network.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.zero_()
                    module.bias.fill_(2)
            model.source_gate.weight.zero_()
            model.prediction_head[0].weight.zero_()
            model.prediction_head[0].bias.zero_()
            model.prediction_head[0].weight[0, 64] = 1
            model.prediction_head[2].weight.zero_()
            model.prediction_head[2].weight[0, 0] = 1
            model.prediction_head[2].bias.zero_()
        inputs = (torch.tensor([[2, 3, 4]]), torch.zeros(1, 3, 5, dtype=torch.long), torch.zeros(1, 20))
        for bias, expected in [(-20., 0.), (0., 1.), (20., 2.)]:
            with torch.no_grad():
                model.source_gate.bias.fill_(bias)
            self.assertAlmostEqual(model(*inputs).item(), expected, places=6)

    def test_attention_and_gate_models_are_finite_and_unordered(self):
        inputs = (torch.tensor([[2, 3, 4], [3, 2, 5]]),
                  torch.tensor([[[2, 3, 0, 0, 0]] * 3, [[0, 0, 0, 0, 0]] * 3]),
                  torch.randn(2, 20))
        for gated in [False, True]:
            torch.manual_seed(42)
            model = self.v2.OptimizedCTRModel(self.cats, self.seqs, self.groups, self.sizes, attention=True, source_gate=gated, dropout=0, unknown_dropout=0)
            model.eval()
            logits = model(*inputs)
            reversed_logits = model(inputs[0], inputs[1].flip(-1), inputs[2])
            torch.testing.assert_close(logits, reversed_logits)
            self.assertEqual(tuple(logits.shape), (2,))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.]))
            loss.backward()
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))
            for table in model.embeddings.values():
                if table.weight.grad is not None:
                    torch.testing.assert_close(table.weight.grad[0], torch.zeros(8))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
