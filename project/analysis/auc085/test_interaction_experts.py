import importlib
import unittest

import torch


class InteractionExpertTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("interaction_architectures")
        except ModuleNotFoundError:
            self.fail("Explicit high-order interaction expert is pending")

    def test_compressed_interactions_form_second_and_third_order_products(self):
        network = self.module.CompressedInteraction(2, [1, 1])
        with torch.no_grad():
            for weight in network.weights:
                weight.zero_()
            for bias in network.biases:
                bias.zero_()
            network.weights[0][0, 0, 1] = 1
            network.weights[1][0, 1, 0] = 1
        result = network(torch.tensor([[[1., 2.], [3., 4.]]]))
        torch.testing.assert_close(result, torch.tensor([[11., 41.]]))

    def test_mixture_returns_member_logits_and_backpropagates_finitely(self):
        torch.manual_seed(42)
        model = self.module.InteractionExpertTabM(["user_id", "task_id"], ["history"],
            {"user_id": "user_id", "task_id": "ad_task", "history": "ad_task"},
            {"user_id": 140, "ad_task": 120}, numeric_dim=2, embedding_dim=4,
            n_blocks=2, d_block=16, k=4, dropout=0., cin_widths=[4, 4], initial_ctr=.01)
        logits = model(torch.tensor([[2, 3], [1, 4]]), torch.zeros(2, 1, 5, dtype=torch.long), torch.zeros(2, 2))
        self.assertEqual(tuple(logits.shape), (2, 4))
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([[1.], [0.]]).expand_as(logits)).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == "__main__":
    unittest.main()
