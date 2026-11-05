import importlib
import unittest

import torch
from torch.nn import functional as F


class RankingObjectiveTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("ranking_objective")
        except ModuleNotFoundError:
            self.fail("Optional member-wise ranking objective is pending")

    def test_every_member_ranks_positives_above_negatives(self):
        logits = torch.tensor([[2., 1.], [-1., 0.], [0., -2.]], requires_grad=True)
        label = torch.tensor([1., 0., 0.])
        loss = self.module.member_ranking_loss(logits, label, max_negatives=128)
        expected = F.softplus(logits[1:] - logits[0]).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue((logits.grad[0] < 0).all())
        self.assertTrue((logits.grad[1:] > 0).all())

    def test_single_class_batches_return_differentiable_zero(self):
        for value in [0., 1.]:
            logits = torch.randn(5, 4, requires_grad=True)
            loss = self.module.member_ranking_loss(logits, torch.full((5,), value))
            self.assertEqual(loss.item(), 0.)
            loss.backward()
            torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_negative_cap_is_deterministic_and_common_logit_shift_is_irrelevant(self):
        logits = torch.tensor([[0., 1.], [-1., 0.], [100., 100.]])
        labels = torch.tensor([1., 0., 0.])
        expected = F.softplus(logits[1] - logits[0]).mean()
        torch.testing.assert_close(self.module.member_ranking_loss(logits, labels, max_negatives=1), expected)
        torch.testing.assert_close(self.module.member_ranking_loss(logits + 100, labels, max_negatives=1), expected)


if __name__ == "__main__":
    unittest.main()
