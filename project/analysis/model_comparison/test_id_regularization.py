"""Check targeted masking without altering baseline initialization or padding."""
import importlib
import unittest

import torch

from mainstream_models import DCNV2Model


class IDRegularizationTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("id_regularized_models")
        except ModuleNotFoundError:
            self.fail("Targeted ID regularization is pending")
        self.arguments = (["user_id", "task_id", "age"], ["history"],
                          {"user_id": "user_id", "task_id": "ad_task", "age": "age", "history": "ad_task"},
                          {"user_id": 140, "ad_task": 150, "age": 8})
        self.options = {"numeric_dim": 2, "embedding_dim": 4, "hidden_dims": [8, 4],
                        "dropout": .1, "unknown_dropout": .01, "initial_ctr": .01}

    def test_empty_override_exactly_preserves_seeded_baseline(self):
        torch.manual_seed(42)
        baseline = DCNV2Model(*self.arguments, **self.options)
        torch.manual_seed(42)
        candidate = self.module.IDRegularizedDCN(*self.arguments, group_dropout_rates={}, **self.options)
        self.assertEqual(set(baseline.state_dict()), set(candidate.state_dict()))
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
        categories = torch.tensor([[2, 3, 4], [5, 6, 2]])
        sequences = torch.tensor([[[2, 4, 0]], [[1, 0, 0]]])
        numeric = torch.tensor([[.4, -.2], [0., .1]])
        torch.manual_seed(97)
        expected = baseline(categories, sequences, numeric)
        torch.manual_seed(97)
        actual = candidate(categories, sequences, numeric)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_mask_is_group_specific_and_training_only(self):
        model = self.module.IDRegularizedDCN(*self.arguments, group_dropout_rates={"user_id": 1., "ad_task": 0.},
                                            **self.options)
        values = torch.tensor([0, 1, 2, 3, 139])
        user_table = model.encoder.embeddings["user_id"]
        task_table = model.encoder.embeddings["ad_task"]
        model.train()
        torch.testing.assert_close(model.encoder._indices(values, user_table), torch.tensor([0, 1, 1, 1, 1]))
        torch.testing.assert_close(model.encoder._indices(values, task_table), values)
        model.eval()
        torch.testing.assert_close(model.encoder._indices(values, user_table), values)

    def test_invalid_configuration_fails_loudly(self):
        for rates in [{"user_id": -0.1}, {"user_id": 1.1}, {"missing_group": .2}]:
            with self.subTest(rates=rates), self.assertRaises(ValueError):
                self.module.IDRegularizedDCN(*self.arguments, group_dropout_rates=rates, **self.options)


if __name__ == "__main__":
    unittest.main()
