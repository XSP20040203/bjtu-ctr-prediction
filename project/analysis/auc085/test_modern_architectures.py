import importlib
import unittest

import numpy as np
import torch


class ModernArchitectureTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("modern_architectures")
        except ModuleNotFoundError:
            self.fail("Modern architecture adapter is pending")

    def test_numeric_bins_handle_constant_features_and_use_only_given_rows(self):
        bins = self.module.fit_numeric_bins(np.array([[0., 7.], [1., 7.], [2., 7.]], dtype=np.float32), 4)
        self.assertEqual(bins[0][0], 0.)
        self.assertEqual(bins[0][-1], 2.)
        self.assertLess(bins[1][0], 7.)
        self.assertGreater(bins[1][-1], 7.)
        self.assertTrue(all(np.all(np.diff(edges) > 0) for edges in bins))

    def test_prediction_averages_member_probabilities(self):
        logits = torch.tensor([[0., 2.], [-2., 2.]])
        torch.testing.assert_close(self.module.member_probability(logits), torch.tensor([.6903985, .5]))

    def test_explicit_empty_id_caps_allow_embedding_magnitude_to_affect_output(self):
        torch.manual_seed(42)
        model = self.module.TabMFieldModel(["user_id"], ["history"],
                    {"user_id": "user_id", "history": "user_id"}, {"user_id": 140},
                    numeric_dim=1, embedding_dim=4, n_blocks=2, d_block=16, k=4,
                    dropout=0., group_norm_caps={}, initial_ctr=.01).eval()
        categories, history, numeric = torch.tensor([[2]]), torch.zeros(1, 1, 5, dtype=torch.long), torch.zeros(1, 1)
        with torch.no_grad():
            model.encoder.embeddings["user_id"].weight[2].copy_(torch.tensor([1., 2., 3., 4.]))
            first = model(categories, history, numeric)
            model.encoder.embeddings["user_id"].weight[2].mul_(2)
            second = model(categories, history, numeric)
        self.assertFalse(torch.allclose(first, second))

    def test_tabm_adapter_supports_empty_history_and_independent_member_gradients(self):
        torch.manual_seed(42)
        model = self.module.TabMFieldModel(["user_id", "task_id"], ["history"],
                    {"user_id": "user_id", "task_id": "ad_task", "history": "ad_task"},
                    {"user_id": 140, "ad_task": 120}, numeric_dim=2, embedding_dim=4,
                    n_blocks=2, d_block=16, k=4, dropout=0., initial_ctr=.01,
                    numeric_bin_edges=[[-1., 0., 1.], [6., 8.]], numeric_embedding_dim=4)
        categories = torch.tensor([[2, 3], [1, 4]])
        history = torch.tensor([[[2, 0, 3]], [[0, 0, 0]]])
        numeric = torch.tensor([[.4, 7.], [-.5, 7.]])
        model.eval()
        logits = model(categories, history, numeric)
        self.assertEqual(tuple(logits.shape), (2, 4))
        torch.testing.assert_close(logits, model(categories, history.flip(-1), numeric))
        self.assertTrue(torch.isfinite(logits).all())
        model.train()
        logits = model(categories, history, numeric)
        target = torch.tensor([[1.], [0.]]).expand_as(logits)
        torch.nn.functional.binary_cross_entropy_with_logits(logits, target).backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())
        self.assertTrue((model.head.weight.grad.abs().sum(dim=(1, 2)) > 0).all())

    def test_clipped_numeric_encoder_saturates_beyond_training_bin_edges(self):
        self.assertTrue(hasattr(self.module, "ClippedTabMFieldModel"), "Bounded numerical encoder is pending")
        torch.manual_seed(42)
        model = self.module.ClippedTabMFieldModel(["user_id", "task_id"], ["history"],
                    {"user_id": "user_id", "task_id": "ad_task", "history": "ad_task"},
                    {"user_id": 140, "ad_task": 120}, numeric_dim=2, embedding_dim=4,
                    n_blocks=2, d_block=16, k=4, dropout=0., initial_ctr=.01,
                    numeric_bin_edges=[[-1., 0., 1.], [6., 7., 8.]], numeric_embedding_dim=4).eval()
        categories = torch.tensor([[2, 3], [2, 3]])
        sequences = torch.zeros(2, 1, 5, dtype=torch.long)
        output = model(categories, sequences, torch.tensor([[1., 8.], [100., 700.]]))
        torch.testing.assert_close(output[0], output[1])

    def test_transformer_has_finite_forward_and_gradients_with_empty_histories(self):
        model = self.module.FieldTransformerModel(["user_id", "task_id", "adv_id"], ["history"],
                    {"user_id": "user_id", "task_id": "ad_task", "adv_id": "adv_id", "history": "ad_task"},
                    {"user_id": 140, "ad_task": 120, "adv_id": 130}, numeric_dim=2,
                    embedding_dim=4, d_token=8, n_layers=1, n_heads=2, dropout=0.)
        logits = model(torch.tensor([[2, 3, 4], [1, 4, 3]]), torch.zeros(2, 1, 5, dtype=torch.long), torch.zeros(2, 2))
        self.assertEqual(tuple(logits.shape), (2, 1))
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([[1.], [0.]])).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == "__main__":
    unittest.main()
