"""Behavioral checks for metadata priors and exposure-supported ID residuals."""
import importlib
import unittest

import numpy as np
import torch


class HierarchicalIDTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("hierarchical_id_model")
        except ModuleNotFoundError:
            self.fail("Hierarchical metadata/ID architecture is pending")
        self.fields = ["user_id", "age", "task_id", "adv_id", "adv_prim_id", "spread_app_id"]
        self.sequences = ["ad_click_list_v001", "feed_clicked_topics"]
        self.groups = {"user_id": "user_id", "age": "age", "task_id": "ad_task", "adv_id": "adv_id",
                       "adv_prim_id": "advertiser", "spread_app_id": "promoted_app",
                       "ad_click_list_v001": "ad_task", "feed_clicked_topics": "news_interest"}
        self.sizes = {"user_id": 140, "age": 10, "ad_task": 120, "adv_id": 130,
                      "advertiser": 20, "promoted_app": 10, "news_interest": 30}

    def make_model(self, mode):
        return self.module.HierarchicalIDTabM(self.fields, self.sequences, self.groups, self.sizes,
            numeric_dim=2, embedding_dim=4, n_blocks=2, d_block=16, k=4,
            dropout=0., unknown_dropout=0., initial_ctr=.01, group_norm_caps={},
            hierarchy_mode=mode, support_temperature=100.)

    def inputs(self):
        return torch.tensor([[2, 2, 2, 2, 2, 2], [3, 2, 2, 2, 2, 2]]), torch.zeros(2, 2, 5, dtype=torch.long), torch.zeros(2, 2)

    def test_support_fitting_uses_only_training_inputs_and_never_counts_unknown_or_padding(self):
        category = np.array([[2, 2], [2, 3], [3, 3], [4, 4]], dtype=np.int32)
        manifest = {"categorical_fields": ["user_id", "task_id"], "group_for_field": {"user_id": "user_id", "task_id": "ad_task"},
                    "vocabulary_sizes": {"user_id": 6, "ad_task": 6}}
        first = self.module.fit_exposure_support(category, np.array([0, 1]), manifest)
        category[2:] = 2
        second = self.module.fit_exposure_support(category, np.array([0, 1]), manifest)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
            np.testing.assert_array_equal(first[name][:2], [0, 0])
        np.testing.assert_array_equal(first["user_id"], [0, 0, 2, 0, 0, 0])
        np.testing.assert_array_equal(first["task_id"], [0, 0, 1, 1, 0, 0])

    def test_neutral_metadata_mode_matches_raw_control_at_initialization(self):
        torch.manual_seed(42)
        raw = self.make_model("raw").eval()
        torch.manual_seed(42)
        metadata = self.make_model("metadata").eval()
        torch.testing.assert_close(raw(*self.inputs()), metadata(*self.inputs()), rtol=0, atol=0)

    def test_rare_id_effect_is_smaller_than_well_supported_id_effect(self):
        model = self.make_model("shrink").eval()
        with torch.no_grad():
            model.support_counts_user_id[2] = 1
            model.support_counts_user_id[3] = 1000
            model.encoder.embeddings["user_id"].weight[2:4].fill_(.1)
        category, history, _ = self.inputs()
        before = model.encode_fields(category, history)[:, 0]
        with torch.no_grad():
            model.encoder.embeddings["user_id"].weight[2:4].add_(1.)
        delta = (model.encode_fields(category, history)[:, 0] - before).norm(dim=-1)
        self.assertLess(delta[0].item(), delta[1].item() / 10)

    def test_shared_mode_blocks_user_id_memorization_but_metadata_can_learn(self):
        torch.manual_seed(42)
        model = self.make_model("shared")
        logits = model(*self.inputs())
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([[1.], [0.]]).expand_as(logits)).backward()
        gradient = model.encoder.embeddings["user_id"].weight.grad
        self.assertTrue(gradient is None or not gradient.abs().sum().item())
        self.assertGreater(model.prior_encoders["user_id"][-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue((model.head.weight.grad.abs().sum(dim=(1, 2)) > 0).all())

    def test_cold_or_unknown_ids_fall_back_to_the_same_metadata_prior(self):
        model = self.make_model("shrink").eval()
        category, history, numeric = self.inputs()
        category[:, 0] = torch.tensor([1, 3])
        with torch.no_grad():
            model.encoder.embeddings["user_id"].weight[1].fill_(100.)
            model.encoder.embeddings["user_id"].weight[3].fill_(-100.)
            model.prior_encoders["user_id"][-1].bias.fill_(.3)
        fields = model.encode_fields(category, history)
        torch.testing.assert_close(fields[0, 0], fields[1, 0])
        self.assertTrue(torch.isfinite(model(category, history, numeric)).all())

    def test_checkpoint_restores_count_buffers_and_adaptive_predictions(self):
        model = self.make_model("shrink").eval()
        with torch.no_grad():
            model.support_counts_user_id[2:4] = torch.tensor([5., 100.])
            model.prior_encoders["task_id"][-1].weight.normal_(std=.05)
        restored = self.make_model("shrink").eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(*self.inputs()), model(*self.inputs()), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
