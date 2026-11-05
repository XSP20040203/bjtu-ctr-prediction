import importlib
import unittest

import torch


class GraphInterestModelTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("graph_interest_model")
        except ModuleNotFoundError:
            self.fail("Graph-interest TabM adapter is pending")

    def make_model(self, protected):
        model = self.module.GraphInterestTabM(["user_id", "task_id", "spread_app_id", "adv_prim_id", "app_second_class"],
            ["history"], {"user_id": "user_id", "task_id": "ad_task", "spread_app_id": "app", "adv_prim_id": "advertiser",
                          "app_second_class": "app_class", "history": "ad_task"},
            {"user_id": 140, "ad_task": 120, "app": 10, "advertiser": 20, "app_class": 8},
            numeric_dim=2, embedding_dim=4, n_blocks=2, d_block=16, k=4, dropout=0., graph_dim=4,
            protected_source=protected, initial_ctr=.01)
        with torch.no_grad():
            model.source_user_vectors[2:4].copy_(torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]]))
            model.source_positive_count[2:4] = 10
            model.graph_score_scale.fill_(1.)
        return model.eval()

    def test_protected_source_blocks_ad_gradients_but_projection_learns(self):
        torch.manual_seed(42)
        model = self.make_model(True)
        output = model(torch.tensor([[2, 2, 2, 2, 2], [3, 2, 2, 2, 2]]), torch.zeros(2, 1, 5, dtype=torch.long), torch.zeros(2, 2))
        torch.nn.functional.binary_cross_entropy_with_logits(output, torch.tensor([[1.], [0.]]).expand_as(output)).backward()
        self.assertIsNone(model.source_user_vectors.grad)
        self.assertIsNone(model.encoder.embeddings["user_id"].weight.grad)
        self.assertGreater(model.user_projection.weight.grad.abs().sum().item(), 0)

    def test_finetuning_can_update_the_source_user_vectors(self):
        torch.manual_seed(42)
        model = self.make_model(False)
        output = model(torch.tensor([[2, 2, 2, 2, 2], [3, 2, 2, 2, 2]]), torch.zeros(2, 1, 5, dtype=torch.long), torch.zeros(2, 2))
        torch.nn.functional.binary_cross_entropy_with_logits(output, torch.tensor([[1.], [0.]]).expand_as(output)).backward()
        self.assertGreater(model.source_user_vectors.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(model.source_user_vectors.grad).all())

    def test_missing_source_representation_is_finite_and_reloadable(self):
        model = self.make_model(True)
        categories = torch.tensor([[0, 2, 2, 2, 2], [1, 2, 2, 2, 2]])
        history, numeric = torch.zeros(2, 1, 5, dtype=torch.long), torch.zeros(2, 2)
        first = model(categories, history, numeric)
        self.assertTrue(torch.isfinite(first).all())
        restored = self.make_model(True)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(categories, history, numeric), first)


if __name__ == "__main__":
    unittest.main()
