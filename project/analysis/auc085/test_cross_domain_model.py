import importlib
import unittest

import torch


class CrossDomainModelTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("cross_domain_model")
        except ModuleNotFoundError:
            self.fail("Cross-domain gated TabM is pending")
        self.schema = {
            "categorical_fields": ["u_userId", "i_cat", "i_docId"],
            "sequence_fields": ["u_newsCatInterests", "u_newsCatInterestsST", "u_click_ca2_news", "u_newsCatDislike"],
            "numeric_fields": ["i_dislikeTimes", "i_upTimes"],
            "group_for_field": {"u_userId": "user_id", "i_cat": "news_interest", "i_docId": "source_i_docId",
                "u_newsCatInterests": "news_interest", "u_newsCatInterestsST": "news_interest",
                "u_click_ca2_news": "news_interest", "u_newsCatDislike": "news_interest"},
            "vocabulary_sizes": {"user_id": 140, "news_interest": 12, "source_i_docId": 9}, "positive_rate": .1}

    def make_model(self, schema=None):
        return self.module.CrossDomainGatedTabM(["user_id", "task_id"],
            ["feed_long_interest", "u_newsCatInterestsST", "feed_click_ca2"],
            {"user_id": "user_id", "task_id": "ad_task", "feed_long_interest": "news_interest",
             "u_newsCatInterestsST": "news_interest", "feed_click_ca2": "news_interest"},
            {"user_id": 140, "ad_task": 120, "news_interest": 12}, source_schema=schema or self.schema,
            numeric_dim=2, embedding_dim=4, n_blocks=2, d_block=16, k=4, dropout=0., profile_dim=16, initial_ctr=.01)

    def test_source_loss_updates_shared_interests_but_not_ad_task_table(self):
        torch.manual_seed(42)
        model = self.make_model()
        categories = torch.tensor([[2, 3, 2], [3, 4, 3]])
        sequences = torch.tensor([[[2, 3, 0], [3, 0, 0], [4, 0, 0], [0, 0, 0]],
                                  [[3, 4, 0], [4, 0, 0], [2, 0, 0], [0, 0, 0]]])
        logits = model.forward_source(categories, sequences, torch.zeros(2, 2))
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0.])).backward()
        self.assertGreater(model.encoder.embeddings["news_interest"].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.encoder.embeddings["ad_task"].weight.grad)
        self.assertGreater(model.profile_tower[0].weight.grad.abs().sum().item(), 0)

    def test_ad_forward_and_reload_preserve_ensemble_predictions(self):
        torch.manual_seed(42)
        model = self.make_model().eval()
        categories = torch.tensor([[2, 3], [3, 4]])
        sequences = torch.zeros(2, 3, 5, dtype=torch.long)
        numeric = torch.zeros(2, 2)
        first = model(categories, sequences, numeric)
        self.assertEqual(tuple(first.shape), (2, 4))
        self.assertTrue(torch.isfinite(first).all())
        restored = self.make_model().eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(categories, sequences, numeric), first)

    def test_source_post_click_inputs_are_rejected(self):
        invalid = {**self.schema, "numeric_fields": ["pro", "i_upTimes"]}
        with self.assertRaises(ValueError):
            self.make_model(invalid)


if __name__ == "__main__":
    unittest.main()
