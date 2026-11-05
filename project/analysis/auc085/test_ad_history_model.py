"""Behavior checks for candidate-conditioned click/close histories."""
import importlib
import unittest

import torch


class AdHistoryTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("ad_history_model")
        except ModuleNotFoundError:
            self.fail("Candidate-conditioned advertisement history model is pending")

    def test_attention_query_changes_weights_and_padding_cannot_affect_them(self):
        attention = self.module.AdHistoryAttention(2, hidden=2, max_slots=3, mode="query")
        with torch.no_grad():
            attention.score[0].weight.zero_()
            attention.score[0].bias.zero_()
            attention.score[0].weight[0, 6] = 1.
            attention.score[0].weight[1, 7] = 1.
            attention.score[2].weight.fill_(1.)
            attention.score[2].bias.zero_()
        history = torch.tensor([[[1., 0.], [0., 1.], [100., 100.]]])
        mask, match = torch.tensor([[True, True, False]]), torch.zeros(1, 3, dtype=torch.bool)
        first = attention(history, torch.tensor([[1., 0.]]), mask, match)
        second = attention(history, torch.tensor([[0., 1.]]), mask, match)
        torch.testing.assert_close(first, torch.tensor([[.7310586, .2689414]]))
        torch.testing.assert_close(second, torch.tensor([[.2689414, .7310586]]))

    def test_position_mode_can_distinguish_supplied_slots(self):
        attention = self.module.AdHistoryAttention(2, max_slots=3, mode="position")
        with torch.no_grad():
            attention.position_bias.copy_(torch.tensor([2., 0., -1.]))
        history = torch.tensor([[[1., 0.], [0., 1.], [0., 0.]]])
        query = torch.zeros(1, 2)
        mask, match = torch.tensor([[True, True, False]]), torch.zeros(1, 3, dtype=torch.bool)
        first = attention(history, query, mask, match)
        second = attention(history[:, [1, 0, 2]], query, mask, match)
        self.assertGreater(first[0, 0].item(), first[0, 1].item())
        self.assertGreater(second[0, 1].item(), second[0, 0].item())

    def test_empty_history_is_zero_and_finite_with_half_inputs(self):
        attention = self.module.AdHistoryAttention(2, max_slots=3, mode="position")
        history = torch.randn(2, 3, 2).half()
        pooled = attention(history, torch.randn(2, 2).half(), torch.zeros(2, 3, dtype=torch.bool), torch.ones(2, 3, dtype=torch.bool))
        torch.testing.assert_close(pooled, torch.zeros(2, 2).half())
        pooled.float().sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in attention.parameters() if p.grad is not None))

    def make_model(self, mode):
        return self.module.AdHistoryTabM(["user_id", "task_id", "adv_prim_id", "spread_app_id"],
            ["ad_click_list_v001", "ad_close_list_v001"],
            {"user_id": "user_id", "task_id": "ad_task", "adv_prim_id": "advertiser", "spread_app_id": "app",
             "ad_click_list_v001": "ad_task", "ad_close_list_v001": "ad_task"},
            {"user_id": 140, "ad_task": 120, "advertiser": 20, "app": 10},
            numeric_dim=2, embedding_dim=4, n_blocks=2, d_block=16, k=4, dropout=0., unknown_dropout=0.,
            history_mode=mode, initial_ctr=.01)

    def test_neutral_initialization_equals_mean_control_and_all_members_receive_gradients(self):
        torch.manual_seed(42)
        control = self.make_model("mean").eval()
        torch.manual_seed(42)
        adaptive = self.make_model("position").eval()
        categories = torch.tensor([[2, 2, 2, 2], [3, 3, 2, 2]])
        sequences = torch.tensor([[[2, 3, 0, 0, 0], [4, 0, 0, 0, 0]], [[3, 4, 0, 0, 0], [0, 0, 0, 0, 0]]])
        numeric = torch.randn(2, 2)
        expected, logits = control(categories, sequences, numeric), adaptive(categories, sequences, numeric)
        torch.testing.assert_close(logits, expected, rtol=1e-5, atol=1e-6)
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([[1.], [0.]]).expand_as(logits)).backward()
        self.assertTrue((adaptive.head.weight.grad.abs().sum(dim=(1, 2)) > 0).all())
        self.assertGreater(adaptive.history_attention["ad_click_list_v001"].match_values.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in adaptive.parameters() if p.grad is not None))

    def test_checkpoint_reload_preserves_candidate_conditioned_predictions(self):
        model = self.make_model("position").eval()
        with torch.no_grad():
            model.history_attention["ad_click_list_v001"].match_values.normal_()
        categories = torch.tensor([[2, 2, 2, 2], [1, 3, 2, 2]])
        history, numeric = torch.tensor([[[2, 4, 0, 0, 0], [0, 0, 0, 0, 0]]] * 2), torch.zeros(2, 2)
        restored = self.make_model("position").eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(categories, history, numeric), model(categories, history, numeric))


if __name__ == "__main__":
    unittest.main()
