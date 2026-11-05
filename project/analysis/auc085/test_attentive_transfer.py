import importlib

import torch

import test_cross_domain_model as contracts


class AttentiveTransferTests(contracts.CrossDomainModelTests):
    def setUp(self):
        super().setUp()
        try:
            self.attentive = importlib.import_module("attentive_transfer")
        except ModuleNotFoundError:
            self.fail("Item-conditioned cross-domain attention is pending")

    def make_model(self, schema=None):
        return self.attentive.AttentiveCrossDomainTabM(["user_id", "task_id"],
            ["feed_long_interest", "u_newsCatInterestsST", "feed_click_ca2"],
            {"user_id": "user_id", "task_id": "ad_task", "feed_long_interest": "news_interest",
             "u_newsCatInterestsST": "news_interest", "feed_click_ca2": "news_interest"},
            {"user_id": 140, "ad_task": 120, "news_interest": 12}, source_schema=schema or self.schema,
            numeric_dim=2, embedding_dim=4, n_blocks=2, d_block=16, k=4, dropout=0., profile_dim=16, initial_ctr=.01)

    def test_query_changes_interest_weights_and_padding_is_ignored(self):
        pool = self.attentive.SharedInterestAttention(2, 2)
        with torch.no_grad():
            pool.score[0].weight.zero_()
            pool.score[0].bias.zero_()
            pool.score[0].weight[0, 6] = 1
            pool.score[0].weight[1, 7] = 1
            pool.score[2].weight.fill_(1)
            pool.score[2].bias.zero_()
        history = torch.tensor([[[1., 0.], [0., 1.], [99., 99.]]])
        mask = torch.tensor([[True, True, False]])
        first = pool(history, torch.tensor([[1., 0.]]), mask)
        second = pool(history, torch.tensor([[0., 1.]]), mask)
        torch.testing.assert_close(first, torch.tensor([[.7310586, .2689414]]))
        torch.testing.assert_close(second, first.flip(-1))

    def test_all_padding_is_finite_with_half_precision_inputs(self):
        pool = self.attentive.SharedInterestAttention(2, 4)
        history = torch.zeros(2, 5, 2, dtype=torch.float16, requires_grad=True)
        output = pool(history, torch.ones(2, 2, dtype=torch.float16), torch.zeros(2, 5, dtype=torch.bool))
        torch.testing.assert_close(output, torch.zeros_like(output))
        output.float().sum().backward()
        self.assertTrue(torch.isfinite(history.grad).all())
