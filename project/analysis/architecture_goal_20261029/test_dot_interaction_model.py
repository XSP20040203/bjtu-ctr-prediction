"""Synthetic CPU checks for DLRM dots, masking, caps, gradients and runner API."""
import io
import math
import unittest
from unittest import mock

import torch
from torch import nn

try:
    from .dot_interaction_model import DotInteractionCTR, PairwiseDotInteraction
except ImportError:
    from dot_interaction_model import DotInteractionCTR, PairwiseDotInteraction


class DotInteractionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        self.cats = ["user_id", "task_id", "adv_id", "adv_prim_id", "spread_app_id", "age"]
        self.seqs = ["ad_click_list_v001", "ad_click_list_v002", "ad_click_list_v003", "news"]
        self.groups = dict(zip(self.cats + self.seqs,
                               ["user_id", "ad_task", "adv_id", "advertiser", "app", "age",
                                "ad_task", "advertiser", "app", "news"]))
        self.sizes = {"user_id": 140, "ad_task": 150, "adv_id": 130,
                      "advertiser": 20, "app": 18, "age": 10, "news": 12}
        self.categories = torch.tensor([[2, 2, 3, 4, 5, 6], [3, 3, 4, 5, 6, 7],
                                        [4, 4, 5, 6, 7, 8]])
        self.sequences = torch.tensor([
            [[8, 9, 0, 0, 0], [9, 10, 0, 0, 0], [10, 11, 0, 0, 0], [2, 3, 0, 0, 0]],
            [[0, 0, 0, 0, 0]] * 4,
            [[2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]],
        ])
        self.numeric = torch.randn(3, 4)

    def model(self, **kwargs):
        options = dict(numeric_dim=4, initial_ctr=.013, dropout=0., unknown_dropout=0.)
        options.update(kwargs)
        return DotInteractionCTR(self.cats, self.seqs, self.groups, self.sizes, **options)

    def test_exact_lower_triangle_dots_are_separate_not_fm_sum(self):
        fields = torch.tensor([[[1., 2.], [3., 4.], [5., 6.], [7., 8.]]], requires_grad=True)
        interaction = PairwiseDotInteraction(4)
        expected = torch.stack([(fields[:, i] * fields[:, j]).sum(-1)
                                for i in range(4) for j in range(i)], dim=1) / math.sqrt(2)
        actual = interaction(fields)
        self.assertEqual(tuple(actual.shape), (1, 6))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        (actual * torch.arange(1, 7)).sum().backward()
        self.assertTrue(torch.isfinite(fields.grad).all())
        self.assertGreater(fields.grad.abs().sum().item(), 0.)
        with self.assertRaises(ValueError):
            interaction(torch.ones(2, 3, 8))

    def test_finite_prior_loss_and_gradients_at_both_embedding_widths(self):
        for dim in (8, 16):
            with self.subTest(embedding_dim=dim):
                model = self.model(embedding_dim=dim)
                numeric = self.numeric.clone().requires_grad_()
                logits = model(self.categories, self.sequences, numeric)
                self.assertEqual(tuple(logits.shape), (3,))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertLess((logits - math.log(.013 / .987)).abs().max().item(), .1)
                nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.])).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(numeric.grad.abs().sum().item(), 0.)
                self.assertGreater(model.bottom[0].weight.grad.abs().sum().item(), 0.)
                self.assertGreater(model.numeric_linear.weight.grad.abs().sum().item(), 0.)
                self.assertGreater(model.field_scale_logits.grad.abs().sum().item(), 0.)
                for table in model.encoder.embeddings.values():
                    torch.testing.assert_close(table.weight.grad[0], torch.zeros(dim))

    def test_shared_candidate_history_table_and_attention(self):
        model = self.model()
        self.assertEqual(len(model.encoder.embeddings), len(self.sizes))
        self.assertEqual(set(model.encoder.history_attention), set(self.seqs[:3]))
        fields, _, _ = model.encoder(self.categories, self.sequences)
        table = model.encoder.embeddings["ad_task"].weight
        category_grad = torch.autograd.grad(fields[0, 1].sum(), table, retain_graph=True)[0]
        history_grad = torch.autograd.grad(fields[0, len(self.cats)].sum(), table)[0]
        self.assertGreater(category_grad[2].abs().sum().item(), 0.)
        self.assertGreater(history_grad[8].abs().sum().item(), 0.)
        torch.testing.assert_close(history_grad[0], torch.zeros(16))
        no_attention = self.model(target_attention=False).eval()
        self.assertEqual(len(no_attention.encoder.history_attention), 0)
        self.assertTrue(torch.isfinite(no_attention(self.categories, self.sequences, self.numeric)).all())

    def test_caps_survive_rms_and_extreme_field_gains_without_table_mutation(self):
        model = self.model().eval()
        with torch.no_grad():
            model.field_scale_logits.fill_(30.)
            for group in ("user_id", "ad_task", "adv_id"):
                model.encoder.embeddings[group].weight.normal_(std=20.)
        before = model.encoder.embeddings["ad_task"].weight.detach().clone()
        tokens, _ = model.field_tokens(self.categories, self.sequences, self.numeric)
        for index in (0, 1, 2, len(self.cats)):
            self.assertLessEqual(tokens[:, index].norm(dim=-1).max().item(), .200001)
        self.assertLessEqual((2 * model.field_scale_logits.sigmoid()).max().item(), 2.)
        torch.testing.assert_close(model.encoder.embeddings["ad_task"].weight, before)
        # Uncapped nonempty fields have RMS at most the bounded gain.
        self.assertLessEqual(tokens[:, 5].square().mean(-1).sqrt().max().item(), 2.000001)
        torch.testing.assert_close(tokens[1, len(self.cats):-1], torch.zeros(4, 16))

    def test_history_permutation_padding_empty_and_row_independence(self):
        model = self.model().eval()
        expected = model(self.categories, self.sequences, self.numeric)
        permuted = self.sequences[:, :, torch.tensor([4, 1, 3, 0, 2])]
        torch.testing.assert_close(expected, model(self.categories, permuted, self.numeric))
        padded = nn.functional.pad(self.sequences, (0, 3))
        with torch.no_grad():
            for table in model.encoder.embeddings.values():
                table.weight[0].fill_(999.)
        torch.testing.assert_close(expected, model(self.categories, padded, self.numeric))
        order = torch.tensor([2, 0, 1])
        torch.testing.assert_close(expected[order], model(self.categories[order], padded[order], self.numeric[order]))
        singleton = model(self.categories[:1], padded[:1], self.numeric[:1])
        torch.testing.assert_close(singleton, expected[:1])
        empty_tokens, _ = model.field_tokens(torch.zeros_like(self.categories), torch.zeros_like(padded), self.numeric)
        torch.testing.assert_close(empty_tokens[:, :-1], torch.zeros_like(empty_tokens[:, :-1]))
        self.assertTrue(torch.isfinite(model(torch.zeros_like(self.categories), torch.zeros_like(padded), self.numeric)).all())

    def test_numeric_linear_residual_bottom_compression_and_clipping(self):
        model = self.model().eval()
        self.assertEqual(model.top[0].in_features, model.interaction.output_dim + model.embedding_dim)
        with torch.no_grad():
            model.output.weight.zero_()
            model.numeric_linear.weight.copy_(torch.tensor([[.1, -.2, .3, -.4]]))
        actual = model(self.categories, self.sequences, self.numeric)
        torch.testing.assert_close(actual, model.bias + (self.numeric * model.numeric_linear.weight).sum(-1))
        raw = torch.tensor([[float("inf"), float("-inf"), float("nan"), 1e12]]).expand(3, -1)
        clipped = torch.tensor([[8., -8., 0., 8.]]).expand(3, -1)
        torch.testing.assert_close(model(self.categories, self.sequences, raw),
                                   model(self.categories, self.sequences, clipped))
        for layer_norm in (False, True):
            zero = self.model(numeric_dim=0, bottom_layer_norm=layer_norm)
            self.assertTrue(torch.isfinite(zero(self.categories, self.sequences, torch.empty(3, 0))).all())
        self.assertEqual(self.model(numeric_dim=None).numeric_dim, 166)
        default = DotInteractionCTR(self.cats, self.seqs, self.groups, self.sizes)
        self.assertEqual(default.numeric_dim, 166)
        self.assertEqual(default.embedding_dim, 16)

    def test_checkpoint_training_eval_and_forward_has_no_io(self):
        options = dict(dropout=.15, unknown_dropout=.2, group_dropout_rates={"ad_task": .3})
        model = self.model(**options).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        loss = nn.functional.binary_cross_entropy_with_logits(
            model(self.categories, self.sequences, self.numeric), torch.tensor([0., 1., 0.]))
        loss.backward()
        optimizer.step()
        model.eval()
        expected = model(self.categories, self.sequences, self.numeric)
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = self.model(**options)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
        restored.eval()
        with mock.patch("builtins.open", side_effect=AssertionError("Forward must not read files")):
            actual = restored(self.categories, self.sequences, self.numeric)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(restored(self.categories, self.sequences, self.numeric), actual, rtol=0, atol=0)

    def test_actual_schema_budget_780_pairs_and_cpu_batch_1024(self):
        # Snapshot of the 39-field supplied schema; no filesystem or labels used.
        cats = ["user_id", "age", "sex", "residence", "city", "city_rank", "series_dev", "series_group",
                "device_dev", "device_name", "device_size", "net_type", "task_id", "adv_id", "creat_type_cd",
                "adv_prim_id", "inter_type_cd", "slot_id", "spread_app_id", "hispace_app_tags", "app_second_class",
                "app_score", "u_refreshTimes", "u_feedLifeCycle", "feed_phone_price", "feed_browser_lifecycle",
                "feed_browser_mode"]
        seqs = ["ad_click_list_v001", "ad_click_list_v002", "ad_click_list_v003", "ad_close_list_v001",
                "ad_close_list_v002", "ad_close_list_v003", "u_newsCatInterestsST", "feed_long_interest",
                "feed_dislike_topics", "feed_clicked_topics", "feed_click_ca2", "feed_exposed_topics"]
        sizes = dict(ad_task=12721, adv_id=8256, advertiser=839, age=10, app_score=5, app_second_class=22,
                     city=325, city_rank=6, creat_type_cd=11, device_dev=25, device_name=197, device_size=821,
                     hispace_app_tags=42, inter_type_cd=6, net_type=8, news_interest=214, promoted_app=197,
                     residence=36, series_dev=25, series_group=9, sex=5, slot_id=61, u_feedLifeCycle=10,
                     u_refreshTimes=12, user_id=8558, feed_phone_price=10, feed_browser_lifecycle=10, feed_browser_mode=8)
        groups = {name: name for name in cats}
        groups.update(task_id="ad_task", adv_prim_id="advertiser", spread_app_id="promoted_app")
        for i, group in enumerate(("ad_task", "advertiser", "promoted_app")):
            groups[seqs[i]] = groups[seqs[i + 3]] = group
        groups.update({name: "news_interest" for name in seqs[6:]})
        for dim in (8, 16):
            with self.subTest(embedding_dim=dim):
                model = DotInteractionCTR(cats, seqs, groups, sizes, embedding_dim=dim, unknown_dropout=0.)
                count = sum(p.numel() for p in model.parameters())
                self.assertLess(count, 1_000_000)
                self.assertEqual(model.field_count, 40)
                self.assertEqual(model.interaction.output_dim, 780)
                self.assertEqual(sum(p.numel() for p in model.encoder.embeddings.parameters()), dim * sum(sizes.values()))
                logits = model(torch.full((1024, len(cats)), 2, dtype=torch.long),
                               torch.full((1024, len(seqs), 5), 2, dtype=torch.long), torch.randn(1024, 166))
                self.assertEqual(tuple(logits.shape), (1024,))
                logits.square().mean().backward()
                self.assertTrue(torch.isfinite(logits).all())
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                print(f"\nDotInteractionCTR dim={dim}: {count:,} parameters, 780 pairs, CPU batch 1024 backward OK")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
