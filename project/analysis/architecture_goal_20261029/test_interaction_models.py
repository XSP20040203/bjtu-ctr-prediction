"""CPU-only behavioral checks; all inputs are synthetic and no data is opened."""
import io
import math
import unittest
from unittest import mock

import torch
from torch import nn

from interaction_models import (
    AttentiveInteractionCTR,
    CappedInteractionEncoder,
    ContextFeatureSelection,
    FinalMLPCTR,
    GatedCrossCTR,
    GatedLowRankCross,
    MultiHeadBilinear,
    NumericFieldGate,
    TargetHistoryAttention,
)


class InteractionModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.cats = ["user_id", "task_id", "adv_id", "adv_prim_id", "spread_app_id", "age"]
        self.seqs = ["ad_click_list_v001", "ad_click_list_v002", "ad_click_list_v003", "news"]
        self.groups = dict(zip(self.cats + self.seqs,
                               ["user_id", "ad_task", "adv_id", "advertiser", "app", "age",
                                "ad_task", "advertiser", "app", "news"]))
        self.sizes = {"user_id": 140, "ad_task": 150, "adv_id": 130,
                      "advertiser": 20, "app": 18, "age": 10, "news": 12}
        self.arguments = (self.cats, self.seqs, self.groups, self.sizes)
        self.categories = torch.tensor([[2, 2, 3, 4, 5, 6], [3, 3, 4, 5, 6, 7],
                                        [4, 4, 5, 6, 7, 8]])
        self.sequences = torch.tensor([
            [[8, 9, 0, 0, 0], [9, 10, 0, 0, 0], [10, 11, 0, 0, 0], [2, 3, 0, 0, 0]],
            [[0, 0, 0, 0, 0]] * 4,
            [[2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]],
        ])
        self.numeric = torch.randn(3, 4)
        self.models = (GatedCrossCTR, AttentiveInteractionCTR, FinalMLPCTR)

    def make_model(self, model_class, **kwargs):
        options = dict(numeric_dim=4, initial_ctr=.13, dropout=0., unknown_dropout=0.)
        options.update(kwargs)
        return model_class(*self.arguments, **options)

    def test_shapes_prior_finite_loss_and_gradients(self):
        for model_class in self.models:
            for normalize in ((False,) if model_class is FinalMLPCTR else (False, True)):
                with self.subTest(model=model_class.__name__, normalize=normalize):
                    option = ("cross_layer_norm" if model_class is GatedCrossCTR
                              else "attention_layer_norm")
                    model = self.make_model(model_class, **({option: normalize}
                                                           if model_class is not FinalMLPCTR else {}))
                    numeric = self.numeric.clone().requires_grad_()
                    logits = model(self.categories, self.sequences, numeric)
                    self.assertEqual(tuple(logits.shape), (3,))
                    self.assertTrue(torch.isfinite(logits).all())
                    prior = math.log(.13 / .87)
                    self.assertLess((logits - prior).abs().max().item(), .2)
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.]))
                    self.assertLess(loss.item(), 2.)
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(numeric.grad.abs().sum().item(), 0.)
                    if model_class is FinalMLPCTR:
                        self.assertGreater(model.gate_left.field_output.weight.grad.abs().sum().item(), 0.)
                        self.assertGreater(model.gate_right.numeric_output.weight.grad.abs().sum().item(), 0.)
                        self.assertGreater(model.bilinear.weight.grad.abs().sum().item(), 0.)
                        self.assertGreater(model.linear_tables["task_id"].weight.grad[2].abs().sum().item(), 0.)
                        self.assertIsNot(model.stream_left[0].weight, model.stream_right[0].weight)
                    else:
                        self.assertGreater(model.field_gate.network[-1].weight.grad.abs().sum().item(), 0.)
                    for table in model.encoder.embeddings.values():
                        torch.testing.assert_close(table.weight.grad[0], torch.zeros(8))
                    if model_class is GatedCrossCTR:
                        self.assertGreater(model.cross[0].router.weight.grad.abs().sum().item(), 0.)
                    else:
                        attention = model.encoder.history_attention["ad_click_list_v001"]
                        self.assertGreater(attention.query.weight.grad.abs().sum().item(), 0.)

    def test_target_attention_changes_with_candidate_and_masks_pad(self):
        attention = TargetHistoryAttention(2, 2)
        with torch.no_grad():
            attention.query.weight.copy_(torch.eye(2))
            attention.key.weight.copy_(torch.eye(2))
        history = torch.tensor([[[1., 0.], [0., 1.], [999., -999.]]], requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        first, weights = attention(history, torch.tensor([[1., 0.]]), mask, return_weights=True)
        second = attention(history, torch.tensor([[0., 1.]]), mask)
        self.assertGreater(first[0, 0].item(), second[0, 0].item())
        torch.testing.assert_close(weights[:, 2], torch.zeros(1))
        torch.testing.assert_close(weights.sum(-1), torch.ones(1))
        torch.testing.assert_close(first, attention(history[:, :2], torch.tensor([[1., 0.]]), mask[:, :2]))
        empty, empty_weights = attention(history, torch.tensor([[1., 0.]]), torch.zeros_like(mask), True)
        torch.testing.assert_close(empty, torch.zeros(1, 2))
        torch.testing.assert_close(empty_weights, torch.zeros(1, 3))
        (first.sum() + empty.sum()).backward()
        self.assertTrue(torch.isfinite(history.grad).all())
        torch.testing.assert_close(history.grad[:, 2], torch.zeros(1, 2))

    def test_shared_tables_receive_category_and_history_gradients(self):
        for model_class in self.models:
            with self.subTest(model=model_class.__name__):
                model = self.make_model(model_class)
                self.assertEqual(len(model.encoder.embeddings), len(self.sizes))
                fields, _, _ = model.encoder(self.categories, self.sequences)
                table = model.encoder.embeddings["ad_task"].weight
                category_gradient = torch.autograd.grad(fields[0, 1].sum(), table, retain_graph=True)[0]
                history_gradient = torch.autograd.grad(fields[0, len(self.cats)].sum(), table)[0]
                self.assertGreater(category_gradient[2].abs().sum().item(), 0.)
                self.assertGreater(history_gradient[8].abs().sum().item(), 0.)
                self.assertEqual(category_gradient[8].abs().sum().item(), 0.)
                self.assertEqual(history_gradient[0].abs().sum().item(), 0.)

    def test_every_id_read_and_pooled_vector_is_capped_without_weight_mutation(self):
        for model_class in self.models:
            with self.subTest(model=model_class.__name__):
                model = self.make_model(model_class).eval()
                encoder = model.encoder
                with torch.no_grad():
                    for group in ("user_id", "ad_task", "adv_id"):
                        encoder.embeddings[group].weight.fill_(30.)
                    encoder.embeddings["age"].weight.fill_(2.)
                old_weight = encoder.embeddings["ad_task"].weight.detach().clone()
                fields, _, _ = encoder(self.categories, self.sequences)
                for index in (0, 1, 2, len(self.cats)):
                    self.assertLessEqual(fields[:, index].norm(dim=-1).max().item(), .200001)
                self.assertGreater(fields[0, 5].norm().item(), 1.)
                reads, _ = encoder.read("ad_click_list_v001", self.sequences[:, 0])
                self.assertLessEqual(reads.norm(dim=-1).max().item(), .200001)
                torch.testing.assert_close(reads[self.sequences[:, 0].eq(0)],
                                           torch.zeros_like(reads[self.sequences[:, 0].eq(0)]))
                torch.testing.assert_close(encoder.embeddings["ad_task"].weight, old_weight)
                fields.sum().backward()
                self.assertTrue(torch.isfinite(encoder.embeddings["ad_task"].weight.grad).all())

    def test_caps_can_be_disabled_or_overridden(self):
        for caps, expected in (({}, 8.), ({"ad_task": .05}, .05)):
            encoder = CappedInteractionEncoder(*self.arguments, embedding_dim=4,
                                                unknown_dropout=0, group_norm_caps=caps)
            with torch.no_grad():
                encoder.embeddings["ad_task"].weight.fill_(4.)
            read, _ = encoder.read("task_id", torch.tensor([2]))
            self.assertAlmostEqual(read.norm().item(), expected, places=5)
        with self.assertRaises(ValueError):
            self.make_model(GatedCrossCTR, group_norm_caps={"ad_task": -1.})

    def test_histories_are_unordered_padding_invariant_and_all_empty_safe(self):
        permutation = torch.tensor([4, 1, 3, 0, 2])
        for model_class in self.models:
            with self.subTest(model=model_class.__name__):
                model = self.make_model(model_class).eval()
                expected = model(self.categories, self.sequences, self.numeric)
                torch.testing.assert_close(expected, model(self.categories, self.sequences[:, :, permutation], self.numeric))
                extended = nn.functional.pad(self.sequences, (0, 3))
                torch.testing.assert_close(expected, model(self.categories, extended, self.numeric))
                with torch.no_grad():
                    for table in model.encoder.embeddings.values():
                        table.weight[0].fill_(999.)
                torch.testing.assert_close(expected, model(self.categories, extended, self.numeric))
                empty = model(torch.zeros_like(self.categories), torch.zeros_like(self.sequences), self.numeric)
                self.assertTrue(torch.isfinite(empty).all())
                self.assertEqual(tuple(model(self.categories[:1], self.sequences[:1], self.numeric[:1]).shape), (1,))

    def test_group_dropout_and_candidate_query_reuse(self):
        for model_class in self.models:
            with self.subTest(model=model_class.__name__):
                model = self.make_model(model_class, group_dropout_rates={"ad_task": 1.})
                model.train()
                with mock.patch.object(model.encoder, "_indices", wraps=model.encoder._indices) as reads:
                    fields, cat_indices, seq_indices = model.encoder(self.categories, self.sequences)
                    self.assertEqual(reads.call_count, len(self.cats) + len(self.seqs))
                torch.testing.assert_close(cat_indices[1], torch.ones(3, dtype=torch.long))
                expected = torch.where(self.sequences[:, 0].eq(0), 0, 1)
                torch.testing.assert_close(seq_indices[0], expected)
                self.assertTrue(torch.isfinite(fields).all())
                model.eval()
                _, cat_indices, seq_indices = model.encoder(self.categories, self.sequences)
                torch.testing.assert_close(cat_indices[1], self.categories[:, 1])
                torch.testing.assert_close(seq_indices[0], self.sequences[:, 0])

    def test_cross_router_depends_on_raw_content_and_residual_preserves_scale(self):
        cross = GatedLowRankCross(2, rank=1, experts=2)
        with torch.no_grad():
            cross.down.zero_()
            cross.core.zero_()
            cross.up.zero_()
            cross.expert_bias.copy_(torch.tensor([[1., 0.], [-1., 0.]]))
            cross.router.weight.copy_(torch.tensor([[2., 0.], [-2., 0.]]))
            cross.router.bias.zero_()
        original = torch.tensor([[2., .1]])
        first = cross(original, original, torch.tensor([[1., 0.]]))
        second = cross(original, original, torch.tensor([[-1., 0.]]))
        self.assertGreater(first[0, 0].item(), second[0, 0].item())
        for normalize in (False, True):
            layer = GatedLowRankCross(2, rank=1, experts=1, layer_norm=normalize)
            with torch.no_grad():
                layer.up.zero_()
            torch.testing.assert_close(layer(original, original, original), original)

    def test_se_field_gate_uses_numeric_summary(self):
        gate = NumericFieldGate(2, 1, hidden_dim=1)
        with torch.no_grad():
            gate.network[0].weight.zero_()
            gate.network[0].weight[0, -1] = 1.
            gate.network[0].bias.zero_()
            gate.network[2].weight.copy_(torch.tensor([[1.], [-1.]]))
            gate.network[2].bias.zero_()
        fields = torch.ones(1, 2, 2)
        first = gate(fields, torch.tensor([[-2.]]))
        second = gate(fields, torch.tensor([[2.]]))
        self.assertGreater(second[0, 0, 0].item(), first[0, 0, 0].item())
        self.assertLess(second[0, 1, 0].item(), first[0, 1, 0].item())

    def test_checkpoint_replay_eval_and_no_label_or_io_dependency(self):
        for model_class in self.models:
            with self.subTest(model=model_class.__name__):
                model = self.make_model(model_class, dropout=.2, unknown_dropout=.3,
                                        group_dropout_rates={"ad_task": .4})
                # Forward/backward/optimizer steps use training tensors and labels only.
                model.train()
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
                restored = self.make_model(model_class, dropout=.2, unknown_dropout=.3,
                                           group_dropout_rates={"ad_task": .4})
                restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                restored.eval()
                with mock.patch("builtins.open", side_effect=AssertionError("Forward must not read data")):
                    actual = restored(self.categories, self.sequences, self.numeric)
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                    torch.manual_seed(901)
                    torch.testing.assert_close(actual, restored(self.categories, self.sequences, self.numeric), rtol=0, atol=0)
                    singleton = restored(self.categories[:1], self.sequences[:1], self.numeric[:1])
                    torch.testing.assert_close(singleton, actual[:1])

    def test_numeric_clipping_and_zero_numeric_dim(self):
        numeric = torch.tensor([[float("inf"), float("-inf"), float("nan"), 1.e12]]).expand(3, -1)
        clipped = torch.tensor([[8., -8., 0., 8.]]).expand(3, -1)
        for model_class in self.models:
            with self.subTest(model=model_class.__name__):
                model = self.make_model(model_class).eval()
                actual = model(self.categories, self.sequences, numeric)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, model(self.categories, self.sequences, clipped))
                no_numeric = self.make_model(model_class, numeric_dim=0).eval()
                logits = no_numeric(self.categories, self.sequences, torch.empty(3, 0))
                self.assertTrue(torch.isfinite(logits).all())

    def test_synthetic_vocabulary_budget_and_batch_1024(self):
        # 57,000 distinct table rows: count the shared tables once, including PAD.
        cats = ["user_id", "task_id", "adv_id", "adv_prim_id", "spread_app_id"]
        cats += [f"context_{i}" for i in range(19)]
        seqs = ["ad_click_list_v001", "ad_click_list_v002", "ad_click_list_v003"]
        groups = {name: name for name in cats}
        groups.update({"task_id": "ad_task", "adv_prim_id": "advertiser", "spread_app_id": "app",
                       "ad_click_list_v001": "ad_task", "ad_click_list_v002": "advertiser",
                       "ad_click_list_v003": "app"})
        sizes = {group: 100 for group in set(groups.values())}
        sizes.update(user_id=40000, ad_task=10000, adv_id=5000)
        for model_class in self.models:
            with self.subTest(model=model_class.__name__):
                model = model_class(cats, seqs, groups, sizes, numeric_dim=80, unknown_dropout=0)
                count = sum(parameter.numel() for parameter in model.parameters())
                self.assertLessEqual(count, 1_000_000)
                inputs = (torch.full((1024, len(cats)), 2, dtype=torch.long),
                          torch.full((1024, len(seqs), 5), 2, dtype=torch.long), torch.randn(1024, 80))
                logits = model(*inputs)
                self.assertEqual(tuple(logits.shape), (1024,))
                logits.mean().backward()
                self.assertTrue(torch.isfinite(logits).all())
                print(f"\n{model_class.__name__}: {count:,} parameters; CPU batch=1024 forward/backward OK")

    def test_finalmlp_bilinear_identity_blocks_and_gradients(self):
        head = MultiHeadBilinear(width=4, heads=2)
        with torch.no_grad():
            head.weight.copy_(torch.eye(2).expand(2, -1, -1))
        left = torch.tensor([[1., 2., 3., 4.]], requires_grad=True)
        right = torch.tensor([[5., 6., 7., 8.]], requires_grad=True)
        expected = (left * right).sum(-1) / math.sqrt(2.)
        torch.testing.assert_close(head(left, right), expected)
        with torch.no_grad():
            head.weight[1].zero_()
        torch.testing.assert_close(head(left, right), (left[:, :2] * right[:, :2]).sum(-1) / math.sqrt(2.))
        head(left, right).sum().backward()
        self.assertTrue(torch.isfinite(head.weight.grad).all())
        self.assertGreater(left.grad.abs().sum().item(), 0.)
        self.assertGreater(right.grad.abs().sum().item(), 0.)
        with self.assertRaises(ValueError):
            MultiHeadBilinear(width=63, heads=4)

    def test_finalmlp_context_gates_can_select_fields_and_numeric_independently(self):
        gate = ContextFeatureSelection(2, 2, context_dim=3, hidden_dim=1)
        fields, numeric = torch.ones(1, 2, 2), torch.ones(1, 2)
        with torch.no_grad():
            gate.context[0].weight.fill_(1.)
            gate.context[0].bias.zero_()
            gate.field_output.weight.copy_(torch.tensor([[1.], [-1.]]))
            gate.field_output.bias.zero_()
            gate.numeric_output.weight.copy_(torch.tensor([[-1.], [1.]]))
            gate.numeric_output.bias.zero_()
        for axis in range(3):  # User, candidate, and numeric context can each affect selection.
            context = torch.zeros(1, 3)
            baseline_fields, baseline_numeric = gate(fields, numeric, context)
            context[0, axis] = 2.
            selected_fields, selected_numeric = gate(fields, numeric, context)
            self.assertGreater(selected_fields[0, 0, 0].item(), baseline_fields[0, 0, 0].item())
            self.assertLess(selected_fields[0, 1, 0].item(), baseline_fields[0, 1, 0].item())
            self.assertLess(selected_numeric[0, 0].item(), baseline_numeric[0, 0].item())
            self.assertGreater(selected_numeric[0, 1].item(), baseline_numeric[0, 1].item())
        with torch.no_grad():
            gate.field_output.weight.zero_()
            gate.numeric_output.weight.zero_()
            gate.field_output.bias.fill_(-30.)
            gate.numeric_output.bias.fill_(30.)
        selected_fields, selected_numeric = gate(fields, numeric, torch.zeros(1, 3))
        self.assertLess(selected_fields.abs().max().item(), 1.e-10)
        torch.testing.assert_close(selected_numeric, 2 * numeric)

    def test_finalmlp_numeric_scale_balances_500_columns_and_optional_attention(self):
        model = self.make_model(FinalMLPCTR, numeric_dim=500)
        fields, _, _ = model.encoder(self.categories, self.sequences)
        clipped, summary = model.numeric(torch.full((3, 500), 8.))
        numeric_features = model._numeric_features(clipped)
        self.assertLessEqual(numeric_features.abs().max().item(), .020001)
        # Even all saturated numeric columns have bounded total energy.
        budget = .02 * math.sqrt(fields.shape[1] * fields.shape[2])
        self.assertLessEqual(numeric_features.norm(dim=-1).max().item(), budget + 1.e-6)
        context = model._context(fields, summary)
        self.assertLessEqual(context.abs().max().item(), 1.)
        self.assertEqual(len(model.encoder.history_attention), 3)
        without_attention = self.make_model(FinalMLPCTR, target_attention=False)
        self.assertEqual(len(without_attention.encoder.history_attention), 0)
        self.assertTrue(torch.isfinite(without_attention(self.categories, self.sequences, self.numeric)).all())

    def test_finalmlp_linear_branch_is_field_specific_pad_safe_and_capped(self):
        model = self.make_model(FinalMLPCTR).eval()
        with torch.no_grad():
            model.head_left.weight.zero_()
            model.head_right.weight.zero_()
            model.bilinear.weight.zero_()
            model.numeric_linear.weight.zero_()
            model.linear_tables["task_id"].weight[2] = .07
            model.linear_tables["ad_click_list_v001"].weight[8] = .03
            model.linear_tables["ad_click_list_v001"].weight[9] = .05
            model.linear_tables["ad_click_list_v001"].weight[0] = 999.
        actual = model(self.categories[:1], self.sequences[:1], self.numeric[:1])
        torch.testing.assert_close(actual, model.bias.detach().reshape(1) + .07 + .04)
        with torch.no_grad():
            model.linear_tables["task_id"].weight[2] = 10.
        actual = model(self.categories[:1], self.sequences[:1], self.numeric[:1])
        torch.testing.assert_close(actual, model.bias.detach().reshape(1) + .2 + .04)
        torch.testing.assert_close(model.linear_tables["task_id"].weight[2], torch.tensor([10.]))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
