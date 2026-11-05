"""Synthetic CPU checks; never opens task data or invokes training runners."""
import inspect
import io
import math
import unittest
from unittest import mock

import torch
from torch import nn

from piecewise_gated_model import GatedPiecewiseNumeric, PiecewiseGatedCTR, fit_numeric_bin_edges


torch.set_num_threads(2)


class PiecewiseGatedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.cats = ["user_id", "task_id", "adv_id"]
        self.seqs = ["clicktask"]
        self.groups = dict(user_id="user_id", task_id="ad_task", adv_id="adv_id", clicktask="ad_task")
        self.sizes = dict(user_id=30, ad_task=25, adv_id=20)
        self.arguments = (self.cats, self.seqs, self.groups, self.sizes)
        self.categories = torch.tensor([[2, 3, 4], [3, 4, 5], [4, 5, 6]])
        self.sequences = torch.tensor([[[8, 9, 0, 0, 0]], [[0, 0, 0, 0, 0]], [[2, 3, 4, 0, 0]]])
        sample = torch.randn(101, 166)
        sample[:, -1] = .3
        self.edges = fit_numeric_bin_edges(sample)
        self.numeric = torch.randn(3, 166)
        self.numeric[:, -1] = .3

    def make_model(self, **kwargs):
        options = dict(numeric_bin_edges=self.edges, initial_ctr=.13, dropout=0., unknown_dropout=0.)
        options.update(kwargs)
        return PiecewiseGatedCTR(*self.arguments, **options)

    def test_shapes_finite_gradients_and_initial_prior(self):
        for normalize in (False, True):
            with self.subTest(normalize=normalize):
                model = self.make_model(numeric_layer_norm=normalize)
                values = self.numeric.clone().requires_grad_()
                logits = model(self.categories, self.sequences, values)
                self.assertEqual(tuple(logits.shape), (3,))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertLess((logits - math.log(.13 / .87)).abs().max().item(), .2)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.]))
                loss.backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                for parameter in (model.piecewise_numeric.gate_logits,
                                  model.piecewise_numeric.feature_scale,
                                  model.piecewise_numeric.embeddings.linear.weight,
                                  model.piecewise_numeric.projection.weight,
                                  model.numeric_residual.weight,
                                  model.cross[0].router.weight):
                    self.assertGreater(parameter.grad.abs().sum().item(), 0.)
                self.assertTrue(torch.isfinite(values.grad).all())
                self.assertGreater(values.grad.abs().sum().item(), 0.)
                for table in model.encoder.embeddings.values():
                    torch.testing.assert_close(table.weight.grad[0], torch.zeros(8))
                self.assertEqual(model.numeric.projection[0].in_features, 64)
                self.assertEqual(model.cross[0].down.shape, (3, 96, 32))
                self.assertEqual(model.numeric_residual.bias, None)
                torch.testing.assert_close(model.numeric_residual.weight, torch.zeros(1, 166))

    def test_quantiles_are_train_sample_only_and_duplicates_collapse(self):
        sample = torch.stack([torch.arange(17.), torch.ones(17), torch.arange(17.) % 2], dim=1)
        edges = fit_numeric_bin_edges(sample, numeric_clip=20.)
        torch.testing.assert_close(edges[0], torch.arange(0., 17., 2.))
        torch.testing.assert_close(edges[1], torch.tensor([1.]))
        torch.testing.assert_close(edges[2], torch.tensor([0., 1.]))
        # No labels, paths, validation data, or implicit fitting in model APIs.
        self.assertEqual(list(inspect.signature(fit_numeric_bin_edges).parameters),
                         ["training_sample", "n_bins", "numeric_clip"])
        self.assertEqual(list(inspect.signature(PiecewiseGatedCTR.forward).parameters),
                         ["self", "categories", "sequences", "numeric"])
        before = [edge.clone() for edge in self.edges]
        with mock.patch("builtins.open", side_effect=AssertionError("No data access")), \
             mock.patch("torch.quantile", side_effect=AssertionError("No fitting in model")):
            model = self.make_model().eval()
            model(self.categories, self.sequences, self.numeric)
        for supplied, original in zip(self.edges, before):
            torch.testing.assert_close(supplied, original)

    def test_gates_scales_constant_fields_and_no_extrapolation(self):
        encoder = GatedPiecewiseNumeric(3, [[-1., 0., 1.], [2.], [-2., 2.]])
        values = torch.tensor([[.5, 2., 1.]])
        embedded = encoder.embed_fields(values)
        self.assertEqual(tuple(embedded.shape), (1, 3, 2))
        torch.testing.assert_close(embedded[:, 1], torch.zeros(1, 2))
        torch.testing.assert_close(2 * encoder.gate_logits.sigmoid(), torch.ones(3))
        with torch.no_grad():
            encoder.gate_logits[0] = math.log(3.)
            encoder.feature_scale[2] = .25
        actual = encoder.embed_fields(values)
        torch.testing.assert_close(actual[:, 0], embedded[:, 0] * 1.5)
        torch.testing.assert_close(actual[:, 2], embedded[:, 2] * .25)
        extremes = torch.tensor([[-1.e20, -1.e20, 1.e20], [-1., 2., 2.]])
        torch.testing.assert_close(encoder.embed_fields(extremes)[0], encoder.embed_fields(extremes)[1])
        encoder(values).sum().backward()
        self.assertEqual(encoder.gate_logits.grad[1].item(), 0.)
        torch.testing.assert_close(encoder.embeddings.linear.weight.grad[1],
                                   torch.zeros_like(encoder.embeddings.linear.weight.grad[1]))

    def test_pad_histories_shared_caps_and_empty_rows(self):
        model = self.make_model().eval()
        expected = model(self.categories, self.sequences, self.numeric)
        extended = nn.functional.pad(self.sequences, (0, 3))
        permuted = self.sequences[:, :, [4, 2, 0, 3, 1]]
        torch.testing.assert_close(expected, model(self.categories, extended, self.numeric))
        torch.testing.assert_close(expected, model(self.categories, permuted, self.numeric))
        self.assertEqual(len(model.encoder.embeddings), 3)
        with torch.no_grad():
            for table in model.encoder.embeddings.values():
                table.weight[0].fill_(999.)
        torch.testing.assert_close(expected, model(self.categories, extended, self.numeric))
        empty = model(torch.zeros_like(self.categories), torch.zeros_like(self.sequences), self.numeric)
        self.assertTrue(torch.isfinite(empty).all())
        with torch.no_grad():
            model.encoder.embeddings["ad_task"].weight[2:].fill_(30.)
        fields, _, _ = model.encoder(self.categories, self.sequences)
        self.assertLessEqual(fields.norm(dim=-1).max().item(), .200001)

    def test_checkpoint_replay_and_fixed_bins(self):
        model = self.make_model(dropout=.2, unknown_dropout=.3)
        loss = model(self.categories, self.sequences, self.numeric).square().mean()
        loss.backward()
        torch.optim.AdamW(model.parameters(), lr=.001).step()
        model.eval()
        expected = model(self.categories, self.sequences, self.numeric)
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        # Same topology but different endpoints: state_dict restores the fitted
        # transformations and bounds, independent of the placeholder values.
        other_edges = [edge * .5 for edge in self.edges]
        restored = self.make_model(numeric_bin_edges=other_edges, dropout=.2, unknown_dropout=.3)
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        restored.eval()
        with mock.patch("builtins.open", side_effect=AssertionError("No forward IO")), \
             mock.patch("torch.quantile", side_effect=AssertionError("No forward fitting")):
            actual = restored(self.categories, self.sequences, self.numeric)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(actual[:1], restored(self.categories[:1], self.sequences[:1], self.numeric[:1]))
        torch.testing.assert_close(restored.piecewise_numeric.lower_bounds, model.piecewise_numeric.lower_bounds)

    def test_numeric_residual_uses_original_clipped_values_with_one_prior(self):
        model = self.make_model().eval()
        with torch.no_grad():
            model.output.weight.zero_()
            model.numeric_residual.weight.fill_(.1)
        values = self.numeric.clone()
        values[0, :4] = torch.tensor([float("nan"), float("inf"), float("-inf"), 1.e20])
        clipped = torch.nan_to_num(values, nan=0., posinf=8., neginf=-8.).clamp(-8, 8)
        actual = model(self.categories, self.sequences, values)
        torch.testing.assert_close(actual, model.bias + .1 * clipped.sum(-1))
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, model(self.categories, self.sequences, clipped))

    def test_edge_validation_and_input_shape(self):
        invalid = [None, [], [[0., 0.]], [[1., 0.]], [[float("nan")]],
                   [[float("inf")]], [[-9., 0.]], [[]], [[[0., 1.]]],
                   [list(range(10))], [[0., 1.e-40]]]
        for edges in invalid:
            with self.subTest(edges=edges), self.assertRaises((ValueError, TypeError)):
                GatedPiecewiseNumeric(1, edges)
        with self.assertRaises(ValueError):
            self.make_model(numeric_bin_edges=self.edges[:-1])
        with self.assertRaises(ValueError):
            self.make_model().piecewise_numeric(torch.randn(3, 165))
        for sample in (torch.ones(1, 2), torch.ones(2), torch.empty(3, 0)):
            with self.assertRaises(ValueError):
                fit_numeric_bin_edges(sample)
        for bins in (1, 9, 2.5, True):
            with self.assertRaises(ValueError):
                fit_numeric_bin_edges(torch.randn(10, 2), n_bins=bins)
        for clip in (0., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fit_numeric_bin_edges(torch.randn(10, 2), numeric_clip=clip)

    def test_representative_vocabulary_parameter_budget_and_batch(self):
        cats = ["user_id", "task_id", "adv_id", "adv_prim_id", "spread_app_id"]
        cats += [f"context_{i}" for i in range(19)]
        seqs = ["clicktask", "clickprim", "clickapp"]
        groups = {name: name for name in cats}
        groups.update(task_id="ad_task", adv_prim_id="advertiser", spread_app_id="app",
                      clicktask="ad_task", clickprim="advertiser", clickapp="app")
        sizes = {group: 100 for group in set(groups.values())}
        sizes.update(user_id=40000, ad_task=10000, adv_id=5000)
        model = PiecewiseGatedCTR(cats, seqs, groups, sizes, numeric_bin_edges=self.edges, unknown_dropout=0.)
        count = sum(parameter.numel() for parameter in model.parameters())
        self.assertLessEqual(count, 1_000_000)
        logits = model(torch.full((1024, len(cats)), 2, dtype=torch.long),
                       torch.full((1024, len(seqs), 5), 2, dtype=torch.long),
                       torch.randn(1024, 166))
        self.assertEqual(tuple(logits.shape), (1024,))
        logits.mean().backward()
        self.assertTrue(torch.isfinite(logits).all())
        print(f"\nPiecewiseGatedCTR: {count:,} parameters; CPU batch=1024 forward/backward OK")


if __name__ == "__main__":
    unittest.main(verbosity=2)
