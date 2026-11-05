"""CPU-only invariance, numerical, and runner-contract tests.

Run with: python -m unittest discover -s analysis/architecture_goal_20261029
          -p test_set_context_model.py -v
"""
from __future__ import annotations

import io
import unittest

import torch

try:
    from .set_context_model import SetContextCTR
    from .interaction_models import CappedInteractionEncoder
except ImportError:
    from set_context_model import SetContextCTR
    from interaction_models import CappedInteractionEncoder


class SetContextCTRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261029)
        self.kwargs = dict(
            categorical_fields=["task_id", "user_id", "country"],
            sequence_fields=["clicktask"],
            group_for_field={"task_id": "ad_task", "clicktask": "ad_task",
                             "user_id": "user_id", "country": "country"},
            vocabulary_sizes={"ad_task": 127, "user_id": 151, "country": 17},
            numeric_dim=500, initial_ctr=.13, unknown_dropout=0.,
            history_queries={"clicktask": "task_id"})
        self.model = SetContextCTR(**self.kwargs).cpu().eval()
        self.inputs = self.make_inputs(12, torch.tensor([0, 1, 0, 2, 1, 0, 2, 1, 2, 0, 2, 2]))

    def make_inputs(self, rows, groups=None):
        categories = torch.stack((torch.randint(0, 127, (rows,)),
                                  torch.randint(0, 151, (rows,)),
                                  torch.randint(0, 17, (rows,))), dim=1)
        sequences = torch.randint(0, 127, (rows, 1, 5))
        sequences[::3] = 0
        numeric = torch.randn(rows, 500)
        if groups is None:
            groups = torch.arange(rows) % 16
        return categories, sequences, numeric, groups

    def assert_close(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_runner_shapes_and_parameter_budget(self):
        logits, counts = self.model(*self.inputs)
        self.assertEqual(logits.shape, (12,))
        self.assertEqual(counts.shape, (3,))
        self.assertLess(sum(p.numel() for p in self.model.parameters()), 1_000_000)
        self.assertIsInstance(self.model.encoder, CappedInteractionEncoder)
        self.assertEqual(len(self.model.encoder.embeddings), 3)
        print(f"\nfixture parameters: {self.model.parameter_count:,}; embedding_dim={self.model.embedding_dim}")

    def test_row_permutation_equivariance_and_count_invariance(self):
        permutation = torch.randperm(12)
        with torch.no_grad():
            expected_logits, expected_counts = self.model(*self.inputs)
            logits, counts = self.model(*(x[permutation] for x in self.inputs))
        self.assert_close(logits, expected_logits[permutation])
        self.assert_close(counts, expected_counts)

    def test_group_id_relabeling(self):
        categories, sequences, numeric, groups = self.inputs
        remap = torch.tensor([2, 0, 1])
        with torch.no_grad():
            expected_logits, expected_counts = self.model(*self.inputs)
            logits, counts = self.model(categories, sequences, numeric, remap[groups])
        self.assert_close(logits, expected_logits)
        self.assert_close(counts[remap], expected_counts)

    def test_group_isolation_and_standalone_replay(self):
        categories, sequences, numeric, groups = self.inputs
        selected = groups == 1
        changed_categories, changed_sequences, changed_numeric = [x.clone() for x in self.inputs[:3]]
        changed_categories[~selected] = 0
        changed_sequences[~selected] = 0
        changed_numeric[~selected] = 1e3
        with torch.no_grad():
            expected_logits, expected_counts = self.model(*self.inputs)
            logits, counts = self.model(changed_categories, changed_sequences, changed_numeric, groups)
            alone_logits, alone_counts = self.model(categories[selected], sequences[selected],
                                                    numeric[selected], torch.zeros(int(selected.sum()), dtype=torch.long))
        self.assert_close(logits[selected], expected_logits[selected])
        self.assert_close(counts[1], expected_counts[1])
        self.assert_close(alone_logits, expected_logits[selected])
        self.assert_close(alone_counts[0], expected_counts[1])

    def test_context_has_effect_and_stays_within_group(self):
        categories, sequences, numeric, groups = self.inputs
        changed = numeric.clone()
        changed[0] = torch.arange(500).float() / 100
        with torch.no_grad():
            original, _ = self.model(*self.inputs)
            updated, _ = self.model(categories, sequences, changed, groups)
        peers = (groups == 0) & (torch.arange(12) != 0)
        self.assertGreater((original[peers] - updated[peers]).abs().max().item(), 1e-7)
        self.assert_close(original[groups != 0], updated[groups != 0])

    def test_padding_empty_histories_and_corrupted_padding_checkpoint(self):
        categories, sequences, numeric, groups = self.inputs
        with torch.no_grad():
            for table in self.model.encoder.embeddings.values():
                table.weight[0].fill_(100)
            fields, _, _ = self.model.encoder(torch.zeros_like(categories), torch.zeros_like(sequences))
            self.assertEqual(fields.count_nonzero().item(), 0)
            outputs = self.model(torch.zeros_like(categories), torch.zeros_like(sequences), numeric, groups)
            empty_history = torch.empty((12, 1, 0), dtype=torch.long)
            empty_outputs = self.model(categories, empty_history, numeric, groups)
        self.assertTrue(all(torch.isfinite(x).all() for x in outputs + empty_outputs))

    def test_finite_gradients_reach_slots_and_count_head(self):
        self.model.train()
        logits, log_lambda = self.model(*self.inputs)
        labels = torch.arange(12).remainder(2).float()  # Targets used only in the loss.
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss = loss + (log_lambda.exp() - torch.tensor([1., 2., 3.]) * log_lambda).mean()
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(self.model.slots.grad.abs().sum().item(), 0.)
        self.assertGreater(self.model.count_head[-1].weight.grad.abs().sum().item(), 0.)

    def test_checkpoint_replay(self):
        with torch.no_grad():
            # Exercise checkpointed, learned count and group biases as well.
            self.model.count_head[-1].weight.normal_(std=.01)
            self.model.group_bias[-1].weight.normal_(std=.01)
            expected = self.model(*self.inputs)
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = SetContextCTR(**self.kwargs).eval()
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        with torch.no_grad():
            actual = restored(*self.inputs)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_fixed_prior_and_count_bound(self):
        groups = self.inputs[-1]
        sizes = torch.bincount(groups).float()
        with torch.no_grad():
            _, log_lambda = self.model(*self.inputs)
        self.assert_close(log_lambda.exp(), sizes * .13)
        self.assertFalse(self.model.initial_logit.requires_grad)
        self.assertNotIn("initial_logit", dict(self.model.named_parameters()))
        with torch.no_grad():
            self.model.count_head[-1].bias.fill_(-100_000.)
            _, negative = self.model(*self.inputs)
            self.model.count_head[-1].bias.fill_(100_000.)
            _, positive = self.model(*self.inputs)
        self.assertTrue(torch.isfinite(negative).all())
        self.assertTrue(torch.isfinite(positive).all())
        self.assertTrue((positive.exp() <= sizes + 1e-6).all())

    def test_1024_rows_forward_backward_and_singletons(self):
        inputs = self.make_inputs(1024)
        logits, counts = self.model(*inputs)
        self.assertEqual(logits.shape, (1024,))
        self.assertEqual(counts.shape, (16,))
        (logits.square().mean() + counts.square().mean()).backward()
        self.assertTrue(torch.isfinite(logits).all() and torch.isfinite(counts).all())
        inputs = self.make_inputs(12, torch.arange(12))
        with torch.no_grad():
            logits, counts = self.model(*inputs)
        self.assertEqual(counts.shape, (12,))
        self.assert_close(counts.exp(), torch.full((12,), .13))

    def test_row_only_control_ignores_other_rows_but_retains_count_head(self):
        model = SetContextCTR(**self.kwargs, use_set_context=False).eval()
        categories, sequences, numeric, groups = self.inputs
        changed = numeric.clone()
        changed[0] = torch.arange(500).float() / 100
        with torch.no_grad():
            original, counts = model(*self.inputs)
            updated, _ = model(categories, sequences, changed, groups)
            regrouped, _ = model(categories, sequences, numeric, torch.arange(12))
            fields, _, _ = model.encoder(categories, sequences)
            tokens = model.row_mlp(torch.cat((fields.flatten(1), model.numeric_projection(numeric)), dim=-1))
            expected = model.initial_logit + model.row_head(tokens).squeeze(-1)
        self.assertEqual(counts.shape, (3,))
        self.assert_close(original[1:], updated[1:])
        self.assert_close(original, regrouped)
        self.assert_close(original, expected)
        logits, _ = model(*self.inputs)
        logits.square().mean().backward()
        self.assertIsNotNone(model.row_head.weight.grad)
        self.assertIsNone(model.slots.grad)
        self.assertIsNone(model.count_head[-1].weight.grad)

    def test_4096_rows_64_groups(self):
        inputs = self.make_inputs(4096, torch.arange(4096) % 64)
        with torch.no_grad():
            logits, counts = self.model(*inputs)
        self.assertEqual(logits.shape, (4096,))
        self.assertEqual(counts.shape, (64,))
        self.assertTrue(torch.isfinite(logits).all() and torch.isfinite(counts).all())

    def test_invalid_empty_or_noncontiguous_groups(self):
        categories, sequences, numeric, _ = self.inputs
        invalid = [torch.ones(12, dtype=torch.long), torch.tensor([0] * 11 + [2]),
                   torch.tensor([0] * 11 + [-1]), torch.arange(12).float(),
                   torch.tensor([0] * 11 + [10**9]), torch.zeros(12, 1, dtype=torch.long)]
        for groups in invalid:
            with self.subTest(groups=groups.tolist()), self.assertRaises(ValueError):
                self.model(categories, sequences, numeric, groups)
        with self.assertRaisesRegex(ValueError, "Empty group"):
            self.model(categories[:0], sequences[:0], numeric[:0], torch.empty(0, dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model(categories, sequences, numeric[:, :499], torch.zeros(12, dtype=torch.long))

    def test_no_max_option_and_adaptive_total_budget(self):
        no_max = SetContextCTR(**self.kwargs, include_max=False).eval()
        self.assertTrue(all(torch.isfinite(x).all() for x in no_max(*self.inputs)))
        large_vocab = dict(self.kwargs, vocabulary_sizes={"ad_task": 80_000, "user_id": 90_000, "country": 17})
        budgeted = SetContextCTR(**large_vocab)
        self.assertLess(budgeted.parameter_count, 1_000_000)
        self.assertLess(budgeted.embedding_dim, 8)
        with self.assertRaises(ValueError):
            SetContextCTR(**large_vocab, embedding_dim=8)
        for prior in (0., 1., float("nan")):
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                SetContextCTR(**dict(self.kwargs, initial_ctr=prior))


if __name__ == "__main__":
    unittest.main(verbosity=2)
