"""CPU tests for count allocation and the runner-compatible set-model variant."""
from __future__ import annotations

import io
import math
import unittest

import torch
from torch.nn import functional as F

try:
    from .allocated_set_model import AllocatedSetContextCTR, allocate_intensity
    from .set_context_model import SetContextCTR
except ImportError:
    from allocated_set_model import AllocatedSetContextCTR, allocate_intensity
    from set_context_model import SetContextCTR


class AllocatedSetContextCTRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20261029)
        self.kwargs = dict(
            categorical_fields=["task_id", "user_id"], sequence_fields=["clicktask"],
            group_for_field={"task_id": "ad_task", "clicktask": "ad_task", "user_id": "user_id"},
            vocabulary_sizes={"ad_task": 127, "user_id": 151}, numeric_dim=500,
            initial_ctr=.13, unknown_dropout=0., history_queries={"clicktask": "task_id"},
            use_set_context=True)
        self.model = AllocatedSetContextCTR(**self.kwargs).cpu().eval()
        self.inputs = self.make_inputs(12, 3)

    @staticmethod
    def make_inputs(rows, group_count):
        categories = torch.stack((torch.randint(0, 127, (rows,)),
                                  torch.randint(0, 151, (rows,))), dim=1)
        sequences = torch.randint(0, 127, (rows, 1, 5))
        sequences[::3] = 0
        return categories, sequences, torch.randn(rows, 500), torch.arange(rows) % group_count

    def test_allocation_conserves_predicted_mass(self):
        groups = torch.tensor([0, 1, 0, 2, 1, 2, 2])
        scores = torch.tensor([-900., 900., -901., 1., 899., 0., -1.], dtype=torch.float64)
        log_lambda = torch.tensor([math.log(1e-20), math.log(1.9), math.log(2.8)], dtype=torch.float64)
        intensity = allocate_intensity(scores, log_lambda, groups)
        actual = torch.zeros(3, dtype=torch.float64).index_add_(0, groups, intensity)
        torch.testing.assert_close(actual, log_lambda.exp(), rtol=1e-12, atol=1e-30)
        shifted = scores + torch.tensor([100_000., -200_000., 300_000.], dtype=scores.dtype)[groups]
        torch.testing.assert_close(allocate_intensity(shifted, log_lambda, groups), intensity,
                                   rtol=1e-12, atol=1e-30)

    def test_probability_mapping_count_shape_and_budget(self):
        with torch.no_grad():
            scores, expected_log_lambda = SetContextCTR.forward(self.model, *self.inputs)
            intensity = allocate_intensity(scores, expected_log_lambda, self.inputs[-1])
            logits, log_lambda = self.model(*self.inputs)
        self.assertEqual(logits.shape, (12,))
        self.assertEqual(log_lambda.shape, (3,))
        torch.testing.assert_close(log_lambda, expected_log_lambda)
        torch.testing.assert_close(logits.sigmoid(), -torch.expm1(-intensity), rtol=2e-6, atol=1e-7)
        actual_mass = torch.zeros(3).index_add_(0, self.inputs[-1], intensity)
        torch.testing.assert_close(actual_mass, log_lambda.exp())
        self.assertLess(self.model.parameter_count, 1_000_000)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),
                         sum(p.numel() for p in SetContextCTR(**self.kwargs).parameters()))
        print(f"\nallocated fixture parameters: {self.model.parameter_count:,}")

    def test_row_bce_supervises_count_head_and_joint_gradients_are_finite(self):
        logits, log_lambda = self.model(*self.inputs)
        targets = torch.zeros(12)
        targets[:8] = 1  # A burst is a training target, never an input feature.
        row_loss = F.binary_cross_entropy_with_logits(logits, targets)
        gradient = torch.autograd.grad(row_loss, self.model.count_head[-1].weight, retain_graph=True)[0]
        self.assertGreater(gradient.abs().sum().item(), 0.)
        group_targets = torch.zeros(3).index_add_(0, self.inputs[-1], targets)
        count_loss = F.poisson_nll_loss(log_lambda, group_targets, log_input=True)
        (row_loss + count_loss).backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(self.model.row_head.weight.grad.abs().sum().item(), 0.)

    def test_small_large_counts_and_concentrated_scores_have_finite_gradients(self):
        groups = torch.arange(1024) % 2
        scores = torch.zeros(1024, requires_grad=True)
        with torch.no_grad():
            scores[0], scores[1] = 10_000., 10_000.
        log_lambda = torch.tensor([-10_000., math.log(512.)], requires_grad=True)
        intensity = allocate_intensity(scores, log_lambda, groups)
        clipped = intensity.clamp(1e-8, 80.)
        logits = clipped + torch.log(-torch.expm1(-clipped))
        loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        loss = loss + F.poisson_nll_loss(log_lambda, torch.ones(2), log_input=True)
        loss.backward()
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue(torch.isfinite(log_lambda.grad).all())
        self.assertTrue((logits <= 80.).all())
        self.assertGreater(float(intensity.max()), 80.)
        self.assertEqual(float(logits.max()), 80.)
        for extreme in (-100_000., 100_000.):
            self.model.zero_grad(set_to_none=True)
            with torch.no_grad():
                self.model.count_head[-1].bias.fill_(extreme)
            actual, counts = self.model(*self.inputs)
            (actual.mean() + counts.mean()).backward()
            self.assertTrue(torch.isfinite(actual).all() and torch.isfinite(counts).all())
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                for p in self.model.parameters()))

    def test_permutation_and_group_relabeling(self):
        permutation = torch.randperm(12)
        categories, sequences, numeric, groups = self.inputs
        remap = torch.tensor([2, 0, 1])
        with torch.no_grad():
            expected, expected_counts = self.model(*self.inputs)
            permuted, counts = self.model(*(x[permutation] for x in self.inputs))
            relabeled, relabeled_counts = self.model(categories, sequences, numeric, remap[groups])
        torch.testing.assert_close(permuted, expected[permutation], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(counts, expected_counts)
        torch.testing.assert_close(relabeled, expected)
        torch.testing.assert_close(relabeled_counts[remap], expected_counts)

    def test_cross_group_isolation_and_standalone_replay(self):
        categories, sequences, numeric, groups = self.inputs
        selected = groups == 1
        changed = numeric.clone()
        changed[~selected] = torch.randn_like(changed[~selected]) * 100
        with torch.no_grad():
            expected, expected_counts = self.model(*self.inputs)
            actual, counts = self.model(categories, sequences, changed, groups)
            alone, alone_counts = self.model(categories[selected], sequences[selected], numeric[selected],
                                             torch.zeros(int(selected.sum()), dtype=torch.long))
        torch.testing.assert_close(actual[selected], expected[selected])
        torch.testing.assert_close(counts[1], expected_counts[1])
        torch.testing.assert_close(alone, expected[selected], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(alone_counts[0], expected_counts[1])

    def test_checkpoint_replays_nondefault_temperature(self):
        self.model.allocation_temperature.fill_(.7)
        with torch.no_grad():
            self.model.count_head[-1].weight.normal_(std=.1)
            expected = self.model(*self.inputs)
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        replay = AllocatedSetContextCTR(**self.kwargs).eval()
        replay.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        self.assertAlmostEqual(float(replay.allocation_temperature), .7, places=6)
        with torch.no_grad():
            actual = replay(*self.inputs)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_temperature_and_double_precision_gradcheck(self):
        scores = torch.tensor([-.5, .2, 1., .7], dtype=torch.float64, requires_grad=True)
        counts = torch.tensor([-.4, -.2], dtype=torch.float64, requires_grad=True)
        groups = torch.tensor([0, 1, 0, 1])
        self.assertTrue(torch.autograd.gradcheck(lambda s, c: allocate_intensity(s, c, groups, .7),
                                                (scores, counts)))
        sharp = allocate_intensity(scores, counts, groups, .5)
        flat = allocate_intensity(scores, counts, groups, 2.)
        self.assertGreater(float(sharp[2] / sharp[0]), float(flat[2] / flat[0]))
        for temperature in (0., -1., float("inf"), float("nan")):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                AllocatedSetContextCTR(**self.kwargs, temperature=temperature)
            with self.assertRaises(ValueError):
                allocate_intensity(scores, counts, groups, temperature)

    def test_empty_invalid_groups_and_padding(self):
        scores = torch.ones(4)
        for groups, counts in ((torch.tensor([0, 0, 2, 2]), torch.zeros(3)),
                               (torch.ones(4, dtype=torch.long), torch.zeros(2)),
                               (torch.tensor([0, 0, -1, 1]), torch.zeros(2)),
                               (torch.zeros(4), torch.zeros(1)),
                               (torch.empty(0, dtype=torch.long), torch.empty(0))):
            with self.subTest(groups=groups.tolist()), self.assertRaises(ValueError):
                allocate_intensity(scores[:groups.numel()], counts, groups)
        categories, sequences, numeric, groups = self.inputs
        with torch.no_grad():
            result = self.model(torch.zeros_like(categories), torch.zeros_like(sequences), numeric, groups)
        self.assertTrue(all(torch.isfinite(x).all() for x in result))

    def test_1024_rows_backward_and_4096_rows_64_groups(self):
        logits, counts = self.model(*self.make_inputs(1024, 16))
        (logits.square().mean() + counts.square().mean()).backward()
        self.assertEqual(logits.shape, (1024,))
        self.assertEqual(counts.shape, (16,))
        with torch.no_grad():
            large_logits, large_counts = self.model(*self.make_inputs(4096, 64))
        self.assertEqual(large_logits.shape, (4096,))
        self.assertEqual(large_counts.shape, (64,))
        self.assertTrue(torch.isfinite(large_logits).all() and torch.isfinite(large_counts).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
