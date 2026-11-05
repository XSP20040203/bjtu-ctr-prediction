"""Validate pooled-input exclusion, strict temporal counts, and auxiliary isolation."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch
from day_context import build_context, past_exposure_features, training_targets, HISTOGRAM_FIELDS
from day_model import UserDayCTR, auxiliary_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis/strong_interest_control_20261031"))
from features import read, cache_dir
from model import build as build_backbone


class DayContextTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(3)
        self.raw = pd.DataFrame({"pt_d": [20191111, 20191111, 20191111, 20191112, 20191115],
                                 "user_id": [2, 2, 3, 2, 2], "task_id": [9, 10, 9, 9, 9]})
        names = HISTOGRAM_FIELDS + ["user_id"]
        self.meta = {"categorical_fields": names, "group_for_field": {name: name for name in names},
                     "vocabulary_sizes": {name: 5 for name in names}}
        self.categories = np.full((5, len(names)), 2, dtype="int32")
        self.categories[1, 0] = 3
        self.numeric = np.array([[1, 100, 2, 200], [3, 1000, 4, 2000], [2, 300, 1, 400],
                                 [4, 500, 3, 600], [5, 700, 4, 800]], dtype="float32")
        self.fields = ["one", "target_stat__user__response", "two", "target_stat__task__response"]

    def build(self, raw=None, categories=None, numeric=None):
        return build_context(self.raw if raw is None else raw,
                              self.categories if categories is None else categories,
                              self.numeric if numeric is None else numeric,
                              self.fields, self.meta, 20191115)

    def test_target_stat_values_cannot_enter_context(self):
        first = self.build()
        changed = self.numeric.copy()
        changed[:, [1, 3]] = np.inf
        second = self.build(numeric=changed)
        np.testing.assert_array_equal(first["values"], second["values"])
        self.assertEqual(first["manifest"]["excluded_target_stat_columns"], [self.fields[1], self.fields[3]])
        with self.assertRaises(ValueError):
            self.build(raw=self.raw.assign(label=1))

    def test_history_queries_before_insert_and_expires_by_calendar(self):
        result = past_exposure_features(self.raw)
        np.testing.assert_array_equal(result[:3, :3], 0.)
        np.testing.assert_allclose(result[3, 0], np.log(2.), atol=1e-7)
        self.assertEqual(result[3, 1], 1)
        np.testing.assert_allclose(result[4, 0], np.log(2.), atol=1e-7)
        self.assertEqual(result[4, 2], 1)

    def test_row_permutation_and_future_predictor_isolation(self):
        first = self.build()
        order = np.array([4, 2, 0, 3, 1])
        second = self.build(raw=self.raw.iloc[order], categories=self.categories[order], numeric=self.numeric[order])
        np.testing.assert_array_equal(first["keys"], second["keys"])
        np.testing.assert_allclose(first["values"], second["values"], atol=1e-6)
        np.testing.assert_array_equal(first["rows"][order], second["rows"])
        raw, numeric = self.raw.copy(), self.numeric.copy()
        raw.loc[raw.pt_d == 20191115, "task_id"] = 90000
        numeric[-1, [0, 2]] = 1000.
        future = self.build(raw=raw, numeric=numeric)
        np.testing.assert_array_equal(first["values"][:3], future["values"][:3])

    def test_auxiliary_targets_never_use_validation_labels(self):
        store = self.build()
        labels = np.array([0, 1, 0, 0, 1], dtype="int8")
        first, prior = training_targets(store["keys"], store["rows"], labels, np.arange(4), 20191115)
        labels[-1] = -99
        second, other_prior = training_targets(store["keys"], store["rows"], labels, np.arange(4), 20191115)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(prior, other_prior)
        self.assertEqual(first[-1], -1)
        with self.assertRaises(ValueError):
            training_targets(store["keys"], store["rows"], np.zeros(5), np.arange(5), 20191115)

    def models(self, device="cpu"):
        meta = read(cache_dir(20191115) / "manifest.json")
        torch.manual_seed(42)
        base = build_backbone(meta, "control", .0084).to(device)
        torch.manual_seed(42)
        model = UserDayCTR(meta, .0084, torch.ones(4, 7, device=device), .3, use_context=True).to(device)
        args = (torch.full((4, 27), 2, dtype=torch.long, device=device),
                torch.full((4, 12, 5), 2, dtype=torch.long, device=device), torch.zeros(4, 166, device=device))
        return base, model, args

    def test_initial_predictions_and_cpu_random_stream(self):
        base, model, args = self.models()
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, model.backbone.state_dict()[name]), name)
        torch.manual_seed(9)
        first = base(*args)
        state = torch.random.get_rng_state()
        torch.manual_seed(9)
        second, _ = model(*args, torch.arange(4))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_auxiliary_loss_updates_shared_user_and_rejects_missing_targets(self):
        _, model, args = self.models()
        _, logits = model(*args, torch.arange(4))
        loss = auxiliary_loss(logits, torch.tensor([1., 0., 1., 0.]), torch.tensor([1., 2., 3., 4.]))
        loss.backward()
        self.assertGreater(float(model.backbone.encoder.embeddings["user_id"].weight.grad.norm()), 0.)
        self.assertGreater(float(model.day_encoder[0].weight.grad.norm()), 0.)
        with self.assertRaises(ValueError):
            auxiliary_loss(torch.zeros(2), torch.tensor([-1., 0.]), torch.ones(2))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_initialization_random_stream_and_gradient(self):
        base, model, args = self.models("cuda")
        rows = torch.arange(4, device="cuda")
        torch.cuda.manual_seed_all(19)
        first = base(*args)
        state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed_all(19)
        second, auxiliary = model(*args, rows)
        torch.testing.assert_close(first, second, rtol=0, atol=2e-7)
        self.assertTrue(torch.equal(state, torch.cuda.get_rng_state()))
        auxiliary_loss(auxiliary, torch.tensor([1., 0., 1., 0.], device="cuda"),
                        torch.ones(4, device="cuda")).backward()
        self.assertTrue(torch.isfinite(model.day_encoder[0].weight.grad).all())


if __name__ == "__main__":
    unittest.main()
