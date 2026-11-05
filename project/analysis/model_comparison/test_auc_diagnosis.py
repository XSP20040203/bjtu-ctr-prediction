"""Hand-computed checks for separating ranking errors and probability errors."""
import importlib
import unittest

import numpy as np


class AUCDiagnosisTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("diagnose_auc")
        except ModuleNotFoundError:
            self.fail("AUC diagnosis helpers are pending")

    def test_pair_ranks_count_ties_as_half_and_preserve_positive_order(self):
        labels = np.array([1, 0, 0, 1])
        probability = np.array([.8, .4, .8, .3])
        np.testing.assert_allclose(self.module.positive_pair_ranks(labels, probability), [.75, 0.])

    def test_probability_losses_recombine_with_observed_class_frequencies(self):
        labels = np.array([1, 0, 0, 0])
        probability = np.array([.8, .2, .2, .2])
        result = self.module.prediction_summary(labels, probability)
        self.assertAlmostEqual(result["auc"], 1.)
        self.assertAlmostEqual(result["logloss"], -np.log(.8))
        self.assertAlmostEqual(result["positive_logloss"] * .25 + result["negative_logloss"] * .75,
                               result["logloss"])

    def test_user_mask_neutralizes_identity_without_changing_cached_data(self):
        self.assertTrue(hasattr(self.module, "category_batch"), "User-ID ablation is pending")
        original = np.array([[2, 3], [4, 5]], dtype=np.int32)
        masked = self.module.category_batch(original, np.array([1, 0]), user_column=0)
        np.testing.assert_array_equal(masked, [[1, 5], [1, 3]])
        np.testing.assert_array_equal(original, [[2, 3], [4, 5]])


if __name__ == "__main__":
    unittest.main()
