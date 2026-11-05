import unittest

import numpy as np
import pandas as pd
import torch

from train_weighted import corrected_probability, training_targets


class Contracts(unittest.TestCase):
    def test_weight_correction_preserves_ranking_and_recovers_prior(self):
        for weight in [1., 5., 20.]:
            probabilities = torch.tensor([.00001, .001, .02, .25, .8], dtype=torch.float64)
            tilted = torch.logit(probabilities) + np.log(weight)
            recovered = corrected_probability(tilted, weight)
            torch.testing.assert_close(recovered, probabilities, atol=1e-14, rtol=1e-12)
            self.assertTrue(torch.all(torch.diff(recovered) > 0))

    def test_auxiliary_targets_ignore_held_out_responses(self):
        keys = np.array([[20191115, 1], [20191115, 2], [20191116, 1]])
        groups = np.array([0, 0, 1, 2, 2])
        train = np.array([0, 1, 2])
        for labels in [np.array([1, 0, 0, 1, 1]), np.array([1, 0, 0, -1, -1])]:
            targets, prior = training_targets(keys, groups, labels, train, 20191116)
            np.testing.assert_array_equal(targets, [1, 0, -1])
            self.assertEqual(prior, .5)

    def test_auxiliary_rejects_held_day_training_rows(self):
        with self.assertRaises(ValueError):
            training_targets(np.array([[20191116, 1]]), np.array([0]), np.array([1]),
                             np.array([0]), 20191116)


if __name__ == "__main__":
    unittest.main()
