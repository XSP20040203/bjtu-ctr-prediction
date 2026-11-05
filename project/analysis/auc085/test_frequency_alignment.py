import importlib
import unittest

import numpy as np


class FrequencyAlignmentTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("align_frequency_features")
        except ModuleNotFoundError:
            self.fail("Training-input-only frequency alignment is pending")

    def test_only_training_inputs_determine_counts(self):
        keys = np.array([10, 10, 20, 20, 10, 10, 30])
        result = self.module.training_input_frequency(keys, np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(result, [2, 2, 2, 2, 2, 2, 0])

    def test_validation_frequency_does_not_inflate_training_counts(self):
        keys = np.array([10, 10, 20, 20] + [10] * 100)
        result = self.module.training_input_frequency(keys, np.arange(4))
        self.assertTrue((result == 2).all())


if __name__ == "__main__":
    unittest.main()
