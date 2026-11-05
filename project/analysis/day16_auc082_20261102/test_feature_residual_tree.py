import unittest
import numpy as np
import pandas as pd
from feature_residual_tree import CATEGORIES, HISTORIES, scalar_and_history


class RawFeatureContracts(unittest.TestCase):
    def test_exact_tokens_missing_values_and_row_alignment(self):
        raw = pd.DataFrame({name:[2,3] for name in CATEGORIES})
        raw["task_id"] = [12,123]
        for name in HISTORIES:
            raw[name] = ["12^2^8","-1"]
        x,names = scalar_and_history(raw)
        np.testing.assert_array_equal(x[:,names.index("ad_click_list_v001_candidate_match")],[1,0])
        np.testing.assert_array_equal(x[:,names.index("ad_click_list_v001_length")],[3,0])
        shifted,shift_names = scalar_and_history(raw.iloc[::-1])
        self.assertEqual(names,shift_names)
        np.testing.assert_array_equal(x[::-1],shifted)

    def test_outcomes_rejected(self):
        with self.assertRaises(ValueError):
            scalar_and_history(pd.DataFrame({"label":[1]}))


if __name__ == "__main__":
    unittest.main()
