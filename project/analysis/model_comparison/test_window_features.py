"""Check past-only calendar windows and missing-history sentinel semantics."""
import importlib
import unittest

import numpy as np
import pandas as pd


class WindowFeatureTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("window_features")
        except ModuleNotFoundError:
            self.fail("Fixed-window feature preparation is pending")

    def test_boundary_includes_three_days_ago_excludes_today_and_old_history(self):
        feeds = pd.DataFrame({"u_userId": [8, 8, 8, 8], "e_et": [20191110, 20191112, 20191113, 20191115],
                              "label": [1, -1, 1, 1], "pro": [100, 0, 80, 100], "i_cat": [1, 2, 3, 4]})
        requests = pd.DataFrame({"user_id": [8, 99], "pt_d": [20191115, 20191115]})
        result = self.module.fixed_window_source_features(feeds, requests, window_days=3)
        np.testing.assert_allclose(result[0], [2, 1, .5, 40, 2, 2, 2])
        np.testing.assert_allclose(result[1], [0, 0, .5, 0, 0, 0, 4])

    def test_same_day_and_future_rows_cannot_change_window_features(self):
        feeds = pd.DataFrame({"u_userId": [8, 8], "e_et": [20191114, 20191116],
                              "label": [-1, 1], "pro": [0, 90], "i_cat": [2, 3]})
        requests = pd.DataFrame({"user_id": [8], "pt_d": [20191115]})
        before = self.module.fixed_window_source_features(feeds, requests, window_days=3)
        feeds.loc[1, ["e_et", "label", "pro"]] = [20191115, -1, 0]
        after = self.module.fixed_window_source_features(feeds, requests, window_days=3)
        np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
