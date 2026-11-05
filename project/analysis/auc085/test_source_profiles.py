import importlib
import unittest

import pandas as pd


class SourceProfileTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("source_profiles")
        except ModuleNotFoundError:
            self.fail("Daily source-profile features are pending")
        self.feeds = pd.DataFrame({
            "u_userId": [8, 8, 8], "e_et": [20191114, 20191115, 20191115],
            "label": [1, -1, 1], "pro": [100, 0, 80], "cillabel": [-1, -1, 1],
            "i_cat": [10, 20, 30], "i_s_sourceId": ["A", "B", "C"],
            "e_rn": [1, 1, 3], "e_po": [1, 2, 4],
            "u_phonePrice": [2, 2, 2], "u_browserLifeCycle": [3, 3, 3], "u_browserMode": [4, 4, 4],
            "u_newsCatInterests": ["99", "0^10^20", "0^10^30"],
            "u_newsCatDislike": ["99", "20", "20"], "u_click_ca2_news": ["99", "10", "30"]})

    def test_profiles_use_only_matching_day_and_preserve_real_zero_topic(self):
        result = self.module.build_profiles(self.feeds)
        row = result.loc[20191115 * 10000000 + 8]
        self.assertEqual(row["feed_day_impressions"], 2)
        self.assertEqual(row["feed_day_clicks"], 1)
        self.assertEqual(row["feed_day_progress_mean"], 40)
        self.assertEqual(row["feed_clicked_topics"], "30")
        self.assertEqual(row["feed_long_interest"], "0^10^20^30")
        self.assertAlmostEqual(row["feed_day_ctr_smoothed"], .5)

    def test_future_source_changes_do_not_modify_an_earlier_profile(self):
        before = self.module.build_profiles(self.feeds).loc[20191115 * 10000000 + 8]
        extra = self.feeds.iloc[[0]].copy()
        extra["e_et"] = 20191117
        after = self.module.build_profiles(pd.concat([self.feeds, extra], ignore_index=True)).loc[20191115 * 10000000 + 8]
        pd.testing.assert_series_equal(before, after)

    def test_overlap_counts_unique_topics_and_real_zero(self):
        self.assertEqual(self.module.interest_overlap("0^10^10", "10^30", "0^10", "20"), (.5, 1., 0.))


if __name__ == "__main__":
    unittest.main()
