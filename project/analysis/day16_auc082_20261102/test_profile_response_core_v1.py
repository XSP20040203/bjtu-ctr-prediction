"""Verify time boundaries, original row ordering, and independent pair counts."""
import unittest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import logit
from profile_response_core_v1 import CHANNELS, TARGETS, NAMES, build_day, pooled_features


class ProfileResponseTest(unittest.TestCase):
    def setUp(self):
        self.ads = pd.DataFrame({channel: ['1^2', '2', '-1', '1', '3', '-1', '999'] for channel in CHANNELS})
        for target in TARGETS:
            self.ads[target] = [1, 1, 2, 1, 2, 1, 999]
        self.ads['pt_d'] = [10, 10, 10, 11, 11, 11, 12]
        self.y = np.array([1, 0, 1, -1, -1, -1, -1])
        self.query = np.array([5, 3, 4])

    def test_independent_statistics_and_order(self):
        a = build_day(self.ads, self.y, 11, self.query)
        b = build_day(self.ads, self.y, 11, self.query, True)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[::-1], build_day(self.ads, self.y, 11, self.query[::-1]))
        self.assertEqual(a.shape, (3, len(NAMES)))
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_array_equal(a[0, 7::8], 1)

    def test_future_profiles_do_not_change_features(self):
        a = build_day(self.ads, self.y, 11, self.query)
        changed = self.ads.copy()
        changed.loc[6, CHANNELS] = '1^2^3'
        changed.loc[6, TARGETS] = 1
        np.testing.assert_array_equal(a, build_day(changed, self.y, 11, self.query))

    def test_current_labels_are_rejected(self):
        y = self.y.copy()
        y[3] = 1
        with self.assertRaisesRegex(ValueError, 'must be masked'):
            build_day(self.ads, y, 11, self.query)

    def test_hand_computed_single_token_response(self):
        exposure = csr_matrix(np.array([[1., 0.], [2., 0.]]))
        clicks = csr_matrix(np.array([[1., 0.], [1., 0.]]))
        query = csr_matrix(np.array([[1., 0.], [0., 0.]]))
        global_rate = 2.5 / 53
        result = pooled_features(query, np.array([0, 0]), exposure, clicks,
                                 np.array([2, 1]), np.array([1, 1]), global_rate)
        target_prior = (1 + 50 * global_rate) / 52
        token_prior = (1 + 20 * global_rate) / 21
        expected = logit((1 + 25 * (target_prior + token_prior)) / 51) - logit(target_prior)
        np.testing.assert_allclose(result[0, :3], expected, rtol=0, atol=1e-14)
        self.assertAlmostEqual(result[0, 4], 1 / 51)
        self.assertAlmostEqual(result[0, 5], np.log(2))
        np.testing.assert_array_equal(result[1], [0, 0, 0, 0, 0, 0, 0, 1])


if __name__ == '__main__':
    unittest.main()
