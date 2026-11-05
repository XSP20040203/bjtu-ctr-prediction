import unittest
import numpy as np
import pandas as pd
from conditional_daily import objective, standardize, mask_ad, activity, daily_targets


class ConditionalDailyBoundaries(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[.4, -.1], [.2, .5], [-.3, .8], [.1, -.2], [.7, .4]])
        self.y = np.array([0., 2., 1., 1., 0.])
        self.groups = np.array([0, 0, 0, 1, 1])
        self.starts = np.array([0, 3])
        self.totals = np.array([3., 1.])
        self.offset = np.log(np.array([.5, .8, 1., .3, .9]))
        self.beta = np.array([.2, -.15])

    def loss(self, beta, offset=None, x=None):
        return objective(beta, self.x if x is None else x, self.y, self.groups,
                         self.starts, self.totals, self.offset if offset is None else offset, .05)

    def test_gradient_matches_finite_differences(self):
        _, gradient = self.loss(self.beta)
        finite = np.empty_like(gradient)
        for column in range(len(self.beta)):
            delta = np.zeros(len(self.beta))
            delta[column] = 1e-6
            finite[column] = (self.loss(self.beta+delta)[0]-self.loss(self.beta-delta)[0])/2e-6
        np.testing.assert_allclose(gradient, finite, rtol=0, atol=2e-9)

    def test_user_constants_cancel_from_conditional_likelihood(self):
        before = self.loss(self.beta)
        after = self.loss(self.beta, offset=self.offset+np.array([4., -2.])[self.groups])
        self.assertAlmostEqual(before[0], after[0], places=13)
        np.testing.assert_allclose(before[1], after[1], rtol=0, atol=1e-13)
        shifted = self.loss(self.beta, x=self.x+np.array([[3., 7.], [-5., 2.]])[self.groups])
        self.assertAlmostEqual(before[0], shifted[0], places=13)
        np.testing.assert_allclose(before[1], shifted[1], rtol=0, atol=1e-13)

    def test_query_features_do_not_change_training_standardization(self):
        users = np.array([0, 0, 1, 1, 1])
        training = np.array([True, False, True, True, False])
        before, means, scale = standardize(self.x, users, training)
        changed = self.x.copy()
        changed[~training] = 100000.
        after, means2, scale2 = standardize(changed, users, training)
        np.testing.assert_array_equal(means, means2)
        np.testing.assert_array_equal(scale, scale2)
        np.testing.assert_array_equal(before[training], after[training])

    def test_current_and_future_ad_targets_are_masked(self):
        raw = pd.DataFrame({'pt_d': [20191110, 20191111, 20191112], 'user_id': [1, 1, 2],
                            'spread_app_id': [2, 2, 3], 'slot_id': [4, 4, 5], 'label': [1, 0, 1]})
        before = mask_ad(raw, 20191111)
        changed = raw.copy()
        changed.loc[changed.pt_d >= 20191111, 'label'] = [1, 0]
        pd.testing.assert_frame_equal(before, mask_ad(changed, 20191111))
        self.assertEqual(before.label.tolist(), [1, -1])

    def test_future_news_is_excluded(self):
        source = pd.DataFrame({'e_et': [20191110, 20191110, 20191111], 'u_userId': [1, 2, 1],
                               'label': [1, -1, 1], 'cillabel': [-1, -1, 1], 'pro': [50, 0, 100]})
        keys = np.array([[20191110, 1], [20191110, 2], [20191110, 3]])
        before, names = activity(source, keys, np.array([10., 5., 8.]), 20191110)
        changed = source.copy()
        changed.loc[changed.e_et == 20191111, ['label', 'pro']] = [-1, 0]
        after, names2 = activity(changed, keys, np.array([10., 5., 8.]), 20191110)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(names, names2)
        self.assertEqual(before[-1, -1], 1.)

    def test_zero_effect_gives_identity_relative_target(self):
        data = {'x': self.x, 'training': np.array([True, True, True, False, False]),
                'users': np.array([0, 0, 1, 0, 2]), 'mass': np.array([.2, .4, .3, .5, .8]),
                'counts': np.array([1., 0., 0., -1., -1.]),
                'row_groups': np.array([3, 3, 4]), 'contract': {'activity_columns': 2}}
        base = np.array([.02, .03, .1])
        targets = daily_targets(data, np.zeros(2), {'scope': 'activity'}, base)
        expected = np.bincount(data['row_groups'], weights=-np.log1p(-base), minlength=5)
        np.testing.assert_array_equal(targets['relative_dynamic'], expected)


if __name__ == '__main__':
    unittest.main()
