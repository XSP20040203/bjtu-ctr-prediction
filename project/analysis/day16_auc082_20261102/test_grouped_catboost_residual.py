import unittest
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from grouped_catboost_residual import components, predict, independent, group_codes


class GroupedCatboostMath(unittest.TestCase):
    def setUp(self):
        self.base = np.array([.001, .01, .2, .004, .03, .1])
        self.forecast = np.array([.002, .08, .1, .02, .004, .14])
        self.groups = np.array([0, 0, 1, 1, 1, 2])

    def test_equal_weights_reproduce_uniform_normalized_logit_blending(self):
        a, mean, centered = components(self.base, self.forecast, self.groups)
        for weight in [-.1, 0., .1, .5]:
            q = predict((a, mean, centered), weight, weight)
            np.testing.assert_allclose(q, expit(a+weight*(mean+centered)), rtol=0, atol=1e-16)
        np.testing.assert_allclose(predict((a, mean, centered), 0., 0.), self.base, rtol=0, atol=1e-16)

    def test_centered_residual_has_zero_mean_and_is_permutation_invariant(self):
        before = components(self.base, self.forecast, self.groups)
        np.testing.assert_allclose(np.bincount(self.groups, weights=before[2]), 0., rtol=0, atol=1e-14)
        order = np.array([5, 1, 3, 0, 4, 2])
        after = components(self.base[order], self.forecast[order], self.groups[order])
        np.testing.assert_allclose(predict(after, .1, .5), predict(before, .1, .5)[order], rtol=0, atol=1e-16)

    def test_numpy_and_independent_pandas_aggregations_agree(self):
        frame = pd.DataFrame({'user_id':self.groups,'task_id':[7,8,7,8,7,8],
            'slot_id':[1,1,2,1,2,1],'spread_app_id':[5,6,5,6,5,6]})
        for group, fields in [('user',['user_id']),('user_app',['user_id','spread_app_id']),('task',['task_id'])]:
            codes = group_codes(frame, fields)
            parts = components(self.base, self.forecast, codes)
            for between, within in [(-.1,.5),(.1,.25),(.5,0.)]:
                recipe = dict(group=group, between=between, within=within)
                np.testing.assert_allclose(predict(parts, between, within),
                    independent(frame, self.base, self.forecast, recipe), rtol=0, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
