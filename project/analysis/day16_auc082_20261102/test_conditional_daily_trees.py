import unittest
import numpy as np
from conditional_daily_trees import conditional_loss, lgb


class ConditionalTreeObjective(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0., 2., 1., 1., 0.])
        self.groups = np.array([0, 0, 0, 1, 1])
        self.starts = np.array([0, 3])
        self.totals = np.array([3., 1.])
        self.offset = np.log(np.array([.5, .8, 1., .3, .9]))

    def test_gradient_and_diagonal_hessian_match_finite_differences(self):
        z = np.array([.2, -.1, .4, -.3, .7])
        def objective(values):
            return conditional_loss(values, self.y, self.groups, self.starts, self.totals, self.offset)
        _, gradient, hessian = objective(z)
        for row in range(len(z)):
            delta = np.zeros(len(z))
            delta[row] = 1e-5
            plus = objective(z+delta)
            minus = objective(z-delta)
            self.assertAlmostEqual(gradient[row], (plus[0]-minus[0])/2e-5, places=8)
            self.assertAlmostEqual(hessian[row], (plus[1][row]-minus[1][row])/2e-5, places=8)
        np.testing.assert_allclose(np.bincount(self.groups, weights=gradient), 0., atol=1e-14)

    def test_per_user_score_shifts_leave_derivatives_unchanged(self):
        base = conditional_loss(np.zeros(5), self.y, self.groups, self.starts, self.totals, self.offset)
        shifted = conditional_loss(np.array([3., -5.])[self.groups], self.y, self.groups,
                                   self.starts, self.totals, self.offset)
        self.assertAlmostEqual(base[0], shifted[0], places=12)
        np.testing.assert_allclose(base[1], shifted[1], atol=1e-14)
        np.testing.assert_allclose(base[2], shifted[2], atol=1e-14)

    def test_coupled_booster_learns_and_serializes(self):
        rng = np.random.default_rng(15239)
        x = rng.normal(size=(500, 3))
        groups = np.repeat(np.arange(100), 5)
        starts = np.arange(0, 500, 5)
        y = np.zeros(500)
        for start in starts:
            score = np.exp(1.3*(x[start:start+5, 0] > 0))
            y[start:start+5] = rng.multinomial(3, score/score.sum())
        totals = np.full(100, 3.)
        offset = np.zeros(500)
        def objective(prediction, dataset):
            _, gradient, hessian = conditional_loss(prediction, y, groups, starts, totals, offset)
            return gradient, hessian
        params = {'objective': objective, 'learning_rate': .05, 'num_leaves': 7,
                  'min_data_in_leaf': 20, 'lambda_l2': 10., 'num_threads': 2,
                  'verbosity': -1, 'seed': 32, 'force_col_wise': True}
        model = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=80)
        p = model.predict(x, raw_score=True, num_threads=2)
        initial = conditional_loss(np.zeros(500), y, groups, starts, totals, offset)[0]
        final = conditional_loss(p, y, groups, starts, totals, offset)[0]
        self.assertLess(final, .95*initial)
        replay = lgb.Booster(model_str=model.model_to_string()).predict(x, raw_score=True, num_threads=2)
        np.testing.assert_array_equal(p, replay)


if __name__ == '__main__':
    unittest.main()
