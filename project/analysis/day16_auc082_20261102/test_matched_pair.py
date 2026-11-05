import unittest
import numpy as np
import torch
from matched_pair_objective import make_pool, sample, objective


class MatchedPairContracts(unittest.TestCase):
    def test_matching_and_temporal_isolation(self):
        labels = np.array([1, 0, 1, 0, 0, 1, -1])
        dates = np.array([20191114, 20191114, 20191115, 20191115, 20191115, 20191115, 20191116])
        users = np.array([1, 1, 1, 2, 2, 3, 1])
        for scope in ["user", "userday"]:
            pool = make_pool(labels, np.arange(6), dates, users, scope)
            rows = sample(pool, np.random.default_rng(9), 16, 5, 12)
            pos, neg = rows[:16], rows[16:96].reshape(16, 5)
            np.testing.assert_array_equal(users[neg], np.repeat(users[pos, None], 5, axis=1))
            if scope == "userday":
                np.testing.assert_array_equal(dates[neg], np.repeat(dates[pos, None], 5, axis=1))
            self.assertTrue(np.all(labels[pos] == 1) and np.all(labels[rows[16:]] == 0))
            self.assertTrue(np.all(dates[rows] < 20191116))
        with self.assertRaises(ValueError):
            make_pool(labels, np.arange(7), dates, users, "user")

    def test_pair_gradient_and_exact_formula(self):
        x = torch.tensor([.1, -.2, .8, -.4, .3, -.7, .5, .2], dtype=torch.float64, requires_grad=True)
        total, pair, _ = objective(x, 2, 2, .02, .7)
        expected = torch.stack([torch.nn.functional.softplus((x[j]-x[i])/.7)
                                for i, ids in enumerate([[2, 3], [4, 5]]) for j in ids]).mean()
        torch.testing.assert_close(pair, expected)
        self.assertTrue(torch.autograd.gradcheck(lambda z: objective(z, 2, 2, .02, .7)[0], (x,)))
        self.assertTrue(bool(torch.isfinite(total)))


if __name__ == "__main__":
    unittest.main()
