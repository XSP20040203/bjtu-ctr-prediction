import unittest
import numpy as np
import torch
from pairwise_neural_objective import class_pools, sample_rows, ranking_loss


class RankingContracts(unittest.TestCase):
    def test_exact_pair_risk_gradient_shift_and_population_bce(self):
        values = torch.tensor([.2, -.4, .6, -1.1, .3], dtype=torch.float64,
                              requires_grad=True)
        total, pair, bce = ranking_loss(values, 2, .02, .7, .25, 5.)
        expected = sum(torch.nn.functional.softplus((values[j]-values[i])/.7)
                       for i in range(2) for j in range(2, 5)) / 6
        torch.testing.assert_close(pair, expected)
        labels = torch.tensor([1., 1., 0., 0., 0.], dtype=torch.float64)
        weights = torch.tensor([.02*5/2]*2 + [.98/3]*3, dtype=torch.float64)
        ordinary = (weights * torch.nn.functional.binary_cross_entropy_with_logits(
            values, labels, reduction="none")).sum() / 1.08
        torch.testing.assert_close(bce, ordinary)
        grad = torch.autograd.grad(pair, values, retain_graph=True)[0]
        self.assertTrue(bool((grad[:2] < 0).all() and (grad[2:] > 0).all()))
        self.assertAlmostEqual(float(grad.sum()), 0., places=12)
        shifted = ranking_loss(values+2., 2, .02, .7, 0., 5.)[1]
        torch.testing.assert_close(pair, shifted)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: ranking_loss(x, 2, .02, .7, .25, 5.)[0], (values,)))
        self.assertTrue(bool(torch.isfinite(total)))

    def test_sampler_excludes_held_labels_and_is_reproducible(self):
        labels = np.array([0, 1, 0, 1, -1])
        dates = np.array([20191114, 20191115, 20191115, 20191115, 20191116])
        pos, neg = class_pools(labels, np.arange(4), dates)
        first = sample_rows(pos, neg, np.random.default_rng(42), 20, 30)
        second = sample_rows(pos, neg, np.random.default_rng(42), 20, 30)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(labels[first[:20]] == 1))
        self.assertTrue(np.all(labels[first[20:]] == 0))
        with self.assertRaises(ValueError):
            class_pools(labels, np.arange(5), dates)
        with self.assertRaises(ValueError):
            class_pools(labels, np.array([1, 1, 2]), dates)


if __name__ == "__main__":
    unittest.main()
