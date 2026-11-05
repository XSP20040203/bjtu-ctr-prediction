import unittest
import numpy as np
import torch
from recurrent_userday import sequences, RecurrentDaily, SequencePack


class RecurrentContracts(unittest.TestCase):
    def test_current_and_future_targets_do_not_enter_current_inputs(self):
        keys = np.array([[20191113, 1], [20191115, 1], [20191116, 1], [20191115, 2]])
        values = np.arange(12, dtype="float32").reshape(4, 3)
        counts = np.array([1., 2., -1., 0.])
        sizes = np.array([20, 30, 40, 10])
        first = sequences(values, keys, counts, sizes, 20191116)
        changed = counts.copy()
        changed[1] = 19.
        second = sequences(values, keys, changed, sizes, 20191116)
        np.testing.assert_array_equal(first[0][0, :2], second[0][0, :2])
        self.assertNotEqual(first[0][0, 2, -4], second[0][0, 2, -4])
        changed[2] = 999.
        third = sequences(values, keys, changed, sizes, 20191116)
        np.testing.assert_array_equal(second[0], third[0])
        np.testing.assert_array_equal(first[1][first[2] == 20191116], [-1.])
        self.assertAlmostEqual(float(first[0][0, 1, -1]), np.log(3), places=6)

    def test_recurrent_outputs_are_causal_and_differentiable(self):
        torch.manual_seed(42)
        model = RecurrentDaily(5, 16).double().eval()
        x = torch.randn(3, 4, 5, dtype=torch.float64, requires_grad=True)
        baseline = model(x)
        changed = x.detach().clone()
        changed[:, 2:] += 5.
        torch.testing.assert_close(baseline[:, :2], model(changed)[:, :2], rtol=0, atol=0)
        baseline.sum().backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        pack = SequencePack((x.detach().numpy(), np.zeros((3,4)), np.zeros((3,4)), np.zeros((12,2))))
        pack.validation_targets = np.zeros((3,4))
        self.assertEqual(pack.validation_targets.shape, (3,4))


if __name__ == "__main__":
    unittest.main()
