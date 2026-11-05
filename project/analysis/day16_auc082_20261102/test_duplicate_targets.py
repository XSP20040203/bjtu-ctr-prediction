import unittest
import numpy as np
import torch
from duplicate_targets import duplicate_targets


def fixture():
    return {"train": np.array([0, 1, 2]), "valid": np.array([3]),
            "categories": np.array([[4], [4], [5], [4]], dtype="int32"),
            "sequences": np.zeros((4, 1, 2), dtype="int32"),
            "numeric": np.ones((4, 2), dtype="float32"),
            "day_rows": np.array([0, 0, 1, 2], dtype="int32"),
            "label": np.array([0, 1, 0, -1])}


class DuplicateContracts(unittest.TestCase):
    def test_true_outcome_mean_and_mass(self):
        target, info = duplicate_targets(fixture())
        np.testing.assert_array_equal(target, [.5, .5, 0, -1])
        self.assertEqual(info["duplicate_extra_rows"], 1)
        self.assertEqual(info["observed_click_mass"], info["averaged_target_mass"])

    def test_equal_input_bce_value_and_gradient_preserved(self):
        logits = torch.tensor([.3, -.8], dtype=torch.float64, requires_grad=True)
        per_row = logits[torch.tensor([0, 0, 1])]
        hard = torch.nn.functional.binary_cross_entropy_with_logits(per_row, torch.tensor([0., 1., 0.], dtype=torch.float64))
        soft = torch.nn.functional.binary_cross_entropy_with_logits(per_row, torch.tensor([.5, .5, 0.], dtype=torch.float64))
        torch.testing.assert_close(hard, soft)
        a = torch.autograd.grad(hard, logits, retain_graph=True)[0]
        b = torch.autograd.grad(soft, logits)[0]
        torch.testing.assert_close(a, b)

    def test_distinct_day_context_is_not_averaged(self):
        data = fixture()
        data["day_rows"][1] = 9
        target, info = duplicate_targets(data)
        np.testing.assert_array_equal(target, [0, 1, 0, -1])
        self.assertEqual(info["duplicate_extra_rows"], 0)

    def test_held_row_cannot_enter_fit(self):
        data = fixture()
        data["train"] = np.array([0, 1, 2, 3])
        data["label"][3] = 1
        with self.assertRaises(ValueError):
            duplicate_targets(data)

    def test_random_noncontiguous_rows_match_independent_byte_key_groups(self):
        rng = np.random.default_rng(413)
        n = 90
        identities = rng.integers(0, 12, size=n)
        prototypes = rng.normal(size=(12, 5)).astype("float32")
        data = {"categories": identities[:,None].astype("int32"),
                "sequences": np.repeat(identities[:,None,None], 6, axis=2).astype("int32"),
                "numeric": prototypes[identities], "day_rows": identities.astype("int32"),
                "label": rng.integers(0,2,size=n)}
        data["train"] = rng.permutation(np.arange(n)[np.arange(n)%4 != 0])
        data["valid"] = np.arange(n)[np.arange(n)%4 == 0]
        data["label"][data["valid"]] = -1
        members = {}
        for row in data["train"]:
            key = tuple(np.ascontiguousarray(data[field][row]).tobytes()
                        for field in ["categories", "sequences", "numeric", "day_rows"])
            members.setdefault(key, []).append(int(row))
        expected = np.full(n, -1., dtype="float64")
        for rows in members.values():
            expected[rows] = data["label"][rows].mean()
        target, info = duplicate_targets(data)
        np.testing.assert_array_equal(target, expected)
        self.assertEqual(info["unique_exact_input_groups"],len(members))


if __name__ == "__main__":
    unittest.main()
