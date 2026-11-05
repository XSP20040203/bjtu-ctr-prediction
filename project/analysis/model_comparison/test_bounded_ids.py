"""Verify deterministic field-vector caps and finite gradients."""
import importlib
import unittest

import torch


class BoundedIDTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("bounded_id_models")
        except ModuleNotFoundError:
            self.fail("Bounded ID representation is pending")

    def test_caps_selected_field_preserving_direction_and_other_fields(self):
        fields = torch.tensor([[[3., 4.], [6., 8.]], [[0., 0.], [1., 2.]]], requires_grad=True)
        bounded = self.module.bound_field_vectors(fields, ["user_id", "ad_task"], {"user_id": 2.})
        torch.testing.assert_close(bounded, torch.tensor([[[1.2, 1.6], [6., 8.]], [[0., 0.], [1., 2.]]]))
        bounded.sum().backward()
        self.assertTrue(torch.isfinite(fields.grad).all())

    def test_empty_caps_preserve_values_exactly(self):
        fields = torch.randn(3, 2, 4)
        torch.testing.assert_close(self.module.bound_field_vectors(fields, ["user_id", "ad_task"], {}), fields,
                                   rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
