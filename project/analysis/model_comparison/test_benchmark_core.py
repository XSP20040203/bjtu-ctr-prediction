"""Small behavioral checks for time leakage, category handling and the model."""
import importlib
import unittest

import numpy as np
import pandas as pd
import torch


class CoreTests(unittest.TestCase):
    def setUp(self):
        try:
            self.core = importlib.import_module("benchmark_core")
        except ModuleNotFoundError:
            self.fail("benchmark_core is not implemented yet")

    def test_real_zero_is_not_padding_and_validation_category_is_unknown(self):
        vocabulary = self.core.make_vocab(["0", "162", "3", "162"])
        self.assertEqual(vocabulary, {"0": 2, "162": 3, "3": 4})
        self.assertEqual(self.core.category_key(10.0), "10")
        self.assertEqual(self.core.category_key("10.0"), "10")
        self.assertEqual(self.core.encode_sequence("0^162^999", vocabulary, 5).tolist(), [2, 3, 1, 0, 0])
        self.assertEqual(self.core.encode_sequence("", vocabulary, 5).tolist(), [0, 0, 0, 0, 0])

    def test_source_history_excludes_same_day_and_preserves_request_order(self):
        feeds = pd.DataFrame({"u_userId": [7, 7, 7], "e_et": [20191114, 20191115, 20191116], "label": [1, -1, 1], "pro": [40, 0, 100], "i_cat": [3, 4, 5]})
        requests = pd.DataFrame({"user_id": [7, 8, 7, 7], "pt_d": [20191116, 20191116, 20191115, 20191114]})
        actual = self.core.source_history_features(feeds, requests)
        expected = np.array([[2, 1, .5, 20, 2, 2, 1], [0, 0, .5, 0, 0, 0, 0], [1, 1, 1, 40, 1, 1, 1], [0, 0, .1, 0, 0, 0, 0]], dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_future_source_rows_cannot_change_past_features(self):
        feeds = pd.DataFrame({"u_userId": [7], "e_et": [20191114], "label": [-1], "pro": [0], "i_cat": [3]})
        future = pd.DataFrame({"u_userId": [7], "e_et": [20191116], "label": [1], "pro": [100], "i_cat": [8]})
        requests = pd.DataFrame({"user_id": [7], "pt_d": [20191115]})
        first = self.core.source_history_features(feeds, requests)
        second = self.core.source_history_features(pd.concat([feeds, future], ignore_index=True), requests)
        np.testing.assert_array_equal(first, second)

    def test_pool_ignores_padding_and_empty_lists_are_finite(self):
        values = torch.tensor([[[2., 4.], [6., 8.], [99., 99.]], [[99., 99.], [99., 99.], [99., 99.]]])
        mask = torch.tensor([[True, True, False], [False, False, False]])
        actual = self.core.masked_mean(values, mask)
        torch.testing.assert_close(actual, torch.tensor([[4., 6.], [0., 0.]]))

    def test_model_is_order_invariant_and_can_backpropagate(self):
        torch.manual_seed(42)
        model = self.core.CTRModel(["current_app"], ["history_apps"], {"current_app": "app", "history_apps": "app"}, {"app": 8}, numeric_dim=2, embedding_dim=4, hidden_dims=[8, 4], dropout=0, unknown_dropout=0)
        model.eval()
        category = torch.tensor([[3], [4]])
        sequence = torch.tensor([[[2, 3, 0]], [[0, 0, 0]]])
        numeric = torch.tensor([[.5, .2], [.1, .3]])
        first = model(category, sequence, numeric)
        second = model(category, sequence.flip(-1), numeric)
        self.assertEqual(tuple(first.shape), (2,))
        torch.testing.assert_close(first, second)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(first, torch.tensor([1., 0.]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        gradient = model.embeddings["app"].weight.grad
        self.assertTrue(torch.isfinite(gradient).all().item())
        torch.testing.assert_close(gradient[0], torch.zeros(4))

    def test_embedding_dictionary_order_does_not_change_seeded_model(self):
        inputs = (torch.tensor([[2, 3]]), torch.empty((1, 0, 5), dtype=torch.long), torch.tensor([[.5]]))
        outputs = []
        for sizes in [{"alpha": 8, "beta": 9}, {"beta": 9, "alpha": 8}]:
            torch.manual_seed(42)
            model = self.core.CTRModel(["a", "b"], [], {"a": "alpha", "b": "beta"}, sizes, numeric_dim=1, embedding_dim=4, hidden_dims=[8], dropout=0, unknown_dropout=0)
            model.eval()
            outputs.append(model(*inputs))
        torch.testing.assert_close(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
