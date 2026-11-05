import unittest
import io
import numpy as np
import pandas as pd
import torch
from row_deepfm_core_v1 import fit_schema, transform, prior_rows, DeepFM


class TestDeepFM(unittest.TestCase):
    def test_train_only_schema_unknowns_and_future_targets(self):
        train = pd.DataFrame({'cat': [8, 3, 8], 'numeric': [1., 2., 3.]})
        held = pd.DataFrame({'cat': [3, 999], 'numeric': [2., 1e6]})
        schema = fit_schema(train, ['cat'])
        numeric, categories = transform(held, schema)
        np.testing.assert_array_equal(categories[:, 0], [1, 0])
        self.assertEqual(schema['mean'], [2.])
        np.testing.assert_array_equal(numeric, transform(held, schema, True)[0])
        np.testing.assert_array_equal(categories, transform(held, schema, True)[1])
        self.assertEqual(numeric[1, 0], 10.)
        np.testing.assert_array_equal(prior_rows([13, 14, 15], [1, 0, -1], 15), [0, 1])
        with self.assertRaises(ValueError):
            prior_rows([13, 14, 15], [1, 0, 1], 15)

    def test_interaction_gradients_serialization_and_unknown_embedding(self):
        torch.manual_seed(151658)
        model = DeepFM(2, [4, 3], 8)
        numeric = torch.randn(6, 2)
        categories = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 1], [1, 2]])
        model.train()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(numeric, categories), torch.arange(6).float() % 2)
        loss.backward()
        for layer in model.embeddings:
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertEqual(float(layer.weight.grad[0].abs().sum()), 0.)
        model.eval()
        before = model(numeric, categories).detach()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = DeepFM(2, [4, 3], 8)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        restored.eval()
        torch.testing.assert_close(before, restored(numeric, categories), rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
