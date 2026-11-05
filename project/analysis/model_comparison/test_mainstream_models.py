"""Check interaction arithmetic, numeric features and safe history handling."""
import importlib
import unittest

import torch


class MainstreamModelTests(unittest.TestCase):
    def setUp(self):
        try:
            self.models = importlib.import_module("mainstream_models")
        except ModuleNotFoundError:
            self.fail("DeepFM and DCN-v2 implementations are pending")

    def test_fm_counts_each_pair_once_and_excludes_self_interactions(self):
        fields = torch.tensor([[[1., 2.], [3., 4.], [0., 0.]],
                               [[2., 0.], [-1., 3.], [4., 1.]]])
        torch.testing.assert_close(self.models.fm_interaction(fields), torch.tensor([11., 5.]))
        torch.testing.assert_close(self.models.fm_interaction(fields[:, :1]), torch.zeros(2))

    def test_cross_layer_uses_matrix_and_bias_inside_the_product(self):
        layer = self.models.CrossLayerV2(2)
        with torch.no_grad():
            layer.linear.weight.copy_(torch.tensor([[1., 2.], [3., 4.]]))
            layer.linear.bias.copy_(torch.tensor([.5, -1.]))
        result = layer(torch.tensor([[2., 3.]]), torch.tensor([[5., 7.]]))
        torch.testing.assert_close(result, torch.tensor([[44., 133.]]))

    def test_deepfm_includes_numeric_category_interactions(self):
        model = self.models.DeepFMModel(["task_id"], ["history"],
                                       {"task_id": "task", "history": "task"}, {"task": 8},
                                       numeric_dim=1, embedding_dim=2, hidden_dims=(4,),
                                       dropout=0., unknown_dropout=0.)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.encoder.embeddings["task"].weight[2].copy_(torch.tensor([1., 2.]))
            model.numeric_embeddings[0].copy_(torch.tensor([3., 4.]))
        categories = torch.tensor([[2], [2]])
        sequences = torch.zeros((2, 1, 3), dtype=torch.long)
        result = model(categories, sequences, torch.tensor([[2.], [0.]]))
        torch.testing.assert_close(result, torch.tensor([22., 0.]))

    def test_models_ignore_padding_order_and_support_unknown_and_empty_history(self):
        categories = torch.tensor([[2, 3], [1, 2], [4, 1]])
        sequences = torch.tensor([[[2, 3, 0]], [[0, 0, 0]], [[1, 4, 0]]])
        numeric = torch.tensor([[.5, -.3], [0., 1.], [-.2, .4]])
        for builder in (self.models.DeepFMModel, self.models.DCNV2Model):
            with self.subTest(builder=builder.__name__):
                torch.manual_seed(42)
                model = builder(["task_id", "age"], ["history"],
                                {"task_id": "task", "age": "age", "history": "task"},
                                {"task": 8, "age": 6}, numeric_dim=2, embedding_dim=4,
                                hidden_dims=(8, 4), dropout=0., unknown_dropout=.01,
                                initial_ctr=.01)
                model.eval()
                original = model(categories, sequences, numeric)
                moved_padding = torch.cat([sequences.flip(-1), torch.zeros((3, 1, 2), dtype=torch.long)], dim=-1)
                torch.testing.assert_close(original, model(categories, moved_padding, numeric))
                self.assertTrue(torch.isfinite(original).all())
                model.train()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(categories, sequences, numeric), torch.tensor([1., 0., 1.]))
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                for table in model.encoder.embeddings.values():
                    torch.testing.assert_close(table.weight.grad[0], torch.zeros_like(table.weight.grad[0]))


if __name__ == "__main__":
    unittest.main()
