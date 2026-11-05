"""Protecting a trained backbone must actually prevent optimizer updates."""
import importlib
import unittest

import torch


class ProtectedAdapterTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("protected_adapters")
        except ModuleNotFoundError:
            self.fail("Protected hierarchical input adapter is pending")
        from modern_architectures import TabMFieldModel
        from hierarchical_id_model import HierarchicalIDTabM
        self.options = dict(numeric_dim=1, embedding_dim=4, n_blocks=2, d_block=16, k=4, dropout=.2, unknown_dropout=.1, initial_ctr=.01)
        self.schema = (["user_id", "age"], ["feed_clicked_topics"],
                       {"user_id": "user_id", "age": "age", "feed_clicked_topics": "news_interest"},
                       {"user_id": 140, "age": 10, "news_interest": 30})
        self.teacher = TabMFieldModel(*self.schema, **self.options).eval()
        self.model = HierarchicalIDTabM(*self.schema, hierarchy_mode="metadata", **self.options)
        self.inputs = (torch.tensor([[2, 2], [3, 3]]), torch.tensor([[[2, 3, 0]], [[3, 4, 0]]]), torch.zeros(2, 1))

    def test_protected_adapter_initially_preserves_teacher_predictions(self):
        info = self.module.initialize_protected_adapter(self.model, self.teacher.state_dict())
        self.assertGreater(info["trainable_adapter_parameters"], 0)
        torch.testing.assert_close(self.model(*self.inputs), self.teacher(*self.inputs), rtol=0, atol=0)
        self.assertFalse(self.model.training)
        self.assertFalse(self.model.encoder.training)
        self.assertFalse(self.model.backbone.training)

    def test_optimizer_changes_only_shared_adapter_parameters(self):
        self.module.initialize_protected_adapter(self.model, self.teacher.state_dict())
        before = {name: value.clone() for name, value in self.model.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(*self.inputs)
            torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([[1.], [0.]]).expand_as(logits)).backward()
            optimizer.step()
        changed = [name for name, value in self.model.named_parameters() if not torch.equal(value, before[name])]
        self.assertTrue(changed)
        self.assertTrue(all(name.startswith("prior_encoders.") for name in changed), changed)
        for name, value in self.teacher.state_dict().items():
            torch.testing.assert_close(value, self.model.state_dict()[name], rtol=0, atol=0)

    def test_incomplete_teacher_state_is_rejected(self):
        incomplete = dict(self.teacher.state_dict())
        incomplete.pop("head.weight")
        with self.assertRaisesRegex(ValueError, "backbone|teacher|Teacher"):
            self.module.initialize_protected_adapter(self.model, incomplete)


if __name__ == "__main__":
    unittest.main()
