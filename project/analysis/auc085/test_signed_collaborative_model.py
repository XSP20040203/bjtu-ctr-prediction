import importlib
import unittest

import numpy as np
import torch

from collaborative_memory import build_collaborative_memory
from collaborative_model import CollaborativeAttentionTabM
from negative_collaboration import build_negative_memory
from bounded_id_models import bound_field_vectors


class SignedCollaborativeTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("signed_collaborative_model")
        except ModuleNotFoundError:
            self.fail("Positive-versus-nonclick collaboration expert is pending")
        torch.manual_seed(42)
        self.fields = ["user_id", "task_id", "adv_prim_id", "spread_app_id", "age"]
        self.groups = {"user_id": "user_id", "task_id": "ad_task", "adv_prim_id": "advertiser", "spread_app_id": "app", "age": "age", "u_newsCatInterestsST": "news"}
        self.sizes = {name: 12 for name in set(self.groups.values())}
        self.schema = (self.fields, ["u_newsCatInterestsST"], self.groups, self.sizes)
        self.manifest = {"categorical_fields": self.fields, "group_for_field": self.groups, "vocabulary_sizes": self.sizes}
        self.category = np.array([[2, 7, 2, 2, 2], [3, 7, 2, 2, 2], [4, 7, 2, 2, 2], [2, 6, 2, 2, 2]], dtype=np.int32)
        self.labels = np.array([1, 1, 0, 0], dtype=np.uint8)
        positive = build_collaborative_memory(self.category, self.labels, np.arange(4), self.manifest, 2)
        self.options = dict(numeric_dim=1, embedding_dim=4, n_blocks=2, d_block=16, k=4, dropout=.2, unknown_dropout=.1,
            initial_ctr=.01, max_neighbors=2, positive_edge_count=len(positive["positive_keys"]), metadata_fields=positive["metadata_fields"], source_dim=4, graph_dim=4)
        self.teacher = CollaborativeAttentionTabM(*self.schema, **self.options).eval()
        self.teacher.memory.install(positive)
        with torch.no_grad():
            self.teacher.source_user_vectors[2:5] = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [1., 1., 0., 0.]])
            self.teacher.source_positive_count[2:5] = 10
            self.teacher.collaborative["residual"][-1].weight.normal_(std=.1)

    def make(self, labels=None, learn_queries=False):
        labels = self.labels if labels is None else labels
        negative = build_negative_memory(self.category, labels, np.arange(4), self.manifest, 2)
        model = self.module.SignedCollaborativeTabM(*self.schema, **self.options, negative_edge_count=len(negative["positive_keys"]), signed_mode="contrast", learn_queries=learn_queries)
        self.module.initialize_signed_adapter(model, self.teacher.state_dict())
        model.negative_memory.install(negative)
        return model

    def inputs(self):
        return torch.as_tensor(self.category[:2], dtype=torch.long), torch.zeros(2, 1, 3, dtype=torch.long), torch.zeros(2, 1)

    def test_initial_raw_and_cached_predictions_preserve_the_collaborative_teacher(self):
        model = self.make().eval()
        category, history, numeric = self.inputs()
        expected = self.teacher(category, history, numeric)
        fields, _, _ = self.teacher.encoder(category, history)
        fields = bound_field_vectors(fields, self.teacher.field_groups, self.teacher.group_norm_caps)
        base = self.teacher._base_logits(fields, numeric)
        torch.testing.assert_close(model(category, history, numeric), expected, rtol=0, atol=0)
        torch.testing.assert_close(model.forward_with_base(category, history, base), expected, rtol=0, atol=0)

    def test_only_new_contrast_expert_can_update(self):
        model = self.make().train()
        self.assertFalse(model.collaborative.training)
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.signed.training)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = model(*self.inputs())
            torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([[1.], [0.]]).expand_as(logits)).backward()
            optimizer.step()
        self.assertGreater(model.signed["residual"][-1].weight.abs().sum().item(), 0)
        for name, original in self.teacher.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], original, rtol=0, atol=0)

    def test_current_label_flip_cannot_change_contrast_prediction(self):
        first = self.make().eval()
        second = self.make(np.array([0, 1, 0, 0], dtype=np.uint8)).eval()
        with torch.no_grad():
            first.signed["residual"][-1].weight.normal_(std=.1)
            other = dict(second.named_parameters())
            for name, parameter in first.named_parameters():
                other[name].copy_(parameter)
        category, history, numeric = self.inputs()
        torch.testing.assert_close(first(category[:1], history[:1], numeric[:1]), second(category[:1], history[:1], numeric[:1]), rtol=0, atol=0)

    def test_empty_negative_relation_and_reload_are_finite(self):
        model = self.make(np.ones(4, dtype=np.uint8)).eval()
        self.assertEqual(model.negative_memory.positive_keys.numel(), 0)
        result = model(*self.inputs())
        self.assertTrue(torch.isfinite(result).all())
        restored = self.make(np.ones(4, dtype=np.uint8)).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(*self.inputs()), result, rtol=0, atol=0)

    def test_adaptive_query_space_learns_without_changing_positive_teacher(self):
        import inspect
        self.assertIn("learn_queries", inspect.signature(self.module.SignedCollaborativeTabM.__init__).parameters,
                      "Independent adaptive query space is pending")
        model = self.make(learn_queries=True).train()
        with torch.no_grad():
            model.signed["residual"][-1].weight.normal_(std=.1)
        original = model.signed["user_context"][-1].weight.detach().clone()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        optimizer.zero_grad(set_to_none=True)
        logits = model(*self.inputs())
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([[1.], [0.]]).expand_as(logits)).backward()
        optimizer.step()
        self.assertFalse(torch.equal(original, model.signed["user_context"][-1].weight))
        for name, value in self.teacher.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
