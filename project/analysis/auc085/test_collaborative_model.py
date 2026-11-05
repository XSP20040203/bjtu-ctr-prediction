import importlib
import unittest

import numpy as np
import torch


class CollaborativeModelTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("collaborative_model")
        except ModuleNotFoundError:
            self.fail("Collaborative attention TabM is pending")
        self.fields = ["user_id", "task_id", "adv_prim_id", "spread_app_id", "age"]
        self.sequences = ["u_newsCatInterestsST"]
        self.groups = {"user_id": "user_id", "task_id": "ad_task", "adv_prim_id": "advertiser", "spread_app_id": "app", "age": "age", "u_newsCatInterestsST": "news"}
        self.sizes = {"user_id": 12, "ad_task": 12, "advertiser": 12, "app": 12, "age": 12, "news": 12}
        self.schema = (self.fields, self.sequences, self.groups, self.sizes)
        self.options = dict(numeric_dim=1, embedding_dim=4, n_blocks=2, d_block=16, k=4, dropout=.2, unknown_dropout=.1, initial_ctr=.01)
        self.manifest = {"categorical_fields": self.fields, "group_for_field": self.groups, "vocabulary_sizes": self.sizes}
        self.category = np.array([[2, 7, 2, 2, 2], [3, 7, 2, 2, 2], [4, 7, 2, 2, 2], [2, 6, 2, 2, 2]], dtype=np.int32)

    def make(self, labels=None, mode="neighbors", k=4):
        from collaborative_memory import build_collaborative_memory
        labels = np.ones(4, dtype=np.uint8) if labels is None else labels
        memory = build_collaborative_memory(self.category, labels, np.arange(4), self.manifest, 2)
        options = {**self.options, "k": k}
        model = self.module.CollaborativeAttentionTabM(*self.schema, **options, memory_mode=mode,
            max_neighbors=2, positive_edge_count=len(memory["positive_keys"]), metadata_fields=memory["metadata_fields"], source_dim=4, graph_dim=4)
        model.memory.install(memory)
        with torch.no_grad():
            model.source_user_vectors[2:5] = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [1., 1., 0., 0.]])
            model.source_positive_count[2:5] = 10
        return model

    def inputs(self):
        return torch.tensor(self.category[:2], dtype=torch.long), torch.zeros(2, 1, 3, dtype=torch.long), torch.zeros(2, 1)

    def test_query_changes_neighbor_attention_and_padding_is_ignored(self):
        attention = self.module.NeighborAttention(2)
        history = torch.tensor([[[1., 0.], [0., 1.], [100., 100.]]])
        mask = torch.tensor([[True, True, False]])
        first = attention(history, torch.tensor([[1., 0.]]), mask)
        second = attention(history, torch.tensor([[0., 1.]]), mask)
        torch.testing.assert_close(first, torch.tensor([[.7310586, .2689414]]))
        torch.testing.assert_close(second, torch.tensor([[.2689414, .7310586]]))
        empty = attention(history.half(), torch.zeros(1, 2).half(), torch.zeros_like(mask))
        torch.testing.assert_close(empty, torch.zeros(1, 2).half())

    def test_mean_neighbor_control_is_query_invariant_and_ignores_padding(self):
        import inspect
        attention = self.module.NeighborAttention(2)
        self.assertIn("pooling", inspect.signature(attention.forward).parameters, "Mean-pooling ablation is pending")
        history = torch.tensor([[[1., 0.], [0., 1.], [999., 999.]]])
        mask = torch.tensor([[True, True, False]])
        first = attention(history, torch.tensor([[1., 0.]]), mask, pooling="mean")
        second = attention(history, torch.tensor([[0., 1.]]), mask, pooling="mean")
        torch.testing.assert_close(first, torch.tensor([[.5, .5]]))
        torch.testing.assert_close(first, second)

    def test_degree_only_control_cannot_use_other_users_source_vectors(self):
        try:
            model = self.make(mode="degrees").eval()
        except ValueError:
            self.fail("Degree-only collaborative ablation is pending")
        with torch.no_grad():
            model.collaborative["residual"][-1].weight.fill_(.1)
        category, history, numeric = self.inputs()
        before = model(category[:1], history[:1], numeric[:1])
        with torch.no_grad():
            model.source_user_vectors[3:5].mul_(-10.)
        torch.testing.assert_close(before, model(category[:1], history[:1], numeric[:1]), rtol=0, atol=0)

    def test_neutral_branch_preserves_teacher_and_cached_base_path(self):
        from modern_architectures import TabMFieldModel
        teacher = TabMFieldModel(*self.schema, **self.options).eval()
        model = self.make()
        self.module.initialize_collaborative_adapter(model, teacher.state_dict())
        expected = teacher(*self.inputs())
        torch.testing.assert_close(model(*self.inputs()), expected, rtol=0, atol=0)
        torch.testing.assert_close(model.forward_with_base(self.inputs()[0], self.inputs()[1], expected), expected, rtol=0, atol=0)

    def test_training_only_updates_new_branch_and_preserves_frozen_teacher(self):
        from modern_architectures import TabMFieldModel
        teacher = TabMFieldModel(*self.schema, **self.options).eval()
        model = self.make()
        self.module.initialize_collaborative_adapter(model, teacher.state_dict())
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.collaborative.training)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = model(*self.inputs())
            torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([[1.], [0.]]).expand_as(logits)).backward()
            optimizer.step()
        self.assertGreater(model.collaborative["residual"][-1].weight.abs().sum().item(), 0)
        for name, original in teacher.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], original, rtol=0, atol=0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_current_pair_label_flip_cannot_change_nonzero_graph_prediction(self):
        model = self.make().eval()
        with torch.no_grad():
            model.collaborative["residual"][-1].weight.normal_(std=.1)
        changed = self.make(np.array([0, 1, 1, 1], dtype=np.uint8)).eval()
        with torch.no_grad():
            for name, value in model.named_parameters():
                dict(changed.named_parameters())[name].copy_(value)
        category, history, numeric = self.inputs()
        torch.testing.assert_close(model(category[:1], history[:1], numeric[:1]), changed(category[:1], history[:1], numeric[:1]), rtol=0, atol=0)

    def test_checkpoint_restores_neighbors_source_vectors_and_predictions(self):
        model = self.make().eval()
        with torch.no_grad():
            model.collaborative["residual"][-1].weight.normal_(std=.1)
        restored = self.make().eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(*self.inputs()), model(*self.inputs()), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA AMP layout regression")
    def test_equal_base_logits_with_different_strides_give_identical_amp_fusion(self):
        # This mean lies at a float16 rounding boundary. Row/column-strided
        # float32 reductions otherwise select different half-precision gates.
        values = [-3.8840341568, -3.8150639534, -4.0568175316, -3.8276863098,
            -4.3430366516, -4.0394358635, -4.2219014168, -4.4455151558,
            -4.0563244820, -3.9985866547, -4.1534605026, -3.9403605461,
            -4.2484779358, -3.8811888695, -4.1557521820, -3.7010011673,
            -4.1421880722, -3.5846095085, -4.0250329971, -4.1780467033,
            -4.3671021461, -4.0614995956, -4.3954043388, -3.5785384178,
            -4.0894770622, -4.1238737106, -4.6761465073, -4.1590309143,
            -3.9876265526, -4.0573554039, -4.0407538414, -3.5771794319]
        model = self.make(k=32).cuda().eval()
        category, history, _ = self.inputs()
        category, history = category[:1].repeat(2048, 1).cuda(), history[:1].repeat(2048, 1, 1).cuda()
        contiguous = torch.tensor(values, device="cuda")[None].repeat(2048, 1)
        strided = contiguous.T.contiguous().T
        with torch.no_grad():
            model.collaborative["residual"][-1].bias.fill_(1.)
            model.collaborative["gate"].weight.zero_()
            model.collaborative["gate"].weight[0, -2] = 1.
            model.collaborative["gate"].bias.fill_(-float(contiguous[0].mean()))
            with torch.cuda.amp.autocast():
                first = model.forward_with_base(category, history, contiguous)
                second = model.forward_with_base(category, history, strided)
        torch.testing.assert_close(first, second, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
