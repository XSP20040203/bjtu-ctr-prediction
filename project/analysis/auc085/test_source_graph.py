import importlib
import unittest

import numpy as np
import pandas as pd
import torch


class SourceGraphTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("source_graph")
        except ModuleNotFoundError:
            self.fail("Typed source graph is pending")
        self.frame = pd.DataFrame({"u_userId": [8, 8, 8, 9, 9], "e_et": [20191115] * 5,
            "i_docId": ["A", "A", "B", "B", "C"], "i_cat": [1, 1, 2, 2, 3],
            "i_entities": ["common^onceA", "common^onceA", "common^onceB", "common^onceB", "onceC"],
            "label": [1, -1, -1, 1, -1]})

    def test_relation_is_symmetric_normalized_and_deduplicated(self):
        adjacency, active = self.module.normalized_relation(np.array([[0, 1], [0, 2], [0, 1]]), 4)
        expected = torch.tensor([[0., 2 ** -.5, 2 ** -.5, 0.], [2 ** -.5, 0., 0., 0.],
                                 [2 ** -.5, 0., 0., 0.], [0., 0., 0., 0.]])
        torch.testing.assert_close(adjacency.to_dense(), expected)
        torch.testing.assert_close(active, torch.tensor([1., 1., 1., 0.]))

    def test_graph_omits_singleton_entities_and_clicked_items_from_negatives(self):
        graph = self.module.build_source_graph(self.frame, cutoff_day=20191115, min_entity_docs=2)
        self.assertEqual(graph["vocabularies"]["entities"], ["common"])
        u = graph["vocabularies"]["users"].index(8)
        doc_b = graph["vocabularies"]["documents"].index("B")
        begin, end = graph["negative_indptr"][u:u + 2]
        np.testing.assert_array_equal(graph["negative_documents"][begin:end], [doc_b])

    def test_future_source_rows_do_not_change_cutoff_graph(self):
        before = self.module.build_source_graph(self.frame, cutoff_day=20191115, min_entity_docs=2)
        extra = self.frame.iloc[[0]].copy()
        extra["e_et"], extra["i_docId"], extra["i_entities"] = 20191117, "FUTURE", "future"
        after = self.module.build_source_graph(pd.concat([self.frame, extra]), cutoff_day=20191115, min_entity_docs=2)
        self.assertEqual(before["vocabularies"], after["vocabularies"])
        np.testing.assert_array_equal(before["positive_pairs"], after["positive_pairs"])

    def test_typed_propagation_preserves_constant_features_and_has_finite_gradients(self):
        relations = {"click": np.array([[0, 1]]), "category": np.array([[1, 2]]), "entity": np.array([[2, 3]])}
        model = self.module.SourceGraphEncoder(4, relations, dimension=3, n_layers=2)
        with torch.no_grad():
            model.embeddings.weight.fill_(1.)
        output = model()
        torch.testing.assert_close(output, torch.ones(4, 3))
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(model.embeddings.weight.grad).all())
        self.assertTrue(torch.isfinite(model.relation_logits.grad).all())


if __name__ == "__main__":
    unittest.main()
