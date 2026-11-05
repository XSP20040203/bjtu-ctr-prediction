import importlib
import unittest

import numpy as np
import torch


class CollaborativeMemoryTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("collaborative_memory")
        except ModuleNotFoundError:
            self.fail("Query-excluded collaborative memory is pending")
        self.manifest = {"categorical_fields": ["user_id", "task_id", "adv_prim_id", "spread_app_id"],
                         "group_for_field": {"user_id": "user_id", "task_id": "ad_task", "adv_prim_id": "advertiser", "spread_app_id": "app"},
                         "vocabulary_sizes": {"user_id": 12, "ad_task": 12, "advertiser": 12, "app": 12}}

    def fit(self, categories, labels, rows=None, k=2):
        rows = np.arange(len(labels)) if rows is None else rows
        return self.module.build_collaborative_memory(np.asarray(categories, dtype=np.int32), np.asarray(labels, dtype=np.uint8), rows, self.manifest, k)

    def make_memory(self, data, k=2):
        memory = self.module.QueryExcludedMemory(12, 12, k, len(data["positive_keys"]), len(data["metadata_fields"]))
        memory.install(data)
        return memory

    def assert_same_query(self, first, second, user, task):
        a = self.make_memory(first).query(torch.tensor([user]), torch.tensor([task]))
        b = self.make_memory(second).query(torch.tensor([user]), torch.tensor([task]))
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)

    def test_only_training_labels_and_inputs_determine_memory(self):
        categories = np.array([[2, 3, 2, 2], [3, 4, 3, 3], [4, 5, 4, 4], [5, 6, 5, 5]])
        labels = np.array([1, 0, 0, 1])
        first = self.fit(categories, labels, np.array([0, 1]))
        categories[2:] = 11
        labels[2:] = 1 - labels[2:]
        second = self.fit(categories, labels, np.array([0, 1]))
        for name in first:
            if isinstance(first[name], np.ndarray):
                np.testing.assert_array_equal(first[name], second[name])
        np.testing.assert_array_equal(first["positive_keys"], [2 * 12 + 3])

    def test_self_label_invariance_inside_at_and_beyond_task_neighbor_cap(self):
        categories = [[user, 7, 2, 2] for user in [2, 3, 4, 5, 6]]
        for selected in [0, 2, 4]:
            labels = np.ones(5, dtype=np.uint8)
            present = self.fit(categories, labels)
            labels[selected] = 0
            absent = self.fit(categories, labels)
            self.assert_same_query(present, absent, categories[selected][0], 7)

    def test_self_label_invariance_inside_at_and_beyond_user_neighbor_cap(self):
        categories = [[7, task, 2, 2] for task in [2, 3, 4, 5, 6]]
        for selected in [0, 2, 4]:
            labels = np.ones(5, dtype=np.uint8)
            present = self.fit(categories, labels)
            labels[selected] = 0
            absent = self.fit(categories, labels)
            self.assert_same_query(present, absent, 7, categories[selected][1])

    def test_degree_exclusion_works_when_query_is_outside_stored_neighbors(self):
        data = self.fit([[user, 7, 2, 2] for user in [2, 3, 4, 5, 6]], [1] * 5)
        result = self.make_memory(data).query(torch.tensor([6]), torch.tensor([7]))
        torch.testing.assert_close(result["task_users"], torch.tensor([[2, 3]]))
        self.assertEqual(result["task_degree"].item(), 4)
        self.assertEqual(result["user_degree"].item(), 0)

    def test_padding_and_unknown_never_create_hubs_but_encoded_real_zero_is_retained(self):
        data = self.fit([[0, 2, 2, 2], [1, 2, 2, 2], [2, 0, 2, 2], [2, 1, 2, 2], [2, 2, 2, 2]], [1] * 5)
        np.testing.assert_array_equal(data["positive_keys"], [2 * 12 + 2])
        self.assertFalse(data["user_tasks"][:2].any())
        self.assertFalse(data["task_users"][:2].any())
        result = self.make_memory(data).query(torch.tensor([2, 1]), torch.tensor([2, 1]))
        self.assertFalse(result["user_tasks"].any())
        self.assertFalse(result["task_users"].any())
        self.assertFalse(result["user_degree"].any())

    def test_metadata_mode_uses_training_x_including_nonclicks_with_deterministic_ties(self):
        data = self.fit([[2, 3, 4, 5], [2, 3, 4, 6], [3, 3, 7, 6]], [1, 0, 0])
        np.testing.assert_array_equal(data["task_metadata"][3], [4, 6])

    def test_query_and_reload_do_not_mutate_saved_graph(self):
        data = self.fit([[2, 3, 2, 2], [2, 4, 2, 2], [4, 3, 2, 2]], [1, 1, 1])
        memory = self.make_memory(data)
        before = {name: tensor.clone() for name, tensor in memory.state_dict().items()}
        result = memory.query(torch.tensor([2]), torch.tensor([3]))
        torch.testing.assert_close(result["user_tasks"], torch.tensor([[4, 0]]))
        torch.testing.assert_close(result["task_users"], torch.tensor([[4, 0]]))
        for name, tensor in memory.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        restored = self.make_memory(data)
        restored.load_state_dict(memory.state_dict())
        for name, tensor in result.items():
            torch.testing.assert_close(restored.query(torch.tensor([2]), torch.tensor([3]))[name], tensor)


if __name__ == "__main__":
    unittest.main()
