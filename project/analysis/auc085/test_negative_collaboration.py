import importlib
import unittest

import numpy as np
import torch

from collaborative_memory import build_collaborative_memory, QueryExcludedMemory, MEMORY_ARRAYS


class NegativeCollaborationTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module("negative_collaboration")
        except ModuleNotFoundError:
            self.fail("Never-clicked collaboration memory is pending")
        self.manifest = {"categorical_fields": ["user_id", "task_id", "adv_prim_id", "spread_app_id"],
                         "group_for_field": {"user_id": "user_id", "task_id": "ad_task", "adv_prim_id": "advertiser", "spread_app_id": "app"},
                         "vocabulary_sizes": {"user_id": 12, "ad_task": 12, "advertiser": 12, "app": 12}}

    def query(self, data, user, task):
        memory = QueryExcludedMemory(12, 12, 2, len(data["positive_keys"]), len(data["metadata_fields"]))
        memory.install(data)
        return memory.query(torch.tensor([user]), torch.tensor([task]))

    def test_clicked_pairs_cannot_be_negative_neighbors_even_with_nonclick_rows(self):
        x = np.array([[2, 3, 2, 2], [2, 3, 2, 2], [2, 4, 2, 2], [3, 4, 2, 2]])
        data = self.module.build_negative_memory(x, np.array([1, 0, 0, 0]), np.arange(4), self.manifest, 2)
        np.testing.assert_array_equal(data["positive_keys"], [2 * 12 + 4, 3 * 12 + 4])
        self.assertEqual(data["mixed_pairs_excluded"], 1)

    def test_heldout_labels_and_inputs_do_not_change_negative_memory(self):
        x = np.array([[2, 3, 2, 2], [3, 4, 3, 3], [4, 5, 4, 4]])
        y = np.array([1, 0, 1])
        before = self.module.build_negative_memory(x, y, np.array([0, 1]), self.manifest, 2)
        x[2] = 11
        y[2] = 0
        after = self.module.build_negative_memory(x, y, np.array([0, 1]), self.manifest, 2)
        for name in MEMORY_ARRAYS:
            np.testing.assert_array_equal(before[name], after[name])

    def test_flipping_current_label_preserves_both_excluded_relation_types(self):
        x = np.array([[user, 7, 2, 2] for user in [2, 3, 4, 5, 6]])
        for selected in [0, 2, 4]:
            y = np.zeros(5, dtype=np.uint8)
            y[selected] = 1
            y[(selected + 1) % 5] = 1
            changed = y.copy()
            changed[selected] = 0
            for builder in [build_collaborative_memory, self.module.build_negative_memory]:
                first = self.query(builder(x, y, np.arange(5), self.manifest, 2), x[selected, 0], 7)
                second = self.query(builder(x, changed, np.arange(5), self.manifest, 2), x[selected, 0], 7)
                for name in first:
                    torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
