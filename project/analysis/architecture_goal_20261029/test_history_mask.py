"""Tests for optional field-specific close-history default handling."""
import json
from pathlib import Path
import unittest

import numpy as np
from history_mask import create_mask_spec, apply_sequence_mask, feature_column_selection, mask_metadata, MISSING_FLAG_NAMES


class HistoryMaskTests(unittest.TestCase):
    def fixture(self):
        fields = ['click_task', 'ad_close_list_v001', 'ad_close_list_v002', 'ad_close_list_v003']
        meta = {'sequence_fields': fields, 'group_for_field': dict(zip(fields, ['task', 'task', 'advertiser', 'app'])),
                'vocabulary_sizes': {'task': 10, 'advertiser': 10, 'app': 10}}
        vocabs = {'task': {'24107': 2}, 'advertiser': {'1218': 3}, 'app': {'173': 4}}
        return meta, vocabs

    def test_field_specific_copy_missing_and_unknown(self):
        meta, vocabs = self.fixture()
        values = np.array([[[2, 0, 0], [2, 0, 0], [3, 1, 0], [4, 5, 0]],
                           [[2, 2, 0], [2, 6, 0], [3, 0, 0], [4, 0, 0]]], dtype='int32')
        original = values.copy()
        result, missing = apply_sequence_mask(values, create_mask_spec(meta, vocabs))
        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(result[:, 0], original[:, 0])
        np.testing.assert_array_equal(missing, [[1, 0, 0], [0, 1, 1]])
        self.assertEqual(result[0, 2, 1], 1)
        self.assertEqual(result[1, 1, 1], 6)

    def test_invalid_default_never_masks_unknown(self):
        meta, vocabs = self.fixture()
        vocabs['task'] = {}
        with self.assertRaises(ValueError):
            create_mask_spec(meta, vocabs)

    def test_empty_batch_and_slot_shape(self):
        meta, vocabs = self.fixture()
        spec = create_mask_spec(meta, vocabs)
        values, missing = apply_sequence_mask(np.empty((0, 4, 5), dtype='int64'), spec)
        self.assertEqual(values.shape, (0, 4, 5))
        self.assertEqual(missing.shape, (0, 3))
        _, missing = apply_sequence_mask(np.empty((2, 4, 0), dtype='int32'), spec)
        np.testing.assert_array_equal(missing, np.ones((2, 3)))

    def test_column_selection_and_json_metadata(self):
        meta, vocabs = self.fixture()
        fields = ['hist_close_task_match', 'ad_close_list_v001__length', 'snapshot_close_1__new',
                  'snapshot_click_1__new', 'history_input__close_1__missing', 'hist_click_task_match']
        np.testing.assert_array_equal(feature_column_selection(fields), [3, 4, 5])
        record = mask_metadata(create_mask_spec(meta, vocabs), fields)
        self.assertEqual(record['output_numeric_count'], 6)
        self.assertEqual(record['removed_numeric_count'], 3)
        self.assertFalse(record['labels_used'])
        json.dumps(record)

    def test_current_schema_and_vocab_smoke(self):
        root = Path(__file__).resolve().parents[2]
        cache = root / 'analysis/user_personalization/cache_day15_stats'
        if not cache.exists():
            self.skipTest('Local dataset not present')
        meta = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))
        vocabs = json.loads((cache / 'vocabs.json').read_text(encoding='utf-8'))
        fields = json.loads((root / 'analysis/architecture_goal_20261029/cache/day15_oof/manifest.json').read_text(encoding='utf-8'))['numeric_fields']
        record = mask_metadata(create_mask_spec(meta, vocabs), fields)
        self.assertEqual(record['removed_numeric_count'], 30)
        self.assertEqual(record['output_numeric_count'], 473)
        sample = np.load(cache / 'sequences.npy', mmap_mode='r')[:100]
        masked, missing = apply_sequence_mask(sample, create_mask_spec(meta, vocabs))
        self.assertEqual(masked.shape, sample.shape)
        self.assertEqual(missing.shape, (100, 3))
        np.testing.assert_array_equal(feature_column_selection(fields[:166]).shape, (136,))


if __name__ == '__main__':
    unittest.main(verbosity=2)
