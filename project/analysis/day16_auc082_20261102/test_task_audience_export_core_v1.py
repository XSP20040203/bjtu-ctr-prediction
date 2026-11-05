"""Exercise a category first observed after the model's vocabulary cutoff."""
import unittest
import numpy as np
import pandas as pd
from task_audience_core import FIELDS, build
from task_audience_export_core_v1 import frozen_schema


class FrozenVocabularyTest(unittest.TestCase):
    def test_new_category_remains_unknown_for_fitted_model(self):
        ads = pd.DataFrame({name: np.ones(5, dtype='int32') for name in FIELDS})
        ads['pt_d'] = [20191115, 20191116, 20191117, 20191117, 20191117]
        ads['age'] = [1, 9, 1, 9, 10]
        ads['creat_type_cd'] = [1, 7, 1, 7, 8]
        keys = np.array([[20191115, 1], [20191116, 1], [20191117, 1]])
        quality = np.zeros((3, 2), dtype='float32')
        old, _, _ = build(ads, np.array([1, -1, -1, -1, -1]), keys[:2], quality[:2], 20191116)
        new, _, meta = build(ads, np.array([1, 0, -1, -1, -1]), keys, quality, 20191117)
        self.assertIn('fraction__age__9', new)
        self.assertNotIn('fraction__age__9', old)
        primary = frozen_schema(new, ads, meta, list(old))
        replay = frozen_schema(new, ads, meta, list(old), independent=True)
        pd.testing.assert_frame_equal(primary, replay)
        for field in ['age', 'creat_type_cd']:
            np.testing.assert_array_equal(primary[f'fraction__{field}__unknown'],
                                          np.array([0, 1, .66667], dtype='float32'))
        self.assertEqual(primary.loc[2, 'category__creat_type_cd'], 1)
        self.assertEqual(list(primary), list(old))

    def test_missing_model_feature_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing fitted audience features'):
            frozen_schema(pd.DataFrame(), pd.DataFrame(), {}, ['absent'])


if __name__ == '__main__':
    unittest.main()
