import unittest
import numpy as np
from dense_context_rows import dense_codes, schema, lgb


class DenseCategoryBoundaries(unittest.TestCase):
    def test_unseen_and_negative_values_stay_missing(self):
        vocabulary = np.array([0, 100, 1516217, 1524772])
        result = dense_codes(np.array([-1, 0, 99, 100, 1516217, 1524772, 1524773]), vocabulary)
        np.testing.assert_array_equal(result, [-1, 0, -1, 1, 2, 3, -1])

    def test_held_inputs_do_not_define_dictionary_or_feature_mask(self):
        x = np.array([[100, 2, 5], [200, 3, 6], [900, 4, 7]], dtype='float32')
        names = ['user_id', 'original__target_stat__user_id__relative_logit', 'quality__x']
        training = np.array([0, 1])
        before, maps = schema(x, training, names, [0], True)
        x[2] = [999999, -100, -500]
        after, new_maps = schema(x, training, names, [0], True)
        self.assertEqual(before, after)
        self.assertEqual(before, [0, 2])
        np.testing.assert_array_equal(maps[0], [100, 200])
        np.testing.assert_array_equal(maps[0], new_maps[0])

    def test_only_high_cardinality_categories_are_removed_by_focus(self):
        x = np.column_stack([np.arange(501), np.arange(501)*0.5, np.arange(501)%3]).astype('float32')
        columns, maps = schema(x, np.arange(501), ['identity', 'numeric', 'small_category'], [0, 2], True)
        self.assertEqual(columns, [1, 2])
        np.testing.assert_array_equal(maps[1], [0, 1, 2])
        columns, maps = schema(x, np.arange(501), ['identity', 'numeric', 'small_category'], [0, 2], False)
        self.assertEqual(columns, [0, 1, 2])
        self.assertEqual(len(maps[0]), 501)

    def test_dense_relabeling_preserves_known_category_model(self):
        random = np.random.default_rng(151637)
        codes = random.integers(0, 48, 2400)
        raw = np.column_stack([1516217+codes*3, random.normal(size=len(codes))])
        labels = (random.random(len(codes)) < np.where(codes%3 == 0, .8, .15)).astype('int32')
        dense = raw.copy()
        dense[:, 0] = dense_codes(raw[:, 0], np.unique(raw[:, 0]))
        params = {'objective': 'binary', 'num_leaves': 7, 'min_data_in_leaf': 20,
            'min_data_per_group': 5, 'cat_smooth': 5., 'cat_l2': 1., 'max_bin': 31,
            'max_cat_to_onehot': 4, 'num_threads': 1, 'verbosity': -1, 'seed': 151637,
            'deterministic': True, 'force_col_wise': True}
        models = [lgb.train(params, lgb.Dataset(values, label=labels, categorical_feature=[0]),
                            num_boost_round=20) for values in [raw, dense]]
        np.testing.assert_allclose(models[0].predict(raw, num_threads=1),
                                   models[1].predict(dense, num_threads=1), rtol=0, atol=1e-12)
        self.assertLess(len(models[1].model_to_string()), len(models[0].model_to_string())*.1)


if __name__ == '__main__':
    unittest.main()
