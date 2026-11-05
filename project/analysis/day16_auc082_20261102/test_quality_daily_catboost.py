import unittest
import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
from quality_daily_catboost import training_targets


class DailyQualityBoundaries(unittest.TestCase):
    def test_held_counts_are_rejected_unless_masked_and_training_is_strictly_earlier(self):
        keys = np.array([[20191113,1],[20191114,1],[20191115,1],[20191116,1]])
        counts = np.array([0,3,-1,-1])
        training, target = training_targets(keys, counts, 20191115, 'poisson')
        np.testing.assert_array_equal(training, [0,1])
        np.testing.assert_array_equal(target, [0,3])
        np.testing.assert_array_equal(training_targets(keys, counts, 20191115, 'binary')[1], [0,1])
        for position in [2,3]:
            changed = counts.copy(); changed[position] = 0
            with self.assertRaises(ValueError):
                training_targets(keys, changed, 20191115, 'poisson')

    def test_unknown_earlier_counts_and_undeclared_objectives_are_rejected(self):
        keys = np.array([[20191113,1],[20191115,1]])
        with self.assertRaises(ValueError):
            training_targets(keys, np.array([-1,-1]), 20191115, 'binary')
        with self.assertRaises(ValueError):
            training_targets(keys, np.array([0,-1]), 20191115, 'rmse')

    def test_catboost_baseline_is_added_once_and_not_stored_in_the_trees(self):
        frame = pd.DataFrame({'x':np.arange(40)%5})
        baseline = np.linspace(-4., -1., len(frame))
        target = (np.arange(len(frame))%7 == 0).astype('float64')
        model = CatBoost(dict(loss_function='Logloss', depth=2, iterations=20, learning_rate=.05,
            thread_count=2, random_seed=42, allow_writing_files=False, verbose=False))
        model.fit(Pool(frame, label=target, baseline=baseline))
        tree_only = model.predict(Pool(frame), prediction_type='RawFormulaVal')
        with_baseline = model.predict(Pool(frame, baseline=baseline), prediction_type='RawFormulaVal')
        np.testing.assert_allclose(with_baseline, tree_only+baseline, rtol=0, atol=3e-7)


if __name__ == '__main__':
    unittest.main()
