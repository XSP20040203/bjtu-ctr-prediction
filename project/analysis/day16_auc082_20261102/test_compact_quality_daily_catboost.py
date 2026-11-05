import unittest
import numpy as np
from compact_quality_daily_catboost import selected_columns, independent_correction
from rich_userday_activity import correct


class CompactBoundaries(unittest.TestCase):
    def test_quality_and_categories_survive_zero_importance(self):
        names=['user_id','numeric_a','quality__q','numeric_b','age']
        columns=selected_columns(names,['user_id','age'],[0,10,0,5,0],1)
        self.assertEqual(columns,['user_id','numeric_a','quality__q','age'])

    def test_invalid_training_importance_is_rejected(self):
        for values in [[1],[1,float('nan')]]:
            with self.assertRaises(ValueError):
                selected_columns(['a','b'],[],values,1)

    def test_independent_count_correction_handles_clipping(self):
        base=np.array([1e-10,.01,.2,.5])
        groups=np.array([0,0,1,1])
        response=np.array([0.,1000.])
        for power in [.0625,.125,.5]:
            np.testing.assert_allclose(independent_correction(base,groups,response,power),
                correct(base,response,groups,'poisson',power),rtol=0,atol=2e-16)


if __name__ == '__main__':
    unittest.main()
