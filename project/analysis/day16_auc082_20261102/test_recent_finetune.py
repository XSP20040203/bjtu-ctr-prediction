import unittest
import numpy as np
from recent_finetune import select_recent_rows, recipe_prediction


class RecentContracts(unittest.TestCase):
    def test_recent_dates_and_unsorted_rows(self):
        dates = np.array([20191113,20191115,20191116,20191114,20191115])
        np.testing.assert_array_equal(select_recent_rows(dates,np.array([4,0,3,1]),1),[4,1])
        np.testing.assert_array_equal(select_recent_rows(dates,np.array([4,0,3,1]),2),[4,3,1])
        with self.assertRaises(ValueError):
            select_recent_rows(dates,np.array([0,1,2]),1)

    def test_fixed_score_recipe(self):
        p,base,v7,core = (np.array([v,.1]) for v in [.2,.3,.4,.5])
        np.testing.assert_allclose(recipe_prediction("reference_blend_0.25",p,base,v7,core),.75*base+.25*p)
        np.testing.assert_allclose(recipe_prediction("core_replacement_030",p,base,v7,core),v7+.3*(p-core))


if __name__ == "__main__":
    unittest.main()
