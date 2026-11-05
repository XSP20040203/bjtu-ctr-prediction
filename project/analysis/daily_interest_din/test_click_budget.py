"""Historical budget cannot use same-day or validation advertisement outcomes."""
import unittest
import numpy as np
import pandas as pd
from prepare_click_budget import build_click_budget,SPECS


class ClickBudgetTests(unittest.TestCase):
    def data(self):
        d=pd.DataFrame({'pt_d':[20200130,20200131,20200201,20200201,20200202],'label':[1,0,1,0,1]})
        for f in {f for fs in SPECS for f in fs}:d[f]=2
        return d

    def test_future_and_current_labels_do_not_change_inputs(self):
        d=self.data();before,_=build_click_budget(d,20200201)
        d.loc[d.pt_d>=20200201,'label']=0
        after,_=build_click_budget(d,20200201)
        np.testing.assert_array_equal(before,after)

    def test_two_today_rows_share_a_budget_divided_by_two_exposures(self):
        d=self.data();v,n=build_click_budget(d,20200201)
        np.testing.assert_array_equal(v[2],v[3])
        self.assertTrue(np.isfinite(v).all())
        self.assertEqual(v.shape[1],81)


if __name__=='__main__':unittest.main()
