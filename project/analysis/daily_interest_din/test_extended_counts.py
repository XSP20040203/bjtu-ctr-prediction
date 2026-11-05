"""Fine-grained volume features depend on current-day inputs, never targets."""
import unittest
import numpy as np
import pandas as pd
from prepare_extended_counts import build_extended_counts,SPECS


class ExtendedCountTests(unittest.TestCase):
    def frame(self):
        fields={field for spec in SPECS for field in spec}|{'user_id','task_id','adv_id','spread_app_id','slot_id','creat_type_cd','inter_type_cd','device_size','net_type'}
        d=pd.DataFrame({'pt_d':[1,1,2]})
        for f in fields:d[f]=['1^2']*3 if ('list' in f or f=='u_newsCatInterestsST') else [2,2,2]
        return d

    def test_future_rows_and_targets_cannot_change_earlier_day(self):
        d=self.frame();a,n=build_extended_counts(d);d['label']=[1,0,1];d.loc[2,'user_id']=9;b,_=build_extended_counts(d)
        np.testing.assert_array_equal(a[:2],b[:2]);self.assertTrue(np.isfinite(a).all())

    def test_counts_preserve_permuted_row_identity(self):
        d=self.frame();a,n=build_extended_counts(d);b,_=build_extended_counts(d.iloc[[2,0,1]])
        np.testing.assert_array_equal(a[[2,0,1]],b)
        self.assertAlmostEqual(a[0,n.index('fine_exposure__user_id__creat_type_cd__logcount')],np.log(3),places=6)


if __name__=='__main__':unittest.main()
