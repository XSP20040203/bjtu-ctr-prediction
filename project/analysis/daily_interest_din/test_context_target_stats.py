"""Historical context rates cannot access current/future ad outcomes."""
import unittest
import numpy as np
import pandas as pd
from prepare_context_target_stats import build_context_statistics,BASE_SPECS


class ContextTargetTests(unittest.TestCase):
    def frame(self):
        fields={f for fs in BASE_SPECS for f in fs}|{'pt_d','label','user_id','task_id','spread_app_id','slot_id'}
        d=pd.DataFrame({'pt_d':[20200131,20200201,20200201,20200202],'label':[1,0,1,1]})
        for f in fields-{'pt_d','label'}:d[f]=2
        return d

    def test_current_future_outcomes_cannot_affect_features(self):
        d=self.frame();a,n=build_context_statistics(d,20200201)
        d.loc[d.pt_d>=20200201,'label']=0
        b,_=build_context_statistics(d,20200201)
        np.testing.assert_array_equal(a,b)
        self.assertTrue(np.isfinite(a).all())
        self.assertEqual(a.shape[1],50)


if __name__=='__main__':unittest.main()
