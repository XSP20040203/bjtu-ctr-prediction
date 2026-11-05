"""Check past-count restoration, exposure alignment, and exclusion of future labels."""
import unittest
import numpy as np
import pandas as pd
from day17_refit_core import restore_prior_counts


class RefitChecks(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame(dict(pt_d=[20191115,20191116,20191116,20191117],user_id=[2,1,1,3],label=[1,0,1,1]))

    def test_restores_newly_available_prior_day_in_key_order(self):
        value=restore_prior_counts([[20191116,1],[20191115,2]],[2,1],[-1,1],self.fixture(),20191117)
        np.testing.assert_array_equal(value,[1,1])

    def test_future_label_perturbation_cannot_affect_restoration(self):
        raw=self.fixture()
        a=restore_prior_counts([[20191116,1]],[2],[-1],raw,20191117)
        raw.loc[raw.pt_d.eq(20191117),'label']=-999
        np.testing.assert_array_equal(a,restore_prior_counts([[20191116,1]],[2],[-1],raw,20191117))

    def test_prediction_day_keys_and_exposure_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            restore_prior_counts([[20191117,3]],[1],[-1],self.fixture(),20191117)
        with self.assertRaises(ValueError):
            restore_prior_counts([[20191116,1]],[3],[-1],self.fixture(),20191117)

    def test_known_counts_and_nonbinary_earlier_labels_rejected(self):
        with self.assertRaises(ValueError):
            restore_prior_counts([[20191115,2]],[1],[0],self.fixture(),20191117)
        raw=self.fixture()
        raw.loc[raw.pt_d.eq(20191116),'label']=2
        with self.assertRaises(ValueError):
            restore_prior_counts([[20191116,1]],[2],[-1],raw,20191117)


if __name__=='__main__':
    unittest.main()
