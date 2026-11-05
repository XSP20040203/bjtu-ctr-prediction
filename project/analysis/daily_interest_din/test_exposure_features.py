"""Within-day unlabeled volume is invariant to targets and subsequent dates."""
import unittest
import numpy as np
import pandas as pd
from prepare_exposure_features import build_exposure_features, SINGLE


class ExposureTests(unittest.TestCase):
    def data(self):
        data=pd.DataFrame({'pt_d':[20200131,20200131,20200131,20200201]})
        for field in SINGLE:data[field]=[2,2,3,2]
        data['task_id']=[10,20,10,10]
        return data

    def test_counts_and_user_fractions(self):
        values,names=build_exposure_features(self.data())
        self.assertAlmostEqual(values[0,names.index('day_exposure__user_id__log_count')],np.log(3),places=6)
        self.assertEqual(values[0,names.index('day_exposure__user_id__task_id__fraction_first')],.5)
        self.assertEqual(values[3,names.index('day_exposure__user_id__task_id__fraction_first')],1.)

    def test_labels_and_future_dates_do_not_affect_previous_inputs(self):
        data=self.data();before,_=build_exposure_features(data)
        data['label']=[1,0,1,0];data.loc[3,'user_id']=9
        after,_=build_exposure_features(data)
        np.testing.assert_array_equal(before[:3],after[:3])

    def test_row_permutation_preserves_corresponding_features(self):
        data=self.data();before,_=build_exposure_features(data)
        order=[2,0,3,1];after,_=build_exposure_features(data.iloc[order])
        np.testing.assert_array_equal(before[order],after)


if __name__=='__main__':unittest.main()
