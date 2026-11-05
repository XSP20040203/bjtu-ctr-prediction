"""Input-only prefix volume changes preserve row order and exclude future dates."""
import unittest
import pandas as pd
import numpy as np
from prepare_context_volume import build_context_volume,FIELDS,PATTERNS


class ContextVolumeTests(unittest.TestCase):
    def data(self):
        data=pd.DataFrame({'pt_d':[20200201,20200131,20200201,20200202]})
        for f in FIELDS:data[f]=[2,2,2,2]
        for f in PATTERNS:data[f]=['1^2']*4
        return data

    def test_current_and_prior_count_align_with_unsorted_rows(self):
        values,names=build_context_volume(self.data())
        np.testing.assert_allclose(values[:,names.index('input_volume__user_id__today_logcount')],np.log1p([2,1,2,1]))
        np.testing.assert_allclose(values[:,names.index('input_volume__user_id__past_logcount')],np.log1p([1,0,1,3]))
        np.testing.assert_allclose(values[:,names.index('input_volume__user_id__logratio_average')],[np.log(1.5),np.log(2),np.log(1.5),np.log(.8)],atol=1e-6)

    def test_future_inputs_and_targets_cannot_change_earlier_rows(self):
        data=self.data();before,_=build_context_volume(data)
        data['label']=[1,0,0,1];data.loc[3,'user_id']=9
        after,_=build_context_volume(data)
        np.testing.assert_array_equal(before[:3],after[:3])


if __name__=='__main__':unittest.main()
