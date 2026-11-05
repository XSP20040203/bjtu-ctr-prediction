import unittest
import numpy as np
import pandas as pd
from scipy.special import logit
from compose_quality_daily_transfer import predict


class CompositionMath(unittest.TestCase):
    def setUp(self):
        values = dict(current=np.array([.01,.03,.02,.09]),daily=np.array([.03,.02,.04,.1]),
                      anchor=np.array([.02,.02,.02,.08]))
        self.data = dict(values=values,logits={name:logit(p) for name,p in values.items()},
            groups={'user':np.array([0,0,1,1])},frame=pd.DataFrame(dict(user_id=[1,1,2,2])))

    def test_zero_transfer_is_identity(self):
        p = predict(self.data,dict(formula='transfer_increment',weight=0))
        np.testing.assert_allclose(p,self.data['values']['current'],rtol=0,atol=2e-16)

    def test_group_split_equal_weights_matches_transfer(self):
        p = predict(self.data,dict(formula='group_transfer',group='user',between=1.5,within=1.5))
        q = predict(self.data,dict(formula='transfer_increment',weight=1.5))
        np.testing.assert_allclose(p,q,rtol=0,atol=2e-16)

    def test_independent_group_aggregation(self):
        record=dict(formula='group_transfer',group='user',between=.5,within=1.5)
        np.testing.assert_allclose(predict(self.data,record),predict(self.data,record,True),rtol=0,atol=2e-16)


if __name__ == '__main__':
    unittest.main()
