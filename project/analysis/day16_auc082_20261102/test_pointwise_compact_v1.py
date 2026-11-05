import unittest
import numpy as np
import pandas as pd
from pointwise_compact_v1 import pool_from,PARAMS


class PointwiseTests(unittest.TestCase):
    def test_binary_targets_without_pair_filtering(self):
        frame=pd.DataFrame({'cat':[1,2,1,3],'value':[.1,.2,.3,.4]})
        info=dict(target=np.array([0,0,1,0]),groups=np.array([0,0,1,2]),pairs=np.array([[2,3]]))
        pool=pool_from(frame,['cat'],info)
        np.testing.assert_array_equal(pool.get_label(),info['target'])
        self.assertEqual(pool.num_row(),4)
        self.assertEqual(PARAMS['loss_function'],'Logloss')
        self.assertEqual(PARAMS['class_weights'],[1.,3.])


if __name__=='__main__':unittest.main()
