import unittest
import numpy as np
from within_user_ranker import checked_layout,core,lgb


class FullCohortLayout(unittest.TestCase):
    def setUp(self):
        self.dates=np.array([20191114,20191115,20191114,20191115,20191116])
        self.users=np.array([1,1,2,2,3])
        self.log_ids=np.arange(100,105)

    def test_full_held_cohort_keeps_raw_order(self):
        positions,rows,groups,audit=checked_layout(np.array([0,1,2,3]),self.dates,self.users,self.log_ids,20191115)
        np.testing.assert_array_equal(positions,[1,3])
        np.testing.assert_array_equal(rows,[1,3])
        np.testing.assert_array_equal(groups,[0,1])
        self.assertEqual(audit['rows'],2)

    def test_missing_held_record_is_rejected(self):
        with self.assertRaises(ValueError):
            checked_layout(np.array([0,1,2]),self.dates,self.users,self.log_ids,20191115)

    def test_later_date_or_duplicate_source_record_is_rejected(self):
        for rows in [np.array([0,1,2,3,4]),np.array([0,1,1,2,3])]:
            with self.assertRaises(ValueError):
                checked_layout(rows,self.dates,self.users,self.log_ids,20191115)

    def test_constructed_dataset_supports_both_ranking_objectives(self):
        rng=np.random.default_rng(3)
        values=rng.normal(size=(40,3)).astype('float32')
        target=np.tile([0,1,0,0,0],8)
        params=dict(core.PARAMS,min_data_in_leaf=2,num_leaves=3,feature_fraction=1.)
        dataset=lgb.Dataset(values,label=target,group=[5]*8,params=params,free_raw_data=True)
        dataset.construct()
        for objective in ['lambdarank','rank_xendcg']:
            model=lgb.train(dict(params,objective=objective),dataset,num_boost_round=5)
            prediction=model.predict(values,raw_score=True,num_threads=3)
            self.assertEqual(prediction.shape,(40,))
            self.assertTrue(np.isfinite(prediction).all())


if __name__=='__main__':
    unittest.main()
