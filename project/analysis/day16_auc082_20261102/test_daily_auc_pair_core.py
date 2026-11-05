"""Verify click/nonclick exposure weighting and date/label boundaries."""
import unittest
import numpy as np
from daily_auc_pair_core import sample_pairs


class PairChecks(unittest.TestCase):
    def test_negative_draws_follow_nonclick_exposures(self):
        keys=np.array([[20191112,1],[20191112,2],[20191112,3]])
        training,pairs,groups=sample_pairs(keys,[1,1,9],[1,0,0],20191115,10000)
        self.assertTrue(np.all(pairs[:,0]==0))
        self.assertLess(abs(np.mean(pairs[:,1]==2)-.9),.02)

    def test_pairs_remain_in_prior_dates(self):
        keys=np.array([[20191114,2],[20191113,2],[20191115,3],[20191113,1],[20191114,1]])
        training,pairs,groups=sample_pairs(keys,[4,4,5,3,3],[0,0,-1,1,1],20191115)
        self.assertTrue(np.all(keys[training,0]<20191115))
        self.assertTrue(np.all(groups[pairs[:,0]]==groups[pairs[:,1]]))
        self.assertTrue(np.all(pairs[:,0]!=pairs[:,1]))

    def test_held_labels_and_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            sample_pairs([[20191114,1],[20191115,2]],[4,4],[1,0],20191115)
        with self.assertRaises(ValueError):
            sample_pairs([[20191114,1],[20191114,1]],[4,4],[1,0],20191115)

    def test_noninteger_or_impossible_counts_rejected(self):
        for counts in [[1.5,0],[5,0],[-1,0]]:
            with self.assertRaises(ValueError):
                sample_pairs([[20191114,1],[20191114,2]],[4,4],counts,20191115)


if __name__=='__main__':
    unittest.main()
