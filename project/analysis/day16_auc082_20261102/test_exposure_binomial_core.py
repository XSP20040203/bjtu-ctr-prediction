"""Prove compressed binomial loss matches original binary exposure observations."""
import unittest
import numpy as np
from scipy.special import expit
from exposure_binomial_core import training_targets


class BinomialChecks(unittest.TestCase):
    def test_weighted_soft_loss_and_gradient_match_binary_rows(self):
        keys=np.array([[20191113,1],[20191114,2],[20191115,3]])
        counts=np.array([2,1,-1])
        sizes=np.array([5,7,10])
        training,target,weights=training_targets(keys,counts,sizes,20191115)
        z=np.array([-.3,-2.])
        compressed=np.sum(weights*(np.logaddexp(0,z)-target*z))
        expanded=0.
        gradient=[]
        for i in training:
            y=np.r_[np.ones(counts[i]),np.zeros(sizes[i]-counts[i])]
            expanded+=np.sum(np.logaddexp(0,z[i])-y*z[i])/sizes[training].mean()
            gradient.append(np.sum(expit(z[i])-y)/sizes[training].mean())
        self.assertAlmostEqual(compressed,expanded,places=14)
        np.testing.assert_allclose(weights*(expit(z)-target),gradient,rtol=0,atol=1e-15)

    def test_recency_uses_only_prior_dates_and_preserves_exposure_ratios(self):
        keys=np.array([[20191113,1],[20191114,1],[20191114,2],[20191115,3]])
        train,target,weights=training_targets(keys,[1,0,1,-1],[10,10,20,30],20191115,1.)
        self.assertTrue(np.all(keys[train,0]<20191115))
        self.assertAlmostEqual(weights[1]/weights[0],2.)
        self.assertAlmostEqual(weights[2]/weights[1],2.)

    def test_unmasked_future_and_bad_counts_rejected(self):
        for counts in [[1,0],[1,2],[-1,-1],[1.5,-1],[11,-1]]:
            with self.assertRaises(ValueError):
                training_targets([[20191114,1],[20191115,2]],counts,[10,10],20191115)

    def test_fractional_or_zero_exposures_rejected(self):
        for sizes in [[0,10],[1.5,10]]:
            with self.assertRaises(ValueError):
                training_targets([[20191114,1],[20191115,2]],[1,-1],sizes,20191115)


if __name__=='__main__':
    unittest.main()
