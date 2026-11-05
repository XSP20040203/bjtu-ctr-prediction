"""Check row CTR offsets, exposure invariance, and group-constant logit updates."""
import unittest
import numpy as np
from scipy.special import logit,expit
from daily_auc_pair_math import rate_baseline,daily_prediction


class DailyMathChecks(unittest.TestCase):
    def test_prior_is_per_exposure_and_float32_pool_exact(self):
        np.testing.assert_array_equal(rate_baseline([.8,8],[100,1000]),
            np.repeat(np.float64(np.float32(logit(.008))),2))
        self.assertNotAlmostEqual(rate_baseline([.8],[100])[0],logit(-np.expm1(-.8)))

    def test_invalid_offsets_rejected(self):
        for mass,sizes in [([0],[10]),([1],[0]),([np.nan],[10]),([1,2],[10])]:
            with self.assertRaises(ValueError):
                rate_baseline(mass,sizes)

    def test_preserves_within_user_odds_and_pandas_replay(self):
        base=np.array([.003,.02,.007,.07,.013,.002])
        groups=np.array([2,0,2,0,3,3])
        residual=np.array([.3,0,-.4,.7])
        total=residual+rate_baseline([.1,.01,.3,.4],[10,2,60,15])
        for mode in ['residual_add','rate_replace','count_mass']:
            p=daily_prediction(base,residual,total,[10,2,60,15],groups,.25,mode)
            q=daily_prediction(base,residual,total,[10,2,60,15],groups,.25,mode,True)
            np.testing.assert_allclose(p,q,rtol=0,atol=2e-14)
            delta=logit(p)-logit(base)
            for group in np.unique(groups):
                self.assertLess(np.ptp(delta[groups==group]),1e-12)

    def test_zero_weight_identity_and_constant_score_stability(self):
        base=np.array([.01,.03,.005,.02])
        groups=np.array([0,0,1,1])
        np.testing.assert_array_equal(daily_prediction(base,[0,0],[-5,-5],[2,2],groups,0.,'rate_replace'),base)
        p=daily_prediction(base,[0,0],[-5,-5],[2,2],groups,.5,'residual_add')
        np.testing.assert_allclose(p,base,rtol=0,atol=1e-15)
        self.assertTrue(np.isfinite(p).all())


if __name__=='__main__':
    unittest.main()
