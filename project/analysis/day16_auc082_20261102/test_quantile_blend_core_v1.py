import unittest
import numpy as np
from scipy.special import expit, logit
from quantile_blend_core_v1 import matched_logit, blend


class QuantileBlendTests(unittest.TestCase):
    def test_order_ties_and_independent_interpolation(self):
        base = np.array([.1,.2,.3,.4,.5])
        margin = np.array([8.,2.,2.,9.,1.])
        result = matched_logit(base, margin)
        expected = np.array([logit(.4), .5*(logit(.2)+logit(.3)),
                             .5*(logit(.2)+logit(.3)), logit(.5), logit(.1)])
        np.testing.assert_allclose(result, expected, atol=1e-14, rtol=0)
        np.testing.assert_allclose(matched_logit(base, margin, True), expected, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(matched_logit(base, margin*3+10), result)

    def test_probability_formula_and_unchanged_unselected_rows(self):
        base = np.array([.001,.1,.2,.9])
        target = np.array([1.,-1.,2.,-2.])
        mask = np.array([False,True,True,False])
        result = blend(base, target, mask, .25)
        np.testing.assert_array_equal(result[~mask], base[~mask])
        np.testing.assert_allclose(result[mask], expit(.75*logit(base[mask])+.25*target[mask]))
        np.testing.assert_allclose(result, blend(base,target,mask,.25,True), atol=1e-14, rtol=0)
        np.testing.assert_array_equal(blend(base,target,mask,0),base)


if __name__ == '__main__':
    unittest.main()
