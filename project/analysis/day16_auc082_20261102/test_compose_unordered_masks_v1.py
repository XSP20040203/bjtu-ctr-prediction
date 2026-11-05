import unittest
import numpy as np
from compose_unordered_masks_v1 import masked_blend


class BlendTests(unittest.TestCase):
    def test_mask_and_independent_odds(self):
        base=np.array([.01,.05,.2,.7])
        margin=np.array([-2.,2.,1.,-1.])
        mask=np.array([True,False,True,False])
        p=masked_blend(base,margin,mask,.35)
        np.testing.assert_array_equal(p[~mask],base[~mask])
        np.testing.assert_allclose(p,masked_blend(base,margin,mask,.35,True),rtol=0,atol=1e-14)
        np.testing.assert_allclose(masked_blend(base,margin,mask,0),base,rtol=0,atol=1e-15)
        with self.assertRaises(ValueError): masked_blend(base,margin,mask,1.5)


if __name__=='__main__': unittest.main()
