import unittest
import numpy as np
from label_boundary_audit import signed_labels


class SignedMaskContracts(unittest.TestCase):
    def test_unsigned_input_is_masked_without_touching_training(self):
        labels = np.array([0,1,1,0],dtype="uint8")
        fixed = signed_labels(labels,[0,1],[2,3])
        np.testing.assert_array_equal(fixed,[0,1,-1,-1])
        np.testing.assert_array_equal(labels,[0,1,1,0])
        self.assertTrue(np.issubdtype(fixed.dtype,np.signedinteger))

    def test_overlap_and_unknown_training_rejected(self):
        with self.assertRaises(ValueError):
            signed_labels(np.array([0,1]),[0,1],[1])
        with self.assertRaises(ValueError):
            signed_labels(np.array([0,255],dtype="uint8"),[0,1],[])


if __name__ == "__main__":
    unittest.main()
