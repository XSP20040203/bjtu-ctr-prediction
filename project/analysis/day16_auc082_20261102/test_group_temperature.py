import unittest
import numpy as np
import pandas as pd
from group_temperature import transform, group_codes


class GroupTemperatureBoundaries(unittest.TestCase):
    def setUp(self):
        self.p = np.array([.001, .01, .2, .004, .03, .1])
        self.groups = np.array([0, 0, 0, 1, 1, 1])

    def test_unit_temperature_is_exact_identity(self):
        for mode in ['logit_mean', 'preserve_hazard_mass']:
            np.testing.assert_array_equal(transform(self.p, self.groups, 1., mode), self.p)

    def test_hazard_mass_and_within_group_order_are_preserved(self):
        q = transform(self.p, self.groups, 1.5, 'preserve_hazard_mass')
        np.testing.assert_allclose(np.bincount(self.groups, weights=-np.log1p(-q)),
            np.bincount(self.groups, weights=-np.log1p(-self.p)), rtol=0, atol=1e-14)
        for group in [0, 1]:
            self.assertTrue(np.all(np.diff(q[self.groups == group]) > 0))

    def test_row_permutation_does_not_change_transform(self):
        order = np.array([5, 1, 4, 0, 2, 3])
        for mode in ['logit_mean', 'preserve_hazard_mass']:
            before = transform(self.p, self.groups, .75, mode)
            after = transform(self.p[order], self.groups[order], .75, mode)
            np.testing.assert_allclose(after, before[order], rtol=0, atol=1e-15)

    def test_label_column_is_rejected(self):
        frame = pd.DataFrame({'user_id': [1], 'task_id': [2], 'slot_id': [3], 'spread_app_id': [4], 'label': [1]})
        with self.assertRaises(ValueError):
            group_codes(frame, ['user_id'])


if __name__ == '__main__':
    unittest.main()
