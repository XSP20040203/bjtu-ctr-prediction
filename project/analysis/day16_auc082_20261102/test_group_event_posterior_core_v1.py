"""Event semantics, temporal exclusion and conditional group marginals."""
import unittest
import numpy as np
import pandas as pd
from scipy.special import logit
from group_event_posterior_core_v1 import CHANNELS, FIELDS, events, fit, deltas, predict


def sample():
    frame = pd.DataFrame(dict(pt_d=[20191113] * 3 + [20191114] * 4,
        user_id=[1, 1, 1, 2, 2, 2, 2], task_id=[7, 8, 9, 7, 7, 8, 9],
        adv_prim_id=[7, 8, 9, 7, 7, 8, 9], spread_app_id=[7, 8, 9, 7, 7, 8, 9]))
    for field, _ in CHANNELS:
        frame[field] = ['-1', '7', '7', '-1', '-1', '7', '7']
    return frame[FIELDS]


class EventTests(unittest.TestCase):
    def test_groups_ignore_row_order_and_hidden_targets(self):
        frame = sample()
        first, other = events(frame), events(frame, True)
        permutation = np.array([6, 1, 4, 0, 3, 5, 2])
        shuffled = events(frame.iloc[permutation])
        for channel, group in first.items():
            for name, value in group.items():
                np.testing.assert_array_equal(value, other[channel][name])
                np.testing.assert_array_equal(value[permutation] if name == 'ids' else value, shuffled[channel][name])
            np.testing.assert_array_equal(group['sizes'], [1, 2])
            np.testing.assert_array_equal(group['bins'], [4, 2])
        dates = frame.pt_d.to_numpy()
        y = np.array([1, 0, 0, -1, -1, -1, -1])
        fitted = fit(first, dates, y, 20191114, 5.)
        self.assertEqual(fitted, fit(other, dates, y, 20191114, 5., independent=True))
        with self.assertRaises(ValueError):
            fit(first, dates, np.zeros(7), 20191114, 5.)

    def test_noisy_or_distributes_probability_and_empty_is_safe(self):
        frame = sample().iloc[3:].reset_index(drop=True)
        group = events(frame)
        record = dict(global_row_rate=.02, channels={field: dict(rates=[.5, .5, .5, .5]) for field, _ in CHANNELS})
        base = np.array([.01, .03, .02, .02])
        first, second = deltas(base, group, record), deltas(base, group, record, True)
        for name in first:
            np.testing.assert_allclose(first[name], second[name], rtol=0, atol=1e-12)
            p = predict(base, first[name], .5)
            np.testing.assert_allclose(p, predict(base, second[name], .5, True), rtol=0, atol=1e-12)
            self.assertGreater(p[0], base[0])
            self.assertGreater(p[1], p[0])
            np.testing.assert_allclose(p[2:], base[2:], rtol=0, atol=1e-15)
        empty = frame.copy()
        for field, _ in CHANNELS:
            empty[field] = '-1'
        zero = deltas(base, events(empty), record)
        np.testing.assert_array_equal(zero['maximum'], np.zeros(4))


if __name__ == '__main__':
    unittest.main()
