"""Boundary, deterministic ties, and exact transition semantics checks."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import prepare_transition_features as transition


def fixture():
    frame = pd.DataFrame({'pt_d': [20191115] * 5, 'user_id': [1] * 5,
                          'log_id': [10, 20, 30, 40, 50],
                          'task_id': [3, 3, 7, 2, 2],
                          'adv_prim_id': [30, 30, 70, 20, 20],
                          'spread_app_id': [300, 300, 700, 200, 200],
                          'slot_id': [1, 2, 1, 2, 1],
                          'ad_click_list_v001': ['1^2'] * 3 + ['3^1'] * 2,
                          'ad_click_list_v002': ['10^20'] * 3 + ['30^10'] * 2,
                          'ad_click_list_v003': ['100^200'] * 3 + ['300^100'] * 2})
    return frame


class TransitionFeaturesTest(unittest.TestCase):
    def values(self, frame):
        values, names = transition.build_features(frame)
        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(values.shape, (len(frame), 56))
        self.assertTrue(np.isfinite(values).all())
        return values, names

    def column(self, values, names, metric, family='ad_click_list_v001'):
        return values[:, names.index('transition__' + family + '__' + metric)]

    def test_exact_matches_and_proximity(self):
        values, names = self.values(fixture())
        expected = {'candidate_in_own': [0, 0, 0, 0, 0],
                    'candidate_in_other_snapshot': [1, 1, 0, 1, 1],
                    'candidate_next_added': [1, 1, 0, 0, 0],
                    'candidate_previous_removed': [0, 0, 0, 1, 1],
                    'candidate_last_before_addition': [0, 1, 0, 0, 0],
                    'candidate_first_after_removal': [0, 0, 0, 1, 0],
                    'next_add_matching_remaining_fraction': [1, 0, 0, 0, 0],
                    'previous_remove_matching_elapsed_fraction': [0, 0, 0, 0, 1],
                    'next_add_rank_proximity': [0, .5, 0, 0, 0],
                    'previous_remove_rank_proximity': [0, 0, 0, 1, 0],
                    'next_add_signed_log_id_gap_fraction': [.75, .5, 0, 0, 0],
                    'previous_remove_signed_log_id_gap_fraction': [0, 0, 0, -.25, -.5],
                    'block_position_fraction': [0, .5, 1, 0, 1]}
        for family, _ in transition.FAMILIES:
            for metric, vector in expected.items():
                np.testing.assert_allclose(self.column(values, names, metric, family), vector)
        self.assertEqual(self.column(values, names, 'next_add_log_matching_count')[1], np.float32(np.log(3)))

    def test_label_unused_and_permutation(self):
        frame = fixture().assign(label=[0, 1, 1, 0, 0])
        first, _ = self.values(frame)
        changed, _ = self.values(frame.assign(label=-999))
        np.testing.assert_array_equal(first, changed)
        permutation = [4, 0, 3, 2, 1]
        shuffled, _ = self.values(frame.iloc[permutation])
        np.testing.assert_array_equal(shuffled, first[permutation])

    def test_user_and_date_boundaries(self):
        base = fixture()
        future = base.assign(pt_d=20191116)
        other = base.assign(user_id=2)
        full = pd.concat([base, future, other], ignore_index=True)
        first, _ = self.values(full)
        expected, _ = self.values(base)
        np.testing.assert_array_equal(first[:len(base)], expected)
        full.loc[full.pt_d == 20191116, 'ad_click_list_v001'] = '999^888'
        full.loc[full.user_id == 2, 'task_id'] = 999
        changed, _ = self.values(full)
        np.testing.assert_array_equal(changed[:len(base)], first[:len(base)])

    def test_equal_id_matches_all_nearest_and_permutation(self):
        frame = fixture()
        frame.loc[0, 'log_id'] = 20
        values, names = self.values(frame)
        np.testing.assert_array_equal(self.column(values, names, 'candidate_last_before_addition'), [1, 1, 0, 0, 0])
        np.testing.assert_array_equal(values[:, 0], [1, 1, 0, 0, 0])
        np.testing.assert_array_equal(values[0], values[1])
        permutation = [3, 1, 4, 0, 2]
        shuffled, _ = self.values(frame.iloc[permutation])
        np.testing.assert_array_equal(shuffled, values[permutation])

    def test_conflicting_id_snapshots_suppress_order(self):
        frame = fixture()
        frame.loc[3, 'log_id'] = 30
        values, names = self.values(frame)
        for field, _ in transition.FAMILIES:
            np.testing.assert_array_equal(self.column(values, names, 'ambiguous_snapshot_tie'), np.ones(5))
            for metric in transition.METRICS[3:17]:
                np.testing.assert_array_equal(self.column(values, names, metric, field), np.zeros(5))
        permutation = [4, 3, 2, 0, 1]
        shuffled, _ = self.values(frame.iloc[permutation])
        np.testing.assert_array_equal(shuffled, values[permutation])

    def test_repeated_own_candidate_other_control(self):
        frame = fixture()
        frame.loc[:, 'task_id'] = 1
        values, names = self.values(frame)
        np.testing.assert_array_equal(self.column(values, names, 'candidate_in_own'), np.ones(5))
        np.testing.assert_array_equal(self.column(values, names, 'candidate_in_other_snapshot'), np.ones(5))
        np.testing.assert_array_equal(self.column(values, names, 'candidate_absent_own_in_union'), np.zeros(5))
        np.testing.assert_array_equal(self.column(values, names, 'candidate_next_added'), np.zeros(5))

    def test_csv_usecols_and_row_positions(self):
        frame = fixture()
        frame.loc[1, 'pt_d'] = 20191116
        frame = frame.assign(label=[1, 1, 1, 1, 1], unrelated='not_loaded')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'train.csv'
            frame.to_csv(path, index=False)
            original = pd.read_csv
            with patch.object(transition.pd, 'read_csv', wraps=original) as reader:
                selected, positions = transition.read_day(path, 20191115, 2)
            self.assertEqual(reader.call_args.kwargs['usecols'], transition.INPUT_FIELDS)
            self.assertNotIn('label', selected.columns)
            np.testing.assert_array_equal(positions, [0, 2, 3, 4])
            values, _ = self.values(selected)
            expected, _ = self.values(frame.iloc[positions])
            np.testing.assert_array_equal(values, expected)

    def test_zero_is_a_token_and_minus_one_missing(self):
        frame = fixture()
        frame.loc[:, 'task_id'] = 0
        frame.loc[:, 'ad_click_list_v001'] = '0^-1'
        values, names = self.values(frame)
        np.testing.assert_array_equal(self.column(values, names, 'candidate_in_own'), np.ones(5))
        frame.loc[:, 'task_id'] = -1
        values, names = self.values(frame)
        np.testing.assert_array_equal(self.column(values, names, 'candidate_in_own'), np.zeros(5))

    def test_empty(self):
        self.values(fixture().iloc[:0])


if __name__ == '__main__':
    unittest.main()
