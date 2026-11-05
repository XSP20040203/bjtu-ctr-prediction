"""Small boundary/leakage tests; no raw dataset or training is required."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

from prepare_multiscale_stats import (GLOBAL_STRENGTH, INITIAL_PRIOR, METRICS, SPECS,
                                      SUPPORT_PARENTS, build_multiscale_statistics,
                                      feature_names, normalize_day)


def sample_frame():
    rows = []
    for day in (10, 11, 12, 14, 15, 16, 17):
        for entity in range(3):
            rows.append({"pt_d": 20191100 + day, "label": int((day + entity) % 3 == 0),
                         "user_id": entity if day < 16 else entity + 5,
                         "app_second_class": entity % 2, "spread_app_id": entity % 2,
                         "adv_prim_id": entity, "adv_id": entity + (20 if day >= 15 else 0),
                         "task_id": entity + (100 if day >= 14 else 0), "slot_id": (day + entity) % 2})
    # Row order and a non-RangeIndex must have no effect on alignment.
    frame = pd.DataFrame(rows).sample(frac=1, random_state=42)
    frame.index = np.arange(len(frame)) * 7 + 100
    return frame


def brute_force(frame, validation_day):
    """Independent per-row reference, recomputing history without prefix state."""
    output = np.zeros((len(frame), len(feature_names())), dtype=np.float32)

    def logit(value):
        value = np.clip(value, 1e-6, 1 - 1e-6)
        return np.log(value / (1 - value))

    for position, (_, row) in enumerate(frame.iterrows()):
        if row.pt_d > validation_day:
            continue
        history = frame.loc[frame.pt_d < row.pt_d]
        global_rate = (history.label.sum() + GLOBAL_STRENGTH * INITIAL_PRIOR) / (len(history) + GLOBAL_STRENGTH)
        parent_rates = {}
        current_date = pd.to_datetime(str(int(row.pt_d)), format="%Y%m%d")

        def counts(fields, days=None):
            matched = history
            for field in fields:
                matched = matched.loc[matched[field] == row[field]]
            if days is not None:
                minimum = int((current_date - pd.Timedelta(days=days)).strftime("%Y%m%d"))
                matched = matched.loc[matched.pt_d >= minimum]
            return len(matched), matched.label.sum()

        for fields, alpha in SUPPORT_PARENTS:
            count, clicks = counts(fields)
            parent_rates[fields] = (clicks + alpha * global_rate) / (count + alpha)
        for group, spec in enumerate(SPECS):
            prior = np.mean([parent_rates[parent] for parent in spec.parents]) if spec.parents else global_rate
            relative_rates = []
            for window, days in enumerate((1, 3, None)):
                count, clicks = counts(spec.fields, days)
                probability = (clicks + spec.alpha * prior) / (count + spec.alpha)
                relative = logit(probability) - logit(global_rate)
                column = group * 13 + window * len(METRICS)
                output[position, column:column + 4] = (np.log1p(count), np.log1p(clicks), relative, count == 0)
                relative_rates.append(relative)
                if days is None:
                    parent_rates[spec.fields] = probability
            output[position, group * 13 + 12] = relative_rates[0] - relative_rates[-1]
        for window, days in enumerate((1, 3, None)):
            count, clicks = counts((), days)
            rate = global_rate if days is None else (clicks + GLOBAL_STRENGTH * global_rate) / (count + GLOBAL_STRENGTH)
            output[position, 156 + window] = logit(rate)
    return output


class MultiscaleStatisticsTest(unittest.TestCase):
    def setUp(self):
        self.frame = sample_frame()

    def test_reference_and_calendar_boundaries(self):
        for day in (20191115, 20191116):
            actual, names = build_multiscale_statistics(self.frame, day)
            self.assertEqual(actual.shape, (len(self.frame), 159))
            self.assertEqual(len(set(names)), 159)
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_allclose(actual, brute_force(self.frame, day), rtol=1e-5, atol=2e-6)
            self.assertTrue(np.isfinite(actual).all())
            self.assertTrue((actual[self.frame.pt_d.to_numpy() > day] == 0).all())
        # Calendar day 13 is absent: day 14's 1-day window must be empty.
        names = feature_names()
        column = names.index("past_multiscale__app_second_class__1d__log_count")
        self.assertTrue((actual[self.frame.pt_d.to_numpy() == 20191114, column] == 0).all())

    def test_current_and_future_label_perturbations(self):
        baseline, _ = build_multiscale_statistics(self.frame, 16)
        for pivot in (20191110, 20191111, 20191112, 20191114, 20191115, 20191116):
            changed = self.frame.copy()
            changed.loc[changed.pt_d >= pivot, "label"] = 1 - changed.loc[changed.pt_d >= pivot, "label"]
            actual, _ = build_multiscale_statistics(changed, 16)
            np.testing.assert_array_equal(actual[self.frame.pt_d.to_numpy() <= pivot],
                                          baseline[self.frame.pt_d.to_numpy() <= pivot])

    def test_validation_labels_are_not_inspected(self):
        for day in (20191115, 20191116):
            baseline, _ = build_multiscale_statistics(self.frame, day)
            changed = self.frame.copy()
            changed.loc[changed.pt_d >= day, "label"] = -98765
            actual, _ = build_multiscale_statistics(changed, day)
            np.testing.assert_array_equal(actual, baseline)
        changed = self.frame.copy()
        changed.loc[changed.pt_d == 20191110, "label"] = 2
        with self.assertRaisesRegex(ValueError, "Historical advertisement labels"):
            build_multiscale_statistics(changed, 16)

    def test_row_order_and_new_entity_backoff(self):
        original, names = build_multiscale_statistics(self.frame, 16)
        reordered = self.frame.iloc[::-1]
        flipped, _ = build_multiscale_statistics(reordered, 16)
        np.testing.assert_array_equal(original, flipped[::-1])
        first = self.frame.pt_d.to_numpy() == self.frame.pt_d.min()
        for group in range(len(SPECS)):
            for window in range(3):
                self.assertTrue((original[first, group * 13 + window * 4 + 3] == 1).all())
        # Tasks first appear on day 14: all-history posterior equals their adv parent.
        day14 = self.frame.pt_d.to_numpy() == 20191114
        child = names.index("past_multiscale__task_id__all__relative_logit")
        parent = names.index("past_multiscale__adv_id__all__relative_logit")
        np.testing.assert_allclose(original[day14, child], original[day14, parent], atol=1e-6)

    def test_memmap_and_cli_day_aliases(self):
        expected, _ = build_multiscale_statistics(self.frame, 16)
        with tempfile.TemporaryDirectory() as directory:
            for column_major in (False, True):
                path = Path(directory) / ("features_" + str(column_major) + ".npy")
                mapped = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                                 shape=expected.shape, fortran_order=column_major)
                mapped[:] = 123  # Existing data must not survive on future rows.
                actual, _ = build_multiscale_statistics(self.frame, 16, output=mapped)
                actual.flush()
                np.testing.assert_array_equal(np.load(path, mmap_mode="r"), expected)
                del actual, mapped
        self.assertEqual(normalize_day("15"), 20191115)
        self.assertEqual(normalize_day("16"), 20191116)
        self.assertEqual(normalize_day("20191116"), 20191116)


if __name__ == "__main__":
    unittest.main(verbosity=2)
