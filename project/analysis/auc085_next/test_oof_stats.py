"""CPU-only independent oracle and protocol tests; never train a model."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from prepare_oof_stats import (FIELDS, SPECS, build_oof_statistics, feature_names,
                               fold_assignments, raw_predictor_hash)
from train_oof_dcn import materialize_numeric, original_first39_scalers


DAY = 20191115


def toy():
    # Non-monotonic DataFrame indices expose accidental pandas alignment/reindex.
    rng = np.random.default_rng(781)
    n = 75
    frame = pd.DataFrame({field: rng.integers(0, 4, n) for field in FIELDS})
    frame["pt_d"] = np.repeat([20191111, 20191112, 20191113, 20191114, DAY, 20191116], [13, 13, 13, 13, 15, 8])
    frame["label"] = (np.arange(n) % 7 == 0).astype(np.int8)
    frame["log_id"] = np.arange(n) + 1000
    frame["raw_sequence"] = ["1^2" if index % 2 else "2^1" for index in range(n)]
    for target, source in ((7, 3), (8, 3), (25, 19)):
        for field in frame.columns:
            if field not in ("label", "log_id"):
                frame.loc[target, field] = frame.loc[source, field]
    frame.loc[7, "label"] = 1 - frame.loc[3, "label"]
    frame.index = np.arange(n)[::-1] * 11 + 5
    return frame


def independent_oracle(frame, hashes):
    """Brute force dictionaries and positional rows, no production factorizer."""
    dates = frame.pt_d.to_numpy()
    labels = frame.label.to_numpy()
    train = np.flatnonzero(dates < DAY)
    expected = np.zeros((len(frame), len(feature_names())), dtype=np.float64)
    for row in np.flatnonzero(dates <= DAY):
        fit = [int(index) for index in train
               if dates[row] == DAY or int(hashes[index] % np.uint64(5)) != int(hashes[row] % np.uint64(5))]
        global_rate = sum(int(labels[index]) for index in fit) / len(fit)
        rates = {}
        for number, spec in enumerate(SPECS):
            table = {}
            for index in fit:
                key = tuple(frame.iloc[index][field] for field in spec.fields)
                count, clicks = table.get(key, (0, 0))
                table[key] = (count + 1, clicks + int(labels[index]))
            key = tuple(frame.iloc[row][field] for field in spec.fields)
            count, clicks = table.get(key, (0, 0))
            prior = sum(rates[parent] for parent in spec.parents) / len(spec.parents) if spec.parents else global_rate
            rate = (clicks + spec.alpha * prior) / (count + spec.alpha)
            rates[spec.fields] = rate
            expected[row, number * 2:number * 2 + 2] = (np.log1p(count), rate)
        expected[row, -1] = global_rate
    return expected


class OOFTests(unittest.TestCase):
    def test_independent_oracle_row_permutation_and_fulltrain_validation(self):
        frame = toy()
        hashes = raw_predictor_hash(frame)
        train = frame.pt_d.to_numpy() < DAY
        values, names, folds = build_oof_statistics(frame, frame.label.to_numpy()[train], DAY, hashes)
        self.assertEqual(len(SPECS), 60)
        self.assertEqual(len(names), 121)
        np.testing.assert_allclose(values, independent_oracle(frame, hashes), atol=4e-7, rtol=4e-7)
        self.assertTrue(np.all(values[frame.pt_d.to_numpy() > DAY] == 0))
        self.assertTrue(np.all(folds[~train] == -1))
        order = np.random.default_rng(42).permutation(len(frame))
        permuted = frame.iloc[order]
        new_train = permuted.pt_d.to_numpy() < DAY
        result, _, new_folds = build_oof_statistics(permuted, permuted.label.to_numpy()[new_train], DAY, raw_predictor_hash(permuted))
        np.testing.assert_array_equal(result, values[order])
        np.testing.assert_array_equal(new_folds, folds[order])

    def test_outer_label_poison_and_complete_duplicate_group_exclusion(self):
        frame = toy()
        dates = frame.pt_d.to_numpy()
        train = dates < DAY
        hashes = raw_predictor_hash(frame)
        values, _, folds = build_oof_statistics(frame, frame.label.to_numpy()[train], DAY, hashes)
        self.assertEqual(hashes[3], hashes[7])
        self.assertEqual(hashes[3], hashes[8])
        self.assertEqual(folds[3], folds[7])
        self.assertEqual(folds[3], folds[8])
        altered = frame.copy()
        altered["label"] = altered["label"].astype(object)
        altered.iloc[np.flatnonzero(~train), altered.columns.get_loc("label")] = "OUTER_LABEL_MUST_NEVER_BE_READ"
        altered["log_id"] = -999
        np.testing.assert_array_equal(raw_predictor_hash(altered), hashes)
        actual, _, _ = build_oof_statistics(altered, altered.label.to_numpy()[train].astype(np.int8), DAY, hashes)
        np.testing.assert_array_equal(actual, values)
        # Flip every outcome in this held fold, including differently labelled
        # exact duplicates. The fold's own features, parents and global prior
        # must all remain bit-identical, while other folds are allowed to change.
        held = folds == folds[3]
        changed_labels = frame.label.to_numpy().copy()
        changed_labels[held] = 1 - changed_labels[held]
        changed, _, _ = build_oof_statistics(frame, changed_labels[train], DAY, hashes)
        np.testing.assert_array_equal(changed[held], values[held])
        self.assertTrue(np.any(changed[train & ~held] != values[train & ~held]))
        with self.assertRaises(ValueError):
            build_oof_statistics(frame, frame.label.to_numpy(), DAY, hashes)

    def test_main_preparation_protocol_smoke_without_training(self):
        from argparse import Namespace
        import json
        from unittest.mock import patch
        import prepare_oof_stats as generator
        import train_oof_dcn as trainer
        frame = toy()
        dates = frame.pt_d.to_numpy(dtype=np.int32)
        train = np.flatnonzero(dates < DAY).astype(np.int32)
        valid = np.flatnonzero(dates == DAY).astype(np.int32)
        labels = frame.label.to_numpy(dtype=np.uint8)
        stats, names, folds = build_oof_statistics(frame, labels[train], DAY, raw_predictor_hash(frame))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache, daily = root / "base", root / "daily"
            cache.mkdir()
            (daily / "cache_day15").mkdir(parents=True)
            numeric_names = ["binary_without_scaler"] + ["base_" + str(i) for i in range(38)]
            manifest = {"numeric_fields": numeric_names + ["OLD_TARGET"] * 36,
                        "numeric_scaler": {"base_0": {"mean": 4., "std": 2.}}}
            (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            for name, value in {"date": dates, "label": labels, "train_indices": train, "valid_indices": valid,
                                "log_id": np.arange(len(frame)), "categories": np.zeros((len(frame), 1), dtype=np.int32),
                                "sequences": np.zeros((len(frame), 1, 1), dtype=np.int32)}.items():
                np.save(cache / (name + ".npy"), value)
            strict = np.zeros((len(frame), 75), dtype=np.float32)
            strict[:, 39:] = np.nan
            np.save(daily / "cache_day15/numeric.npy", strict)
            for stem, width in (("exposure_features", 43), ("snapshot_features", 48)):
                np.save(daily / (stem + ".npy"), np.ones((len(frame), width), dtype=np.float32))
                (daily / (stem + ".json")).write_text(json.dumps({"features": [stem + str(i) for i in range(width)],
                    "advertisement_labels_loaded": False}), encoding="utf-8")
            path = root / "stats.npy"
            np.save(path, stats)
            np.save(path.with_suffix(".folds.npy"), folds)
            meta = {"protocol": generator.PROTOCOL, "status": "complete", "validation_day": DAY,
                    "advertisement_validation_labels_used": False, "advertisement_validation_labels_loaded": False,
                    "train_oof_allowed_across_train_dates": True, "per_row_strictly_historical": False,
                    "duplicate_group_fold_check": "passed", "raw_predictor_field_count": 33, "fold_count": 5,
                    "features": names, "shape": list(stats.shape), "date_sha256": generator.array_digest(dates),
                    "train_indices_sha256": generator.array_digest(train.astype(np.int64)),
                    "train_label_sha256": generator.array_digest(labels[train]),
                    "fold_sha256": generator.sha256(path.with_suffix(".folds.npy")),
                    "source_sha256": "toy", "outer_split": "whole date toy"}
            path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
            args = Namespace(day=15, tag="smoke", pos_weight=1., stats=path, model="dcn", source_context=False,
                             seed=42, epochs=2, patience=4, learning_rate=.001)
            def no_training(*arguments):
                dense, protocol = arguments[4], arguments[6]
                self.assertEqual(dense.shape[1], 251)
                self.assertFalse(protocol["old_target_statistics_columns_used"])
                self.assertEqual(protocol["original_first39_scalers"]["binary_without_scaler"], {"mean": 0., "std": 1.})
                self.assertEqual(protocol["original_first39_scalers"]["base_0"], {"mean": 4., "std": 2.})
                dense._mmap.close()
                return {"maximum_reload_error": 0.}
            with patch.object(trainer, "DAILY", daily), patch.object(trainer, "base_cache", return_value=cache), \
                    patch.object(trainer, "train_dcn", side_effect=no_training) as train_stub:
                trainer.main(args)
                train_stub.assert_called_once()
                for array in train_stub.call_args.args[2].values():
                    array._mmap.close()
                train_stub.reset_mock()
            protocol = json.loads((daily / "dense_results/smoke/protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(len(protocol["original_first39_scalers"]), 39)
            self.assertTrue((daily / "dense_results/smoke/metrics.json").exists())
            self.assertEqual(original_first39_scalers(manifest)["binary_without_scaler"], {"mean": 0., "std": 1.})

    def test_memmap_replaces_all_old_targets_and_scaler_fits_only_train(self):
        frame = toy()
        train = np.flatnonzero(frame.pt_d.to_numpy() < DAY)
        stats, names, _ = build_oof_statistics(frame, frame.label.to_numpy()[train], DAY, raw_predictor_hash(frame))
        rng = np.random.default_rng(42)
        strict = rng.normal(size=(len(frame), 75)).astype(np.float32)
        strict[:, 39:75] = np.nan  # Any legacy target-stat access fails finiteness.
        extra = rng.normal(size=(len(frame), 2)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            mapped = np.lib.format.open_memmap(Path(tmp) / "stats.npy", mode="w+", shape=stats.shape,
                                              dtype="float32", fortran_order=True)
            mapped[:] = stats
            values, scalers = materialize_numeric(Path(tmp) / "dense.npy", strict, train, mapped, names,
                                                 [(extra, ["extra_a", "extra_b"])], chunk_rows=7)
            self.assertIsInstance(values, np.memmap)
            self.assertEqual(values.shape, (len(frame), 39 + 121 + 2))
            np.testing.assert_array_equal(values[:, :39], strict[:, :39])
            self.assertTrue(np.isfinite(values).all())
            for column, name in enumerate(names):
                self.assertAlmostEqual(scalers[name]["mean"], stats[train, column].astype(np.float64).mean(), places=12)
            changed = stats.copy()
            changed[frame.pt_d.to_numpy() >= DAY] = 1e6
            perturbed, altered_scalers = materialize_numeric(Path(tmp) / "changed.npy", strict, train, changed, names, chunk_rows=9)
            self.assertEqual({name: scalers[name] for name in names}, altered_scalers)
            np.testing.assert_array_equal(perturbed[train, 39:], values[train, 39:39 + 121])
            raw, _ = materialize_numeric(Path(tmp) / "raw.npy", strict, train, stats, names, standardize=False, chunk_rows=4)
            np.testing.assert_array_equal(raw[:, 39:], stats)
            np.testing.assert_array_equal(raw[:, :39], strict[:, :39])
            # Explicit close lets Windows remove the temporary mapped files.
            for array in (mapped, values, perturbed, raw):
                array._mmap.close()


if __name__ == "__main__":
    unittest.main()
