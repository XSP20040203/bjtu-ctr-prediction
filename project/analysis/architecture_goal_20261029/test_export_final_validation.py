"""Regression checks for source integrity and nonfinite verification failures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import export_best_architecture_final as export


class FinalValidationTests(unittest.TestCase):
    def test_probability_comparison_rejects_nan_inf_shape_and_range(self):
        good = np.array([.2, .8])
        self.assertEqual(export.probability_error(good, good, 2), 0.)
        for bad in [np.array([np.nan, .8]), np.array([.2, np.inf]),
                    np.array([-.1, .8]), np.array([.2]), np.array([[.2, .8]])]:
            with self.subTest(bad=bad):
                with self.assertRaises(AssertionError):
                    export.probability_error(good, bad, 2)

    def test_failed_or_interrupted_verification_replaces_stale_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with mock.patch.object(export, 'OUT', out):
                for error in [AssertionError('bad source'), KeyboardInterrupt()]:
                    export.write(out / 'verification.json', {'status': 'passed'})
                    with mock.patch.object(export, '_verify', side_effect=error):
                        with self.assertRaises(type(error)):
                            export.verify()
                    record = export.read(out / 'verification.json')
                    self.assertEqual(record['status'], 'failed')
                    self.assertEqual(record['error_type'], type(error).__name__)

    def fixtures(self, root):
        v6, full, out = root / 'v6', root / 'full', root / 'out'
        v6.mkdir(); full.mkdir(); out.mkdir()
        p = np.full(404182, .1, dtype='float32')
        np.save(v6 / 'ensemble_final_prediction.npy', p)
        export.write(v6 / 'verification.json', {'status': 'passed', 'final_label_column_read': False})
        export.write(v6 / 'manifest.json', {'files': {'ensemble_final_prediction.npy':
            {'sha256': export.digest(v6 / 'ensemble_final_prediction.npy')}}})
        sources = {'v6': {'path': str(v6 / 'ensemble_final_prediction.npy'),
                           'sha256': export.digest(v6 / 'ensemble_final_prediction.npy')}}
        for name in ['gated500', 'allocated', 'gated166']:
            folder = full / name
            folder.mkdir()
            (folder / 'model.pt').write_bytes(b'fixture checkpoint')
            np.save(folder / 'final_prediction.npy', p)
            prediction_hash = export.digest(folder / 'final_prediction.npy')
            export.write(folder / 'verification.json', {'status': 'complete_and_verified',
                'final_labels_read': False, 'train_rows': 2526320, 'final_rows': 404182,
                'model_sha256': export.digest(folder / 'model.pt'), 'prediction_sha256': prediction_hash})
            sources[name] = {'path': str(folder / 'final_prediction.npy'), 'sha256': prediction_hash}
        export.write(out / 'manifest.json', {'components': sources})
        return v6, full, out

    def test_changed_valid_probability_vector_cannot_receive_fresh_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            v6, full, out = self.fixtures(Path(temporary))
            with mock.patch.object(export, 'V6', v6), mock.patch.object(export, 'FULL', full), mock.patch.object(export, 'OUT', out):
                self.assertEqual(set(export.source_predictions()), {'v6', 'gated500', 'allocated', 'gated166'})
                np.save(full / 'allocated/final_prediction.npy', np.full(404182, .2, dtype='float32'))
                with self.assertRaisesRegex(AssertionError, 'recorded hash'):
                    export.source_predictions()

    def test_prior_manifest_and_model_hashes_are_enforced(self):
        for target in ['prior', 'model']:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                v6, full, out = self.fixtures(Path(temporary))
                with mock.patch.object(export, 'V6', v6), mock.patch.object(export, 'FULL', full), mock.patch.object(export, 'OUT', out):
                    if target == 'prior':
                        np.save(v6 / 'ensemble_final_prediction.npy', np.full(404182, .2, dtype='float32'))
                    else:
                        (full / 'gated500/model.pt').write_bytes(b'changed checkpoint')
                    with self.assertRaisesRegex(AssertionError, 'recorded hash'):
                        export.source_predictions()

    def test_legacy_gate_hash_comes_from_original_export_or_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            v6, full, out = self.fixtures(Path(temporary))
            path = full / 'gated166/verification.json'
            record = export.read(path)
            record.pop('prediction_sha256')
            export.write(path, record)
            with mock.patch.object(export, 'V6', v6), mock.patch.object(export, 'FULL', full), mock.patch.object(export, 'OUT', out):
                export.source_predictions()
                np.save(full / 'gated166/final_prediction.npy', np.full(404182, .3, dtype='float32'))
                with self.assertRaisesRegex(AssertionError, 'recorded hash'):
                    export.source_predictions()
                np.save(full / 'gated166/final_prediction.npy', np.full(404182, .1, dtype='float32'))
                (out / 'manifest.json').unlink()
                with self.assertRaisesRegex(AssertionError, 'recorded prediction hash'):
                    export.source_predictions()

    def test_atomic_json_does_not_write_nan(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'record.json'
            export.write(path, {'status': 'passed'})
            with self.assertRaises(ValueError):
                export.write(path, {'error': float('nan')})
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {'status': 'passed'})


if __name__ == '__main__':
    unittest.main()
