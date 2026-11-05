"""Small CPU fixtures for the full gated refit's fail-closed input checks."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import refit_gated_final as refit


class RefitValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache = self.root / 'cache'
        self.cache.mkdir()
        (self.root / '2026ml').mkdir()
        self.meta = {'train_rows': 3, 'final_rows': 2}
        self.train = pd.DataFrame({'log_id': [10, 11, 12], 'user_id': [2, 3, 2],
                                   'pt_d': [20191114, 20191115, 20191116], 'label': [0, 1, 0]})
        # Poison final labels: identity checks must never request or parse them.
        self.final = pd.DataFrame({'log_id': [13, 14], 'user_id': [3, 4],
                                   'pt_d': [20191117, 20191117], 'label': ['secret', 'secret']})
        self.train.to_csv(self.root / '2026ml/train.csv', index=False)
        self.final.to_csv(self.root / '2026ml/final_test.csv', index=False)
        for field, name in [('log_id', 'log_id'), ('user_id', 'user_id'), ('pt_d', 'date')]:
            np.save(self.cache / f'{name}.npy', np.concatenate([self.train[field], self.final[field]]))
        np.save(self.cache / 'label.npy', self.train['label'].to_numpy())
        self.identities = {field: hashlib.sha256(np.ascontiguousarray(
            np.concatenate([self.train[field], self.final[field]]), dtype='<i8').tobytes()).hexdigest()
            for field in refit.IDENTITY_COLUMNS}

    def tearDown(self):
        self.temp.cleanup()

    def alignment(self):
        with patch.object(refit, 'ROOT', self.root), patch.object(refit, 'CACHE', self.cache):
            return refit.raw_alignment(self.meta)

    def test_all_raw_identities_and_training_labels(self):
        self.assertEqual(self.alignment(), self.identities)
        np.save(self.cache / 'label.npy', np.array([1, 0, 0]))
        with self.assertRaises(AssertionError):
            self.alignment()
        np.save(self.cache / 'label.npy', self.train['label'].to_numpy())
        values = np.load(self.cache / 'log_id.npy')
        values[[0, 1]] = values[[1, 0]]
        np.save(self.cache / 'log_id.npy', values)
        with self.assertRaises(AssertionError):
            self.alignment()

    def test_final_order_and_row_counts(self):
        self.final.iloc[::-1].to_csv(self.root / '2026ml/final_test.csv', index=False)
        with self.assertRaises(AssertionError):
            self.alignment()
        self.final.iloc[:1].to_csv(self.root / '2026ml/final_test.csv', index=False)
        with self.assertRaises(AssertionError):
            self.alignment()

    def input_fixture(self, width):
        path = self.root / f'numeric{width}.npy'
        np.save(path, np.zeros((5, width), dtype='float32'))
        names = [f'f{i}' for i in range(width)]
        scalers = {name: {'mean': 0., 'std': 1.} for name in names}
        info = {'status': 'complete_and_verified', 'final_labels_read': False,
            'numeric_fields': names, 'train_rows': 3, 'final_rows': 2, 'clip': [-8, 8],
            'identity_hash_rule': refit.IDENTITY_RULE, 'row_identity_sha256': self.identities,
            'numeric_sha256': {'oof': refit.digest(path)}, 'source_scalers_oof': scalers,
            'scalers': scalers, 'scaler_fit_rows': 3, 'scaler_fit_indices': [0, 3],
            'raw_sources_finite_all_rows': True}
        refit.write(self.root / ('manifest.json' if width == 500 else 'input_manifest.json'), info)
        return path, info, names

    def validate(self, path, info, names, width):
        return refit.validate_input(path, info, names, self.meta, self.identities, width)

    def test_hash_identity_scaler_and_clip_validation(self):
        for width in (166, 500):
            path, info, names = self.input_fixture(width)
            numeric, preprocessing = self.validate(path, info, names, width)
            numeric._mmap.close()
            self.assertEqual(preprocessing['scalers'], info['scalers'])
            for field, value in [('row_identity_sha256', {}), ('numeric_sha256', {'oof': '0' * 64}),
                                 ('clip', [-9, 9]), ('scaler_fit_indices', [0, 5])]:
                bad = dict(info, **{field: value})
                with self.assertRaises(AssertionError):
                    self.validate(path, bad, names, width)
            bad = copy.deepcopy(info)
            bad['scalers'][names[0]]['std'] = float('nan')
            with self.assertRaises(AssertionError):
                self.validate(path, bad, names, width)
            values = np.load(path)
            values[-1, -1] = 9.
            np.save(path, values)
            with self.assertRaises(AssertionError):
                self.validate(path, info, names, width)
            info['numeric_sha256']['oof'] = refit.digest(path)
            with self.assertRaises(AssertionError):
                self.validate(path, info, names, width)

    def test_every_chunk_checked_and_inference_rejects_nonfinite(self):
        values = np.zeros((16386, 2), dtype='float32')
        for bad in (np.nan, np.inf, -np.inf):
            values[-1, -1] = bad
            with self.assertRaises(AssertionError):
                refit.scan_numeric(values, values.shape)
            with self.assertRaises(AssertionError):
                refit.checked_inputs(np.zeros((len(values), 1)), np.zeros((len(values), 1, 2)),
                                     values, np.array([len(values) - 1]), torch.device('cpu'))

    def protocols(self):
        protocol = {'model': 'GatedCrossCTR', 'numeric_fields': [f'f{i}' for i in range(166)],
            'options': {'numeric_dim': 166}, 'seed': 42, 'learning_rate': .001, 'batch_size': 1024,
            'weight_decay': .0001, 'epochs': 3, 'statistics': 'oof', 'numeric_scale': 1.,
            'numeric_width': 166, 'recent_days': 0, 'slow_id_learning_rate_factor': 1.}
        for day in (15, 16):
            folder = self.root / f'results/gated166_day{day}'
            folder.mkdir(parents=True)
            source = 'class GatedCrossCTR:\n    pass\n'
            if day == 16:
                source += '\nclass Unrelated:\n    pass\n__all__ = ["GatedCrossCTR", "Unrelated"]\n'
            else:
                source += '__all__ = ["GatedCrossCTR"]\n'
            (folder / 'model_snapshot.py').write_text(source, encoding='utf-8')
            refit.write(folder / 'protocol.json', dict(protocol, model_source_sha256=refit.digest(folder / 'model_snapshot.py')))
            refit.write(folder / 'metrics.json', {'status': 'complete_and_verified', 'auc': .7, 'best_fractional_epoch': 2.})
        return self.root / 'results/gated166_day16'

    def test_config_snapshot_hash_and_gate_definition(self):
        folder = self.protocols()
        with patch.object(refit, 'HERE', self.root):
            self.assertEqual(refit.protocol_for(166)[1], {'numeric_dim': 166})
            original = refit.read(folder / 'protocol.json')
            for key in ('seed', 'learning_rate', 'batch_size', 'weight_decay'):
                changed = dict(original)
                changed[key] *= 2
                refit.write(folder / 'protocol.json', changed)
                with self.assertRaises(AssertionError):
                    refit.protocol_for(166)
            refit.write(folder / 'protocol.json', original)
            (folder / 'model_snapshot.py').write_text('class GatedCrossCTR:\n    changed = True\n', encoding='utf-8')
            with self.assertRaises(AssertionError):
                refit.protocol_for(166)
            original['model_source_sha256'] = refit.digest(folder / 'model_snapshot.py')
            refit.write(folder / 'protocol.json', original)
            with self.assertRaises(AssertionError):
                refit.protocol_for(166)

    def test_replay_rejects_nan_shape_range_and_drift(self):
        p = np.array([.2, .8], dtype='float32')
        self.assertEqual(refit.verify_replay(p, p.copy(), 2), 0.)
        for bad in (np.array([np.nan, .8]), np.array([np.inf, .8]), np.array([[.2, .8]]),
                    np.array([.2]), np.array([.2, 1.1]), np.array([.2, .81])):
            with self.assertRaises(AssertionError):
                refit.verify_replay(p, bad, 2)
            with self.assertRaises(AssertionError):
                refit.verify_replay(bad, p, 2)

    def test_checkpoint_reconstruction_uses_saved_manifest(self):
        source = self.root / 'model_snapshot.py'
        source.write_text('import torch\nclass GatedCrossCTR(torch.nn.Module):\n'
            '    def __init__(self, cats, seqs, groups, vocab, initial_ctr, numeric_dim):\n'
            '        super().__init__()\n        self.received = (cats, seqs, groups, vocab, initial_ctr, numeric_dim)\n', encoding='utf-8')
        scalers = {'oof_feature': {'mean': 2., 'std': 3.}}
        saved = {'manifest': {'categorical_fields': ['id'], 'sequence_fields': [], 'group_for_field': {'id': 'id'},
            'vocabulary_sizes': {'id': 5}, 'numeric_fields': ['oof_feature'],
            'numeric_scaler': scalers, 'numeric_clip': [-8, 8]}, 'numeric_fields': ['oof_feature'],
            'preprocessing_manifest': {'numeric_fields': ['oof_feature'], 'scalers': scalers, 'clip': [-8, 8]},
            'options': {'numeric_dim': 1}, 'initial_ctr': .3, 'model_source_sha256': refit.digest(source)}
        model = refit.build_model(saved, source, torch.device('cpu'))
        self.assertEqual(model.received, (['id'], [], {'id': 'id'}, {'id': 5}, .3, 1))
        bad = copy.deepcopy(saved)
        bad['manifest']['numeric_fields'] = ['stale_past_feature']
        with self.assertRaises(AssertionError):
            refit.build_model(bad, source, torch.device('cpu'))
        bad = copy.deepcopy(saved)
        bad['manifest']['numeric_scaler']['oof_feature']['std'] = 4.
        # Break the saved metadata copies independently, as a corrupt checkpoint would.
        bad['preprocessing_manifest'] = copy.deepcopy(saved['preprocessing_manifest'])
        with self.assertRaises(AssertionError):
            refit.build_model(bad, source, torch.device('cpu'))

    def test_core_uses_first75_of_166_source_columns(self):
        previous = self.root / 'previous'
        (previous / 'full_best/oof').mkdir(parents=True)
        out = self.root / 'out'
        names = [f'f{i}' for i in range(166)]
        base = np.arange(5 * 166, dtype='float32').reshape(5, 166)
        # Unconsumed columns do not participate in core preparation.
        base[:, 75:] = np.nan
        np.save(previous / 'full_best/oof/numeric.npy', base)
        exposure = np.arange(5 * 43, dtype='float32').reshape(5, 43)
        snapshot = np.arange(5 * 48, dtype='float32').reshape(5, 48)
        np.save(self.cache / 'exposure_raw.npy', exposure)
        np.save(self.cache / 'snapshot_raw.npy', snapshot)
        refit.write(self.cache / 'manifest.json', self.meta)
        with patch.object(refit, 'ROOT', self.root), patch.object(refit, 'CACHE', self.cache), \
                patch.object(refit, 'PREVIOUS', previous), patch.object(refit, 'OUT', out), \
                patch.object(refit, 'protocol_for', return_value=(names, None, None, None)):
            refit.prepare_core()
        path = out / 'gated166/numeric.npy'
        actual = np.load(path)
        expected_raw = np.concatenate([base[:, :75], exposure, snapshot], axis=1)
        mean = expected_raw[:3].astype('float64').mean(axis=0)
        std = np.maximum(expected_raw[:3].astype('float64').std(axis=0), 1e-5)
        np.testing.assert_array_equal(actual, np.clip((expected_raw - mean) / std, -8, 8).astype('float32'))
        info = refit.read(out / 'gated166/input_manifest.json')
        self.assertEqual(info['numeric_fields'], names)
        self.assertEqual(list(info['scalers']), names)
        self.assertEqual(info['numeric_sha256']['oof'], refit.digest(path))
        self.assertEqual(info['row_identity_sha256'], self.identities)
        self.assertTrue(info['raw_sources_finite_all_rows'])
        self.assertEqual(base.shape, (5, 166))

    def test_core_prescan_rejects_nonfinite_before_output_creation(self):
        previous = self.root / 'previous'
        (previous / 'full_best/oof').mkdir(parents=True)
        out = self.root / 'out'
        np.save(previous / 'full_best/oof/numeric.npy', np.zeros((5, 75), dtype='float32'))
        np.save(self.cache / 'exposure_raw.npy', np.zeros((5, 43), dtype='float32'))
        bad = np.zeros((5, 48), dtype='float32')
        bad[-1, -1] = np.inf
        np.save(self.cache / 'snapshot_raw.npy', bad)
        refit.write(self.cache / 'manifest.json', self.meta)
        with patch.object(refit, 'ROOT', self.root), patch.object(refit, 'CACHE', self.cache), \
                patch.object(refit, 'PREVIOUS', previous), patch.object(refit, 'OUT', out), \
                patch.object(refit, 'protocol_for', return_value=([f'f{i}' for i in range(166)], None, None, None)):
            with self.assertRaises(AssertionError):
                refit.prepare_core()
        self.assertFalse((out / 'gated166').exists())


if __name__ == '__main__':
    unittest.main()
