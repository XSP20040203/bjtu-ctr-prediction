"""Audit original delivered v7 inputs without changing trained models or predictions.

Legacy gate checkpoints store a base cache manifest. This report binds their
actual preprocessing to the separately saved training protocol and source maps.
No final label column is opened; only train.csv labels are compared.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FULL = HERE / 'full_refit'
CACHE = ROOT / 'analysis/daily_interest_din/full_refit_cache'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'analysis/model_comparison'))
import numpy as np
import pandas as pd
import torch


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    started = time.perf_counter()
    meta = read(CACHE / 'manifest.json')
    features = read(FULL / 'features/manifest.json')
    ntrain, nfinal = meta['train_rows'], meta['final_rows']
    train = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['label', 'user_id', 'pt_d', 'log_id'], dtype='int64')
    final = pd.read_csv(ROOT / '2026ml/final_test.csv', usecols=['user_id', 'pt_d', 'log_id'], dtype='int64')
    assert len(train) == ntrain and len(final) == nfinal and 'label' not in final
    raw = pd.concat([train[['user_id', 'pt_d', 'log_id']], final], ignore_index=True)
    identity = {}
    for field, cached in [('user_id', 'user_id'), ('pt_d', 'date'), ('log_id', 'log_id')]:
        values = raw[field].to_numpy()
        np.testing.assert_array_equal(values, np.load(CACHE / f'{cached}.npy', mmap_mode='r'))
        identity[field] = hashlib.sha256(np.ascontiguousarray(values, dtype='<i8').tobytes()).hexdigest()
    assert identity == features['row_identity_sha256']
    labels = np.load(CACHE / 'label.npy', mmap_mode='r')
    np.testing.assert_array_equal(train.label, labels[:ntrain])
    assert np.isin(labels[:ntrain], [0, 1]).all()
    assert np.all(train.pt_d < 20191117) and np.all(final.pt_d == 20191117)
    del train, final, raw
    oof_path = FULL / 'features/numeric_oof500.npy'
    past_path = FULL / 'features/numeric_past500.npy'
    if digest(oof_path) != features['numeric_sha256']['oof'] or digest(past_path) != features['numeric_sha256']['past']:
        raise AssertionError('Feature files changed after original verification')
    oof = np.load(oof_path, mmap_mode='r')
    past = np.load(past_path, mmap_mode='r')
    core = np.load(FULL / 'gated166/numeric.npy', mmap_mode='r')
    for start in range(0, len(oof), 8192):
        stop = min(start + 8192, len(oof))
        for values in [oof[start:stop], past[start:stop], core[start:stop]]:
            if not np.isfinite(values).all() or np.any(np.abs(values) > 8):
                raise AssertionError('Existing numeric input contains nonfinite or out-of-clip values')
        np.testing.assert_array_equal(core[start:stop], oof[start:stop, :166])
    torch.set_num_threads(3)
    categories = np.load(CACHE / 'categories.npy', mmap_mode='r')
    sequences = np.load(CACHE / 'sequences.npy', mmap_mode='r')
    export = read(ROOT / 'output/best_nextday_arch_v7/manifest.json')
    results = {}
    for name, width, numeric in [('gated166', 166, core), ('gated500', 500, oof)]:
        folder = FULL / name
        protocol, verification = read(folder / 'protocol.json'), read(folder / 'verification.json')
        if digest(folder / 'model.pt') != verification['model_sha256']:
            raise AssertionError('Legacy checkpoint changed')
        if digest(folder / 'final_prediction.npy') != export['components'][name]['sha256']:
            raise AssertionError('Legacy gate prediction changed since delivery')
        if protocol['numeric_fields'] != features['numeric_fields'][:width]:
            raise AssertionError('Actual feature schema differs from original protocol')
        if digest(folder / 'trainer_snapshot.py') != protocol['source_sha256'] or digest(folder / 'model_snapshot.py') != protocol['model_source_sha256']:
            raise AssertionError('Original source snapshot hash mismatch')
        saved = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
        if saved['options'] != protocol['options'] or saved['options']['numeric_dim'] != width:
            raise AssertionError('Saved model options differ from training protocol')
        module_name = 'v7_original_' + name
        spec = importlib.util.spec_from_file_location(module_name, folder / 'model_snapshot.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        model = module.GatedCrossCTR(meta['categorical_fields'], meta['sequence_fields'], meta['group_for_field'],
                                    meta['vocabulary_sizes'], initial_ctr=float(labels[:ntrain].mean()), **saved['options'])
        model.load_state_dict(saved['state_dict'], strict=True)
        model.eval()
        selected = np.linspace(ntrain, ntrain + nfinal - 1, 1024, dtype='int64')
        with torch.no_grad():
            p = torch.sigmoid(model(torch.as_tensor(categories[selected].astype('int64')),
                                    torch.as_tensor(sequences[selected].astype('int64')),
                                    torch.as_tensor(np.array(numeric[selected])))).numpy()
        reference = np.load(folder / 'final_prediction.npy', mmap_mode='r')[selected - ntrain]
        error = float(np.max(np.abs(p - reference)))
        if not np.isfinite(error) or error > 2e-6:
            raise AssertionError('Original checkpoint reconstruction differs')
        results[name] = {'status': 'passed', 'numeric_columns': width,
            'input_path': str(protocol['numeric_path']), 'numeric_fields_match_training_protocol': True,
            'preprocessing_scalers': features['source_scalers_oof'] if width == 500 else read(folder / 'input_manifest.json')['scalers'],
            'numeric_sha256': digest(Path(protocol['numeric_path'])),
            'model_sha256': verification['model_sha256'],
            'prediction_sha256': export['components'][name]['sha256'],
            'checkpoint_manifest_note': 'Legacy manifest describes base cache only; actual numeric schema/scalers bound here and in original protocol',
            'replayed_final_rows': len(selected), 'maximum_cpu_saved_gpu_error': error}
        print('EXISTING_GATE_AUDIT', name, error, flush=True)
    record = {'status': 'passed', 'final_labels_read': False, 'delivered_predictions_modified': False,
        'raw_training_labels_and_all_row_identities_equal_cache': True, 'row_identity_sha256': identity,
        'all_numeric_values_finite_and_clipped': True, 'all_core166_values_equal_oof500_prefix': True,
        'full_numeric_hashes_match_original_feature_manifest': True, 'gate_artifacts': results,
        'submission_sha256': digest(ROOT / 'output/best_nextday_arch_v7/submission.csv'),
        'seconds': time.perf_counter() - started}
    (FULL / 'existing_v7_input_audit.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('EXISTING_V7_INPUT_AUDIT_PASSED', record['seconds'], flush=True)


if __name__ == '__main__':
    main()
