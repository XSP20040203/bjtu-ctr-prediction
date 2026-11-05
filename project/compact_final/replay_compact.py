"""Reload all ten retained models using only the independent compact final bundle."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for variable in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(variable, '3')
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import torch


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(bundle, device_name, component_tolerance):
    started = time.perf_counter()
    info = read(bundle / 'manifest.json')
    for name in ['verification.json', 'replay_verification.json']:
        (bundle / name).write_text(json.dumps({'status': 'checking', 'final_labels_read': False}), encoding='utf-8')
    if info['status'] != 'complete' or info['rows'] != 404182 or info['final_labels_read'] is not False:
        raise AssertionError('Compact bundle not complete')
    for relative, record in info['files'].items():
        path = bundle / relative
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise AssertionError('Compact file changed: ' + relative)
    print('COMPACT_HASHES_VERIFIED', len(info['files']), flush=True)
    sys.path.insert(0, str(bundle / 'source'))
    sys.path.insert(0, str(bundle / 'vendor'))
    from bounded_id_models import BoundedIDDCN
    from modern_architectures import ClippedTabMFieldModel, member_probability
    from allocated_set_model import AllocatedSetContextCTR
    from catboost import CatBoostClassifier, Pool
    import lightgbm as lgb
    torch.set_num_threads(3)
    torch.manual_seed(42)
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)
    arrays = {name: np.load(bundle / 'arrays' / (name + '.npy'), mmap_mode='r', allow_pickle=False)
              for name in ['categories', 'sequences', 'base166', 'multiscale_extra159',
                           'oof_stats36', 'tabm535', 'gate_oof500', 'past500', 'ids', 'context_count']}
    n = info['rows']
    if any(len(values) != n for values in arrays.values()):
        raise AssertionError('Final array lengths differ')
    if arrays['categories'].shape != (n, 27) or arrays['sequences'].shape != (n, 12, 5):
        raise AssertionError('Final encoded schema differs')
    if arrays['ids'].shape != (n, 3) or not np.all(arrays['ids'][:, 2] == 20191117):
        raise AssertionError('Final IDs or dates differ')
    meta = read(bundle / 'metadata/cache_manifest.json')
    output = bundle / 'replayed'
    output.mkdir(exist_ok=True)
    predictions, errors = {}, {}
    def commit(name, prediction):
        prediction = np.asarray(prediction)
        if prediction.shape != (n,) or not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 1)):
            raise AssertionError('Invalid prediction: ' + name)
        reference = np.load(bundle / 'reference' / (name + '.npy'), allow_pickle=False)
        error = float(np.max(np.abs(prediction.astype('float64') - reference.astype('float64'))))
        tolerance = 1e-12 if name in ['tree', 'lgb', 'shared_lgb'] else component_tolerance
        if error > tolerance:
            raise AssertionError(f'{name} compact replay error {error} > {tolerance}')
        predictions[name] = prediction.astype('float64')
        errors[name] = error
        np.save(output / (name + '.npy'), prediction)
        print('REPLAY_VERIFIED', name, 'maximum_error', error, flush=True)
    def numeric(name, rows):
        if name == 'dcn':
            return np.array(arrays['base166'][rows])
        if name == 'multiscale_dcn':
            return np.concatenate([arrays['base166'][rows], arrays['multiscale_extra159'][rows]], axis=1)
        if name == 'oof':
            return np.concatenate([arrays['base166'][rows, :39], arrays['oof_stats36'][rows],
                                   arrays['base166'][rows, 75:]], axis=1)
        if name == 'tabm':
            return np.array(arrays['tabm535'][rows])
        if name == 'gated166':
            return np.array(arrays['gate_oof500'][rows, :166])
        if name == 'gated500':
            return np.array(arrays['gate_oof500'][rows])
        if name == 'allocated':
            return np.array(arrays['past500'][rows])
        raise KeyError(name)
    def tensors(name, rows):
        return (torch.as_tensor(arrays['categories'][rows].astype('int64'), device=device),
                torch.as_tensor(arrays['sequences'][rows].astype('int64'), device=device),
                torch.as_tensor(numeric(name, rows), device=device))
    for name in ['dcn', 'multiscale_dcn', 'oof', 'tabm', 'gated166', 'gated500', 'allocated']:
        checkpoint = torch.load(bundle / info['models'][name], map_location='cpu', weights_only=False)
        schema = checkpoint['manifest']
        options = checkpoint['options']
        prior = checkpoint.get('initial_ctr', schema.get('train_clicks', 21354) / schema.get('train_rows', 2526320))
        args = [schema['categorical_fields'], schema['sequence_fields'], schema['group_for_field'], schema['vocabulary_sizes']]
        if name.startswith('gated'):
            module = load_module(bundle / 'source' / (name + '_snapshot.py'), 'compact_' + name)
            constructor = module.GatedCrossCTR
        elif name == 'allocated':
            constructor = AllocatedSetContextCTR
        elif name == 'tabm':
            constructor = ClippedTabMFieldModel
        else:
            constructor = BoundedIDDCN
        model = constructor(*args, initial_ctr=prior, **options).to(device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        prediction = np.empty(n, dtype='float32')
        print('REPLAY_START', name, str(device), flush=True)
        with torch.no_grad():
            if name == 'allocated':
                # Exact original grouping and packing: first occurrence group order,
                # stable row order inside each complete raw user/day group.
                keys = pd.MultiIndex.from_arrays([arrays['ids'][:, 2], arrays['ids'][:, 1]])
                codes, unique = pd.factorize(keys, sort=False)
                order = np.argsort(codes, kind='stable')
                counts = np.bincount(codes)
                edges = np.r_[0, np.cumsum(counts)]
                groups = [order[edges[g]:edges[g + 1]] for g in range(len(counts))]
                def packed():
                    batch, size = [], 0
                    for group in groups:
                        if batch and (size + len(group) > 4096 or len(batch) >= 64):
                            yield batch
                            batch, size = [], 0
                        batch.append(group)
                        size += len(group)
                    if batch:
                        yield batch
                seen = np.zeros(n, dtype=bool)
                for batch in packed():
                    rows = np.concatenate(batch)
                    group_codes = np.repeat(np.arange(len(batch)), [len(group) for group in batch])
                    logits, _ = model(*tensors(name, rows), torch.as_tensor(group_codes, dtype=torch.long, device=device))
                    prediction[rows] = torch.sigmoid(logits).cpu().numpy()
                    if seen[rows].any():
                        raise AssertionError('Duplicate allocated rows')
                    seen[rows] = True
                if not seen.all():
                    raise AssertionError('Missing allocated rows')
            else:
                batchsize = 1024 if name in ['tabm', 'gated166', 'gated500'] else 4096
                for start in range(0, n, batchsize):
                    stop = min(start + batchsize, n)
                    logits = model(*tensors(name, slice(start, stop)))
                    probability = member_probability(logits) if name == 'tabm' else torch.sigmoid(logits)
                    prediction[start:stop] = probability.cpu().numpy()
        commit(name, prediction)
        del model, checkpoint, prediction
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    lgb_protocol = read(bundle / 'metadata/lgb_protocol.json')
    shared_protocol = read(bundle / 'metadata/shared_lgb_protocol.json')
    def tree_frame(start, stop, shared=False, catboost=False):
        cats = arrays['categories'][start:stop]
        raw = arrays['tabm535'][start:stop]
        if shared:
            selected = [i for i, field in enumerate(meta['categorical_fields']) if field not in ['user_id', 'task_id', 'adv_id']]
            matrix = np.concatenate([cats[:, selected], raw[:, :325]], axis=1).astype('float32', order='F')
            return pd.DataFrame(matrix, columns=shared_protocol['features'], copy=False)
        if catboost:
            columns = {}
            for i, field in enumerate(meta['categorical_fields']):
                columns[field] = pd.Categorical(cats[:, i])
            for i, field in enumerate(meta['numeric_fields'][:75]):
                columns[field] = raw[:, i]
            for i, field in enumerate(meta['sequence_fields']):
                for token in range(5):
                    columns[field + '_token' + str(token)] = pd.Categorical(arrays['sequences'][start:stop, i, token])
            for i, field in enumerate(meta['numeric_fields'][75:166]):
                columns[field] = raw[:, 75 + i]
            return pd.DataFrame(columns)
        matrix = np.concatenate([cats, raw[:, :75], arrays['sequences'][start:stop].reshape(stop - start, -1),
                                 raw[:, 75:166]], axis=1).astype('float32', order='F')
        return pd.DataFrame(matrix, columns=lgb_protocol['features'], copy=False)
    for name in ['tree', 'lgb', 'shared_lgb']:
        print('REPLAY_START', name, 'cpu', flush=True)
        if name == 'tree':
            model = CatBoostClassifier()
            model.load_model(str(bundle / info['models'][name]))
        else:
            model = lgb.Booster(model_str=(bundle / info['models'][name]).read_text(encoding='utf-8'))
        prediction = np.empty(n, dtype='float64')
        for start in range(0, n, 25000):
            stop = min(start + 25000, n)
            frame = tree_frame(start, stop, shared=name == 'shared_lgb', catboost=name == 'tree')
            if name == 'tree':
                if frame.columns.tolist() != model.feature_names_:
                    raise AssertionError('CatBoost feature order differs')
                prediction[start:stop] = model.predict_proba(Pool(frame, cat_features=lgb_protocol['categorical_fields']), thread_count=3)[:, 1]
            else:
                protocol = shared_protocol if name == 'shared_lgb' else lgb_protocol
                if model.feature_name() != protocol['features']:
                    raise AssertionError('LightGBM feature order differs')
                prediction[start:stop] = model.predict(frame, num_iteration=protocol['rounds'], num_threads=3)
        commit(name, prediction)
        del model, frame
        gc.collect()
    p = predictions
    # Preserve the exact operation order from the original export scripts.
    v3 = .3 * p['dcn'] + .3 * p['tree'] + .15 * p['lgb'] + .25 * p['multiscale_dcn']
    v4 = .8 * v3 + .2 * p['shared_lgb']
    v6_base = .9 * (.75 * v4 + .25 * p['oof']) + .1 * p['tabm']
    clipped = np.clip(v6_base, 1e-7, 1 - 1e-7)
    odds = clipped / (1 - clipped) * arrays['context_count'].astype('float32') ** .1
    v6 = odds / (1 + odds)
    v7 = .7 * (.9 * (.8 * v6 + .2 * p['gated500']) + .1 * p['allocated']) + .3 * p['gated166']
    ensembles = {'v4': v4, 'v6': v6, 'v7': v7}
    ensemble_errors = {}
    for name, prediction in ensembles.items():
        reference = np.load(bundle / 'reference' / (name + '.npy'))
        error = float(np.max(np.abs(prediction - reference)))
        if error > component_tolerance:
            raise AssertionError(f'{name} ensemble replay differs: {error}')
        ensemble_errors[name] = error
        np.save(output / (name + '.npy'), prediction)
        print('ENSEMBLE_VERIFIED', name, 'maximum_error', error, flush=True)
    pd.DataFrame({'log_id': arrays['ids'][:, 0], 'pctr': v7}).to_csv(output / 'result.csv', index=False, float_format='%.6f')
    pd.DataFrame({'id': np.arange(n), 'pctr': v7}).to_csv(output / 'submission.csv', index=False, float_format='%.6f')
    original_csv_errors = {}
    for filename, identity, expected in [('result.csv', 'log_id', arrays['ids'][:, 0]),
                                         ('submission.csv', 'id', np.arange(n))]:
        original = pd.read_csv(bundle / 'reference' / ('v7_' + filename), float_precision='round_trip')
        actual = pd.read_csv(output / filename, float_precision='round_trip')
        np.testing.assert_array_equal(original[identity].to_numpy(), expected)
        np.testing.assert_array_equal(actual[identity].to_numpy(), expected)
        error = float(np.max(np.abs(original.pctr.to_numpy() - v7)))
        if error > 5.000001e-7 + ensemble_errors['v7']:
            raise AssertionError('Original CSV probability disagreement')
        if float(np.max(np.abs(actual.pctr.to_numpy() - v7))) > 5.000001e-7:
            raise AssertionError('Replay CSV rounding differs')
        original_csv_errors[filename] = error
    report = {'status': 'passed', 'rows': n, 'model_count': 10, 'device': str(device),
              'all_bundle_hashes_verified': True, 'all_models_reloaded': True,
              'all_final_rows_replayed': True, 'training_arrays_used': False,
              'final_labels_read': False, 'final_auc': None,
              'maximum_component_errors': errors, 'maximum_ensemble_errors': ensemble_errors,
              'component_absolute_tolerance': component_tolerance,
              'original_v7_csv_vs_model_replay_error': original_csv_errors,
              'raw_user_day_complete_groups_preserved': True,
              'seconds': time.perf_counter() - started}
    report_text = json.dumps(report, ensure_ascii=False, indent=2)
    (bundle / 'verification.json').write_text(report_text, encoding='utf-8')
    (bundle / 'replay_verification.json').write_text(report_text, encoding='utf-8')
    print('COMPACT_FULL_REPLAY_PASSED', json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--component-tolerance', type=float, default=2e-6)
    arguments = parser.parse_args()
    main(arguments.bundle.resolve(), arguments.device, arguments.component_tolerance)
