"""Fixed-budget full refits for selected core166 and wide500 gated CTR components.

No final labels are loaded. Budgets are mean selected epochs from development.
Core preparation recreates source blocks before train-only scaling and clipping.
"""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

for variable in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(variable, '3')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / 'analysis/daily_interest_din/full_refit_cache'
PREVIOUS = ROOT / 'analysis/nextday_goal_20261029'
OUT = HERE / 'full_refit'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'analysis/model_comparison'))
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
import torch
from torch import nn

IDENTITY_COLUMNS = ['log_id', 'user_id', 'pt_d']
IDENTITY_RULE = 'SHA256 of contiguous little-endian signed int64 (<i8), no header, original train.csv then final_test.csv rows'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def gated_definition(path):
    # Compare the gate and its module dependencies; unrelated appended classes are harmless.
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    definitions = {node.name: node for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    selected, pending = set(), ['GatedCrossCTR']
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        selected.add(name)
        pending.extend(node.id for node in ast.walk(definitions[name])
                       if isinstance(node, ast.Name) and node.id in definitions and node.id not in selected)
    nodes = [node for node in tree.body
             if not (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == '__all__'
                                                         for target in node.targets))
             and (not isinstance(node, (ast.ClassDef, ast.FunctionDef)) or node.name in selected)]
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)


def protocol_for(width):
    if width not in (166, 500):
        raise AssertionError('Unsupported gate width')
    prefix = 'gated166' if width == 166 else 'gated_oof'
    evidence = []
    fields, options, expected = None, None, None
    config_fields = ['model', 'numeric_fields', 'options', 'seed', 'learning_rate',
                     'batch_size', 'weight_decay', 'epochs', 'statistics', 'numeric_scale',
                     'numeric_width', 'recent_days', 'slow_id_learning_rate_factor']
    for day in [15, 16]:
        folder = HERE / f'results/{prefix}_day{day}'
        protocol = read(folder / 'protocol.json')
        metrics = read(folder / 'metrics.json')
        # Older wide protocols omit these runner defaults.
        for key, value in [('numeric_scale', 1.), ('numeric_width', width),
                           ('slow_id_learning_rate_factor', 1.)]:
            protocol.setdefault(key, value)
        if metrics['status'] != 'complete_and_verified' or protocol['model'] != 'GatedCrossCTR':
            raise AssertionError('Selected development component not verified')
        if protocol['numeric_scale'] != 1. or protocol['statistics'] != 'oof':
            raise AssertionError('Unexpected selected numeric protocol')
        if digest(folder / 'model_snapshot.py') != protocol['model_source_sha256']:
            raise AssertionError('Selected model snapshot hash differs')
        if fields is None:
            fields, options, expected = protocol['numeric_fields'], protocol['options'], protocol
        elif any(expected[key] != protocol[key] for key in config_fields):
            raise AssertionError('Selected configurations differ between dates')
        elif gated_definition(selected_snapshot(width)) != gated_definition(folder / 'model_snapshot.py'):
            raise AssertionError('Selected gate definitions differ between dates')
        epoch = float(metrics['best_fractional_epoch'])
        if not math.isfinite(epoch) or epoch <= 0:
            raise AssertionError('Invalid selected training budget')
        evidence.append({'day': 20191100 + day, 'auc': metrics['auc'],
                         'fractional_epochs': epoch,
                         'metrics_sha256': digest(folder / 'metrics.json'),
                         'model_source_sha256': protocol['model_source_sha256']})
    if len(fields) != width or len(set(fields)) != width or options != {'numeric_dim': width}:
        raise AssertionError('Unexpected selected gate options')
    return fields, options, evidence, expected


def selected_snapshot(width):
    prefix = 'gated166' if width == 166 else 'gated_oof'
    return HERE / f'results/{prefix}_day15/model_snapshot.py'


def scan_numeric(values, shape, clip=None):
    if values.shape != shape or values.dtype != np.dtype('float32'):
        raise AssertionError('Numeric shape/dtype differs')
    for start in range(0, len(values), 16384):
        block = values[start:start + 16384]
        if not np.isfinite(block).all():
            raise AssertionError('Nonfinite numeric source/input')
        if clip is not None and (np.any(block < clip[0]) or np.any(block > clip[1])):
            raise AssertionError('Numeric clip bounds differ')
    return {'finite_all_rows': True, 'clip_all_rows': clip is not None}


def raw_alignment(meta):
    ntrain, nfinal = meta['train_rows'], meta['final_rows']
    cache = {field: np.load(CACHE / f'{name}.npy', mmap_mode='r')
             for field, name in [('log_id', 'log_id'), ('user_id', 'user_id'), ('pt_d', 'date')]}
    if any(values.shape != (ntrain + nfinal,) for values in cache.values()):
        raise AssertionError('Identity cache shape differs')
    labels = np.load(CACHE / 'label.npy', mmap_mode='r')
    if labels.ndim != 1 or len(labels) < ntrain:
        raise AssertionError('Label cache shape differs')
    hashes = {field: hashlib.sha256() for field in IDENTITY_COLUMNS}
    offset = 0
    for filename, count, training in [('train.csv', ntrain, True), ('final_test.csv', nfinal, False)]:
        # The final allowlist deliberately excludes label, even if the CSV contains it.
        columns = IDENTITY_COLUMNS + (['label'] if training else [])
        seen = 0
        with pd.read_csv(ROOT / '2026ml' / filename, usecols=columns, dtype='int64', chunksize=65536) as chunks:
            for raw in chunks:
                stop = offset + len(raw)
                if seen + len(raw) > count:
                    raise AssertionError('Raw row count differs')
                for field in IDENTITY_COLUMNS:
                    values = raw[field].to_numpy()
                    np.testing.assert_array_equal(values, cache[field][offset:stop])
                    hashes[field].update(np.ascontiguousarray(values, dtype='<i8').tobytes())
                dates = raw['pt_d'].to_numpy()
                if training:
                    target = raw['label'].to_numpy()
                    if not np.isin(target, [0, 1]).all() or not np.all(dates < 20191117):
                        raise AssertionError('Invalid training label/date')
                    np.testing.assert_array_equal(target, labels[offset:stop])
                elif not np.all(dates == 20191117):
                    raise AssertionError('Invalid final date')
                seen += len(raw)
                offset = stop
        if seen != count:
            raise AssertionError('Raw row count differs')
    return {field: value.hexdigest() for field, value in hashes.items()}


def validate_input(path, info, names, meta, identities, width):
    ntrain, nfinal = meta['train_rows'], meta['final_rows']
    if (info.get('status') != 'complete_and_verified' or info.get('final_labels_read') is not False
            or info.get('numeric_fields') != names or info.get('train_rows') != ntrain
            or info.get('final_rows') != nfinal or info.get('clip') != [-8, 8]):
        raise AssertionError('Full input metadata differs')
    if info.get('identity_hash_rule') != IDENTITY_RULE or info.get('row_identity_sha256') != identities:
        raise AssertionError('Full input row identity differs')
    expected_hash = info.get('numeric_sha256', {}).get('oof')
    if not expected_hash or digest(path) != expected_hash:
        raise AssertionError('Full numeric hash differs')
    scalers = info.get('source_scalers_oof') if width == 500 else info.get('scalers')
    if not isinstance(scalers, dict) or set(scalers) != set(names):
        raise AssertionError('Full input scalers differ')
    for value in scalers.values():
        if not math.isfinite(value['mean']) or not math.isfinite(value['std']) or value['std'] < 1e-5:
            raise AssertionError('Invalid input scaler')
    if width == 500:
        if info.get('scaler_fit_rows') != ntrain or info.get('scaler_fit_indices') != [0, ntrain]:
            raise AssertionError('Wide scaler fit scope differs')
    elif info.get('scaler_fit_indices') != [0, ntrain] or info.get('raw_sources_finite_all_rows') is not True:
        raise AssertionError('Core scaler fit/source validation differs')
    numeric = np.load(path, mmap_mode='r')
    prescan = scan_numeric(numeric, (ntrain + nfinal, width), [-8, 8])
    return numeric, {'numeric_fields': names, 'statistics': 'oof', 'scalers': scalers,
        'clip': [-8, 8], 'std_floor': 1e-5, 'scaler_fit_indices': [0, ntrain],
        'numeric_sha256': expected_hash, 'prescan': prescan,
        'row_identity_sha256': identities, 'identity_hash_rule': IDENTITY_RULE,
        'input_manifest_sha256': digest(Path(path).parent / ('manifest.json' if width == 500 else 'input_manifest.json')),
        'final_labels_read': False}


def prepare_core():
    folder = OUT / 'gated166'
    if folder.exists():
        raise FileExistsError('Preserving existing core preparation')
    started = time.perf_counter()
    meta = read(CACHE / 'manifest.json')
    ntrain, nfinal = meta['train_rows'], meta['final_rows']
    names, _, _, _ = protocol_for(166)
    identities = raw_alignment(meta)
    paths = [PREVIOUS / 'full_best/oof/numeric.npy', CACHE / 'exposure_raw.npy', CACHE / 'snapshot_raw.npy']
    sources = [(np.load(path, mmap_mode='r'), fields, width, path)
               for path, fields, width in zip(paths, [names[:75], names[75:118], names[118:166]], [75, 43, 48])]
    for source, fields, width, path in sources:
        if source.ndim != 2 or source.shape[0] != ntrain + nfinal or source.shape[1] < width:
            raise AssertionError('Core source shape differs')
        scan_numeric(source[:, :width], (ntrain + nfinal, width))
    folder.mkdir(parents=True)
    numeric = open_memmap(folder / 'numeric.npy', mode='w+', dtype='float32', shape=(ntrain + nfinal, 166))
    scalers = {}
    offset = 0
    for source, fields, width, path in sources:
        mean, std = np.empty(width), np.empty(width)
        for col, name in enumerate(fields):
            values = np.asarray(source[:ntrain, col], dtype='float64')
            mean[col], std[col] = values.mean(), max(float(values.std()), 1e-5)
            if not np.isfinite([mean[col], std[col]]).all():
                raise AssertionError('Nonfinite core scaler')
            scalers[name] = {'mean': float(mean[col]), 'std': float(std[col])}
        for start in range(0, len(numeric), 16384):
            stop = min(start + 16384, len(numeric))
            values = np.asarray(source[start:stop, :width], dtype='float32')
            numeric[start:stop, offset:offset + width] = np.clip((values - mean) / std, -8, 8)
        offset += width
        numeric.flush()
    if offset != 166:
        raise AssertionError('Core width mismatch')
    prescan = scan_numeric(numeric, (ntrain + nfinal, 166), [-8, 8])
    write(folder / 'input_manifest.json', {'status': 'complete_and_verified', 'numeric_fields': names,
        'train_rows': ntrain, 'final_rows': nfinal, 'scalers': scalers, 'clip': [-8, 8],
        'scaler_fit_indices': [0, ntrain], 'raw_sources_finite_all_rows': True, 'prescan': prescan,
        'numeric_sha256': {'oof': digest(folder / 'numeric.npy')},
        'raw_source_sha256': {str(path): digest(path) for path in paths},
        'row_identity_sha256': identities, 'identity_hash_rule': IDENTITY_RULE,
        'fit_scope': 'Full labeled training rows only', 'final_labels_read': False,
        'seconds': time.perf_counter() - started})
    print('CORE_PREPARED', ntrain, nfinal, flush=True)


def build_model(saved, snapshot, device):
    if digest(snapshot) != saved['model_source_sha256']:
        raise AssertionError('Inference model snapshot hash differs')
    manifest, options = saved['manifest'], saved['options']
    preprocessing = saved['preprocessing_manifest']
    if (saved['numeric_fields'] != manifest['numeric_fields']
            or saved['numeric_fields'] != preprocessing['numeric_fields']
            or options != {'numeric_dim': len(saved['numeric_fields'])}
            or manifest['numeric_scaler'] != preprocessing['scalers']
            or manifest['numeric_clip'] != preprocessing['clip']):
        raise AssertionError('Checkpoint numeric metadata differs')
    spec = importlib.util.spec_from_file_location('selected_gated_snapshot', snapshot)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GatedCrossCTR(manifest['categorical_fields'], manifest['sequence_fields'],
        manifest['group_for_field'], manifest['vocabulary_sizes'],
        initial_ctr=saved['initial_ctr'], **options).to(device)


def checked_inputs(categories, sequences, numeric, rows, device):
    values = np.array(numeric[rows])
    if not np.isfinite(values).all():
        raise AssertionError('Nonfinite model inference input')
    return (torch.as_tensor(categories[rows].astype('int64'), device=device),
            torch.as_tensor(sequences[rows].astype('int64'), device=device),
            torch.as_tensor(values, device=device))


def verify_replay(prediction, replay, count):
    for values in (prediction, replay):
        if values.shape != (count,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise AssertionError('Full gate probability/reload mismatch')
    error = float(np.max(np.abs(prediction - replay)))
    if not math.isfinite(error) or error > 2e-7:
        raise AssertionError('Full gate probability/reload mismatch')
    return error


def train(width):
    folder = OUT / ('gated166' if width == 166 else 'gated500')
    if (folder / 'model.pt').exists():
        raise FileExistsError('Preserving trained full gate')
    if width == 166:
        if not (folder / 'input_manifest.json').exists():
            prepare_core()
        numeric_path = folder / 'numeric.npy'
        input_meta = read(folder / 'input_manifest.json')
    else:
        if folder.exists():
            raise FileExistsError('Preserving wide gated experiment')
        input_meta = read(OUT / 'features/manifest.json')
        numeric_path = OUT / 'features/numeric_oof500.npy'
    names, options, evidence, selected = protocol_for(width)
    meta = read(CACHE / 'manifest.json')
    ntrain, nfinal = meta['train_rows'], meta['final_rows']
    identities = raw_alignment(meta)
    numeric, preprocessing = validate_input(numeric_path, input_meta, names, meta, identities, width)
    categories = np.load(CACHE / 'categories.npy', mmap_mode='r')
    sequences = np.load(CACHE / 'sequences.npy', mmap_mode='r')
    labels = np.array(np.load(CACHE / 'label.npy', mmap_mode='r')[:ntrain])
    if len(categories) != ntrain + nfinal or len(sequences) != ntrain + nfinal:
        raise AssertionError('Full categorical/sequence row count differs')
    torch.set_num_threads(3)
    seed = selected['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Full gate expects available GPU')
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda')
    folder.mkdir(parents=True, exist_ok=True)
    manifest = dict(meta, numeric_fields=names, numeric_scaler=preprocessing['scalers'],
                    numeric_clip=preprocessing['clip'])
    saved = {'manifest': manifest, 'options': options, 'numeric_fields': names,
        'preprocessing_manifest': preprocessing, 'initial_ctr': float(labels.mean()),
        'model_source_sha256': selected['model_source_sha256']}
    snapshot = selected_snapshot(width)
    model = build_model(saved, snapshot, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=selected['learning_rate'], weight_decay=selected['weight_decay'])
    batchsize = selected['batch_size']
    epochs = float(np.mean([row['fractional_epochs'] for row in evidence]))
    batches = math.ceil(ntrain / batchsize)
    target_steps = int(round(epochs * batches))
    protocol = {'status': 'fixed_before_training', 'model': 'GatedCrossCTR', 'options': options,
        'train_rows': ntrain, 'final_rows': nfinal, 'numeric_path': str(numeric_path.resolve()),
        'numeric_fields': names, 'seed': seed, 'batch_size': batchsize,
        'learning_rate': selected['learning_rate'], 'weight_decay': selected['weight_decay'],
        'fixed_fractional_epochs': epochs, 'fixed_steps': target_steps, 'budget_source': evidence,
        'final_labels_read': False, 'final_auc': None, 'validation_early_stopping': False,
        'preprocessing_manifest': preprocessing,
        'source_sha256': digest(Path(__file__)), 'model_source_sha256': digest(snapshot)}
    write(folder / 'protocol.json', protocol)
    (folder / 'trainer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (folder / 'model_snapshot.py').write_bytes(snapshot.read_bytes())
    def inputs(rows):
        return checked_inputs(categories, sequences, numeric, rows, device)
    started = time.perf_counter()
    steps, seen, total = 0, 0, 0.
    print('FULL_GATE_START', width, target_steps, flush=True)
    model.train()
    for epoch in range(1, math.ceil(epochs) + 1):
        order = np.random.default_rng(seed + epoch).permutation(ntrain)
        for start in range(0, ntrain, batchsize):
            if steps >= target_steps:
                break
            rows = order[start:start + batchsize]
            target = torch.as_tensor(labels[rows].astype('float32'), device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(*inputs(rows)), target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite full gate loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            steps += 1
            seen += len(rows)
            total += float(loss.detach()) * len(rows)
            if steps % 400 == 0:
                print('FULL_GATE_PROGRESS', width, steps, target_steps, total / seen, flush=True)
    if steps != target_steps:
        raise AssertionError('Training budget differs')
    saved.update(state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, steps=steps)
    torch.save(saved, folder / 'model.pt')
    def predict(instance):
        instance.eval()
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(instance(*inputs(np.arange(start, min(start + batchsize, ntrain + nfinal))))).cpu().numpy()
                for start in range(ntrain, ntrain + nfinal, batchsize)])
    p = predict(model)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    loaded = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
    for key in ['manifest', 'options', 'numeric_fields', 'preprocessing_manifest', 'initial_ctr', 'model_source_sha256', 'steps']:
        if loaded[key] != saved[key]:
            raise AssertionError('Reloaded checkpoint metadata differs')
    restored = build_model(loaded, folder / 'model_snapshot.py', device)
    restored.load_state_dict(loaded['state_dict'], strict=True)
    replay = predict(restored)
    error = verify_replay(p, replay, nfinal)
    np.save(folder / 'final_prediction.npy', p)
    np.testing.assert_array_equal(np.load(folder / 'final_prediction.npy', allow_pickle=False), p)
    write(folder / 'verification.json', {'status': 'complete_and_verified', 'model': 'GatedCrossCTR',
        'numeric_width': width, 'train_rows': ntrain, 'final_rows': nfinal, 'fixed_fractional_epochs': epochs,
        'steps': steps, 'maximum_reload_error': error, 'final_labels_read': False, 'final_auc': None,
        'model_sha256': digest(folder / 'model.pt'), 'prediction_sha256': digest(folder / 'final_prediction.npy'),
        'preprocessing_manifest': preprocessing, 'mean_probability': float(p.mean()),
        'minimum_probability': float(p.min()), 'maximum_probability': float(p.max()),
        'seconds': time.perf_counter() - started})
    print('FULL_GATE_COMPLETE', width, error, float(p.mean()), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, choices=[166, 500], required=True)
    args = parser.parse_args()
    train(args.width)
