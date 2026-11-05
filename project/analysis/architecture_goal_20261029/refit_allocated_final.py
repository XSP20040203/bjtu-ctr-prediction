"""Fixed one-epoch, complete-user-day AllocatedSetContextCTR full-data refit.

Only --stage train opens real data or initializes CUDA. Numeric preparation is
owned by the independent full_refit/features builder; an incomplete/missing
manifest fails immediately. Final CSV columns are explicitly allowlisted without
label, and cache labels are sliced to the original training prefix only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '3')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / 'analysis/daily_interest_din/full_refit_cache'
FEATURES = HERE / 'full_refit/features'
OUT = HERE / 'full_refit/allocated'
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch
from torch import nn
from allocated_set_model import AllocatedSetContextCTR
from train_set_context import groups_for_rows, pack_groups

SEED = 42
MAX_ROWS = 4096
MAX_GROUPS = 64
TRAIN_ROWS = 2526320
FINAL_ROWS = 404182
STRICT_REPLAY_ATOL = 2e-7
CUDA_REPLAY_ATOL = 3e-7
IDENTITY_COLUMNS = ['user_id', 'pt_d', 'log_id']
REPLAY_POLICY = (
    'Fresh model, same backend and identical complete-group batches, every final row; '
    'CPU and CUDA strict absolute tolerance 2e-7. CUDA only: if strict comparison '
    'fails, a repeated inference on the restored instance must demonstrate nonzero '
    'reduction jitter, and all three pairwise errors must be <=3e-7. '
    'No CPU/GPU cross-backend comparison or relative tolerance.'
)


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


def source_hashes():
    return {str(path.resolve()): digest(path) for path in (
        Path(__file__), HERE / 'allocated_set_model.py', HERE / 'set_context_model.py',
        HERE / 'interaction_models.py', HERE / 'train_set_context.py',
        HERE / 'train_architecture.py')}


def validate_feature_manifest(info, cache_meta, directory=FEATURES):
    if info.get('status') != 'complete_and_verified':
        raise RuntimeError('Feature manifest is not complete_and_verified; wait for feature preparation')
    expected = {'rows': TRAIN_ROWS + FINAL_ROWS, 'train_rows': TRAIN_ROWS, 'final_rows': FINAL_ROWS}
    for key, value in expected.items():
        if info.get(key) != value or cache_meta.get(key) != value:
            raise AssertionError('Feature/cache row boundary differs: ' + key)
    if info.get('final_labels_read') is not False or cache_meta.get('final_labels_available') is not False:
        raise AssertionError('Feature/cache final-label isolation is not declared')
    fields = info.get('numeric_fields', [])
    if len(fields) != 500 or len(set(fields)) != 500 or info.get('clip') != [-8, 8]:
        raise AssertionError('Expected 500 unique standardized, clipped numeric features')
    expected_path = directory / 'numeric_past500.npy'
    if Path(info.get('numeric_past_path', '')).resolve() != expected_path.resolve():
        raise AssertionError('Past numeric feature path differs')
    scalers = info.get('source_scalers_past', info.get('scalers', {}).get('past', {}))
    if set(scalers) != set(fields):
        raise AssertionError('Training-only numeric scaler schema differs')
    for field in fields:
        mean, std = scalers[field]['mean'], scalers[field]['std']
        if not np.isfinite(mean) or not np.isfinite(std) or std < 1e-5:
            raise AssertionError('Invalid training-only scaler: ' + field)
    if info.get('scaler_fit_scope') != 'Original train.csv rows [0,2526320), never final rows':
        raise AssertionError('Numeric scaler must fit the full original training prefix only')
    if info.get('per_row_strictly_historical', {}).get('past') is not True:
        raise AssertionError('Allocated set model requires strictly earlier-date label statistics')
    if (info.get('shape') != [TRAIN_ROWS + FINAL_ROWS, 500] or
            info.get('dtype') != 'float32' or info.get('storage_order') != 'C' or
            info.get('scaler_fit_rows') != TRAIN_ROWS or info.get('scaler_fit_indices') != [0, TRAIN_ROWS]):
        raise AssertionError('Numeric metadata shape/storage/scaler boundary differs')
    identity_rule = 'SHA256 of contiguous little-endian signed int64 (<i8), no header, original train.csv then final_test.csv rows'
    if info.get('identity_hash_rule') != identity_rule or set(info.get('row_identity_sha256', {})) != set(IDENTITY_COLUMNS):
        raise AssertionError('Numeric feature identity hash schema differs')
    for value in [info.get('numeric_sha256', {}).get('past', ''), *info['row_identity_sha256'].values()]:
        if len(value) != 64 or any(character not in '0123456789abcdef' for character in value):
            raise AssertionError('Invalid feature identity or numeric SHA256')
    return fields


def development_contract(numeric_fields):
    records, options = {}, None
    for day in (15, 16):
        directory = HERE / 'results' / f'set_allocated_day{day}'
        protocol = read(directory / 'protocol.json')
        metrics = read(directory / 'metrics.json')
        saved = torch.load(directory / 'model.pt', map_location='cpu', weights_only=False)
        expected_options = {'numeric_dim': 500, 'use_set_context': True}
        if saved['options'] != expected_options or protocol['options'] != saved['options']:
            raise AssertionError('Development checkpoint model options differ')
        if options is not None and options != saved['options']:
            raise AssertionError('Two development checkpoint options disagree')
        options = dict(saved['options'])
        required = {'model': 'AllocatedSetContextCTR', 'statistics': 'past', 'seed': SEED,
                    'learning_rate': .0005, 'max_rows_per_batch': MAX_ROWS,
                    'max_groups_per_batch': MAX_GROUPS, 'auxiliary_count_weight': .01,
                    'consistency_weight': 0., 'row_only_control': False,
                    'probability_allocation': True, 'final_test_read': False}
        for key, value in required.items():
            if protocol.get(key) != value:
                raise AssertionError('Development protocol differs: ' + key)
        if (metrics.get('status') != 'complete_and_verified' or
                metrics.get('best_fractional_epoch') != 1. or
                saved['record'].get('fractional_epoch') != 1.):
            raise AssertionError('Both development checkpoints must select exactly one epoch')
        for current, snapshot in (('allocated_set_model.py', 'model_snapshot.py'),
                                  ('set_context_model.py', 'base_model_snapshot.py')):
            if digest(HERE / current) != digest(directory / snapshot):
                raise AssertionError('Development model source differs: ' + current)
        if protocol['model_sha256'] != digest(HERE / 'allocated_set_model.py'):
            raise AssertionError('Development allocated model hash differs')
        if digest(HERE / 'train_set_context.py') != digest(directory / 'trainer_snapshot.py'):
            raise AssertionError('Development group/loss trainer semantics changed')
        input_meta = read(HERE / 'cache' / f'day{day}_past' / 'manifest.json')
        if (input_meta['numeric_fields'] != numeric_fields or
                input_meta.get('per_row_strictly_historical') is not True):
            raise AssertionError('Full past numeric feature order differs from development')
        records[str(day)] = {'directory': str(directory.resolve()),
                            'checkpoint_sha256': digest(directory / 'model.pt'),
                            'protocol_sha256': digest(directory / 'protocol.json'),
                            'best_fractional_epoch': saved['record']['fractional_epoch'],
                            'options': options, 'metrics': metrics}
    return options, records


def read_raw_sources(data_dir, ntrain, nfinal):
    train = pd.read_csv(data_dir / 'train.csv', usecols=IDENTITY_COLUMNS + ['label'], dtype='int64')
    final = pd.read_csv(data_dir / 'final_test.csv', usecols=IDENTITY_COLUMNS, dtype='int64')
    if len(train) != ntrain or len(final) != nfinal or 'label' in final.columns:
        raise AssertionError('Raw source rows or final-label isolation differs')
    raw = {field: np.concatenate((train[field].to_numpy(), final[field].to_numpy()))
           for field in IDENTITY_COLUMNS}
    labels = train['label'].to_numpy(dtype='float32')
    if not np.isin(labels, [0., 1.]).all():
        raise AssertionError('Training labels must be binary')
    return raw, labels


def validate_groups(dates, users, rows, groups):
    if not groups or sum(map(len, groups)) != len(rows):
        raise AssertionError('Complete-group partition lost rows')
    np.testing.assert_array_equal(np.sort(np.concatenate(groups)), rows)
    for group in groups:
        if not len(group) or not np.all(dates[group] == dates[group[0]]) or not np.all(users[group] == users[group[0]]):
            raise AssertionError('A complete set mixes raw users or dates')


def batch_inputs(batch, arrays, numeric, device):
    rows = np.concatenate(batch)
    codes = np.repeat(np.arange(len(batch)), [len(group) for group in batch])
    values = np.array(numeric[rows], dtype='float32')
    if not np.isfinite(values).all() or np.any(np.abs(values) > 8.):
        raise AssertionError('Numeric features must be finite and clipped to [-8,8]')
    return rows, (torch.as_tensor(arrays['categories'][rows].astype('int64'), device=device),
                  torch.as_tensor(arrays['sequences'][rows].astype('int64'), device=device),
                  torch.as_tensor(values, device=device),
                  torch.as_tensor(codes, dtype=torch.long, device=device))


def original_objective(logits, log_lambda, target, codes, group_count):
    # Exact original development loss, including per-group Poisson reduction.
    counts = torch.zeros(group_count, device=target.device)
    counts.index_add_(0, codes, target)
    row_loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
    count_loss = nn.functional.poisson_nll_loss(log_lambda, counts, log_input=True, full=False)
    mass = torch.zeros(group_count, device=target.device)
    mass.index_add_(0, codes, torch.sigmoid(logits))
    consistency = (torch.log1p(mass) - torch.log1p(log_lambda.exp())).square().mean()
    return row_loss + .01 * count_loss + 0. * consistency, row_loss, count_loss, consistency


def predict(instance, groups, arrays, numeric, ntrain, nfinal, device, max_rows=MAX_ROWS, max_groups=MAX_GROUPS):
    instance.eval()
    result = np.empty(nfinal, dtype='float32')
    seen = np.zeros(nfinal, dtype=bool)
    with torch.no_grad():
        for batch in pack_groups(groups, np.arange(len(groups)), max_rows, max_groups):
            rows, inputs = batch_inputs(batch, arrays, numeric, device)
            positions = rows - ntrain
            if np.any(positions < 0) or np.any(positions >= nfinal) or seen[positions].any():
                raise AssertionError('Final row-position restoration differs')
            logits, _ = instance(*inputs)
            result[positions] = torch.sigmoid(logits).cpu().numpy()
            seen[positions] = True
    if not seen.all() or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise AssertionError('Final inference row coverage/probabilities differ')
    return result


def replay_verdict(error, backend, repeat_error=None, third_error=None):
    if not np.isfinite(error) or error < 0:
        raise AssertionError('Invalid replay error')
    if error <= STRICT_REPLAY_ATOL:
        return {'accepted_absolute_tolerance': STRICT_REPLAY_ATOL, 'cuda_reduction_allowance_used': False}
    if (backend == 'cuda' and error <= CUDA_REPLAY_ATOL and repeat_error is not None and
            third_error is not None and 0. < repeat_error <= CUDA_REPLAY_ATOL and
            np.isfinite(third_error) and 0. <= third_error <= CUDA_REPLAY_ATOL):
        return {'accepted_absolute_tolerance': CUDA_REPLAY_ATOL, 'cuda_reduction_allowance_used': True}
    raise AssertionError('Fresh-model replay exceeds documented same-backend tolerance')


def train_and_predict(device_name):
    feature_manifest = FEATURES / 'manifest.json'
    if not feature_manifest.is_file():
        raise FileNotFoundError('Feature manifest is not ready: ' + str(feature_manifest))
    if OUT.exists():
        raise FileExistsError('Preserving existing allocated refit directory: ' + str(OUT))
    started = time.perf_counter()
    feature_info, meta = read(feature_manifest), read(CACHE / 'manifest.json')
    numeric_fields = validate_feature_manifest(feature_info, meta)
    input_hashes = {str(path.resolve()): digest(path) for path in (feature_manifest, CACHE / 'manifest.json', CACHE / 'vocabs.json')}
    numeric_hash = digest(FEATURES / 'numeric_past500.npy')
    if numeric_hash != feature_info['numeric_sha256']['past']:
        raise AssertionError('Past numeric NPY bytes differ from verified feature manifest')
    sources_before = source_hashes()
    options, development = development_contract(numeric_fields)
    ntrain, nfinal = meta['train_rows'], meta['final_rows']
    numeric = np.load(FEATURES / 'numeric_past500.npy', mmap_mode='r')
    arrays = {name: np.load(CACHE / f'{name}.npy', mmap_mode='r')
              for name in ('categories', 'sequences', 'date', 'user_id', 'log_id')}
    if numeric.shape != (ntrain + nfinal, 500) or numeric.dtype != np.float32:
        raise AssertionError('Past numeric matrix shape/dtype differs')
    if any(len(value) != ntrain + nfinal for value in arrays.values()):
        raise AssertionError('Full cache array row count differs')
    if (arrays['categories'].shape != (ntrain + nfinal, len(meta['categorical_fields'])) or
            arrays['sequences'].shape[:2] != (ntrain + nfinal, len(meta['sequence_fields']))):
        raise AssertionError('Full cache categorical/sequence schema differs')
    # The final label tail is never indexed, scanned, reduced, or copied.
    labels = np.array(np.load(CACHE / 'label.npy', mmap_mode='r')[:ntrain], dtype='float32')
    raw, raw_labels = read_raw_sources(ROOT / '2026ml', ntrain, nfinal)
    identity_hashes = {}
    for field, cache_field in (('user_id', 'user_id'), ('pt_d', 'date'), ('log_id', 'log_id')):
        np.testing.assert_array_equal(raw[field], arrays[cache_field])
        identity_hashes[field] = hashlib.sha256(np.ascontiguousarray(raw[field], dtype='<i8').tobytes()).hexdigest()
    if identity_hashes != feature_info['row_identity_sha256']:
        raise AssertionError('Raw CSV row identities differ from numeric feature manifest')
    np.testing.assert_array_equal(labels, raw_labels)
    del raw_labels
    dates, users = raw['pt_d'], raw['user_id']
    if not np.all(dates[:ntrain] < 20191117) or not np.all(dates[ntrain:] == 20191117):
        raise AssertionError('Full-data chronological boundary differs')
    train_rows, final_rows = np.arange(ntrain), np.arange(ntrain, ntrain + nfinal)
    train_groups = groups_for_rows(dates, users, train_rows)
    final_groups = groups_for_rows(dates, users, final_rows)
    validate_groups(dates, users, train_rows, train_groups)
    validate_groups(dates, users, final_rows, final_groups)
    torch.set_num_threads(3)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no output directory created')
    device = torch.device(device_name)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(SEED)
    def make_model():
        return AllocatedSetContextCTR(meta['categorical_fields'], meta['sequence_fields'],
            meta['group_for_field'], meta['vocabulary_sizes'], initial_ctr=float(labels.mean()), **options).to(device)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=1e-4)
    order = np.random.default_rng(SEED + 1).permutation(len(train_groups))
    batches = list(pack_groups(train_groups, order, MAX_ROWS, MAX_GROUPS))
    protocol = {'status': 'training', 'model': 'AllocatedSetContextCTR', 'options': options,
        'train_rows': ntrain, 'final_rows': nfinal, 'train_groups': len(train_groups), 'final_groups': len(final_groups),
        'fixed_training_epochs': 1, 'fixed_fractional_epochs': 1., 'training_steps': len(batches),
        'budget_source': 'Both saved development checkpoints select best_fractional_epoch=1.0; fixed before refit',
        'development': development, 'seed': SEED, 'torch_threads': 3, 'device': str(device),
        'learning_rate': .0005, 'weight_decay': 1e-4, 'gradient_norm_clip': 5.,
        'max_rows_per_batch': MAX_ROWS, 'max_groups_per_batch': MAX_GROUPS,
        'group_unit': 'Raw user_id and pt_d, train and final partitions separately',
        'oversize_policy': 'Complete oversized group alone; never split, sample, truncate or cap',
        'maximum_training_group_rows': max(map(len, train_groups)),
        'maximum_final_group_rows': max(map(len, final_groups)),
        'auxiliary_count_weight': .01, 'consistency_weight': 0.,
        'loss': 'Mean row BCE + .01 * mean per-group Poisson(log_input=True,full=False) + 0 * original consistency',
        'model_inputs': ['categories', 'sequences', 'numeric_past500', 'group_index'],
        'training_targets': 'Only original train.csv labels / cache label prefix [0,2526320)',
        'numeric_fields': numeric_fields, 'numeric_features': str((FEATURES / 'numeric_past500.npy').resolve()),
        'feature_metadata': str(feature_manifest.resolve()), 'feature_metadata_sha256': input_hashes[str(feature_manifest.resolve())],
        'feature_scaler_fit_scope': feature_info['scaler_fit_scope'], 'numeric_clip': [-8, 8],
        'numeric_sha256': numeric_hash, 'row_identity_sha256': identity_hashes,
        'categorical_sequence_vocabulary_cache': str(CACHE.resolve()),
        'raw_train_usecols': IDENTITY_COLUMNS + ['label'], 'raw_final_usecols': IDENTITY_COLUMNS,
        'raw_cache_identity_checks': {'user_id': True, 'pt_d': True, 'log_id': True, 'training_labels': True},
        'final_labels_read': False, 'final_auc': None, 'replay_policy': REPLAY_POLICY,
        'source_sha256': sources_before, 'input_metadata_sha256': input_hashes,
        'parameter_count': sum(parameter.numel() for parameter in model.parameters())}
    OUT.mkdir(parents=True)
    write(OUT / 'protocol.json', protocol)
    for source, target in ((Path(__file__), 'trainer_snapshot.py'),
                           (HERE / 'allocated_set_model.py', 'model_snapshot.py'),
                           (HERE / 'set_context_model.py', 'base_model_snapshot.py'),
                           (HERE / 'interaction_models.py', 'interaction_model_snapshot.py'),
                           (HERE / 'train_set_context.py', 'group_trainer_snapshot.py')):
        (OUT / target).write_bytes(source.read_bytes())
    model.train()
    totals, seen = np.zeros(4), 0
    print('FULL_ALLOCATED_START', str(device), len(train_groups), len(batches), ntrain, flush=True)
    for step, batch in enumerate(batches, 1):
        rows, inputs = batch_inputs(batch, arrays, numeric, device)
        target = torch.as_tensor(labels[rows], device=device)
        optimizer.zero_grad(set_to_none=True)
        losses = original_objective(*model(*inputs), target, inputs[-1], len(batch))
        loss = losses[0]
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite allocated objective')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        totals += np.asarray([float(value.detach()) for value in losses]) * len(rows)
        seen += len(rows)
        if step % 100 == 0 or step == len(batches):
            print('FULL_ALLOCATED_PROGRESS', step, len(batches), seen, (totals / seen).tolist(), flush=True)
    if seen != ntrain:
        raise AssertionError('The fixed full training epoch did not visit every row once')
    torch.save({'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                'manifest': meta, 'options': options, 'numeric_fields': numeric_fields,
                'initial_ctr': float(labels.mean()), 'fixed_training_epochs': 1, 'steps': len(batches),
                'feature_metadata_sha256': protocol['feature_metadata_sha256'], 'source_sha256': sources_before}, OUT / 'model.pt')
    prediction = predict(model, final_groups, arrays, numeric, ntrain, nfinal, device)
    del model, optimizer, batches
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    saved = torch.load(OUT / 'model.pt', map_location='cpu', weights_only=False)
    restored = make_model()
    restored.load_state_dict(saved['state_dict'], strict=True)
    replay = predict(restored, final_groups, arrays, numeric, ntrain, nfinal, device)
    error = float(np.max(np.abs(prediction.astype('float64') - replay)))
    repeat_error, third_error = None, None
    if STRICT_REPLAY_ATOL < error <= CUDA_REPLAY_ATOL and device.type == 'cuda':
        repeated = predict(restored, final_groups, arrays, numeric, ntrain, nfinal, device)
        repeat_error = float(np.max(np.abs(replay.astype('float64') - repeated)))
        third_error = float(np.max(np.abs(prediction.astype('float64') - repeated)))
    verdict = replay_verdict(error, device.type, repeat_error, third_error)
    if (sources_before != source_hashes() or
            any(digest(path) != value for path, value in input_hashes.items()) or
            digest(FEATURES / 'numeric_past500.npy') != numeric_hash):
        raise AssertionError('Source, numeric features or input metadata changed during refit')
    np.save(OUT / 'final_prediction.npy', prediction)
    verification = {'status': 'complete_and_verified', 'model': 'AllocatedSetContextCTR',
        'train_rows': ntrain, 'final_rows': nfinal, 'train_groups': len(train_groups), 'final_groups': len(final_groups),
        'fixed_training_epochs': 1, 'fixed_fractional_epochs': 1., 'steps': saved['steps'], 'training_rows_seen': seen,
        'fresh_model_replayed_rows': len(replay), 'maximum_reload_error': error,
        'restored_instance_repeat_error': repeat_error, 'original_vs_third_error': third_error,
        **verdict, 'replay_backend': str(device), 'replay_policy': REPLAY_POLICY,
        'train_loss': float(totals[0] / seen), 'row_loss': float(totals[1] / seen),
        'count_loss': float(totals[2] / seen), 'consistency_loss': float(totals[3] / seen),
        'mean_probability': float(prediction.mean()), 'min_probability': float(prediction.min()),
        'max_probability': float(prediction.max()), 'prediction_shape': list(prediction.shape),
        'model_sha256': digest(OUT / 'model.pt'), 'prediction_sha256': digest(OUT / 'final_prediction.npy'),
        'feature_metadata': str(feature_manifest.resolve()), 'feature_metadata_sha256': protocol['feature_metadata_sha256'],
        'source_immutability_verified': True, 'numeric_immutability_verified': True,
        'numeric_sha256': numeric_hash, 'row_identity_sha256': identity_hashes,
        'raw_cache_identity_checks': protocol['raw_cache_identity_checks'],
        'final_labels_read': False, 'final_auc': None, 'seconds': time.perf_counter() - started}
    protocol['status'] = 'complete_and_verified'
    write(OUT / 'protocol.json', protocol)
    write(OUT / 'verification.json', verification)
    print('FULL_ALLOCATED_COMPLETE', json.dumps(verification), flush=True)


def toy_tests():
    """Synthetic CSV/group/loss/replay tests only; never inspect project data/CUDA."""
    import tempfile
    torch.set_num_threads(3)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    with tempfile.TemporaryDirectory(prefix='allocated_refit_toy_') as temporary:
        directory = Path(temporary)
        pd.DataFrame({'user_id': [7, 7, 8, 7, 7], 'pt_d': [1, 2, 1, 1, 1],
                      'log_id': [1, 2, 3, 4, 5], 'label': [0, 1, 0, 1, 0]}).to_csv(directory / 'train.csv', index=False)
        # Deliberately nonnumeric final labels prove the explicit usecols boundary.
        pd.DataFrame({'user_id': [7, 9, 7, 7], 'pt_d': [3, 3, 3, 3], 'log_id': [6, 7, 8, 9],
                      'label': ['DO_NOT_READ'] * 4}).to_csv(directory / 'final_test.csv', index=False)
        raw, labels = read_raw_sources(directory, 5, 4)
        groups = groups_for_rows(raw['pt_d'], raw['user_id'], np.arange(5))
        final = groups_for_rows(raw['pt_d'], raw['user_id'], np.arange(5, 9))
        validate_groups(raw['pt_d'], raw['user_id'], np.arange(5), groups)
        validate_groups(raw['pt_d'], raw['user_id'], np.arange(5, 9), final)
        assert [group.tolist() for group in groups] == [[0, 3, 4], [1], [2]]
        oversized = list(pack_groups(groups, np.arange(len(groups)), 2, 1))
        assert oversized[0][0].tolist() == [0, 3, 4] and all(len(batch) == 1 for batch in oversized)
        arrays = {'categories': np.random.randint(0, 10, (9, 2), dtype='int32'),
                  'sequences': np.random.randint(0, 10, (9, 1, 5), dtype='int32')}
        numeric = np.clip(np.random.normal(size=(9, 500)), -8, 8).astype('float32')
        kwargs = dict(categorical_fields=['user_id', 'task_id'], sequence_fields=['history'],
                      group_for_field={'user_id': 'user', 'task_id': 'task', 'history': 'task'},
                      vocabulary_sizes={'user': 10, 'task': 10}, numeric_dim=500,
                      use_set_context=True, initial_ctr=float(labels.mean()))
        model = AllocatedSetContextCTR(**kwargs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=1e-4)
        order = np.random.default_rng(SEED + 1).permutation(len(groups))
        covered = []
        for batch in pack_groups(groups, order, 2, 1):
            rows, inputs = batch_inputs(batch, arrays, numeric, torch.device('cpu'))
            covered.extend(rows.tolist())
            target = torch.as_tensor(labels[rows])
            optimizer.zero_grad(set_to_none=True)
            logits, log_lambda = model(*inputs)
            losses = original_objective(logits, log_lambda, target, inputs[-1], len(batch))
            group_targets = torch.zeros(len(batch)).index_add_(0, inputs[-1], target)
            expected = nn.functional.binary_cross_entropy_with_logits(logits, target) + .01 * nn.functional.poisson_nll_loss(log_lambda, group_targets, log_input=True, full=False)
            torch.testing.assert_close(losses[0], expected, rtol=0, atol=0)
            losses[0].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        assert sorted(covered) == list(range(5))
        prediction = predict(model, final, arrays, numeric, 5, 4, torch.device('cpu'), 2, 1)
        torch.save(model.state_dict(), directory / 'toy.pt')
        restored = AllocatedSetContextCTR(**kwargs)
        restored.load_state_dict(torch.load(directory / 'toy.pt', weights_only=True), strict=True)
        replay = predict(restored, final, arrays, numeric, 5, 4, torch.device('cpu'), 2, 1)
        np.testing.assert_array_equal(prediction, replay)
        # Explicitly compare restored original row positions with standalone groups.
        with torch.no_grad():
            for group in final:
                rows, inputs = batch_inputs([group], arrays, numeric, torch.device('cpu'))
                np.testing.assert_array_equal(prediction[rows - 5], restored(*inputs)[0].sigmoid().numpy())
    assert replay_verdict(2e-7, 'cpu')['accepted_absolute_tolerance'] == 2e-7
    assert replay_verdict(2e-7, 'cuda')['accepted_absolute_tolerance'] == 2e-7
    assert replay_verdict(3e-7, 'cuda', 1e-8, 3e-7)['cuda_reduction_allowance_used']
    for arguments in ((np.nextafter(2e-7, np.inf), 'cpu'),
                      (np.nextafter(3e-7, np.inf), 'cuda', 1e-8, 1e-8),
                      (2.5e-7, 'cuda'), (2.5e-7, 'cuda', 0., 2.5e-7),
                      (2.5e-7, 'cuda', 1e-8, np.nextafter(3e-7, np.inf))):
        try:
            replay_verdict(*arguments)
        except AssertionError:
            pass
        else:
            raise AssertionError('Replay threshold accepted an invalid boundary')
    fields = [f'toy_{column}' for column in range(500)]
    toy_info = {'status': 'complete_and_verified', 'rows': TRAIN_ROWS + FINAL_ROWS,
        'train_rows': TRAIN_ROWS, 'final_rows': FINAL_ROWS, 'final_labels_read': False,
        'numeric_fields': fields, 'clip': [-8, 8], 'numeric_past_path': str(FEATURES / 'numeric_past500.npy'),
        'source_scalers_past': {field: {'mean': 0., 'std': 1.} for field in fields},
        'scaler_fit_scope': 'Original train.csv rows [0,2526320), never final rows',
        'per_row_strictly_historical': {'past': True, 'oof': False},
        'shape': [TRAIN_ROWS + FINAL_ROWS, 500], 'dtype': 'float32', 'storage_order': 'C',
        'scaler_fit_rows': TRAIN_ROWS, 'scaler_fit_indices': [0, TRAIN_ROWS],
        'identity_hash_rule': 'SHA256 of contiguous little-endian signed int64 (<i8), no header, original train.csv then final_test.csv rows',
        'row_identity_sha256': {field: 'a' * 64 for field in IDENTITY_COLUMNS}, 'numeric_sha256': {'past': 'b' * 64}}
    toy_meta = {'rows': TRAIN_ROWS + FINAL_ROWS, 'train_rows': TRAIN_ROWS,
                'final_rows': FINAL_ROWS, 'final_labels_available': False}
    assert validate_feature_manifest(toy_info, toy_meta) == fields
    for patch in ({'status': 'incomplete'}, {'final_labels_read': True},
                  {'scaler_fit_indices': [0, TRAIN_ROWS + 1]}, {'clip': [-9, 9]},
                  {'per_row_strictly_historical': {'past': False}},
                  {'numeric_sha256': {'past': 'invalid'}}):
        try:
            validate_feature_manifest({**toy_info, **patch}, toy_meta)
        except (RuntimeError, AssertionError):
            pass
        else:
            raise AssertionError('Invalid feature manifest was accepted: ' + str(patch))
    print('ALLOCATED_REFIT_TOY_TESTS_PASS: CSV label isolation, complete/oversized groups, original loss, fixed row coverage, row restoration, fresh CPU replay, feature manifest gates, exact 2e-7/3e-7 boundaries', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'toy'], required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    arguments = parser.parse_args()
    toy_tests() if arguments.stage == 'toy' else train_and_predict(arguments.device)
