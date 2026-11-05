"""Fixed .75-epoch full multiscale DCN refit, new v3 export and process replay.

Writes only a previously absent output/chronological_multiscale_v3. Supplied
final labels are never loaded: final CSV reads have an explicit column allowlist,
and final label values are synthesized as -1 before any feature builder call.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('MKL_NUM_THREADS', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '3')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DAILY = HERE.parent / 'daily_interest_din'
CACHE = DAILY / 'full_refit_cache'
BASE_MODELS = DAILY / 'full_refit_models'
V2 = ROOT / 'output' / 'chronological_lgb_v2'
EXPORT = ROOT / 'output' / 'chronological_multiscale_v3'
sys.path.insert(0, str(HERE.parent / 'model_comparison'))
sys.path.insert(0, str(HERE))

import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
import torch
from torch import nn
from bounded_id_models import BoundedIDDCN
import prepare_multiscale_stats as multiscale

SEED = 42
SHUFFLE_SEED = 43
BATCH_SIZE = 2048
INFERENCE_BATCH = 4096
CHUNK = 25000
EPOCH_FRACTION = .75
WEIGHTS = {'dcn': .3, 'tree': .3, 'lgb': .15, 'multiscale_dcn': .25}
OPTIONS = {'numeric_dim': 325, 'embedding_dim': 8, 'hidden_dims': [128, 64],
           'dropout': .1, 'unknown_dropout': .01, 'cross_layers': 2,
           'group_norm_caps': {'user_id': .2, 'ad_task': .2, 'adv_id': .2}}
OPTIMIZER = {'name': 'AdamW', 'lr': .001, 'weight_decay': 1e-5,
             'pos_weight': 1., 'gradient_clip_norm': 5.}
DEVELOPMENT = {
    '20191115': {'fixed_v3_auc': .7876002210086661, 'fixed_v2_auc': .786473152710307,
                 'multiscale_dcn_auc': .7761320006022201, 'best_fractional_epoch': .7503045066991474},
    '20191116': {'fixed_v3_auc': .7948963507440203, 'fixed_v2_auc': .79243244179162,
                 'multiscale_dcn_auc': .7857505309802582, 'best_fractional_epoch': .75},
}
WARNINGS = [
    '开发日期15/16反复用于模型和固定融合权重选型，不是独立盲测；开发AUC不代表final AUC。',
    '源域新闻数据、当前日曝光和广告历史快照采用所提供整日离线输入假设，不代表逐条在线曝光前可用。',
    'final标签不读取且强制为-1；final_auc=None，不能声称final达标或AUC达到0.85。',
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}


def log(event, **fields):
    entry = {'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'pid': os.getpid(), 'event': event, **fields}
    text = json.dumps(entry, ensure_ascii=True)
    with (EXPORT / 'run.log').open('a', encoding='utf-8') as stream:
        stream.write(text + '\n')
    print(text, flush=True)


def probabilities(values, rows, name):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (rows,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise AssertionError(name + ': invalid probabilities')
    return values


def load_cache():
    manifest = read(CACHE / 'manifest.json')
    if manifest['final_labels_available'] is not False:
        raise AssertionError('Cache final-label availability differs')
    arrays = {name: np.load(CACHE / (name + '.npy'), mmap_mode='r')
              for name in ['categories', 'sequences', 'numeric', 'date', 'log_id', 'user_id']}
    ntrain, nfinal = manifest['train_rows'], manifest['final_rows']
    if manifest['rows'] != ntrain + nfinal or any(len(value) != manifest['rows'] for value in arrays.values()):
        raise AssertionError('Full cache row counts differ')
    if arrays['numeric'].shape != (manifest['rows'], 166):
        raise AssertionError('Expected existing 166-dimensional full numeric base')
    if arrays['date'][:ntrain].max() != 20191116 or not np.all(arrays['date'][:ntrain] < 20191117):
        raise AssertionError('Training date boundary differs')
    if not np.all(arrays['date'][ntrain:] == 20191117):
        raise AssertionError('Final date boundary differs')
    return manifest, arrays


def final_ids(manifest, arrays):
    ids = pd.read_csv(ROOT / '2026ml' / 'final_test.csv', usecols=['log_id', 'user_id', 'pt_d'], dtype='int64')
    n = manifest['train_rows']
    if len(ids) != manifest['final_rows']:
        raise AssertionError('Final row count differs')
    for name, cached in [('log_id', 'log_id'), ('user_id', 'user_id'), ('pt_d', 'date')]:
        np.testing.assert_array_equal(ids[name].to_numpy(), arrays[cached][n:])
    return ids


def existing_components(nfinal):
    paths = {'dcn': BASE_MODELS / 'dcn_final_prediction.npy',
             'tree': BASE_MODELS / 'tree_final_prediction.npy',
             'lgb': V2 / 'lgb_final_prediction.npy'}
    return {name: probabilities(np.load(path, mmap_mode='r'), nfinal, name) for name, path in paths.items()}


def development_evidence():
    files = {}
    for day, expected in DEVELOPMENT.items():
        path = DAILY / 'dense_results' / ('goal085_day' + day[-2:] + '_multiscale_dcn') / 'metrics.json'
        metrics = read(path)
        if metrics['options'] != OPTIONS or metrics['seed'] != SEED or metrics['learning_rate'] != .001:
            raise AssertionError('Development DCN hyperparameters differ')
        if metrics.get('pos_weight', 1.) != 1. or metrics['recent_days'] != 0:
            raise AssertionError('Development DCN weighting/history differs')
        if abs(metrics['best_fractional_epoch'] - expected['best_fractional_epoch']) > 1e-14:
            raise AssertionError('Development budget source differs')
        if abs(metrics['auc'] - expected['multiscale_dcn_auc']) > 1e-12:
            raise AssertionError('Development DCN AUC differs')
        files[day] = file_record(path)
    results = read(HERE / 'round_results.json')
    for day, expected in DEVELOPMENT.items():
        matches = [entry for entry in results['experiments'] if entry['day'] == int(day)
                   and entry['model'] == 'fixed_lgb_multiscale25_blend']
        if len(matches) != 1 or abs(matches[0]['auc'] - expected['fixed_v3_auc']) > 1e-14:
            raise AssertionError('Fixed v3 development blend evidence differs')
    files['round_results_at_start'] = file_record(HERE / 'round_results.json')
    return files


def aligned_ads(manifest, arrays):
    columns = sorted(set(multiscale.FIELDS + ['pt_d', 'log_id', 'user_id']))
    train = pd.read_csv(ROOT / '2026ml' / 'train.csv', usecols=columns + ['label'], dtype='int32')
    final = pd.read_csv(ROOT / '2026ml' / 'final_test.csv', usecols=columns, dtype='int32')
    final['label'] = np.int32(-1)
    if len(train) != manifest['train_rows'] or len(final) != manifest['final_rows']:
        raise AssertionError('Original train/final row counts differ')
    frame = pd.concat([train, final], ignore_index=True)
    del train, final
    for name, cached in [('log_id', 'log_id'), ('user_id', 'user_id'), ('pt_d', 'date')]:
        np.testing.assert_array_equal(frame[name].to_numpy(), arrays[cached])
    labels = np.load(CACHE / 'label.npy', mmap_mode='r')
    np.testing.assert_array_equal(frame.label.to_numpy()[:manifest['train_rows']], labels[:manifest['train_rows']])
    del labels
    if not np.all(frame.label.to_numpy()[manifest['train_rows']:] == -1):
        raise AssertionError('Synthetic final labels differ')
    alignment = {'all_rows': len(frame), 'full_train_rows': manifest['train_rows'],
                 'final_rows': manifest['final_rows'], 'original_order': 'train.csv then final_test.csv; no sorting/deduplication',
                 'all_log_id_user_id_date_equal_existing_full_cache': True,
                 'all_train_labels_equal_existing_full_cache_training_prefix': True,
                 'final_label_source': 'np.int32(-1), never loaded from CSV/cache final suffix',
                 'train_clicks': int(frame.label.iloc[:manifest['train_rows']].sum()),
                 'first_final_row': frame.iloc[manifest['train_rows']][['log_id', 'user_id', 'pt_d', 'label']].astype(int).to_dict(),
                 'last_final_row': frame.iloc[-1][['log_id', 'user_id', 'pt_d', 'label']].astype(int).to_dict(),
                 'order_sha256': {name: hashlib.sha256(np.ascontiguousarray(frame[name].to_numpy()).tobytes()).hexdigest()
                                  for name in ['log_id', 'user_id', 'pt_d']}}
    write(EXPORT / 'input_alignment.json', alignment)
    return frame, alignment


def prepare_features(manifest, arrays):
    log('phase_start', phase='strict_past_159_feature_generation')
    frame, alignment = aligned_ads(manifest, arrays)
    raw = open_memmap(EXPORT / 'multiscale_raw.npy', mode='w+', dtype='float32',
                      shape=(manifest['rows'], 159), fortran_order=True)
    started = time.perf_counter()
    def progress(index, total, fields, size):
        log('feature_group_complete', group=index, total=total, fields=fields,
            distinct_keys=size, seconds=round(time.perf_counter() - started, 3))
    raw, names = multiscale.build_multiscale_statistics(frame, 20191117, output=raw, progress=progress)
    raw.flush()
    # Strictly past features for every train row must be identical to the
    # validated day16 builder output, including those day's unseen labels.
    prior = np.load(HERE / 'multiscale_day16.npy', mmap_mode='r')
    if prior.shape != (manifest['train_rows'], 159):
        raise AssertionError('Existing day16 raw schema differs')
    for start in range(0, manifest['train_rows'], CHUNK):
        stop = min(start + CHUNK, manifest['train_rows'])
        np.testing.assert_array_equal(raw[start:stop], prior[start:stop])
    del prior, frame
    gc.collect()
    meta = {'shape': list(raw.shape), 'dtype': 'float32', 'storage_order': 'F', 'features': names,
            'validation_day': 20191117, 'advertisement_labels_loaded': True,
            'advertisement_validation_labels_used': False, 'final_labels_read': False,
            'rule': 'Each row uses strictly earlier calendar-date advertisement labels; final labels synthesized as -1',
            'all_train_raw_features_equal_day16_development_raw': True,
            'builder_source': file_record(HERE / 'prepare_multiscale_stats.py'),
            'input_alignment': alignment, 'feature_generation_seconds': time.perf_counter() - started}
    write(EXPORT / 'multiscale_raw.json', meta)
    log('phase_start', phase='full_train_only_scaling_and_325_numeric_memmap')
    scalers = {}
    means = np.empty(159, dtype=np.float32)
    stds = np.empty(159, dtype=np.float32)
    for col, name in enumerate(names):
        values = raw[:manifest['train_rows'], col]
        mean, std = float(values.mean()), max(float(values.std()), 1e-6)
        means[col], stds[col] = mean, std
        scalers[name] = {'mean': mean, 'std': std}
    numeric = open_memmap(EXPORT / 'numeric.npy', mode='w+', dtype='float32',
                          shape=(manifest['rows'], 325))
    for start in range(0, manifest['rows'], CHUNK):
        stop = min(start + CHUNK, manifest['rows'])
        numeric[start:stop, :166] = arrays['numeric'][start:stop]
        numeric[start:stop, 166:] = (raw[start:stop] - means) / stds
        if not np.isfinite(numeric[start:stop]).all():
            raise AssertionError('Nonfinite combined numeric input')
        if start % (CHUNK * 20) == 0:
            numeric.flush()
            log('numeric_chunk', rows=stop, total=manifest['rows'])
    numeric.flush()
    numeric_meta = {'shape': list(numeric.shape), 'dtype': 'float32', 'storage_order': 'C',
                    'features': manifest['numeric_fields'] + names,
                    'base_166_copied_from': str((CACHE / 'numeric.npy').resolve()),
                    'base_166_unchanged': True, 'extra_159_scalers': scalers,
                    'scaler_fit_rows': manifest['train_rows'], 'scaler_fit_selection': '[0, full_train_rows) only',
                    'final_labels_read': False, 'final_auc': None}
    write(EXPORT / 'numeric_meta.json', numeric_meta)
    del raw
    gc.collect()
    log('features_ready', shape=list(numeric.shape), all_train_raw_match_development=True)
    return numeric, numeric_meta, alignment


def setup_device():
    torch.set_num_threads(3)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    memory = list(torch.cuda.mem_get_info()) if device.type == 'cuda' else None
    log('phase_start', phase='fixed_budget_training_device_ready', device=str(device), free_total_gpu_bytes=memory)
    return device


def model_inputs(rows, arrays, numeric, device):
    return (torch.as_tensor(arrays['categories'][rows].astype(np.int64), device=device),
            torch.as_tensor(arrays['sequences'][rows].astype(np.int64), device=device),
            torch.as_tensor(np.array(numeric[rows]), device=device))


def predict(model, manifest, arrays, numeric, device, prefix):
    model.eval()
    values = np.empty(manifest['final_rows'], dtype=np.float64)
    with torch.no_grad():
        for offset in range(0, len(values), INFERENCE_BATCH):
            stop = min(offset + INFERENCE_BATCH, len(values))
            rows = slice(manifest['train_rows'] + offset, manifest['train_rows'] + stop)
            values[offset:stop] = torch.sigmoid(model(*model_inputs(rows, arrays, numeric, device))).cpu().numpy()
            if offset % (INFERENCE_BATCH * 25) == 0 or stop == len(values):
                log(prefix, rows=stop, total=len(values))
    return probabilities(values, manifest['final_rows'], prefix)


def main():
    started = time.perf_counter()
    if EXPORT.exists():
        raise FileExistsError('Preserving existing v3 output: ' + str(EXPORT))
    manifest, arrays = load_cache()
    evidence = development_evidence()
    existing_components(manifest['final_rows'])
    protected_paths = [BASE_MODELS / 'dcn.pt', BASE_MODELS / 'catboost.cbm',
                       BASE_MODELS / 'dcn_final_prediction.npy', BASE_MODELS / 'tree_final_prediction.npy']
    protected_paths += sorted(path for path in V2.iterdir() if path.is_file())
    protected = {str(path.resolve()): file_record(path) for path in protected_paths}
    sources = {str(path.resolve()): file_record(path) for path in
               [Path(__file__), HERE / 'prepare_multiscale_stats.py', DAILY / 'train_dense_signals.py',
                HERE.parent / 'model_comparison' / 'bounded_id_models.py',
                HERE.parent / 'model_comparison' / 'id_regularized_models.py',
                HERE.parent / 'model_comparison' / 'mainstream_models.py']}
    batches = math.ceil(manifest['train_rows'] / BATCH_SIZE)
    steps = math.ceil(batches * EPOCH_FRACTION)
    budget_rows = min(steps * BATCH_SIZE, manifest['train_rows'])
    protocol = {'status': 'fixed_before_features_training_and_final_inference',
                'seed': SEED, 'shuffle_seed': SHUFFLE_SEED, 'batch_size': BATCH_SIZE,
                'fixed_epoch_fraction': EPOCH_FRACTION, 'full_epoch_batches': batches,
                'fixed_optimizer_steps': steps, 'fixed_training_rows_seen': budget_rows,
                'actual_fractional_epoch_by_batches': steps / batches,
                'budget_rule': 'ceil(ceil(full_train_rows/2048)*0.75), exactly the development .75 evaluation checkpoint rule',
                'full_train_rows': manifest['train_rows'], 'final_rows': manifest['final_rows'],
                'options': OPTIONS, 'optimizer': OPTIMIZER, 'ensemble': WEIGHTS,
                'weights_source': 'Day15 chose 20% LGB inside old blend and 25% multiscale; day16 confirmed direction; fixed .3/.3/.15/.25',
                'development_auc': DEVELOPMENT, 'development_evidence_files': evidence,
                'source_files': sources, 'protected_files': protected,
                'cache_manifest': file_record(CACHE / 'manifest.json'),
                'input_sha256': manifest['input_sha256'], 'early_stopping': False,
                'feature_generation': 'Import build_multiscale_statistics(all_train_plus_final_minus1,20191117,F_memmap159)',
                'numeric_inputs': 'Existing full-cache166 unchanged + 159 multiscale columns normalized on full training rows only',
                'same_day_input_assumption': 'Supplied full source-day snapshots, daily exposures and ad-history snapshots available offline',
                'advertisement_statistics': 'Strictly earlier calendar dates; final advertisement labels never loaded',
                'inference_batch_size': INFERENCE_BATCH, 'final_labels_available': False,
                'final_labels_read': False, 'final_label_value': -1, 'final_auc': None, 'warnings': WARNINGS}
    EXPORT.mkdir(parents=True, exist_ok=False)
    write(EXPORT / 'protocol.json', protocol)
    log('phase_start', phase='protocol_committed', fixed_optimizer_steps=steps,
        requested_fraction=.75, actual_fraction=steps / batches, rows_seen=budget_rows)
    numeric, numeric_meta, alignment = prepare_features(manifest, arrays)
    device = setup_device()
    # A GPU allocation error leaves prepared feature files and protocol intact;
    # no reduced or different training budget is silently substituted.
    labels = np.load(CACHE / 'label.npy', mmap_mode='r')
    train_labels = np.array(labels[:manifest['train_rows']], dtype=np.float32, copy=True)
    del labels
    if not np.isin(train_labels, [0, 1]).all() or int(train_labels.sum()) != manifest['train_clicks']:
        raise AssertionError('Full training labels differ')
    prior = float(train_labels.mean(dtype=np.float64))
    model = BoundedIDDCN(manifest['categorical_fields'], manifest['sequence_fields'],
                         manifest['group_for_field'], manifest['vocabulary_sizes'], initial_ctr=prior, **OPTIONS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    train = np.arange(manifest['train_rows'], dtype=np.int32)
    order = np.random.default_rng(SHUFFLE_SEED).permutation(train)
    order_hash = hashlib.sha256(order.tobytes()).hexdigest()
    positive_weight = torch.tensor(1., device=device)
    model.train()
    total, seen = 0., 0
    training_started = time.perf_counter()
    log('phase_start', phase='fixed_075_epoch_training', steps=steps,
        train_rows=manifest['train_rows'], shuffle_order_sha256=order_hash)
    for step in range(steps):
        rows = order[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
        target = torch.as_tensor(train_labels[rows], device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(*model_inputs(rows, arrays, numeric, device)),
                                                             target, pos_weight=positive_weight)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite full multiscale training loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        total += float(loss.detach()) * len(rows)
        seen += len(rows)
        if (step + 1) % 100 == 0 or step == 0 or step + 1 == steps:
            log('train_progress', step=step + 1, total_steps=steps, rows_seen=seen,
                online_logloss=total / seen, seconds=time.perf_counter() - training_started)
    if seen != budget_rows:
        raise AssertionError('Actual row budget differs from committed protocol')
    training_seconds = time.perf_counter() - training_started
    checkpoint = {'state_dict': {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                  'manifest': manifest, 'options': OPTIONS, 'numeric_fields': numeric_meta['features'],
                  'extra_scalers': numeric_meta['extra_159_scalers'], 'seed': SEED, 'shuffle_seed': SHUFFLE_SEED,
                  'pos_weight': 1., 'fixed_epoch_fraction': .75, 'actual_optimizer_steps': steps,
                  'actual_training_rows_seen': seen, 'actual_fractional_epoch': steps / batches,
                  'shuffle_order_sha256': order_hash, 'initial_ctr': prior, 'online_training_logloss': total / seen,
                  'protocol_sha256': digest(EXPORT / 'protocol.json')}
    torch.save(checkpoint, EXPORT / 'multiscale_dcn.pt')
    del checkpoint, optimizer, train, train_labels, order, positive_weight, target, loss
    gc.collect()
    log('phase_start', phase='full_final_multiscale_prediction')
    prediction = predict(model, manifest, arrays, numeric, device, 'final_prediction')
    np.save(EXPORT / 'multiscale_dcn_final_prediction.npy', prediction)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    log('phase_start', phase='fixed_four_component_export')
    components = existing_components(manifest['final_rows'])
    components['multiscale_dcn'] = prediction
    probability = probabilities(sum(WEIGHTS[name] * components[name] for name in WEIGHTS),
                                manifest['final_rows'], 'ensemble')
    np.save(EXPORT / 'ensemble_final_prediction.npy', probability)
    ids = final_ids(manifest, arrays)
    pd.DataFrame({'log_id': ids.log_id.to_numpy(), 'pctr': probability}).to_csv(
        EXPORT / 'result.csv', index=False, float_format='%.6f', encoding='utf-8')
    pd.DataFrame({'id': np.arange(len(ids)), 'pctr': probability}).to_csv(
        EXPORT / 'submission.csv', index=False, float_format='%.6f', encoding='utf-8')
    detail = pd.DataFrame({'id': np.arange(len(ids))})
    for name in ['log_id', 'user_id', 'pt_d']:
        detail[name] = ids[name].to_numpy()
    detail['label'] = np.full(len(ids), -1, dtype=np.int8)
    detail['pctr'] = probability
    for name in WEIGHTS:
        detail[name + '_pctr'] = components[name]
    detail.to_csv(EXPORT / 'prediction_details.csv', index=False, float_format='%.17g', encoding='utf-8')
    log('phase_start', phase='feature_model_export_hashing')
    file_names = ['protocol.json', 'input_alignment.json', 'multiscale_raw.npy', 'multiscale_raw.json',
                  'numeric.npy', 'numeric_meta.json', 'multiscale_dcn.pt', 'multiscale_dcn_final_prediction.npy',
                  'ensemble_final_prediction.npy', 'result.csv', 'submission.csv', 'prediction_details.csv']
    record = {**protocol, 'status': 'predicted_awaiting_independent_verification',
              'files': {name: file_record(EXPORT / name) for name in file_names},
              'models': {'dcn': file_record(BASE_MODELS / 'dcn.pt'), 'tree': file_record(BASE_MODELS / 'catboost.cbm'),
                         'lgb': file_record(V2 / 'lgb_model.txt'), 'multiscale_dcn': file_record(EXPORT / 'multiscale_dcn.pt')},
              'input_alignment': alignment, 'actual_optimizer_steps': steps, 'actual_training_rows_seen': seen,
              'shuffle_order_sha256': order_hash, 'training_seconds': training_seconds,
              'train_clicks': manifest['train_clicks'], 'numeric_dim': 325,
              'device': str(device), 'torch_version': torch.__version__, 'numpy_version': np.__version__,
              'prediction_rows': len(ids), 'duplicate_log_id_rows': int(ids.log_id.duplicated().sum()),
              'probability': {'minimum': float(probability.min()), 'maximum': float(probability.max()), 'mean': float(probability.mean())},
              'final_id_rule': 'result.log_id keeps original final_test.csv rows including duplicates; submission.id is zero-based row position'}
    write(EXPORT / 'manifest.json', record)
    del arrays, numeric, components, prediction, probability, ids, detail
    gc.collect()
    log('phase_start', phase='independent_process_model_reload_and_all_final_verification')
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--verify-only'], check=True)
    verified = read(EXPORT / 'verification.json')
    if verified['status'] != 'passed':
        raise AssertionError('Independent verification failed')
    log('complete', rows=manifest['final_rows'], training_seconds=training_seconds,
        total_seconds=time.perf_counter() - started, verification='passed', final_auc=None)
    record = read(EXPORT / 'manifest.json')
    record['status'] = 'complete_and_verified'
    record['elapsed_seconds'] = time.perf_counter() - started
    record['files']['verification.json'] = file_record(EXPORT / 'verification.json')
    record['files']['run.log'] = file_record(EXPORT / 'run.log')
    write(EXPORT / 'manifest.json', record)


def verify_features(manifest, arrays, numeric):
    raw = np.load(EXPORT / 'multiscale_raw.npy', mmap_mode='r')
    meta = read(EXPORT / 'numeric_meta.json')
    if meta['features'] != manifest['numeric_fields'] + multiscale.feature_names():
        raise AssertionError('Actual model numeric feature order differs')
    if raw.shape != (manifest['rows'], 159) or not raw.flags.f_contiguous or numeric.shape != (manifest['rows'], 325):
        raise AssertionError('Actual numeric storage shape/order differs')
    means, stds = [], []
    for col, name in enumerate(multiscale.feature_names()):
        values = raw[:manifest['train_rows'], col]
        actual = {'mean': float(values.mean()), 'std': max(float(values.std()), 1e-6)}
        if actual != meta['extra_159_scalers'][name]:
            raise AssertionError('Scaler does not reproduce training-only fit: ' + name)
        means.append(actual['mean'])
        stds.append(actual['std'])
    means, stds = np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)
    for start in range(0, manifest['rows'], CHUNK):
        stop = min(start + CHUNK, manifest['rows'])
        np.testing.assert_array_equal(numeric[start:stop, :166], arrays['numeric'][start:stop])
        np.testing.assert_array_equal(numeric[start:stop, 166:], (raw[start:stop] - means) / stds)
    del raw
    return True


def verify():
    started = time.perf_counter()
    if (EXPORT / 'verification.json').exists():
        raise FileExistsError('Preserving existing v3 verification')
    log('phase_start', phase='independent_file_hash_and_feature_alignment_verification')
    record = read(EXPORT / 'manifest.json')
    protocol = read(EXPORT / 'protocol.json')
    if protocol['options'] != OPTIONS or protocol['ensemble'] != WEIGHTS or protocol['fixed_epoch_fraction'] != .75:
        raise AssertionError('Fixed refit protocol differs')
    # round_results.json is shared live experiment evidence and may gain entries;
    # its captured hash is provenance, while fixed model metrics remain immutable.
    immutable_evidence = [item for name, item in record['development_evidence_files'].items()
                          if name != 'round_results_at_start']
    for item in (list(record['files'].values()) + list(record['models'].values()) + list(record['source_files'].values())
                 + list(record['protected_files'].values()) + immutable_evidence + [record['cache_manifest']]):
        if digest(item['path']) != item['sha256']:
            raise AssertionError('Protected/input/output hash changed: ' + item['path'])
    if digest(ROOT / '2026ml' / 'final_test.csv') != record['input_sha256']['final_test.csv']:
        raise AssertionError('Original final input bytes changed')
    manifest, arrays = load_cache()
    numeric = np.load(EXPORT / 'numeric.npy', mmap_mode='r')
    verify_features(manifest, arrays, numeric)
    ids = final_ids(manifest, arrays)
    checkpoint = torch.load(EXPORT / 'multiscale_dcn.pt', map_location='cpu', weights_only=False)
    batches = math.ceil(manifest['train_rows'] / BATCH_SIZE)
    steps = math.ceil(batches * .75)
    if (checkpoint['actual_optimizer_steps'] != steps or checkpoint['actual_training_rows_seen'] != steps * BATCH_SIZE
            or checkpoint['options'] != OPTIONS or checkpoint['seed'] != SEED or checkpoint['shuffle_seed'] != SHUFFLE_SEED
            or checkpoint['protocol_sha256'] != digest(EXPORT / 'protocol.json')):
        raise AssertionError('Actual checkpoint budget/hyperparameters differ')
    order = np.random.default_rng(SHUFFLE_SEED).permutation(np.arange(manifest['train_rows'], dtype=np.int32))
    if hashlib.sha256(order.tobytes()).hexdigest() != checkpoint['shuffle_order_sha256']:
        raise AssertionError('Training shuffle order does not reproduce')
    del order
    device = setup_device()
    restored = BoundedIDDCN(manifest['categorical_fields'], manifest['sequence_fields'], manifest['group_for_field'],
                            manifest['vocabulary_sizes'], initial_ctr=checkpoint['initial_ctr'], **checkpoint['options']).to(device)
    restored.load_state_dict(checkpoint['state_dict'], strict=True)
    log('phase_start', phase='independent_all_final_model_replay')
    replay = predict(restored, manifest, arrays, numeric, device, 'independent_final_prediction')
    saved = probabilities(np.load(EXPORT / 'multiscale_dcn_final_prediction.npy'), len(ids), 'saved multiscale')
    reload_error = float(np.max(np.abs(replay - saved)))
    if reload_error > 2e-7:
        raise AssertionError('Independent full final DCN replay differs')
    components = existing_components(len(ids))
    components['multiscale_dcn'] = replay
    probability = probabilities(sum(WEIGHTS[name] * components[name] for name in WEIGHTS), len(ids), 'independent ensemble')
    saved_ensemble = probabilities(np.load(EXPORT / 'ensemble_final_prediction.npy'), len(ids), 'saved ensemble')
    ensemble_error = float(np.max(np.abs(probability - saved_ensemble)))
    if ensemble_error > 5e-8:
        raise AssertionError('Independent ensemble replay differs')
    result = pd.read_csv(EXPORT / 'result.csv', float_precision='round_trip')
    submission = pd.read_csv(EXPORT / 'submission.csv', float_precision='round_trip')
    detail = pd.read_csv(EXPORT / 'prediction_details.csv', float_precision='round_trip')
    if result.columns.tolist() != ['log_id', 'pctr'] or submission.columns.tolist() != ['id', 'pctr']:
        raise AssertionError('Submission columns differ')
    if any(len(frame) != len(ids) for frame in [result, submission, detail]):
        raise AssertionError('Submission row count differs')
    np.testing.assert_array_equal(result.log_id.to_numpy(), ids.log_id.to_numpy())
    np.testing.assert_array_equal(submission.id.to_numpy(), np.arange(len(ids)))
    np.testing.assert_array_equal(detail.id.to_numpy(), np.arange(len(ids)))
    for name in ['log_id', 'user_id', 'pt_d']:
        np.testing.assert_array_equal(detail[name].to_numpy(), ids[name].to_numpy())
    if not np.all(detail.label.to_numpy() == -1):
        raise AssertionError('Export final labels not forced to -1')
    rounding = {}
    for name, frame in [('result.csv', result), ('submission.csv', submission)]:
        probabilities(frame.pctr.to_numpy(), len(ids), name)
        rounding[name] = float(np.max(np.abs(frame.pctr.to_numpy() - saved_ensemble)))
        if rounding[name] > 5.000001e-7:
            raise AssertionError('CSV six-digit rounding differs')
        with (EXPORT / name).open('rb') as stream:
            if stream.read(3) == b'\xef\xbb\xbf':
                raise AssertionError('Unexpected BOM')
        with (EXPORT / name).open(encoding='utf-8', newline='') as stream:
            reader = csv.reader(stream)
            next(reader)
            if any(len(row) != 2 or not re.fullmatch(r'[01]\.\d{6}', row[1]) for row in reader):
                raise AssertionError('Expected exactly six decimal places')
    np.testing.assert_array_equal(result.pctr.to_numpy(), submission.pctr.to_numpy())
    detail_errors = {'pctr': float(np.max(np.abs(detail.pctr.to_numpy() - saved_ensemble)))}
    for name, values in {**components, 'multiscale_dcn': saved}.items():
        detail_errors[name + '_pctr'] = float(np.max(np.abs(detail[name + '_pctr'].to_numpy() - values)))
    if max(detail_errors.values()) > 1e-14:
        raise AssertionError('Actual details CSV differs from saved component predictions')
    output = {'status': 'passed', 'independent_process': True, 'verification_pid': os.getpid(),
              'full_train_rows': manifest['train_rows'], 'rows': len(ids), 'numeric_dim': 325,
              'full_final_model_reload': True, 'maximum_full_multiscale_reload_error': reload_error,
              'maximum_ensemble_replay_error': ensemble_error, 'maximum_csv_rounding_error': rounding,
              'maximum_detail_error': detail_errors, 'original_ids_and_row_order': True,
              'duplicate_log_id_rows_preserved': int(ids.log_id.duplicated().sum()),
              'submission_ids_are_zero_based_original_row_positions': True,
              'all_166_base_and_159_standardized_features_verified': True, 'all_159_scalers_refit_from_training_prefix': True,
              'budget_fixed_epoch_fraction': .75, 'actual_optimizer_steps': steps,
              'actual_training_rows_seen': steps * BATCH_SIZE, 'actual_fractional_epoch_by_batches': steps / batches,
              'shuffle_order_reproduces': True, 'seed': SEED, 'shuffle_seed': SHUFFLE_SEED,
              'utf8_no_bom': True, 'six_decimals': True, 'all_final_labels_forced_to_minus_one': True,
              'input_model_source_and_export_hashes_match': True, 'old_dcn_tree_and_entire_v2_unchanged': True,
              'final_labels_read': False, 'final_labels_available': False, 'final_auc': None,
              'development_auc': DEVELOPMENT, 'warnings': WARNINGS, 'verification_seconds': time.perf_counter() - started}
    write(EXPORT / 'verification.json', output)
    log('independent_verification_passed', rows=len(ids), maximum_full_multiscale_reload_error=reload_error,
        maximum_ensemble_replay_error=ensemble_error, seconds=time.perf_counter() - started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    verify() if args.verify_only else main()
