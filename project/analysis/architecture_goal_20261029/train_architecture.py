"""Common chronological trainer for gated cross and target-attention structures.

All train rows predate the complete held day. Target statistics are either
strict-past or grouped OOF inside outer training dates; held-day targets enter
metrics only. Input scalers fit only training rows. Final data is never loaded.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key, '3')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / 'analysis/nextday_goal_20261029'
sys.path.insert(0, str(PREVIOUS))
sys.path.insert(0, str(ROOT / 'analysis/model_comparison'))
from train_combined import feature_blocks, read, write, DAILY
from bounded_id_models import BoundedIDDCN
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score


def cache_dir(day):
    return ROOT / ('analysis/user_personalization/cache_day15_stats' if day == 20191115
                   else 'analysis/auc085/cache_rich_stats_time16')


def prepare_numeric(day, statistics, meta, arrays):
    directory = HERE / 'cache' / f'day{day % 100}_{statistics}'
    train, valid = arrays['train_indices'], arrays['valid_indices']
    if (directory / 'manifest.json').exists():
        info = read(directory / 'manifest.json')
        if info['day'] != day or info['statistics'] != statistics or info['train_rows'] != len(train):
            raise AssertionError('Cached architecture input protocol differs')
        np.testing.assert_array_equal(train, np.load(directory / 'train_indices.npy'))
        np.testing.assert_array_equal(valid, np.load(directory / 'valid_indices.npy'))
        return np.load(directory / 'numeric.npy', mmap_mode='r'), info
    if directory.exists():
        raise FileExistsError('Incomplete architecture cache, preserve for diagnosis: ' + str(directory))
    directory.mkdir(parents=True)
    source = (cache_dir(day) / 'numeric.npy' if statistics == 'oof'
              else DAILY / f'cache_day{day % 100}/numeric.npy')
    original = np.load(source, mmap_mode='r')
    blocks = feature_blocks(day, ['history', 'source_context'], len(arrays['date']))
    sources = [(original, meta['numeric_fields'], str(source))] + blocks
    names = [name for _, fields, _ in sources for name in fields]
    selected = np.r_[train, valid]
    matrix = open_memmap(directory / 'numeric.npy', mode='w+', dtype='float32', shape=(len(selected), len(names)))
    scaler = {}
    offset = 0
    for values, fields, path in sources:
        means, stds = np.empty(len(fields)), np.empty(len(fields))
        for column, name in enumerate(fields):
            sample = np.asarray(values[train, column], dtype='float64')
            means[column] = sample.mean()
            stds[column] = max(float(sample.std()), 1e-5)
            scaler[name] = {'mean': float(means[column]), 'std': float(stds[column])}
        for start in range(0, len(selected), 16384):
            stop = min(start + 16384, len(selected))
            block = np.asarray(values[selected[start:stop]], dtype='float32')
            matrix[start:stop, offset:offset + len(fields)] = np.clip((block - means) / stds, -8, 8)
        offset += len(fields)
        matrix.flush()
        print('ARCH_INPUT_BLOCK', path, len(fields), flush=True)
    if len(names) != 500 or offset != 500 or len(set(names)) != 500:
        raise AssertionError('Expected 500 unique numeric features')
    np.save(directory / 'train_indices.npy', train)
    np.save(directory / 'valid_indices.npy', valid)
    info = {'day': day, 'statistics': statistics, 'train_rows': len(train), 'valid_rows': len(valid),
            'numeric_fields': names, 'scalers': scaler, 'clip': [-8, 8],
            'scaler_fit_scope': 'Only outer-training rows', 'base_numeric': str(source),
            'advertisement_validation_labels_used': False,
            'per_row_strictly_historical': statistics == 'past',
            'availability': 'Complete supplied same-day auxiliary snapshot and exposures, offline'}
    write(directory / 'manifest.json', info)
    del original, blocks, sources, sample
    gc.collect()
    return matrix, info


def scores(y, p):
    return {'auc': float(roc_auc_score(y, p)), 'logloss': float(log_loss(y, p)),
            'average_precision': float(average_precision_score(y, p)), 'mean_probability': float(p.mean())}


def main(args):
    out = HERE / 'results' / args.tag
    if out.exists():
        raise FileExistsError('Preserving existing architecture run')
    torch.set_num_threads(3)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Architecture training expects the available GPU')
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    cache = cache_dir(args.day)
    meta = read(cache / 'manifest.json')
    arrays = {name: np.load(cache / (name + '.npy'), mmap_mode='r') for name in
              ['categories', 'sequences', 'date', 'label', 'train_indices', 'valid_indices']}
    train, valid = arrays['train_indices'], arrays['valid_indices']
    if not np.array_equal(train, np.flatnonzero(arrays['date'] < args.day)) or not np.array_equal(valid, np.flatnonzero(arrays['date'] == args.day)):
        raise AssertionError('Require complete future-day split')
    numeric, input_meta = prepare_numeric(args.day, args.statistics, meta, arrays)
    if args.prepare_only:
        return
    feature_columns = np.arange(500) if args.numeric_width == 500 else np.arange(166)
    numeric_fields = [input_meta['numeric_fields'][column] for column in feature_columns]
    mask_spec, mask_protocol = None, None
    if args.mask_close_defaults:
        from history_mask import create_mask_spec, apply_sequence_mask, feature_column_selection, mask_metadata, MISSING_FLAG_NAMES
        mask_spec = create_mask_spec(meta, read(cache / 'vocabs.json'))
        mask_protocol = mask_metadata(mask_spec, numeric_fields)
        keep = feature_column_selection(numeric_fields)
        feature_columns = feature_columns[keep]
        numeric_fields = [numeric_fields[column] for column in keep] + list(MISSING_FLAG_NAMES)
    transition_values, transition_scalers = None, None
    if args.transition:
        transition_meta = read(HERE / 'transition_features.json')
        if transition_meta.get('advertisement_labels_loaded') is not False:
            raise AssertionError('Transition inputs must never load advertisement labels')
        transition_values = np.load(HERE / 'transition_features.npy', mmap_mode='r')
        if len(transition_values) != len(arrays['date']) or transition_values.shape[1] != len(transition_meta['features']):
            raise AssertionError('Transition row/schema alignment differs')
        transition_means = np.empty(transition_values.shape[1], dtype='float32')
        transition_stds = np.empty_like(transition_means)
        transition_scalers = {}
        for column, name in enumerate(transition_meta['features']):
            values = np.asarray(transition_values[train, column], dtype='float64')
            transition_means[column] = values.mean()
            transition_stds[column] = max(float(values.std()), .1)
            transition_scalers[name] = {'mean': float(transition_means[column]), 'std': float(transition_stds[column])}
        numeric_fields += transition_meta['features']
    numeric_dim = len(numeric_fields)
    if args.model == 'dot':
        from dot_interaction_model import DotInteractionCTR
        builder = DotInteractionCTR
        options = {'numeric_dim': numeric_dim}
        model_source = HERE / 'dot_interaction_model.py'
    elif args.model == 'domain':
        from domain_gated_model import DomainGatedCTR
        builder = DomainGatedCTR
        training_dates = np.unique(arrays['date'][train])
        options = {'numeric_dim': numeric_dim, 'domain_classes': len(training_dates)}
        model_source = HERE / 'domain_gated_model.py'
    elif args.model == 'piecewise':
        from piecewise_gated_model import PiecewiseGatedCTR, fit_numeric_bin_edges
        builder = PiecewiseGatedCTR
        sample = np.sort(np.random.default_rng(args.seed).choice(len(train), min(len(train), 120000), replace=False))
        fitting = np.asarray(numeric[sample])[:, feature_columns] * args.numeric_scale
        if mask_spec is not None or transition_values is not None:
            raise ValueError('Piecewise control requires unmodified core feature set')
        bin_edges = [edges.tolist() for edges in fit_numeric_bin_edges(fitting)]
        options = {'numeric_dim': numeric_dim, 'numeric_bin_edges': bin_edges}
        model_source = HERE / 'piecewise_gated_model.py'
    elif args.model == 'dcn':
        builder = BoundedIDDCN
        options = {'numeric_dim': numeric_dim, 'embedding_dim': 8, 'hidden_dims': [128, 64], 'dropout': .15,
                   'unknown_dropout': .01, 'cross_layers': 2,
                   'group_norm_caps': {'user_id': .2, 'ad_task': .2, 'adv_id': .2}}
        model_source = ROOT / 'analysis/model_comparison/bounded_id_models.py'
    else:
        from interaction_models import GatedCrossCTR, AttentiveInteractionCTR, FinalMLPCTR
        builder = {'gated': GatedCrossCTR, 'attention': AttentiveInteractionCTR, 'finalmlp': FinalMLPCTR}[args.model]
        options = {'numeric_dim': numeric_dim}
        if args.id_cap != .2:
            options['group_norm_caps'] = {group: args.id_cap for group in ['user_id', 'ad_task', 'adv_id']}
        if args.id_dropout != .01:
            options['group_dropout_rates'] = {group: args.id_dropout for group in ['user_id', 'ad_task', 'adv_id']}
        model_source = HERE / 'interaction_models.py'
    training_prior = float(arrays['label'][train].mean())
    model_prior = .5 if args.balanced_batches else training_prior
    model = builder(meta['categorical_fields'], meta['sequence_fields'], meta['group_for_field'],
                    meta['vocabulary_sizes'], initial_ctr=model_prior, **options).to(device)
    sampling_logit_offset = math.log(training_prior / (1 - training_prior)) if args.balanced_batches else 0.
    if args.slow_id_factor != 1.:
        slow_groups = ['user_id', 'ad_task', 'adv_id']
        slow_parameters = [model.encoder.embeddings[group].weight for group in slow_groups]
        slow_ids = {id(parameter) for parameter in slow_parameters}
        ordinary_parameters = [parameter for parameter in model.parameters() if id(parameter) not in slow_ids]
        optimizer = torch.optim.AdamW([{'params': ordinary_parameters},
            {'params': slow_parameters, 'lr': args.learning_rate * args.slow_id_factor}],
            lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    out.mkdir(parents=True)
    protocol = {'day': args.day, 'train_rows': len(train), 'valid_rows': len(valid),
                'model': builder.__name__, 'options': options, 'seed': args.seed,
                'input_cache': str((HERE / 'cache' / f'day{args.day % 100}_{args.statistics}').resolve()),
                'numeric_fields': numeric_fields, 'statistics': args.statistics,
                'numeric_scale': args.numeric_scale, 'numeric_width': args.numeric_width,
                'close_history_mask': mask_protocol,
                'transition_features': args.transition, 'transition_scalers': transition_scalers,
                'epochs': args.epochs, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
                'balanced_batches': args.balanced_batches, 'sampling_logit_offset': sampling_logit_offset,
                'batch_sampling': 'Equal positive and negative rows sampled with replacement; steps equivalent to full ordinary epoch' if args.balanced_batches else 'Ordinary shuffled full training rows',
                'weight_decay': args.weight_decay, 'recent_days': args.recent_days,
                'training_date_adversary_weight': args.domain_weight if args.model == 'domain' else 0.,
                'slow_id_learning_rate_factor': args.slow_id_factor,
                'validation_labels_used_for_features_or_optimizer': False,
                'validation_usage': 'Checkpoint selection and metrics only; development date previously reused',
                'final_test_read': False, 'model_source_sha256': hashlib.sha256(model_source.read_bytes()).hexdigest(),
                'trainer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'parameter_count': sum(p.numel() for p in model.parameters())}
    write(out / 'protocol.json', protocol)
    (out / 'trainer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out / 'model_snapshot.py').write_bytes(model_source.read_bytes())
    np.save(out / 'valid_indices.npy', valid)
    ntrain = len(train)
    local_train = np.arange(ntrain)
    if args.recent_days:
        cutoff = int((pd.Timestamp(str(args.day)) - pd.Timedelta(days=args.recent_days)).strftime('%Y%m%d'))
        local_train = local_train[arrays['date'][train] >= cutoff]
    y = arrays['label'][valid]
    def inputs(global_rows, local_rows):
        sequences = arrays['sequences'][global_rows].astype('int64')
        values = np.array(numeric[local_rows])[:, feature_columns]
        if mask_spec is not None:
            sequences, missing = apply_sequence_mask(sequences, mask_spec)
            values = np.concatenate([values, missing], axis=1)
        if transition_values is not None:
            transition = (np.asarray(transition_values[global_rows]) - transition_means) / transition_stds
            values = np.concatenate([values, np.clip(transition, -8, 8)], axis=1)
        return (torch.as_tensor(arrays['categories'][global_rows].astype('int64'), device=device),
                torch.as_tensor(sequences, device=device),
                torch.as_tensor(values * args.numeric_scale, device=device))
    def predict():
        model.eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(valid), args.batch_size):
                stop = min(start + args.batch_size, len(valid))
                logits = model(*inputs(valid[start:stop], slice(ntrain + start, ntrain + stop)))
                parts.append(torch.sigmoid(logits + sampling_logit_offset).cpu().numpy())
        return np.concatenate(parts)
    started = time.perf_counter()
    history, best, bad = [], -1., 0
    stopped = False
    print('ARCH_TRAIN', builder.__name__, len(local_train), len(valid), protocol['parameter_count'], flush=True)
    positive_local = local_train[arrays['label'][train[local_train]] == 1]
    negative_local = local_train[arrays['label'][train[local_train]] == 0]
    for epoch in range(1, args.epochs + 1):
        rng = np.random.default_rng(args.seed + epoch)
        if args.balanced_batches:
            steps = math.ceil(len(local_train) / args.batch_size)
            half = args.batch_size // 2
            sampled = np.empty((steps, args.batch_size), dtype='int64')
            sampled[:, :half] = rng.choice(positive_local, (steps, half), replace=True)
            sampled[:, half:] = rng.choice(negative_local, (steps, args.batch_size - half), replace=True)
            for batch in sampled:
                rng.shuffle(batch)
            order = sampled.reshape(-1)
        else:
            order = rng.permutation(local_train)
        batches = math.ceil(len(order) / args.batch_size)
        checks = {math.ceil(batches * fraction) for fraction in [.25, .5, .75, 1.]}
        total, seen = 0., 0
        model.train()
        for step, start in enumerate(range(0, len(order), args.batch_size), 1):
            local = order[start:start + args.batch_size]
            rows = train[local]
            target = torch.as_tensor(arrays['label'][rows].astype('float32'), device=device)
            optimizer.zero_grad(set_to_none=True)
            if args.model == 'domain':
                logits, domain_logits = model.forward_with_domain(*inputs(rows, local), reversal_strength=1.)
                domain_targets = torch.as_tensor(np.searchsorted(training_dates, arrays['date'][rows]),
                                                  dtype=torch.long, device=device)
                loss = (nn.functional.binary_cross_entropy_with_logits(logits, target)
                        + args.domain_weight * nn.functional.cross_entropy(domain_logits, domain_targets))
            else:
                logits = model(*inputs(rows, local))
                loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite architecture loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
            seen += len(rows)
            if step not in checks:
                continue
            p = predict()
            record = {'fractional_epoch': epoch - 1 + step / batches, **scores(y, p),
                      'train_loss': total / seen, 'seconds': time.perf_counter() - started}
            history.append(record)
            write(out / 'history.json', history)
            print('ARCH_PROGRESS', json.dumps(record), flush=True)
            if record['auc'] > best + 1e-5:
                best, bad = record['auc'], 0
                torch.save({'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                            'manifest': meta, 'options': options, 'model': builder.__name__, 'record': record}, out / 'model.pt')
                np.save(out / 'validation_prediction.npy', p)
            else:
                bad += 1
            model.train()
            if bad >= args.patience:
                stopped = True
                break
        if stopped:
            break
    saved = torch.load(out / 'model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['state_dict'], strict=True)
    p = predict()
    error = float(np.max(np.abs(p - np.load(out / 'validation_prediction.npy'))))
    if error > 2e-7:
        raise AssertionError('Architecture checkpoint replay differs')
    baseline = np.load(PREVIOUS / f'duplicate_multiplicity/day{args.day % 100}_prediction.npy')
    comparisons = [{'weight': weight, **scores(y, (1 - weight) * baseline + weight * p)} for weight in [.05, .1, .2, .3]]
    metrics = {'status': 'complete_and_verified', 'day': args.day, 'model': builder.__name__, **scores(y, p),
               'best_fractional_epoch': saved['record']['fractional_epoch'], 'maximum_reload_error': error,
               'train_rows': len(local_train), 'valid_rows': len(valid), 'validation_clicks': int(y.sum()),
               'seconds': time.perf_counter() - started, 'baseline_auc': float(roc_auc_score(y, baseline)),
               'fixed_blends': comparisons, 'goal_met': bool(roc_auc_score(y, p) > .85), 'final_auc': None}
    write(out / 'metrics.json', metrics)
    print('ARCH_COMPLETE', json.dumps(metrics), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=int, choices=[20191115, 20191116], required=True)
    parser.add_argument('--model', choices=['dcn', 'gated', 'attention', 'finalmlp', 'piecewise', 'domain', 'dot'], required=True)
    parser.add_argument('--statistics', choices=['past', 'oof'], default='oof')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--weight-decay', type=float, default=.0001)
    parser.add_argument('--slow-id-factor', type=float, default=1.)
    parser.add_argument('--id-cap', type=float, default=.2)
    parser.add_argument('--id-dropout', type=float, default=.01)
    parser.add_argument('--numeric-width', type=int, choices=[166, 500], default=500)
    parser.add_argument('--numeric-scale', type=float, default=1.)
    parser.add_argument('--mask-close-defaults', action='store_true')
    parser.add_argument('--transition', action='store_true')
    parser.add_argument('--domain-weight', type=float, default=.01)
    parser.add_argument('--balanced-batches', action='store_true')
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--recent-days', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepare-only', action='store_true')
    main(parser.parse_args())
