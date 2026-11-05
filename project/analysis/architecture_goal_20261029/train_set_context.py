"""Train complete user-day exposure sets with row and count supervision.

The full supplied predictor bag is available at inference under the documented
offline task. Labels supervise training rows and count targets only; no labels
enter the bag encoder. Complete user-day bags are never split across batches.
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

os.environ.setdefault('OMP_NUM_THREADS', '3')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / 'analysis/nextday_goal_20261029'
sys.path.insert(0, str(HERE))
from train_architecture import cache_dir, prepare_numeric, read, write, scores
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score


def groups_for_rows(dates, users, rows):
    keys = pd.MultiIndex.from_arrays([dates[rows], users[rows]])
    codes, unique = pd.factorize(keys, sort=False)
    order = np.argsort(codes, kind='stable')
    counts = np.bincount(codes)
    edges = np.r_[0, np.cumsum(counts)]
    return [rows[order[edges[g]:edges[g + 1]]] for g in range(len(counts))]


def pack_groups(groups, order, max_rows, max_groups):
    batch, size = [], 0
    for g in order:
        group = groups[int(g)]
        if batch and (size + len(group) > max_rows or len(batch) >= max_groups):
            yield batch
            batch, size = [], 0
        batch.append(group)
        size += len(group)
    if batch:
        yield batch


def main(args):
    out = HERE / 'results' / args.tag
    if out.exists():
        raise FileExistsError('Preserving set-context run')
    torch.set_num_threads(3)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Set-context training expects GPU')
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    cache = cache_dir(args.day)
    meta = read(cache / 'manifest.json')
    arrays = {name: np.load(cache / (name + '.npy'), mmap_mode='r') for name in
              ['categories', 'sequences', 'date', 'label', 'train_indices', 'valid_indices']}
    train, valid = arrays['train_indices'], arrays['valid_indices']
    if not np.array_equal(train, np.flatnonzero(arrays['date'] < args.day)) or not np.array_equal(valid, np.flatnonzero(arrays['date'] == args.day)):
        raise AssertionError('Require complete earlier-day and held-day rows')
    if args.statistics != 'past':
        raise ValueError('Set mixing requires strictly historical features; row-level OOF can leak across a user-day bag')
    numeric, info = prepare_numeric(args.day, args.statistics, meta, arrays)
    if info.get('per_row_strictly_historical') is not True:
        raise AssertionError('Set-context statistics must precede every row date')
    selected = np.r_[train, valid]
    local_map = np.full(len(arrays['date']), -1, dtype='int32')
    local_map[selected] = np.arange(len(selected))
    raw_identity = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['user_id', 'pt_d'], dtype='int32')
    np.testing.assert_array_equal(raw_identity.pt_d.to_numpy(), arrays['date'])
    users = raw_identity.user_id.to_numpy()
    train_groups = groups_for_rows(arrays['date'], users, train)
    valid_groups = groups_for_rows(arrays['date'], users, valid)
    if sum(map(len, train_groups)) != len(train) or sum(map(len, valid_groups)) != len(valid):
        raise AssertionError('Set partition lost rows')
    np.testing.assert_array_equal(np.sort(np.concatenate(train_groups)), train)
    np.testing.assert_array_equal(np.sort(np.concatenate(valid_groups)), valid)
    for group in train_groups + valid_groups:
        if not np.all(users[group] == users[group[0]]) or not np.all(arrays['date'][group] == arrays['date'][group[0]]):
            raise AssertionError('Set group mixes raw users or dates')
    from set_context_model import SetContextCTR
    if args.allocated:
        from allocated_set_model import AllocatedSetContextCTR
        builder = AllocatedSetContextCTR
    else:
        builder = SetContextCTR
    options = {'numeric_dim': len(info['numeric_fields']), 'use_set_context': not args.row_only}
    model = builder(meta['categorical_fields'], meta['sequence_fields'], meta['group_for_field'],
                     meta['vocabulary_sizes'], initial_ctr=float(arrays['label'][train].mean()), **options).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.0001)
    out.mkdir(parents=True)
    np.save(out / 'valid_indices.npy', valid)
    model_source = HERE / ('allocated_set_model.py' if args.allocated else 'set_context_model.py')
    protocol = {'day': args.day, 'model': builder.__name__, 'options': options, 'statistics': args.statistics,
                'train_rows': len(train), 'valid_rows': len(valid), 'train_groups': len(train_groups),
                'valid_groups': len(valid_groups), 'group_unit': 'user and calendar date',
                'max_rows_per_batch': args.max_rows, 'max_groups_per_batch': args.max_groups,
                'oversize_policy': 'A complete oversized bag is handled alone, no cap/sampling',
                'auxiliary_count_weight': args.count_weight, 'consistency_weight': args.consistency_weight,
                'loss': 'Mean row BCE + count_weight * mean Poisson(loglambda,target_clickcount) + consistency_weight * squared log mass discrepancy',
                'count_loss_weighting': 'Per group, not per exposure; targets outer-training labels only',
                'row_only_control': args.row_only,
                'maximum_training_group_rows': max(map(len, train_groups)),
                'maximum_validation_group_rows': max(map(len, valid_groups)),
                'epochs': args.epochs, 'learning_rate': args.learning_rate, 'seed': args.seed,
                'validation_labels_used_as_inputs_or_optimizer': False, 'final_test_read': False,
                'source_availability': 'Full supplied user-day advertisement input set plus source snapshots; offline assumption',
                'scope': 'Repeated complete next-day development; checkpoint selected by held-day AUC',
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'trainer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'model_sha256': hashlib.sha256(model_source.read_bytes()).hexdigest(),
                'probability_allocation': args.allocated}
    write(out / 'protocol.json', protocol)
    (out / 'trainer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out / 'model_snapshot.py').write_bytes(model_source.read_bytes())
    if args.allocated:
        (out / 'base_model_snapshot.py').write_bytes((HERE / 'set_context_model.py').read_bytes())
    valid_positions = np.full(len(arrays['date']), -1, dtype='int32')
    valid_positions[valid] = np.arange(len(valid))
    def batch_inputs(batch):
        rows = np.concatenate(batch)
        codes = np.repeat(np.arange(len(batch)), [len(group) for group in batch])
        return rows, (torch.as_tensor(arrays['categories'][rows].astype('int64'), device=device),
                      torch.as_tensor(arrays['sequences'][rows].astype('int64'), device=device),
                      torch.as_tensor(np.array(numeric[local_map[rows]]), device=device),
                      torch.as_tensor(codes, dtype=torch.long, device=device))
    def predict():
        model.eval()
        p = np.empty(len(valid), dtype='float32')
        expected, true_count = [], []
        with torch.no_grad():
            for batch in pack_groups(valid_groups, np.arange(len(valid_groups)), args.max_rows, args.max_groups):
                rows, inputs = batch_inputs(batch)
                logits, log_lambda = model(*inputs)
                p[valid_positions[rows]] = torch.sigmoid(logits).cpu().numpy()
                expected.append(log_lambda.exp().cpu().numpy())
                true_count += [float(arrays['label'][group].sum()) for group in batch]
        return p, np.concatenate(expected), np.asarray(true_count)
    y = arrays['label'][valid]
    baseline = np.load(PREVIOUS / f'duplicate_multiplicity/day{args.day % 100}_prediction.npy')
    started = time.perf_counter()
    best, bad = -1., 0
    history = []
    stop = False
    print('SET_TRAIN', len(train_groups), len(valid_groups), len(train), protocol['parameter_count'], flush=True)
    for epoch in range(1, args.epochs + 1):
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_groups))
        batches = list(pack_groups(train_groups, order, args.max_rows, args.max_groups))
        checks = {math.ceil(len(batches) * frac) for frac in [.25, .5, .75, 1.]}
        model.train()
        totals = np.zeros(4)
        seen = 0
        for step, batch in enumerate(batches, 1):
            rows, inputs = batch_inputs(batch)
            target = torch.as_tensor(arrays['label'][rows].astype('float32'), device=device)
            group_count = torch.zeros(len(batch), device=device)
            group_count.index_add_(0, inputs[-1], target)
            optimizer.zero_grad(set_to_none=True)
            logits, log_lambda = model(*inputs)
            row_loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            count_loss = nn.functional.poisson_nll_loss(log_lambda, group_count, log_input=True, full=False)
            mass = torch.zeros(len(batch), device=device)
            mass.index_add_(0, inputs[-1], torch.sigmoid(logits))
            consistency = (torch.log1p(mass) - torch.log1p(log_lambda.exp())).square().mean()
            loss = row_loss + args.count_weight * count_loss + args.consistency_weight * consistency
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite set-context objective')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            totals += np.array([float(loss.detach()), float(row_loss.detach()), float(count_loss.detach()), float(consistency.detach())]) * len(rows)
            seen += len(rows)
            if step not in checks:
                continue
            p, group_prediction, group_target = predict()
            record = {'fractional_epoch': epoch - 1 + step / len(batches), **scores(y, p),
                'training_loss': float(totals[0] / seen), 'row_loss': float(totals[1] / seen),
                'count_loss': float(totals[2] / seen), 'consistency_loss': float(totals[3] / seen),
                'count_anyclick_auc': float(roc_auc_score(group_target > 0, group_prediction)),
                'count_correlation': float(np.corrcoef(group_target, group_prediction)[0, 1]),
                'seconds': time.perf_counter() - started}
            history.append(record)
            write(out / 'history.json', history)
            print('SET_PROGRESS', json.dumps(record), flush=True)
            if record['auc'] > best + 1e-5:
                best, bad = record['auc'], 0
                torch.save({'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                            'manifest': meta, 'options': options, 'record': record}, out / 'model.pt')
                np.save(out / 'validation_prediction.npy', p)
                np.save(out / 'validation_group_expected.npy', group_prediction)
            else:
                bad += 1
            model.train()
            if bad >= 4:
                stop = True
                break
        if stop:
            break
    saved = torch.load(out / 'model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['state_dict'], strict=True)
    p, group_prediction, group_target = predict()
    error = float(np.max(np.abs(p - np.load(out / 'validation_prediction.npy'))))
    if error > 2e-7:
        raise AssertionError('Set-context model replay differs')
    result = {'status': 'complete_and_verified', 'day': args.day, 'model': builder.__name__, **scores(y, p),
              'best_fractional_epoch': saved['record']['fractional_epoch'], 'maximum_reload_error': error,
              'count_anyclick_auc': float(roc_auc_score(group_target > 0, group_prediction)),
              'count_correlation': float(np.corrcoef(group_target, group_prediction)[0, 1]),
              'baseline_auc': float(roc_auc_score(y, baseline)),
              'fixed_blends': [{'weight': weight, **scores(y, (1 - weight) * baseline + weight * p)}
                               for weight in [.05, .1, .2, .3]],
              'goal_met': bool(roc_auc_score(y, p) > .85), 'final_auc': None,
              'train_rows': len(train), 'valid_rows': len(valid), 'seconds': time.perf_counter() - started}
    write(out / 'metrics.json', result)
    print('SET_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=int, choices=[20191115, 20191116], required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--statistics', choices=['past'], default='past')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=.0005)
    parser.add_argument('--max-rows', type=int, default=4096)
    parser.add_argument('--max-groups', type=int, default=64)
    parser.add_argument('--row-only', action='store_true')
    parser.add_argument('--allocated', action='store_true')
    parser.add_argument('--count-weight', type=float, default=.01)
    parser.add_argument('--consistency-weight', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())
