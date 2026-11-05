"""Input-only, offline same-day history transitions in empirical log_id order.

The provided complete user/day input may contain information recorded after an
exposure. These features are for that offline assumption, not an online claim.
Ascending and descending log_id are both represented; log_id is not a timestamp.
Advertisement outcomes are never selected by the CSV reader or used by build.
"""
import os

for _variable in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_variable] = '2'

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FAMILIES = [('ad_click_list_v001', 'task_id'),
            ('ad_click_list_v002', 'adv_prim_id'),
            ('ad_click_list_v003', 'spread_app_id')]
INPUT_FIELDS = ['pt_d', 'user_id', 'log_id', 'task_id', 'adv_prim_id', 'spread_app_id', 'slot_id'] + [f for f, _ in FAMILIES]
METRICS = ['candidate_in_own', 'candidate_in_other_snapshot',
           'candidate_absent_own_in_union', 'candidate_next_added',
           'candidate_previous_removed', 'candidate_last_before_addition',
           'candidate_first_after_removal', 'next_add_log_matching_count',
           'previous_remove_log_matching_count',
           'next_add_matching_remaining_fraction',
           'previous_remove_matching_elapsed_fraction',
           'next_add_rank_proximity', 'previous_remove_rank_proximity',
           'next_add_signed_log_id_gap_fraction',
           'previous_remove_signed_log_id_gap_fraction',
           'block_position_fraction', 'block_log_size',
           'ambiguous_snapshot_tie']
FEATURE_NAMES = ['transition__exact_log_id_tie', 'transition__log_id_tie_log_size'] + [
    'transition__' + field + '__' + metric
    for field, _ in FAMILIES for metric in METRICS]


def build_features(frame):
    """Return float32 rows in original frame order; ignore all unselected fields.

    Within a user/day/history, equal log IDs get equal positional ranks. Every
    tied last/first matching exposure receives its respective nearest flag.
    Conflicting history strings at equal log_id suppress all transition/block
    features for that user/day/history and set ambiguous_snapshot_tie instead.
    Membership controls remain available for such ambiguous groups.
    """
    data = frame[INPUT_FIELDS].reset_index(drop=True).copy()
    for field, _ in FAMILIES:
        data[field] = data[field].fillna('').astype(str)
    ordered = data.sort_values(['pt_d', 'user_id', 'log_id'], kind='stable')
    result = np.zeros((len(data), len(FEATURE_NAMES)), dtype='float32')
    if len(data) == 0:
        return result, list(FEATURE_NAMES)
    users = ordered.user_id.to_numpy()
    dates = ordered.pt_d.to_numpy()
    ids = ordered.log_id.to_numpy(dtype='int64')
    boundaries = np.r_[0, np.flatnonzero((users[1:] != users[:-1]) | (dates[1:] != dates[:-1])) + 1, len(data)]
    cache = {}

    def tokens(text):
        if text not in cache:
            cache[text] = frozenset(int(x) for x in text.split('^') if x not in ['', 'nan', 'None', '-1'])
        return cache[text]

    texts = [ordered[field].to_numpy() for field, _ in FAMILIES]
    candidates = [ordered[target].to_numpy() for _, target in FAMILIES]
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        local_ids = ids[begin:end]
        _, tie_rank, tie_counts = np.unique(local_ids, return_inverse=True, return_counts=True)
        result[begin:end, 0] = tie_counts[tie_rank] > 1
        result[begin:end, 1] = np.log1p(tie_counts[tie_rank])
        user_span = max(int(local_ids[-1] - local_ids[0]), 1)
        for family, (all_text, all_candidate) in enumerate(zip(texts, candidates)):
            text = all_text[begin:end]
            candidate = all_candidate[begin:end]
            output = result[begin:end, 2 + family * len(METRICS):2 + (family + 1) * len(METRICS)]
            states = {value: tokens(value) for value in set(text)}
            own = np.fromiter((int(value) in states[state] for value, state in zip(candidate, text)), dtype=bool, count=end - begin)
            frequencies = Counter(token for state in states.values() for token in state)
            counts = np.fromiter((frequencies[int(value)] for value in candidate), dtype='int32', count=end - begin)
            output[:, 0] = own
            output[:, 1] = counts - own.astype('int32') > 0
            output[:, 2] = (counts > 0) & ~own
            ambiguous = bool(np.any((local_ids[1:] == local_ids[:-1]) & (text[1:] != text[:-1])))
            if ambiguous:
                output[:, 17] = 1
                continue
            cuts = np.r_[0, np.flatnonzero(text[1:] != text[:-1]) + 1, end - begin]
            for left, right in zip(cuts[:-1], cuts[1:]):
                rank = tie_rank[left:right] - tie_rank[left]
                output[left:right, 15] = rank / max(int(rank[-1]), 1)
                output[left:right, 16] = np.log1p(right - left)
            for left, middle, right in zip(cuts[:-2], cuts[1:-1], cuts[2:]):
                before, after = states[text[left]], states[text[middle]]
                for token in after - before:
                    matched = np.flatnonzero(candidate[left:middle] == token) + left
                    if not len(matched):
                        continue
                    match_ids = local_ids[matched]
                    _, occurrence_rank = np.unique(match_ids, return_inverse=True)
                    last_rank = int(occurrence_rank[-1])
                    output[matched, 3] = 1
                    output[matched, 5] = match_ids == match_ids[-1]
                    output[matched, 7] = np.log1p(len(matched))
                    output[matched, 9] = (last_rank - occurrence_rank) / max(last_rank, 1)
                    distance = tie_rank[middle - 1] - tie_rank[matched]
                    output[matched, 11] = 1 - distance / max(int(tie_rank[middle - 1] - tie_rank[left]), 1)
                    output[matched, 13] = (local_ids[middle] - match_ids) / user_span
                for token in before - after:
                    matched = np.flatnonzero(candidate[middle:right] == token) + middle
                    if not len(matched):
                        continue
                    match_ids = local_ids[matched]
                    _, occurrence_rank = np.unique(match_ids, return_inverse=True)
                    last_rank = int(occurrence_rank[-1])
                    output[matched, 4] = 1
                    output[matched, 6] = match_ids == match_ids[0]
                    output[matched, 8] = np.log1p(len(matched))
                    output[matched, 10] = occurrence_rank / max(last_rank, 1)
                    distance = tie_rank[matched] - tie_rank[middle]
                    output[matched, 12] = 1 - distance / max(int(tie_rank[right - 1] - tie_rank[middle]), 1)
                    output[matched, 14] = -(match_ids - local_ids[middle - 1]) / user_span
    if not np.isfinite(result).all():
        raise AssertionError('Nonfinite transition features')
    restored = np.empty_like(result)
    restored[ordered.index.to_numpy()] = result
    return restored, list(FEATURE_NAMES)


def read_day(source, day, chunksize):
    dtype = {field: str if 'list' in field else 'int32' for field in INPUT_FIELDS}
    parts, positions, offset = [], [], 0
    for chunk in pd.read_csv(source, usecols=INPUT_FIELDS, dtype=dtype, chunksize=chunksize):
        selected = np.flatnonzero(chunk.pt_d.to_numpy() == day)
        if len(selected):
            parts.append(chunk.iloc[selected].copy())
            positions.append(selected + offset)
        offset += len(chunk)
    return pd.concat(parts, ignore_index=True), np.concatenate(positions)


def main(args):
    if args.test_only:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(HERE), pattern='test_transition_features.py')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    output = args.output
    metadata = output.with_suffix('.json')
    if output.exists() or metadata.exists():
        raise FileExistsError('Preserving existing transition output')
    started = time.perf_counter()
    day_counts = Counter()
    for chunk in pd.read_csv(args.source, usecols=['pt_d'], dtype='int32', chunksize=args.chunksize):
        day_counts.update({int(k): int(v) for k, v in chunk.pt_d.value_counts().items()})
    total = sum(day_counts.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    mapped = open_memmap(output, mode='w+', dtype='float32', shape=(total, len(FEATURE_NAMES)))
    mapped.flush()
    mapped._mmap.close()
    del mapped, chunk
    diagnostic = []
    for day, expected in sorted(day_counts.items()):
        if expected > args.max_day_rows:
            raise ValueError(f'Day {day} exceeds bounded day size: {expected}')
        frame, positions = read_day(args.source, day, args.chunksize)
        if len(frame) != expected:
            raise AssertionError('Day row alignment mismatch')
        values, names = build_features(frame)
        mapped = np.load(output, mmap_mode='r+')
        mapped[positions] = values
        mapped.flush()
        mapped._mmap.close()
        stats = {'day': day, 'rows': len(frame), 'seconds': round(time.perf_counter() - started, 3),
                 'next_add_matches': [int(values[:, names.index('transition__' + field + '__candidate_next_added')].sum()) for field, _ in FAMILIES],
                 'previous_remove_matches': [int(values[:, names.index('transition__' + field + '__candidate_previous_removed')].sum()) for field, _ in FAMILIES]}
        diagnostic.append(stats)
        print('TRANSITION_DAY ' + json.dumps(stats), flush=True)
        del mapped, values, frame, positions
        gc.collect()
    meta = {'shape': [total, len(FEATURE_NAMES)], 'features': FEATURE_NAMES,
            'advertisement_labels_loaded': False, 'input_fields': INPUT_FIELDS,
            'rule': 'Complete same-day offline input only; raw user/day groups in log_id order; both directions; no timestamp or online availability claim; no cross-date inputs',
            'tie_rule': 'Equal log IDs share ranks/nearest flags; history conflict at tied ID suppresses transition/block features for that user/day/history',
            'parser_ignored_tokens': ['', 'nan', 'None', '-1'],
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'seconds': round(time.perf_counter() - started, 3), 'days': diagnostic}
    metadata.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('TRANSITION_COMPLETE ' + json.dumps({k: v for k, v in meta.items() if k not in ['features', 'days']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / '2026ml/train.csv')
    parser.add_argument('--output', type=Path, default=HERE / 'transition_features.npy')
    parser.add_argument('--chunksize', type=int, default=60000)
    parser.add_argument('--max-day-rows', type=int, default=500000)
    parser.add_argument('--test-only', action='store_true')
    main(parser.parse_args())
