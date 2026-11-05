"""Input-only adjacency in user/day log-id order, not assumed exposure timestamps."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIELDS = ['task_id', 'adv_id', 'adv_prim_id', 'spread_app_id', 'slot_id',
          'creat_type_cd', 'inter_type_cd', 'device_size', 'net_type']


def build_features(frame):
    data = frame[['pt_d', 'user_id', 'log_id'] + FIELDS].reset_index(drop=True)
    ordered = data.sort_values(['pt_d', 'user_id', 'log_id'], kind='stable')
    grouped = ordered.groupby(['pt_d', 'user_id'], sort=False)
    columns = {}
    ordinal = grouped.cumcount().to_numpy(dtype=np.float32)
    size = grouped.log_id.transform('size').to_numpy(dtype=np.float32)
    columns['local__position_fraction'] = ordinal / np.maximum(size - 1, 1)
    columns['local__log_position'] = np.log1p(ordinal)
    columns['local__log_remaining'] = np.log1p(size - ordinal - 1)
    ids = ordered.log_id.to_numpy()
    for direction, periods in [('previous', 1), ('next', -1)]:
        other = grouped.log_id.shift(periods).to_numpy()
        columns[f'local__{direction}_missing'] = np.isnan(other).astype(np.float32)
        columns[f'local__{direction}_log_gap'] = np.log1p(np.minimum(np.nan_to_num(np.abs(ids - other), nan=100000), 100000))
    for field in FIELDS:
        values = ordered[field]
        previous = grouped[field].shift(1)
        following = grouped[field].shift(-1)
        prefix = 'local__' + field
        columns[prefix + '__same_previous'] = (values == previous).to_numpy(dtype=np.float32)
        columns[prefix + '__same_next'] = (values == following).to_numpy(dtype=np.float32)
        run = (values != previous).cumsum()
        runs = ordered.groupby(run, sort=False)
        run_size = runs[field].transform('size').to_numpy(dtype=np.float32)
        run_position = runs.cumcount().to_numpy(dtype=np.float32)
        columns[prefix + '__log_run_size'] = np.log1p(run_size)
        columns[prefix + '__log_run_position'] = np.log1p(run_position)
        columns[prefix + '__log_run_remaining'] = np.log1p(run_size - run_position - 1)
        same = ordered.groupby(['pt_d', 'user_id', field], sort=False)
        occurrence = same.cumcount().to_numpy(dtype=np.float32)
        count = same.log_id.transform('size').to_numpy(dtype=np.float32)
        columns[prefix + '__occurrence_fraction'] = occurrence / np.maximum(count - 1, 1)
        columns[prefix + '__log_occurrence'] = np.log1p(occurrence)
        columns[prefix + '__log_remaining_occurrences'] = np.log1p(count - occurrence - 1)
        for direction, periods in [('previous', 1), ('next', -1)]:
            other = same.log_id.shift(periods).to_numpy()
            missing = np.isnan(other)
            gap = np.minimum(np.nan_to_num(np.abs(ids - other), nan=100000), 100000)
            columns[prefix + '__' + direction + '_missing'] = missing.astype(np.float32)
            columns[prefix + '__' + direction + '_log_gap'] = np.log1p(gap)
        for distance in [2, 3]:
            columns[prefix + f'__same_previous{distance}'] = (values == grouped[field].shift(distance)).to_numpy(dtype=np.float32)
            columns[prefix + f'__same_next{distance}'] = (values == grouped[field].shift(-distance)).to_numpy(dtype=np.float32)
    result = np.column_stack(list(columns.values())).astype(np.float32)
    inverse = np.empty(len(ordered), dtype=np.int64)
    inverse[ordered.index.to_numpy()] = np.arange(len(ordered))
    result = result[inverse]
    if not np.isfinite(result).all():
        raise AssertionError('Nonfinite local structure')
    return result, list(columns)


def check_boundaries():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({'pt_d': [1, 1, 1, 1, 2, 2], 'user_id': [2, 2, 2, 3, 2, 2],
                          'log_id': [4, 6, 5, 5, 4, 5], **{f: [1, 1, 2, 1, 3, 3] for f in FIELDS}})
    first, names = build_features(frame.assign(label=rng.integers(0, 2, len(frame))))
    second, _ = build_features(frame.assign(label=1))
    np.testing.assert_array_equal(first, second)
    permutation = [5, 0, 3, 2, 4, 1]
    shuffled, _ = build_features(frame.iloc[permutation])
    np.testing.assert_array_equal(shuffled, first[permutation])
    changed = frame.copy()
    changed.loc[changed.pt_d == 2, 'task_id'] = 999
    future, _ = build_features(changed)
    np.testing.assert_array_equal(first[frame.pt_d == 1], future[frame.pt_d == 1])
    assert first[0, names.index('local__task_id__same_next')] == 0
    assert first[1, names.index('local__task_id__same_previous')] == 0
    assert first[4, names.index('local__task_id__same_previous')] == 0
    print('LOCAL_BOUNDARIES_OK label independence, row alignment, user/day boundaries', flush=True)


def main(args):
    check_boundaries()
    if args.test_only:
        return
    if args.output.exists():
        raise FileExistsError('Preserving local features')
    frame = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'user_id', 'log_id'] + FIELDS, dtype='int32')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = None
    names = None
    for day, rows in frame.groupby('pt_d', sort=True).groups.items():
        values, current_names = build_features(frame.loc[rows])
        if result is None:
            names = current_names
            result = open_memmap(args.output, mode='w+', dtype='float32', shape=(len(frame), len(names)))
        if names != current_names:
            raise AssertionError('Column order changed')
        result[np.asarray(rows)] = values
        result.flush()
        print(f'LOCAL_DAY {day} rows={len(rows)} fields={len(names)}', flush=True)
    meta = {'shape': list(result.shape), 'features': names, 'advertisement_labels_loaded': False,
            'rule': 'Within provided full user/day input, stable log_id ordering; no outcome or cross-day neighbors; IDs not asserted to be timestamps',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('LOCAL_COMPLETE ' + json.dumps({k: v for k, v in meta.items() if k != 'features'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'local_structure.npy')
    parser.add_argument('--test-only', action='store_true')
    main(parser.parse_args())
