"""Verify input-only transition row alignment on complete user/day groups."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from prepare_transition_features import build_features, INPUT_FIELDS, FAMILIES


def main():
    meta = json.loads((HERE / 'transition_features.json').read_text(encoding='utf-8'))
    if meta['advertisement_labels_loaded'] is not False:
        raise AssertionError('Transition targets loaded')
    source_hash = hashlib.sha256((HERE / 'prepare_transition_features.py').read_bytes()).hexdigest()
    if meta['source_sha256'] != source_hash:
        raise AssertionError('Transition source changed after build')
    values = np.load(HERE / 'transition_features.npy', mmap_mode='r')
    # Selection depends only on input feature values, never labels.
    selected = np.unique(np.r_[np.arange(7), np.flatnonzero(values[:, 7] > .5)[::40],
                                np.flatnonzero(values[:, 19] > .5)[::100]])[:80]
    keys = set()
    offset = 0
    for chunk in pd.read_csv(ROOT / '2026ml/train.csv', usecols=['user_id', 'pt_d'], dtype='int32', chunksize=60000):
        probes = selected[(selected >= offset) & (selected < offset + len(chunk))] - offset
        for row in probes:
            keys.add(tuple(chunk.iloc[int(row)][['user_id', 'pt_d']]))
        offset += len(chunk)
    parts, positions = [], []
    offset = 0
    dtypes = {field: str if 'list' in field else 'int32' for field in INPUT_FIELDS}
    for chunk in pd.read_csv(ROOT / '2026ml/train.csv', usecols=INPUT_FIELDS, dtype=dtypes, chunksize=60000):
        take = np.fromiter((key in keys for key in zip(chunk.user_id, chunk.pt_d)), dtype=bool, count=len(chunk))
        if take.any():
            parts.append(chunk.loc[take])
            positions.append(np.flatnonzero(take) + offset)
        offset += len(chunk)
    raw = pd.concat(parts, ignore_index=True)
    rows = np.concatenate(positions)
    fresh, names = build_features(raw)
    np.testing.assert_array_equal(fresh, values[rows])
    shuffled = raw.sample(frac=1., random_state=42)
    permuted, _ = build_features(shuffled)
    np.testing.assert_array_equal(permuted, fresh[shuffled.index.to_numpy()])
    raw_label_changed = raw.assign(label=-999)
    independent, _ = build_features(raw_label_changed)
    np.testing.assert_array_equal(independent, fresh)
    snapshot = np.load(ROOT / 'analysis/daily_interest_din/snapshot_features.npy', mmap_mode='r')
    snapshot_names = json.loads((ROOT / 'analysis/daily_interest_din/snapshot_features.json').read_text(encoding='utf-8'))['features']
    comparisons = {}
    for i, (field, _) in enumerate(FAMILIES, 1):
        for transition_name, snapshot_name in [('candidate_in_own', None), ('candidate_in_other_snapshot', f'snapshot_click_{i}__candidate_in_other_snapshot')]:
            if snapshot_name is None:
                continue
            a = fresh[:, names.index('transition__' + field + '__' + transition_name)]
            b = snapshot[rows, snapshot_names.index(snapshot_name)]
            np.testing.assert_array_equal(a, b)
            comparisons[field + '__' + transition_name] = 0.
    result = {'status': 'passed', 'complete_groups': len(keys), 'rows': len(rows), 'features': len(names),
              'maximum_fresh_vs_saved_error': 0., 'permutation_invariance': True, 'label_independence': True,
              'snapshot_membership_control_errors': comparisons, 'source_sha256': source_hash,
              'original_global_position_alignment': True, 'final_test_read': False,
              'scope': 'Complete supplied offline user/day input; log_id empirical sort, no timestamp claim'}
    (HERE / 'transition_build_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
