"""Rebuild the original 166 numeric and15 sequence columns for every historical row."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'
import argparse
import ctypes
import gc
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash
from rich_row_catboost import selection
sys.path.insert(0, str(ROOT / 'analysis/strong_interest_control_20261031'))
from features import prepare as original_prepare, cache_dir

OUT = HERE / 'runs/full_population_original_v1'
NATIVE = HERE / 'runs/rich_augmented_rows_v1'
POPULATION = HERE / 'runs/full_negative_context_v1'
REQUIRED_FREE_BYTES = 8 * 1024 ** 3


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in ('total_physical', 'available_physical',
        'total_pagefile', 'available_pagefile', 'total_virtual', 'available_virtual', 'extended_virtual')]


def wait_for_memory():
    last = None
    while True:
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        if status.available_physical >= REQUIRED_FREE_BYTES:
            return
        bucket = int(status.available_physical // 1024 ** 3)
        if bucket != last:
            emit('full_population_original_waiting_for_memory', free_gib=bucket, required_gib=8)
            last = bucket
        time.sleep(5)


def freeze():
    assert not OUT.exists(), 'Preserve completed or partial feature builds'
    assert read(NATIVE / 'verification.json')['status'] == 'passed'
    paths = [Path(__file__), HERE / 'run_full_population_original_v1.ps1',
             HERE / 'fit_stacker.py', HERE / 'rich_userday_activity.py', HERE / 'rich_row_catboost.py',
             HERE / 'conditional_daily.py', NATIVE / 'verification.json', NATIVE / 'protocol.json',
             POPULATION / 'inputs_locked.json', ROOT / 'analysis/auc085/daily_source_profiles.pkl']
    assert paths[-1].is_file(), 'Reuse the verified saved source-day profile cache'
    for day in (15, 16):
        source = read(NATIVE / f'input_contract{day}.json')
        for filename, expected in source['original_contract']['original_feature_metadata_sha256'].items():
            assert digest(filename) == expected, filename
            paths.append(Path(filename))
        cache = cache_dir(20191100 + day)
        paths += [cache / name for name in ('label.npy', 'date.npy', 'train_indices.npy', 'valid_indices.npy',
                                           'manifest.json', 'vocabs.json', 'target_stat_oof_fold.npy')]
        paths += sorted((cache / 'target_stat_models').glob('*.npz'))
        paths += [NATIVE / f'input_contract{day}.json', NATIVE / f'independent_day{day}/features.npy',
                  POPULATION / f'input_day{day}/contract.json']
        paths += [POPULATION / f'input_day{day}/{name}.npy' for name in ('rows', 'train', 'held', 'labels')]
    expected = read(POPULATION / 'protocol.json')['raw_datasets_sha256']
    for filename, value in expected.items():
        assert digest(ROOT / '2026ml' / filename) == value, filename
    OUT.mkdir()
    write(OUT / 'protocol.json', dict(dates=[20191115, 20191116],
        feature_source='Original saved-fold166 numeric columns and15 first-three click-history token codes',
        population='Every prior positive/negative and every held-date row, in original CSV order',
        targets='Only strictly earlier advertisement labels enter target-stat fit; held targets masked in retained core',
        prior_target_statistics='Saved duplicate-group OOF folds, all earlier rows; independently checked saved counts and clicks',
        purpose='Restore the original fine-grained row information to the complete-negative training population',
        inferred17_labels_read=False, held_ad_labels_used_for_feature_fitting=False,
        raw_datasets_sha256=expected, sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)}))


def check():
    protocol = read(OUT / 'protocol.json')
    for filename, expected in protocol['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    return protocol


def build(day, independent=False):
    wait_for_memory()
    destination = OUT / (f'independent_day{day}' if independent else f'input_day{day}')
    destination.mkdir(parents=True, exist_ok=False)
    core = original_prepare(20191100 + day, destination)
    core['label'] = core['label'].astype('int32')
    core['label'][core['valid']] = -1
    assert np.all(core['label'][core['valid']] == -1)
    folder = POPULATION / f'input_day{day}'
    rows = np.load(folder / 'rows.npy')
    train, held = np.load(folder / 'train.npy'), np.load(folder / 'held.npy')
    np.testing.assert_array_equal(core['global_rows'], rows)
    np.testing.assert_array_equal(core['train'], train)
    np.testing.assert_array_equal(core['valid'], held)
    np.testing.assert_array_equal(core['label'][train], np.load(folder / 'labels.npy'))
    numeric = core['numeric']
    tokens = core['sequences'][:, :3, :].reshape(len(rows), 15)
    sequence_names = core['meta']['sequence_fields'][:3]
    assert sequence_names == [f'ad_click_list_v00{i}' for i in (1, 2, 3)]
    names = ['original__' + name for name in core['feature_manifest']['numeric_fields']]
    names += [f'ordered_click_history__{name}__{i}' for name in sequence_names for i in range(5)]
    assert numeric.shape == (len(rows), 166) and len(names) == len(set(names)) == 181
    matrix = np.lib.format.open_memmap(destination / 'features.npy', mode='w+', dtype='float32', shape=(len(rows), 181))
    for start in range(0, len(rows), 10000):
        stop = min(start + 10000, len(rows))
        matrix[start:stop, :166] = numeric[start:stop]
        matrix[start:stop, 166:] = tokens[start:stop]
        np.testing.assert_array_equal(matrix[start:stop, 166:].astype('int32'), tokens[start:stop])
        assert np.isfinite(matrix[start:stop]).all()
    matrix.flush()
    for name, values in [('rows', rows), ('train', train), ('held', held),
                         ('labels', core['label'][train]), ('user_id', core['user_id'])]:
        np.save(destination / (name + '.npy'), values)
    # Every previously audited sample/held row supplies an independent saved reference.
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    chosen, _, held_rows, _, _, _ = selection(day, raw.label.to_numpy(), raw.pt_d.to_numpy())
    reference_rows = np.sort(np.r_[chosen, held_rows])
    positions = np.searchsorted(rows, reference_rows)
    np.testing.assert_array_equal(rows[positions], reference_rows)
    source_contract = read(NATIVE / f'input_contract{day}.json')
    path = NATIVE / f'independent_day{day}/features.npy'
    assert digest(path) == source_contract['matrix_sha256']
    old = np.load(path, mmap_mode='r')
    columns = [source_contract['feature_names'].index(name) for name in names]
    for start in range(0, len(reference_rows), 10000):
        stop = min(start + 10000, len(reference_rows))
        np.testing.assert_array_equal(matrix[positions[start:stop]],
                                      old[np.ix_(np.arange(start, stop), columns)])
    contract = dict(shape=list(matrix.shape), feature_names=names, categorical_columns=list(range(166, 181)),
        matrix_sha256=digest(destination / 'features.npy'),
        rows_sha256=array_hash(rows), training_rows_sha256=array_hash(rows[train]),
        training_labels_sha256=array_hash(core['label'][train]), held_rows_sha256=array_hash(rows[held]),
        training_rows=len(train), held_rows=len(held), max_training_date=int(raw.pt_d.to_numpy()[rows[train]].max()),
        saved_sampled_reference_rows=len(reference_rows), saved_reference_columns=181,
        every_prior_audited_row_matches=True, all_prior_negative_rows_retained=True,
        held_ad_labels_used_for_feature_fitting=False, held_labels_masked_in_retained_core=True,
        source_metadata_sha256=core['feature_manifest']['metadata_sha256'],
        files_sha256={name + '.npy': digest(destination / (name + '.npy'))
                      for name in ('rows', 'train', 'held', 'labels', 'user_id')}, inferred17_labels_read=False)
    assert contract['max_training_date'] < 20191100 + day
    write(destination / 'contract.json', contract)
    emit('full_population_original_ready', day=day, independent=independent, rows=len(rows),
         training_rows=len(train), matched_reference_rows=len(reference_rows), columns=181)
    del core, numeric, tokens, matrix, old, raw
    gc.collect()
    return contract


def prepare():
    freeze()
    records = {}
    for day in (15, 16):
        records[str(day)] = build(day)
        write(OUT / 'inputs_locked.json', records)
    check()


def verify():
    check()
    assert not (OUT / 'verification.json').exists()
    locked = read(OUT / 'inputs_locked.json')
    for day in (15, 16):
        assert build(day, True) == locked[str(day)]
    check()
    write(OUT / 'verification.json', dict(status='passed', feature_only=True,
        full_population_rebuilt_twice=True, all181_columns_match_previous_audited_rows=True,
        dates={str(20191100 + day): dict(rows=locked[str(day)]['shape'][0],
            training_rows=locked[str(day)]['training_rows'], held_rows=locked[str(day)]['held_rows'],
            max_training_date=locked[str(day)]['max_training_date']) for day in (15, 16)},
        original_datasets_unchanged=True, inferred17_labels_read=False))
    emit('full_population_original_verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['prepare', 'verify'])
    {'prepare': prepare, 'verify': verify}[parser.parse_args().phase]()
