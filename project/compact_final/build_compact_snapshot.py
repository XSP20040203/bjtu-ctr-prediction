"""Build a self-contained, final-only replay bundle; never train or delete inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NTRAIN, NFINAL = 2526320, 404182
CACHE = 'analysis/daily_interest_din/full_refit_cache'
V3 = 'output/chronological_multiscale_v3'
NEXT = 'analysis/nextday_goal_20261029/full_best'
ARCH = 'analysis/architecture_goal_20261029/full_refit'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def main(out):
    started = time.perf_counter()
    if out.exists():
        raise FileExistsError('Preserving existing bundle: ' + str(out))
    for name in ['arrays', 'models', 'source', 'metadata', 'reference', 'vendor']:
        (out / name).mkdir(parents=True, exist_ok=True)
    manifest = {'status': 'building', 'format': 'compact_final_v1', 'rows': NFINAL,
                'original_training_rows': NTRAIN, 'final_day': 20191117,
                'final_labels_read': False, 'final_auc': None, 'files': {},
                'source_row_slice': [NTRAIN, NTRAIN + NFINAL],
                'scope': 'Replay the existing 404182 final rows; training arrays deliberately omitted.'}
    def record(target, source=None, **extra):
        key = target.relative_to(out).as_posix()
        manifest['files'][key] = {'sha256': digest(target), 'bytes': target.stat().st_size,
                                  **({'original_source': str(source)} if source else {}), **extra}
    def copy(source, relative):
        source = ROOT / source
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        record(target, source)
        return target
    def matrix(source, name, columns=None):
        path = ROOT / source
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        if len(values) != NTRAIN + NFINAL:
            raise AssertionError('Unexpected original row count: ' + str(path))
        tail = values[NTRAIN:]
        if columns is not None:
            tail = tail[:, columns]
        target = out / 'arrays' / (name + '.npy')
        result = np.lib.format.open_memmap(target, mode='w+', dtype=tail.dtype, shape=tail.shape)
        for start in range(0, NFINAL, 8192):
            stop = min(start + 8192, NFINAL)
            result[start:stop] = tail[start:stop]
            np.testing.assert_array_equal(result[start:stop], tail[start:stop])
        result.flush()
        shape, dtype = list(result.shape), str(result.dtype)
        del result, tail, values
        record(target, path, shape=shape, dtype=dtype,
               original_columns=[columns.start, columns.stop] if columns is not None else 'all')
        print('COPIED', name, shape, flush=True)
    matrix(CACHE + '/categories.npy', 'categories')
    matrix(CACHE + '/sequences.npy', 'sequences')
    matrix(CACHE + '/numeric.npy', 'base166')
    matrix(V3 + '/numeric.npy', 'multiscale_extra159', slice(166, 325))
    matrix(NEXT + '/oof/numeric.npy', 'oof_stats36', slice(39, 75))
    matrix(NEXT + '/tabm/numeric.npy', 'tabm535')
    matrix(ARCH + '/features/numeric_oof500.npy', 'gate_oof500')
    matrix(ARCH + '/features/numeric_past500.npy', 'past500')
    # Verify every deduplicated final value against its original source.
    base = np.load(out / 'arrays/base166.npy', mmap_mode='r')
    tabm = np.load(out / 'arrays/tabm535.npy', mmap_mode='r')
    gate = np.load(out / 'arrays/gate_oof500.npy', mmap_mode='r')
    checks = [(NEXT + '/oof/numeric.npy', slice(0, 39), base, slice(0, 39)),
              (NEXT + '/oof/numeric.npy', slice(75, 166), base, slice(75, 166)),
              (V3 + '/numeric.npy', slice(0, 166), base, slice(0, 166)),
              (ARCH + '/gated166/numeric.npy', slice(0, 166), gate, slice(0, 166)),
              (CACHE + '/numeric.npy', slice(0, 75), tabm, slice(0, 75)),
              (CACHE + '/exposure_raw.npy', slice(0, 43), tabm, slice(75, 118)),
              (CACHE + '/snapshot_raw.npy', slice(0, 48), tabm, slice(118, 166)),
              (V3 + '/multiscale_raw.npy', slice(0, 159), tabm, slice(166, 325))]
    for source, source_cols, compact, compact_cols in checks:
        original = np.load(ROOT / source, mmap_mode='r')
        for start in range(0, NFINAL, 8192):
            stop = min(start + 8192, NFINAL)
            np.testing.assert_array_equal(original[NTRAIN + start:NTRAIN + stop, source_cols],
                                          compact[start:stop, compact_cols])
        del original
        print('DEDUP_VERIFIED', source, source_cols, flush=True)
    del base, tabm, gate
    ids = pd.read_csv(ROOT / '2026ml/final_test.csv', usecols=['log_id', 'user_id', 'pt_d'], dtype='int64')
    if len(ids) != NFINAL or not np.all(ids.pt_d == 20191117):
        raise AssertionError('Unexpected final source rows/date')
    for raw, cached in [('log_id', 'log_id'), ('user_id', 'user_id'), ('pt_d', 'date')]:
        values = np.load(ROOT / CACHE / (cached + '.npy'), mmap_mode='r')
        np.testing.assert_array_equal(ids[raw].to_numpy(), values[NTRAIN:])
    ids = ids[['log_id', 'user_id', 'pt_d']].to_numpy(dtype='<i8')
    np.save(out / 'arrays/ids.npy', ids)
    record(out / 'arrays/ids.npy', ROOT / '2026ml/final_test.csv', columns=['log_id', 'user_id', 'pt_d'])
    copy('output/chronological_duplicate_v5/context_count.npy', 'arrays/context_count.npy')
    model_sources = {
        'dcn': 'analysis/daily_interest_din/full_refit_models/dcn.pt',
        'tree': 'analysis/daily_interest_din/full_refit_models/catboost.cbm',
        'lgb': 'output/chronological_lgb_v2/lgb_model.txt',
        'multiscale_dcn': V3 + '/multiscale_dcn.pt',
        'shared_lgb': 'output/chronological_shared_lgb_v4/shared_lgb_model.txt',
        'oof': NEXT + '/oof/model.pt', 'tabm': NEXT + '/tabm/model.pt',
        'gated166': ARCH + '/gated166/model.pt', 'gated500': ARCH + '/gated500/model.pt',
        'allocated': ARCH + '/allocated/model.pt'}
    manifest['models'] = {}
    for name, source in model_sources.items():
        extension = Path(source).suffix
        target = copy(source, 'models/' + name + extension)
        manifest['models'][name] = target.relative_to(out).as_posix()
    references = {
        'dcn': 'analysis/daily_interest_din/full_refit_models/dcn_final_prediction.npy',
        'tree': 'analysis/daily_interest_din/full_refit_models/tree_final_prediction.npy',
        'lgb': 'output/chronological_lgb_v2/lgb_final_prediction.npy',
        'multiscale_dcn': V3 + '/multiscale_dcn_final_prediction.npy',
        'shared_lgb': 'output/chronological_shared_lgb_v4/shared_lgb_final_prediction.npy',
        'oof': NEXT + '/oof/final_prediction.npy', 'tabm': NEXT + '/tabm/final_prediction.npy',
        'gated166': ARCH + '/gated166/final_prediction.npy',
        'gated500': ARCH + '/gated500/final_prediction.npy', 'allocated': ARCH + '/allocated/final_prediction.npy',
        'v4': 'output/chronological_shared_lgb_v4/ensemble_final_prediction.npy',
        'v6': 'output/best_nextday_full_v6/ensemble_final_prediction.npy',
        'v7': 'output/best_nextday_arch_v7/ensemble_final_prediction.npy'}
    for name, source in references.items():
        copy(source, 'reference/' + name + '.npy')
    for name in ['result.csv', 'submission.csv']:
        copy('output/best_nextday_arch_v7/' + name, 'reference/v7_' + name)
    metadata = {
        'cache_manifest.json': CACHE + '/manifest.json', 'vocabs.json': CACHE + '/vocabs.json',
        'lgb_protocol.json': 'output/chronological_lgb_v2/protocol.json',
        'shared_lgb_protocol.json': 'output/chronological_shared_lgb_v4/protocol.json',
        'multiscale_numeric_meta.json': V3 + '/numeric_meta.json',
        'architecture_features.json': ARCH + '/features/manifest.json',
        'gated166_input.json': ARCH + '/gated166/input_manifest.json',
        'tabm_protocol.json': NEXT + '/tabm/protocol.json',
        'oof_manifest.json': NEXT + '/oof/manifest.json',
        'duplicate_selection.json': 'analysis/nextday_goal_20261029/duplicate_multiplicity/selection_frozen.json'}
    for target, source in metadata.items():
        copy(source, 'metadata/' + target)
    for name in ['gated166', 'gated500', 'allocated']:
        copy(ARCH + '/' + name + '/protocol.json', 'metadata/' + name + '_protocol.json')
        copy(ARCH + '/' + name + '/verification.json', 'metadata/' + name + '_verification.json')
    sources = {
        'bounded_id_models.py': 'analysis/model_comparison/bounded_id_models.py',
        'id_regularized_models.py': 'analysis/model_comparison/id_regularized_models.py',
        'mainstream_models.py': 'analysis/model_comparison/mainstream_models.py',
        'benchmark_core.py': 'analysis/model_comparison/benchmark_core.py',
        'modern_architectures.py': 'analysis/auc085/modern_architectures.py',
        'interaction_models.py': 'analysis/architecture_goal_20261029/interaction_models.py',
        'set_context_model.py': 'analysis/architecture_goal_20261029/set_context_model.py',
        'allocated_set_model.py': 'analysis/architecture_goal_20261029/allocated_set_model.py',
        'gated166_snapshot.py': ARCH + '/gated166/model_snapshot.py',
        'gated500_snapshot.py': ARCH + '/gated500/model_snapshot.py'}
    for target, source in sources.items():
        copy(source, 'source/' + target)
    # LightGBM is project-local; preserve its native runtime inside the bundle.
    shutil.copytree(ROOT / 'analysis/auc085_next/deps/lightgbm', out / 'vendor/lightgbm')
    for path in sorted((out / 'vendor').rglob('*')):
        if path.is_file():
            record(path)
    copy('analysis/compact_final_20261031/replay_compact.py', 'replay_compact.py')
    copy('analysis/compact_final_20261031/build_compact.py', 'build_compact_snapshot.py')
    manifest['status'] = 'complete'
    manifest['all_deduplicated_final_values_exactly_verified'] = True
    manifest['total_bytes'] = sum(item['bytes'] for item in manifest['files'].values())
    manifest['build_seconds'] = time.perf_counter() - started
    manifest['numeric_inputs'] = {
        'dcn': 'base166', 'tree': 'categories27 + tabm535[:75] + sequences60 + tabm535[75:166]',
        'lgb': 'same 253-column tree frame, original integer category IDs',
        'multiscale_dcn': 'base166 + multiscale_extra159',
        'shared_lgb': 'categories24 excluding user_id/task_id/adv_id + tabm535[:325]',
        'oof': 'base166[:39] + oof_stats36 + base166[75:166]', 'tabm': 'tabm535',
        'gated166': 'gate_oof500[:166]', 'gated500': 'gate_oof500', 'allocated': 'past500'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('COMPACT_BUILD_COMPLETE', manifest['total_bytes'], manifest['build_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'compact_final')
    main(parser.parse_args().output.resolve())
