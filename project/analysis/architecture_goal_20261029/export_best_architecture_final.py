"""Export newest verified complete architecture blend after all full refits.

Selected fixed formula: .7*(.9*(.8*v6+.2*gated500)+.1*allocated)+.3*gated166.
The v6 component already includes input multiplicity correction; no second
correction is applied. Final targets are never accessed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FULL = HERE / 'full_refit'
V6 = ROOT / 'output/best_nextday_full_v6'
OUT = ROOT / 'output/best_nextday_arch_v7'
FORMULA = '.7*(.9*(.8*v6+.2*gated500)+.1*allocated)+.3*gated166'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def blend(v6, gate500, allocated, gate166):
    return .7 * (.9 * (.8 * v6 + .2 * gate500) + .1 * allocated) + .3 * gate166


def check_prob(p, rows=404182):
    p = np.asarray(p)
    if p.shape != (rows,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise AssertionError('Invalid probability shape, values or range')
    return p


def bound_digest(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or digest(path) != expected:
        raise AssertionError('Artifact differs from its recorded hash: ' + str(path))


def probability_error(expected, actual, rows):
    expected = check_prob(expected, rows)
    actual = check_prob(actual, rows)
    error = float(np.max(np.abs(expected - actual)))
    if not np.isfinite(error):
        raise AssertionError('Nonfinite probability comparison')
    return error


def source_predictions():
    prior_verification = read(V6 / 'verification.json')
    prior_manifest = read(V6 / 'manifest.json')
    if prior_verification['status'] != 'passed' or prior_verification.get('final_label_column_read') is not False:
        raise AssertionError('Prior full best not verified')
    prior_path = V6 / 'ensemble_final_prediction.npy'
    bound_digest(prior_path, prior_manifest['files']['ensemble_final_prediction.npy']['sha256'])
    export_manifest = read(OUT / 'manifest.json') if (OUT / 'manifest.json').exists() else None
    result = {'v6': check_prob(np.load(prior_path))}
    for name in ['gated500', 'allocated', 'gated166']:
        folder = FULL / name
        verification = read(folder / 'verification.json')
        if (verification['status'] != 'complete_and_verified' or verification['final_labels_read'] is not False
                or verification.get('final_rows') != 404182 or verification.get('train_rows') != 2526320):
            raise AssertionError('Full-refit component verification differs: ' + name)
        bound_digest(folder / 'model.pt', verification['model_sha256'])
        prediction_path = folder / 'final_prediction.npy'
        expected_prediction = verification.get('prediction_sha256')
        if expected_prediction is None:
            # Legacy gate artifacts predate prediction hashes. Bind them to the
            # already-delivered export instead of trusting their current bytes.
            if export_manifest is None:
                raise AssertionError('Component requires a recorded prediction hash: ' + name)
            record = export_manifest['components'][name]
            if Path(record['path']).resolve() != prediction_path.resolve():
                raise AssertionError('Recorded component path differs: ' + name)
            expected_prediction = record['sha256']
        bound_digest(prediction_path, expected_prediction)
        result[name] = check_prob(np.load(prediction_path))
    if export_manifest is not None:
        paths = {'v6': prior_path, **{name: FULL / name / 'final_prediction.npy'
                                   for name in ['gated500', 'allocated', 'gated166']}}
        for name, path in paths.items():
            record = export_manifest['components'][name]
            if Path(record['path']).resolve() != path.resolve():
                raise AssertionError('Export source path differs: ' + name)
            bound_digest(path, record['sha256'])
    return result


def final_ids():
    ids = pd.read_csv(ROOT / '2026ml/final_test.csv', usecols=['log_id', 'user_id', 'pt_d'], dtype='int64')
    if len(ids) != 404182 or not np.all(ids.pt_d == 20191117):
        raise AssertionError('Final row/date differs')
    cache = ROOT / 'analysis/daily_interest_din/full_refit_cache'
    cache_files = [cache / f'{name}.npy' for name in ['log_id', 'user_id', 'date']]
    if all(path.is_file() for path in cache_files):
        meta = read(cache / 'manifest.json')
        for field, cached in [('log_id', 'log_id'), ('user_id', 'user_id'), ('pt_d', 'date')]:
            np.testing.assert_array_equal(ids[field], np.load(cache / f'{cached}.npy', mmap_mode='r')[meta['train_rows']:])
    else:
        # Large training caches may have been explicitly cleaned. The compact
        # bundle keeps exact final identities bound to its recorded hash.
        compact = ROOT / 'compact_final'
        compact_manifest = read(compact / 'manifest.json')
        identity_record = compact_manifest['files']['arrays/ids.npy']
        identity_path = compact / 'arrays/ids.npy'
        bound_digest(identity_path, identity_record['sha256'])
        compact_ids = np.load(identity_path, mmap_mode='r')
        np.testing.assert_array_equal(compact_ids, ids[['log_id', 'user_id', 'pt_d']].to_numpy())
    details = pd.read_csv(V6 / 'prediction_details.csv', usecols=['id', 'log_id', 'user_id', 'pt_d'])
    np.testing.assert_array_equal(details.id, np.arange(len(ids)))
    for name in ['log_id', 'user_id', 'pt_d']:
        np.testing.assert_array_equal(details[name], ids[name])
    return ids


def development():
    evaluation = ROOT / 'analysis/auc085_next/evaluation'
    prior = ROOT / 'analysis/nextday_goal_20261029/duplicate_multiplicity'
    result = {}
    for day in [15, 16]:
        rows = np.load(evaluation / f'day{day}_valid_indices.npy')
        y = np.load(evaluation / f'day{day}_labels.npy')
        parts = []
        for tag in [f'gated_oof_day{day}', f'set_allocated_day{day}', f'gated166_day{day}']:
            folder = HERE / 'results' / tag
            np.testing.assert_array_equal(rows, np.load(folder / 'valid_indices.npy'))
            if read(folder / 'metrics.json')['status'] != 'complete_and_verified':
                raise AssertionError('Unverified selected component')
            parts.append(np.load(folder / 'validation_prediction.npy'))
        p = blend(np.load(prior / f'day{day}_prediction.npy'), *parts)
        expected = np.load(HERE / f'fixed_gated20_allocated10_core30_day{day}.npy')
        error = probability_error(p, expected, len(y))
        if error > 1e-14:
            raise AssertionError('Selected fixed-development formula mismatch')
        result[str(20191100 + day)] = {'auc': float(roc_auc_score(y, p)), 'logloss': float(log_loss(y, p)),
            'rows': len(y), 'clicks': int(y.sum()), 'formula_replay_error': error}
    return result


def _verify():
    manifest = read(OUT / 'manifest.json')
    values = source_predictions()
    ids = final_ids()
    expected = check_prob(blend(values['v6'], values['gated500'], values['allocated'], values['gated166']))
    p = check_prob(np.load(OUT / 'ensemble_final_prediction.npy'))
    error = probability_error(expected, p, len(ids))
    if error > 1e-15:
        raise AssertionError('Full blend independent replay mismatch')
    submission = pd.read_csv(OUT / 'submission.csv', float_precision='round_trip')
    result = pd.read_csv(OUT / 'result.csv', float_precision='round_trip')
    if submission.columns.tolist() != ['id', 'pctr'] or result.columns.tolist() != ['log_id', 'pctr']:
        raise AssertionError('Submission schema mismatch')
    np.testing.assert_array_equal(submission.id, np.arange(len(ids)))
    np.testing.assert_array_equal(result.log_id, ids.log_id)
    errors = {name: probability_error(p, frame.pctr.to_numpy(), len(ids))
              for name, frame in [('submission.csv', submission), ('result.csv', result)]}
    if not all(np.isfinite(error) for error in errors.values()) or max(errors.values()) > 5.000001e-7:
        raise AssertionError('CSV rounding exceeds six-decimal tolerance')
    details = pd.read_csv(OUT / 'prediction_details.csv', float_precision='round_trip')
    for name in ['log_id', 'user_id', 'pt_d']:
        np.testing.assert_array_equal(details[name], ids[name])
    np.testing.assert_allclose(details.pctr.to_numpy(), p, rtol=0, atol=1e-15)
    for name, record in manifest['files'].items():
        if digest(OUT / name) != record['sha256']:
            raise AssertionError('Output file hash mismatch')
    for name, record in manifest['components'].items():
        if digest(Path(record['path'])) != record['sha256']:
            raise AssertionError('Source prediction hash changed: ' + name)
    record = {'status': 'passed', 'independent_process': True, 'rows': len(ids),
        'source_day': 20191117, 'formula': FORMULA, 'maximum_formula_replay_error': error,
        'maximum_csv_rounding_error': errors, 'all_output_hashes_match': True,
        'full_source_predictions_unchanged': True, 'raw_final_and_cache_row_order_verified': True,
        'duplicate_log_ids_preserved': int(ids.log_id.duplicated().sum()),
        'probabilities_finite_in_range': True, 'final_label_column_read': False,
        'development': development(), 'final_auc': None, 'goal_auc085_met': False,
        'original_source_hashes_bound': True, 'finite_comparisons_enforced': True,
        'verifier_sha256': digest(Path(__file__))}
    write(OUT / 'verification.json', record)
    print('ARCH_FINAL_VERIFIED', json.dumps(record), flush=True)


def verify():
    # Invalidate an earlier pass before reading any mutable source. A failed or
    # interrupted recheck must never leave stale status='passed' behind.
    verification_path = OUT / 'verification.json'
    if verification_path.exists():
        read(verification_path)
    write(verification_path, {'status': 'checking', 'verifier_sha256': digest(Path(__file__))})
    try:
        _verify()
    except BaseException as error:
        write(verification_path, {'status': 'failed', 'error_type': type(error).__name__,
                                 'message': str(error), 'verifier_sha256': digest(Path(__file__))})
        raise


def main():
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError('Preserving existing final architecture export')
    values = source_predictions()
    ids = final_ids()
    p = check_prob(blend(values['v6'], values['gated500'], values['allocated'], values['gated166']))
    dev = development()
    OUT.mkdir(parents=True)
    np.save(OUT / 'ensemble_final_prediction.npy', p)
    pd.DataFrame({'id': np.arange(len(ids)), 'pctr': p}).to_csv(OUT / 'submission.csv', index=False, float_format='%.6f', encoding='utf-8')
    pd.DataFrame({'log_id': ids.log_id, 'pctr': p}).to_csv(OUT / 'result.csv', index=False, float_format='%.6f', encoding='utf-8')
    details = pd.DataFrame({'id': np.arange(len(ids))}).join(ids)
    details['pctr'] = p
    for name, prediction in values.items():
        details[name + '_pctr'] = prediction
    details.to_csv(OUT / 'prediction_details.csv', index=False, float_format='%.17g', encoding='utf-8')
    note = ['# 最新最佳隔天结构融合提交', '', '**请提交 submission.csv**：id,pctr，共404182行，id为final_test.csv原始行的0起始序号。',
        '', '新增门控500维、用户日点击分配、门控166维模型按开发结果确定的固定预算，用全部2526320条有标签数据重训；此前v6全量预测复用。',
        '', '固定融合：' + FORMULA + '。v6内部已做输入重复次数校正，本层不重复校正。',
        '', '对应完整隔天开发AUC：11月15日0.7921069471；11月16日0.7998466417。验证日期已反复参与开发，最终AUC未知，不能保证0.85。',
        '', 'final标签未读取，原始顺序和重复log_id保留。独立进程重新计算融合并核验CSV、哈希和行对齐。',
        '', 'result.csv提供log_id,pctr版本；prediction_details.csv提供各组件概率。',
        '', '当前推荐使用本v7提交，旧v6文件保留供对照。']
    (OUT / '提交说明.md').write_text('\n'.join(note) + '\n', encoding='utf-8')
    manifest = {'status': 'exported', 'model_selection': 'Highest complete fixed blend on latest confirmation day among completed experiments',
        'formula': FORMULA, 'expanded_weights': {'v6': .504, 'gated500': .126, 'allocated': .07, 'gated166': .3},
        'train_rows': 2526320, 'final_rows': len(ids), 'development': dev,
        'full_refit_protocols': {name: read(FULL / name / 'protocol.json') for name in ['gated500', 'allocated', 'gated166']},
        'components': {name: {'path': str(path.resolve()), 'sha256': digest(path)} for name, path in
            {'v6': V6 / 'ensemble_final_prediction.npy', 'gated500': FULL / 'gated500/final_prediction.npy',
             'allocated': FULL / 'allocated/final_prediction.npy', 'gated166': FULL / 'gated166/final_prediction.npy'}.items()},
        'files': {name: {'sha256': digest(OUT / name), 'bytes': (OUT / name).stat().st_size} for name in
            ['submission.csv', 'result.csv', 'prediction_details.csv', 'ensemble_final_prediction.npy', '提交说明.md']},
        'probability': {'mean': float(p.mean()), 'minimum': float(p.min()), 'maximum': float(p.max())},
        'id_rule': 'Zero-based original final row position; preserve nonunique log_id order',
        'final_label_column_read': False, 'final_auc': None, 'goal_auc085_met': False,
        'exporter_sha256': digest(Path(__file__)), 'seconds': time.perf_counter() - started}
    write(OUT / 'manifest.json', manifest)
    print('ARCH_FINAL_EXPORTED', len(ids), flush=True)
    subprocess.run([sys.executable, '-u', str(Path(__file__)), '--verify'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    verify() if args.verify else main()
