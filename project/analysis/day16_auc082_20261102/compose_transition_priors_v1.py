"""Replay frozen strict-prior transition rates against the best completed baseline."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '2'
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash
from verify_stacker import rank_sum_auc
import transition_posterior_v1 as fitted

OUT = HERE / 'runs/compose_transition_priors_v1'
BASE = HERE / 'runs/pointwise_compact_v1'


def dependencies():
    fitted.check_fit()
    assert read(BASE / 'verification.json')['status'] == 'passed'
    paths = [Path(__file__), HERE / 'run_compose_transition_priors_v1.ps1']
    for folder in [fitted.OUT, BASE]:
        paths += [folder / 'protocol.json']
        for filename, expected in read(folder / 'protocol.json')['sources_sha256'].items():
            assert digest(filename) == expected, filename
            paths.append(Path(filename))
    paths += [fitted.OUT / 'rates_locked.json', fitted.OUT / 'inputs.npz', BASE / 'verification.json']
    for day in [15, 16]:
        paths += [BASE / f'day{day}_prediction.npy', BASE / f'day{day}_indices.npy']
    return list(dict.fromkeys(paths))


def check():
    protocol = read(OUT / 'protocol.json')
    for filename, expected in protocol['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    fitted.check_fit()
    return protocol


def inputs(day, saved):
    selected = np.flatnonzero(saved['dates'] == 20191100 + day)
    rows = saved['rows'][selected]
    np.testing.assert_array_equal(rows, np.load(BASE / f'day{day}_indices.npy'))
    return rows, saved['bins'][selected], {'pointwise_compact_v1': np.load(BASE / f'day{day}_prediction.npy')}


def prepare():
    assert not OUT.exists(), 'Preserve previous experiments'
    paths = dependencies()
    OUT.mkdir()
    write(OUT / 'protocol.json', dict(
        baseline=str(BASE), strengths=fitted.STRENGTHS, weights=fitted.WEIGHTS,
        thresholds=[1, 2, 3, 4], new_model_training=False,
        training='Reuses locked rates fitted only on real labels strictly before each prediction date',
        selection='Both dates candidate hashes locked before scoring; dates repeatedly reused',
        evaluation_day_labels_fit=False, inferred17_labels_read=False,
        raw_datasets_sha256=read(fitted.OUT / 'protocol.json')['raw_datasets_sha256'],
        sources_sha256={str(path): digest(path) for path in paths}))
    rates = read(fitted.OUT / 'rates_locked.json')['rates']
    saved = np.load(fitted.OUT / 'inputs.npz')
    locks = {}
    for day in [15, 16]:
        rows, bins, bases = inputs(day, saved)
        np.save(OUT / f'day{day}_indices.npy', rows)
        locks[str(day)] = {fitted.key(r): array_hash(p) for r, p in fitted.candidates(day, bins, bases, rates)}
    write(OUT / 'predictions_locked.json', dict(
        protocol_sha256=digest(OUT / 'protocol.json'), dates=locks, held_labels_opened_for_scoring=False))
    check()
    emit('transition_prior_composition_locked', candidates={d: len(v) for d, v in locks.items()})


def verify():
    check()
    assert not (OUT / 'verification.json').exists(), 'Preserve prior verification'
    locked = read(OUT / 'predictions_locked.json')
    assert locked['protocol_sha256'] == digest(OUT / 'protocol.json')
    rows, dates, labels, bins = fitted.inputs(True)
    saved = np.load(fitted.OUT / 'inputs.npz')
    for name, values in [('rows', rows), ('dates', dates), ('bins', bins)]:
        np.testing.assert_array_equal(saved[name], values)
    rates = read(fitted.OUT / 'rates_locked.json')['rates']
    assert fitted.rates_for(dates, labels, bins, True) == rates
    reports = {}
    for day in [15, 16]:
        local_rows, local_bins, bases = inputs(day, saved)
        y = labels[dates == 20191100 + day]
        assert len(y) == {15: 424780, 16: 421983}[day] and y.sum() == {15: 3523, 16: 3724}[day]
        records, best_value, maximum = [], -1., 0.
        for (record, prediction), (other, replay) in zip(
            fitted.candidates(day, local_bins, bases, rates),
            fitted.candidates(day, local_bins, bases, rates, True)):
            assert record == other
            assert array_hash(prediction) == locked['dates'][str(day)][fitted.key(record)]
            assert np.isfinite(prediction).all() and np.all((prediction > 0) & (prediction < 1))
            error = float(np.max(np.abs(prediction - replay)))
            assert error < 2e-13
            maximum = max(maximum, error)
            value = rank_sum_auc(y, prediction)
            record = dict(record, auc=value)
            records.append(record)
            if value > best_value:
                best_value, best, best_prediction = value, record, prediction.copy()
        assert abs(roc_auc_score(y, best_prediction) - best_value) < 1e-12
        np.save(OUT / f'day{day}_prediction.npy', best_prediction)
        reports[str(20191100 + day)] = dict(
            auc=best_value, best=best, records=records, rows=len(y), clicks=int(y.sum()),
            bin_counts=np.bincount(local_bins, minlength=5).tolist(),
            evaluation_day_labels_fit=False, inferred17_labels_read=False,
            held_labels_used_to_select_fixed_recipe=True,
            all_both_date_candidates_locked_before_scoring=True,
            independent_bins_rates_and_odds_replayed=True,
            maximum_independent_prediction_error=maximum,
            target_met=best_value >= {15: .815, 16: .82}[day])
        emit('transition_prior_composition_scored', day=day, best=best)
    check()
    write(OUT / 'verification.json', dict(status='passed', dates=reports,
        original_datasets_unchanged=True, both_targets_met=all(r['target_met'] for r in reports.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['prepare', 'verify'])
    {'prepare': prepare, 'verify': verify}[parser.parse_args().phase]()
