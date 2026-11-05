"""Recheck frozen model-family means against the latest neural/cohort winner."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash
from full_negative_mask_core_v1 import masks as historical_masks
import compose_family_quantiles_v1 as engine

OUT = HERE / 'runs/refresh_family_quantiles_v1'
BASE = HERE / 'runs/compose_neural_embedding_masks_v1'
INPUT = HERE / 'runs/full_negative_context_v1'
AUDIT = ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json'
MASKS = ['all', 'cold_task', 'user_no_past_ad_click', 'user_with_past_ad_click']
WEIGHTS = [.0125, .025, .05, .1, .2, .35, .5]
original_inputs, original_dependencies = engine.inputs, engine.dependencies
engine.OUT, engine.BASE, engine.WEIGHTS = OUT, BASE, WEIGHTS
engine.emit = lambda name, **values: emit(name.replace('family_quantiles', 'refreshed_family_quantiles'), **values)
CONTRACTS = {}


def inputs(day, independent=False):
    rows, base, old_masks, targets = original_inputs(day, independent)
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'user_id', 'task_id'], dtype='int32')
    directory = INPUT / f'input_day{day}'
    source_rows = np.load(directory / 'rows.npy')
    training = source_rows[np.load(directory / 'train.npy')]
    query = source_rows[np.load(directory / 'held.npy')]
    np.testing.assert_array_equal(rows, query)
    masks = historical_masks(raw.pt_d.to_numpy(), raw.user_id.to_numpy(), raw.task_id.to_numpy(),
        training, np.load(directory / 'labels.npy'), rows, 20191100 + day, independent)
    np.testing.assert_array_equal(masks['cold_task'], old_masks['cold_task'])
    selected = {name: old_masks['all'] if name == 'all' else masks[name] for name in MASKS}
    contract = dict(rows=len(rows), rows_sha256=array_hash(rows),
        source_training_rows_sha256=array_hash(training),
        masks_sha256={name: array_hash(value) for name, value in selected.items()},
        mask_rows={name: int(value.sum()) for name, value in selected.items()},
        max_mask_label_date=20191100 + day - 1,
        complete_strictly_past_training_population=True, inferred17_labels_read=False)
    if independent:
        assert contract == read(OUT / 'inputs_locked.json')[str(day)]
    else:
        CONTRACTS[str(day)] = contract
    return rows, base, selected, targets


engine.inputs = inputs


def prepare():
    assert not OUT.exists(), 'Preserve completed and failed experiments'
    audit = read(AUDIT)
    assert audit['audit_status'] == 'passed' and audit['original_datasets_unchanged']
    verified = read(BASE / 'verification.json')
    assert verified['status'] == 'passed'
    for day in (15, 16):
        date = str(20191100 + day)
        assert verified['dates'][date]['auc'] == audit['dates'][date]['auc']
        assert digest(BASE / f'day{day}_prediction.npy') == audit['dates'][date]['prediction_sha256']
    paths = original_dependencies() + [Path(__file__),
        HERE / 'run_refresh_family_quantiles_v1.ps1', HERE / 'full_negative_mask_core_v1.py']
    for path, expected in read(BASE / 'protocol.json')['sources_sha256'].items():
        if Path(path).suffix == '.py':
            assert digest(path) == expected, path
            paths.append(Path(path))
    for day in (15, 16):
        directory = INPUT / f'input_day{day}'
        contract = read(directory / 'contract.json')
        for name in ('rows.npy', 'train.npy', 'held.npy', 'labels.npy'):
            assert digest(directory / name) == contract['files_sha256'][name]
            paths.append(directory / name)
        paths.append(directory / 'contract.json')
    for filename, expected in audit['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    OUT.mkdir()
    write(OUT / 'selection_audit_snapshot.json', audit)
    paths.append(OUT / 'selection_audit_snapshot.json')
    write(OUT / 'protocol.json', dict(families=engine.FAMILIES, weights=WEIGHTS, masks=MASKS,
        baseline=str(BASE), new_model_training=False,
        score_mapping='Independent tie-averaged model ranks mapped into the latest baseline logit distribution',
        averaging='All checkpoints and late checkpoints within each of six historical families, plus equal late-family mean',
        rationale='Earlier family aggregation used a weaker baseline before complete-negative and neural/cohort improvements; test whether its frozen forecasts still add information on the new baseline, including past-click cohorts.',
        mask_training_labels='Every original strictly earlier row only; no query-day label enters mask construction',
        selection='All730 both-date candidate hashes frozen before scoring; development dates repeatedly reused',
        evaluation_day_labels_fit=False, inferred17_labels_read=False,
        raw_datasets_sha256=audit['raw_datasets_sha256'],
        sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)}))
    locks = {}
    for day in (15, 16):
        data = inputs(day)
        assert len(data[3]) == 13
        np.save(OUT / f'day{day}_indices.npy', data[0])
        locks[str(day)] = {engine.key(record): array_hash(prediction)
                          for record, prediction in engine.candidates(data)}
        assert len(locks[str(day)]) == 1 + 13 * len(MASKS) * len(WEIGHTS) == 365
        emit('refreshed_family_candidates_locked', day=day, candidates=len(locks[str(day)]))
    write(OUT / 'inputs_locked.json', CONTRACTS)
    write(OUT / 'predictions_locked.json', dict(protocol_sha256=digest(OUT / 'protocol.json'),
        dates=locks, held_labels_opened_for_scoring=False,
        all_both_date_candidates_locked_before_scoring=True,
        input_contracts_sha256=digest(OUT / 'inputs_locked.json')))
    engine.check()


def verify():
    lock = read(OUT / 'predictions_locked.json')
    assert digest(OUT / 'inputs_locked.json') == lock['input_contracts_sha256']
    engine.verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['prepare', 'verify'])
    {'prepare': prepare, 'verify': verify}[parser.parse_args().phase]()
