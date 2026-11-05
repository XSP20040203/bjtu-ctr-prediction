"""Restore the first real dense-context checkpoint without held-label scoring."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash, lgb
from dense_categorical_context_core_v1 import frame
import full_negative_dense_context_lgb_v1 as experiment

OUT = HERE / 'runs/dense_context_production_checkpoint_probe_v1'
NAME = 'day15_dense_context_leaf31_w5_trees400'
MODEL = experiment.OUT / (NAME + '.txt')


def structure(path):
    trees = 0
    declared = None
    end_trees = end_parameters = False
    with path.open('rb') as source:
        for line in source:
            if line.startswith(b'Tree='):
                trees += 1
            elif line.startswith(b'tree_sizes='):
                declared = len(line.split(b'=', 1)[1].split())
            elif line.strip() == b'end of trees':
                end_trees = True
            elif line.strip() == b'end of parameters':
                end_parameters = True
    return dict(actual_tree_headers=trees, declared_tree_sizes=declared,
        end_of_trees_present=end_trees, end_of_parameters_present=end_parameters)


def main():
    assert not OUT.exists(), 'Preserve previous probes and failures'
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    experiment_protocol = read(experiment.OUT / 'protocol.json')
    preflight = read(experiment.OUT / 'preflight.json')
    assert preflight['status'] == 'passed'
    assert preflight['protocol_sha256'] == digest(experiment.OUT / 'protocol.json')
    assert preflight['input_lock_sha256'] == digest(experiment.OUT / 'inputs_locked.json')
    saved_journal = read(experiment.OUT / 'checkpoint_save_journal.json')
    assert saved_journal['stage'] == 'native_save_complete'
    assert Path(saved_journal['path']) == MODEL
    assert saved_journal['trees'] == 400 and saved_journal['sha256'] == digest(MODEL)
    expected_structure = dict(actual_tree_headers=400, declared_tree_sizes=400,
        end_of_trees_present=True, end_of_parameters_present=True)
    assert structure(MODEL) == expected_structure
    paths = [Path(__file__), MODEL, experiment.OUT / 'protocol.json',
        experiment.OUT / 'preflight.json', experiment.OUT / 'inputs_locked.json',
        HERE / 'dense_categorical_context_core_v1.py', HERE / 'full_negative_dense_context_lgb_v1.py']
    for filename, expected in experiment_protocol['sources_sha256'].items():
        if Path(filename).suffix == '.py':
            assert digest(filename) == expected, filename
            paths.append(Path(filename))
    for independent in (False, True):
        directory = experiment.folder(15, independent)
        contract = experiment.validate_inputs(15, independent)
        assert contract['max_training_date'] == 20191114
        assert contract['own_held_labels_masked'] and contract['all_historical_negatives_retained']
        paths += [directory / filename for filename in ('features.npy', 'held.npy', 'contract.json')]
        paths.append(experiment.dictionary_path(15, independent))
    OUT.mkdir()
    write(OUT / 'protocol.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        model=str(MODEL), trees=400, feature_columns=395, held_probe_rows=4000,
        row_selection='4000 evenly spaced held feature positions; no held labels read',
        trained_on_complete_historical_population=True, training_max_date=20191114,
        test='Two separate native ASCII-relative file restores; independent raw sources, fitted dictionaries and column mapping; compare raw margins exactly',
        held_advertisement_labels_read=False, auc_evaluated=False, inferred17_labels_read=False,
        original_experiment_kept_running=True,
        sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)}))
    held = np.load(experiment.folder(15) / 'held.npy')
    selected = np.linspace(0, len(held)-1, 4000, dtype='int64')
    primary, categories = frame(experiment.folder(15), 'held', experiment.dictionaries(15), selected)
    independent, other_categories = frame(experiment.folder(15, True), 'held',
        experiment.dictionaries(15, True), selected, independent=True)
    assert categories == other_categories and list(primary) == list(independent)
    np.testing.assert_array_equal(primary.to_numpy(), independent.to_numpy())
    first = lgb.Booster(model_file=experiment.native_path(MODEL))
    assert first.current_iteration() == 400 and first.feature_name() == list(primary)
    second = lgb.Booster(model_file=experiment.native_path(MODEL))
    assert second.current_iteration() == 400 and second.feature_name() == list(independent)
    left = first.predict(primary, raw_score=True, num_threads=2)
    right = second.predict(independent, raw_score=True, num_threads=2)
    np.testing.assert_array_equal(left, right)
    assert left.shape == (4000,) and np.isfinite(left).all()
    np.save(OUT / 'held_feature_positions.npy', selected)
    np.save(OUT / 'restored_raw_margin.npy', left)
    for filename, expected in read(OUT / 'protocol.json')['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in experiment_protocol['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    report = dict(status='passed', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        model=str(MODEL), model_sha256=digest(MODEL), model_bytes=MODEL.stat().st_size,
        checkpoint_structure=expected_structure, native_restored_trees=400, columns=395,
        training_rows=contract['training_rows'], training_max_date=20191114,
        trained_on_complete_historical_population=True, held_probe_rows=4000,
        two_independent_native_file_restores_passed=True,
        independent_source_dictionaries_feature_mapping_and_prediction_exact=True,
        maximum_raw_prediction_error=float(np.max(np.abs(left-right))),
        raw_margin_array_sha256=array_hash(left), original_experiment_kept_running=True,
        complete2400_tree_or_both_date_production_reliability_proven=False,
        held_advertisement_labels_read=False, auc_evaluated=False, inferred17_labels_read=False,
        gpu_memory_allocated=False, original_datasets_unchanged=True,
        raw_datasets_sha256=experiment_protocol['raw_datasets_sha256'],
        protocol_sha256=digest(OUT / 'protocol.json'))
    assert not torch.cuda.is_initialized()
    write(OUT / 'verification.json', report)
    emit('dense_context_real_production_checkpoint_restored', trees=400, bytes=report['model_bytes'],
        train_rows=report['training_rows'], probe_rows=4000, maximum_error=0., auc_evaluated=False)


if __name__ == '__main__':
    main()
