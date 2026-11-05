"""Validate export v10, preserve delivery history and record completed restore evidence."""
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / 'analysis/day16_auc082_20261102'
sys.path.insert(0, str(EXP))
from conditional_daily import read
from fit_stacker import digest, write
from rich_userday_activity import array_hash
from publish_progress import publication_lock
from record_goal_continuation_v9 import observed_processes


def main():
    folder = ROOT / 'output/current_best_20191117_20261103_v10'
    run = HERE / 'current_best_17_run_v10'
    report = read(folder / 'verification.json')
    manifest = read(folder / 'prediction_manifest.json')
    assert report['status'] == 'passed' and report['inferred17_labels_read'] is False
    assert digest(run / 'protocol.json') == manifest['protocol_sha256']
    assert digest(run / 'input_contracts.json') == manifest['input_contracts_sha256']
    audit = read(ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json')
    assert audit['audit_status'] == 'passed'
    assert abs(audit['dates']['20191116']['auc'] - report['development16_auc']) < 1e-12
    csv = folder / 'submission_best_20191117.csv'
    assert digest(csv) == report['submission_sha256'] == manifest['files_sha256'][csv.name]
    assert digest(folder / 'day17_prediction.npy') == manifest['files_sha256']['day17_prediction.npy']
    saved = pd.read_csv(csv, float_precision='round_trip')
    assert list(saved) == ['id', 'pctr'] and len(saved) == report['rows'] == 404182
    np.testing.assert_array_equal(saved.id, np.arange(len(saved)))
    np.testing.assert_array_equal(saved.pctr, np.load(folder / 'day17_prediction.npy'))
    assert np.isfinite(saved.pctr).all() and saved.pctr.between(0, 1, inclusive='neither').all()
    original = pd.read_csv(ROOT / '2026ml/final_test.csv', usecols=['pt_d', 'log_id'], dtype='int64')
    assert original.pt_d.eq(20191117).all() and len(original) == len(saved)
    assert array_hash(original.log_id.to_numpy()) == read(run / 'input_contracts.json')['20191117']['log_ids_sha256']
    for filename, expected in report['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    probe_path = EXP / 'runs/dense_context_production_checkpoint_probe_v1/verification.json'
    probe = read(probe_path)
    assert probe['status'] == 'passed' and probe['inferred17_labels_read'] is False
    assert probe['two_independent_native_file_restores_passed']
    assert probe['independent_source_dictionaries_feature_mapping_and_prediction_exact']
    assert not probe['complete2400_tree_or_both_date_production_reliability_proven']
    assert digest(probe['model']) == probe['model_sha256']
    observed = observed_processes()
    now = datetime.now(timezone.utc)
    delivery = dict(timestamp_utc=now.isoformat(), request_complete=True,
        submission=str(csv), verification=str(folder / 'verification.json'), rows=len(saved),
        development16_auc=report['development16_auc'], selected=report['selected'],
        submission_sha256=digest(csv), inferred17_labels_read=False, platform_auc=None,
        original_target_goal_remains_active=True, independently_replayed=True,
        fresh_csv_probability_id_order_and_original_datasets_checks='passed',
        probability_min=float(saved.pctr.min()), probability_max=float(saved.pctr.max()))
    destination = HERE / 'delivery_current_best_17_v10.json'
    assert not destination.exists(), 'Preserve prior delivery records'
    write(destination, delivery)
    with publication_lock():
        archive = EXP / 'runtime_history' / now.strftime('%Y%m%dT%H%M%S%fZ_delivery17_v10')
        archive.mkdir(parents=True, exist_ok=False)
        for filename in ('STATUS.json', 'resumed_goal_runtime_v1.json', 'resumed_processes_v1.json'):
            path = EXP / filename
            if path.exists():
                shutil.copy2(path, archive / filename)
        runtime = read(EXP / 'resumed_goal_runtime_v1.json')
        previous = runtime.get('latest_day17_delivery')
        history = runtime.setdefault('day17_delivery_history', [])
        if previous and previous not in history:
            history.append(previous)
        runtime['latest_day17_delivery'] = delivery
        runtime['last_day17_delivery_update_utc'] = now.isoformat()
        runtime['observed_processes'] = observed
        runtime['latest17_not_regenerated_during_goal_optimization'] = False
        runtime['latest17_export_reason'] = 'Explicit user request: predict day17 using current audited best model'
        runtime['dense_context_production_restore_probe'] = dict(
            verification=str(probe_path), verification_sha256=digest(probe_path),
            status='passed', trees=400, held_probe_rows=4000, maximum_raw_prediction_error=0.,
            held_advertisement_labels_read=False, auc_evaluated=False,
            inferred17_labels_read=False, full2400_tree_or_both_date_reliability_proven=False)
        runtime['previous_goal_turn_classification'] = runtime.get('last_goal_turn_classification')
        runtime['previous_goal_turn_evidence'] = runtime.get('last_goal_turn_evidence')
        runtime['last_goal_turn_classification'] = 'progress'
        runtime['last_goal_turn_evidence'] = (
            'Delivered current audited day16 history_dot_d6/task_at_most_100_prior_rows/'
            'replace_global/.1 recipe as day17 v10 after full16 and independent17 replay. '
            'Also recorded actual full-population400-tree native checkpoint restore probe: '
            'two independent native loaders and4000 held feature rows matched bitwise. '
            'Probe did not evaluate AUC; no new15/16 AUC claimed. '
            'Original datasets preserved and no inferred17 labels read; optimization goal remains active.')
        runtime['preserved_runtime_snapshot'] = str(archive)
        runtime['next_actions'] = [
            'Keep current dense-context CPU training and post-dense complete-inventory publisher watcher running; revalidate identities before intervention.',
            'After all12 checkpoints complete, lock both-date candidates and finish independent full-cohort replay and audit.',
            'Promote only completed independently audited full15/16 development results; targets remain15>=.815 and16>=.82.',
            'Preserve original datasets, every frozen experiment and submission v10; never read inferred17 labels.']
        write(EXP / 'resumed_processes_v1.json', observed)
        write(EXP / 'resumed_goal_runtime_v1.json', runtime)
        state = read(EXP / 'STATUS.json')
        previous = state.get('latest_best_recipe17_delivery')
        history = state.setdefault('latest_best_recipe17_delivery_history', [])
        if previous and previous not in history:
            history.append(previous)
        state['latest_best_recipe17_delivery'] = delivery
        state['resumed_goal_runtime'] = runtime
        state['dense_context_production_restore_probe'] = runtime['dense_context_production_restore_probe']
        write(EXP / 'STATUS.json', state)
    print(delivery, flush=True)


if __name__ == '__main__':
    main()
