"""Refresh live pipeline evidence without repeating stale progress narratives."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from conditional_daily import read
from fit_stacker import HERE, ROOT, digest, write
from publish_progress import publication_lock
from record_goal_continuation_v11 import observed_processes, reporting

TRACKED = (
    ('full_negative_dense_context_lgb_v1', 'full_negative_dense_context_lgb_progress_v1.json'),
    ('compose_history_dot_embedding_masks_v1', 'compose_history_dot_embedding_masks_progress_v1.json'),
    ('compose_dense_context_checkpoint_masks_v1', 'dense_context_checkpoint_mask_composition_progress_v1.json'))


def main():
    observed = observed_processes()
    summaries = {name: reporting.summary(name, observed) for name in reporting.EXPERIMENTS}
    dense = summaries['full_negative_dense_context_lgb_v1']
    queue = summaries['compose_dense_context_checkpoint_masks_v1']
    # A file alone is insufficient evidence for a verified wait.
    live = [p for p in observed if any(name in p['command_line'] for name, _ in TRACKED)]
    assert live, 'No relevant live processes: investigate terminal handles before recording a wait'
    identities = [p for p in live if p.get('identity_and_creation_time_revalidated')]
    assert identities, 'No known wrapper/fit identity revalidated: inspect actual process creation times'
    plan_path = HERE / 'dense_context_checkpoint_cohort_transfer_queued_v1.json'
    plan = read(plan_path)
    assert plan['total_candidates'] == 2356 and plan['fixture']['status'] == 'passed'
    for filename, expected in plan['sources_sha256'].items():
        assert digest(filename) == expected, filename
    journal_path = HERE / 'runs/full_negative_dense_context_lgb_v1/checkpoint_save_journal.json'
    journal = read(journal_path)
    checkpoint = None
    if journal['stage'] == 'native_save_complete':
        assert digest(journal['path']) == journal['sha256']
        checkpoint = {key: journal[key] for key in ('day', 'arm', 'trees', 'path', 'sha256')}
        checkpoint['bytes'] = Path(journal['path']).stat().st_size
    for filename, expected in plan['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    now = datetime.now(timezone.utc)
    with publication_lock():
        audit_path = ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json'
        audit = read(audit_path)
        assert audit['audit_status'] == 'passed' and audit['original_datasets_unchanged']
        assert audit['raw_datasets_sha256'] == plan['raw_datasets_sha256']
        scores = {day: row['auc'] for day, row in audit['dates'].items()}
        runtime = read(HERE / 'resumed_goal_runtime_v1.json')
        delivery = runtime['latest_day17_delivery']
        assert not delivery['inferred17_labels_read']
        assert digest(delivery['submission']) == delivery['submission_sha256']
        previous_details = runtime.get('last_goal_turn_evidence_details', {})
        previous_journal = previous_details.get('checkpoint_save_journal', {})
        previous_completed = (previous_details.get('completed_checkpoint') or
            (previous_journal if previous_journal.get('stage') == 'native_save_complete' else {}))
        checkpoint_changed = (checkpoint is not None and
            any(checkpoint[key] != previous_completed.get(key) for key in ('day', 'arm', 'trees', 'sha256')))
        scores_changed = runtime.get('scores') != scores
        previous_experiments = runtime.get('current_experiments', {})
        status_changes = {name: dict(previous=previous_experiments.get(name, {}).get('status'),
                                     current=summaries[name]['status'])
            for name, _ in TRACKED
            if previous_experiments.get(name, {}).get('status') != summaries[name]['status']}
        classification = 'progress' if scores_changed or checkpoint_changed or status_changes else 'verified_wait'
        evidence = dict(timestamp_utc=now.isoformat(), observed_live_processes=live,
            known_identities_and_creation_times_revalidated=len(identities),
            sessions=dict(dense=9696, complete_publisher=59427, dense_cohort=65373),
            completed_checkpoint=checkpoint, checkpoint_save_journal=journal,
            completed_checkpoint_changed=checkpoint_changed,
            scores_changed_since_previous_record=scores_changed, experiment_status_changes=status_changes,
            authoritative_audit=str(audit_path), authoritative_audit_sha256=digest(audit_path),
            frozen_dense_cohort_plan=str(plan_path), frozen_dense_cohort_plan_sha256=digest(plan_path),
            current_dense_status=dense['status'], current_dense_cohort_status=queue['status'],
            original_four_csv_hashes_freshly_revalidated=True,
            day17_delivery_preserved=delivery['submission'], inferred17_labels_read=False,
            recorder_sha256=digest(Path(__file__)))
        archive = HERE / 'runtime_history' / now.strftime('%Y%m%dT%H%M%S%fZ_live_dense_pipeline')
        archive.mkdir(parents=True, exist_ok=False)
        for filename in ('STATUS.json', 'resumed_processes_v1.json', 'resumed_goal_runtime_v1.json',
                         *(filename for _, filename in TRACKED)):
            path = HERE / filename
            if path.exists():
                shutil.copy2(path, archive / filename)
        narrative = (
            f'Fresh live process and exact known identity/creation-time observations: '
            f'dense status={dense["status"]}, queued cohort status={queue["status"]}. '
            f'Completed checkpoint changed={checkpoint_changed}; audited scores changed={scores_changed}; '
            f'pipeline status changes={status_changes}. '
            'No checkpoint save alone is an AUC result. Preserve original processes, all frozen '
            'sources and failures; continue full12-model/both-date candidate locking and independent '
            'replay, followed by the2356-candidate sequential cohort composition and complete publisher. '
            'Original CSV hashes freshly match; requested17 delivery preserved; no inferred17 labels.')
        runtime.update(timestamp_utc=now.isoformat(), goal_status='active',
            previous_goal_turn_classification=runtime.get('last_goal_turn_classification'),
            previous_goal_turn_evidence=runtime.get('last_goal_turn_evidence'),
            last_goal_turn_classification=classification, last_goal_turn_evidence=narrative,
            last_goal_turn_evidence_details=evidence,
            scores=scores, targets={day: row['target_auc'] for day, row in audit['dates'].items()},
            both_targets_met=audit['goal_achieved'], original_datasets_unchanged=True,
            advertisement_inferred17_labels_read=False, observed_processes=observed,
            current_experiments=summaries, preserved_runtime_snapshot=str(archive))
        write(HERE / 'resumed_processes_v1.json', observed)
        write(HERE / 'resumed_goal_runtime_v1.json', runtime)
        state = read(HERE / 'STATUS.json')
        state.update(snapshot_utc=now.isoformat(), goal_status='active',
            last_goal_turn_classification=classification, resumed_goal_runtime=runtime,
            current_full_negative_experiments=summaries,
            current_verified15_auc=scores['20191115'], current_verified_best_auc=scores['20191116'],
            target_verified=audit['goal_achieved'])
        write(HERE / 'STATUS.json', state)
        for name, filename in TRACKED:
            write(HERE / filename, dict(timestamp_utc=now.isoformat(), **summaries[name]))
    print(json.dumps(dict(classification=classification, scores=scores,
        goal_achieved=audit['goal_achieved'], completed_checkpoint=checkpoint,
        dense_status=dense['status'], dense_cohort_status=queue['status'],
        pipeline_status_changes=status_changes, archive=str(archive))), flush=True)


if __name__ == '__main__':
    main()
