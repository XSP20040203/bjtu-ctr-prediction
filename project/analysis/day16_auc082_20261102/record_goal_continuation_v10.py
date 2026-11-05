"""Record a verified wait on the existing dense fit and preserve the v10 delivery."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from conditional_daily import read
from fit_stacker import HERE, ROOT, digest, write
from publish_progress import publication_lock
from record_goal_continuation_v9 import observed_processes
import record_goal_continuation_v3 as reporting


def main():
    observed = observed_processes()
    fit = next(p for p in observed if p['process_id'] == 74360)
    wrapper = next(p for p in observed if p['process_id'] == 29148)
    watcher = next(p for p in observed if p['process_id'] == 61976)
    assert all(p['identity_and_creation_time_revalidated'] for p in (fit, wrapper, watcher))
    assert fit['parent_process_id'] == wrapper['process_id']
    assert '--phase fit' in fit['command_line']
    summaries = {name: reporting.summary(name, observed) for name in reporting.EXPERIMENTS}
    dense = summaries['full_negative_dense_context_lgb_v1']
    assert dense['status'] == 'running_fit'
    folder = HERE / 'runs/full_negative_dense_context_lgb_v1'
    journal_path = folder / 'checkpoint_save_journal.json'
    journal = read(journal_path)
    assert journal['stage'] == 'native_save_complete'
    assert digest(journal['path']) == journal['sha256']
    assert journal['trees'] >= 1200
    probe_path = HERE / 'runs/dense_context_production_checkpoint_probe_v1/verification.json'
    probe = read(probe_path)
    assert probe['status'] == 'passed' and not probe['auc_evaluated']
    assert probe['two_independent_native_file_restores_passed']
    assert not probe['complete2400_tree_or_both_date_production_reliability_proven']
    assert not probe['inferred17_labels_read']
    assert digest(probe['model']) == probe['model_sha256']
    delivery_path = ROOT / 'analysis/final_best_recipe_compact_dense_20261102/delivery_current_best_17_v10.json'
    delivery = read(delivery_path)
    assert delivery['request_complete'] and not delivery['inferred17_labels_read']
    assert digest(delivery['submission']) == delivery['submission_sha256']
    now = datetime.now(timezone.utc)
    with publication_lock():
        audit_path = ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json'
        audit = read(audit_path)
        assert audit['audit_status'] == 'passed'
        for name, expected in audit['raw_datasets_sha256'].items():
            assert digest(ROOT / '2026ml' / name) == expected, name
        scores = {date: record['auc'] for date, record in audit['dates'].items()}
        runtime = read(HERE / 'resumed_goal_runtime_v1.json')
        assert runtime['latest_day17_delivery'] == delivery
        archive = HERE / 'runtime_history' / now.strftime('%Y%m%dT%H%M%S%fZ_dense_context_verified_wait')
        archive.mkdir(parents=True, exist_ok=False)
        for filename in ('STATUS.json', 'resumed_processes_v1.json', 'resumed_goal_runtime_v1.json',
                         'full_negative_dense_context_lgb_progress_v1.json',
                         'compose_history_dot_embedding_masks_progress_v1.json'):
            path = HERE / filename
            if path.exists():
                shutil.copy2(path, archive / filename)
        evidence = dict(session_id=9696, watcher_session_id=59427,
            live_fit=fit, live_wrapper=wrapper, live_publisher_watcher=watcher,
            checkpoint_save_journal=journal,
            checkpoint_file_bytes=Path(journal['path']).stat().st_size,
            checkpoint_file_hash_freshly_verified=True,
            saved_models=dense['saved_model_count'], expected_models=dense['expected_models'],
            current_fit_has_no_completed_auc_evaluation=True,
            authoritative_audit=str(audit_path), authoritative_audit_sha256=digest(audit_path),
            scores_changed_since_previous_record=runtime['scores'] != scores,
            preceding_turn_request_completed=delivery_path.as_posix(),
            production_restore_verification=str(probe_path),
            production_restore_verification_sha256=digest(probe_path),
            recorder_sha256=digest(Path(__file__)))
        runtime.update(timestamp_utc=now.isoformat(), goal_status='active',
            previous_goal_turn_classification=runtime.get('last_goal_turn_classification'),
            previous_goal_turn_evidence=runtime.get('last_goal_turn_evidence'),
            last_goal_turn_classification='verified_wait',
            last_goal_turn_evidence=(
                'Previous turn made progress: delivered requested current-best17 v10 and recorded '
                'the completed400-tree production restore verification. This turn freshly polled '
                'live sessions9696 and59427, revalidated fit74360/wrapper29148/watcher61976 '
                'identities and exact creation times, and observed dense day15 leaf31/weight5 '
                'advance to the1200-tree checkpoint. Its28MB file hash matches the native save '
                'journal. No new AUC is claimed. Keep the existing fit and complete-inventory '
                'publisher watcher running; do not restart or launch competing heavy fits.'),
            last_goal_turn_evidence_details=evidence,
            scores=scores, targets={day: row['target_auc'] for day, row in audit['dates'].items()},
            both_targets_met=audit['goal_achieved'], original_datasets_unchanged=True,
            advertisement_inferred17_labels_read=False, observed_processes=observed,
            current_experiments=summaries, preserved_runtime_snapshot=str(archive),
            latest_day17_delivery=delivery,
            next_actions=[
                'Continue observing existing dense-context session9696; preserve exact process identities and do not restart after observation timeouts.',
                'Complete all12 saved checkpoints and freeze both-date candidate hashes before scoring; independently replay full-cohort models and formulas before promotion.',
                'Keep watcher59427/wrapper61976 alive so publisher v3 includes both history-dot and dense-context results after dense wrapper29148 exits.',
                'Targets remain complete15>=.815 and16>=.82; repeated development sets are not blind tests.',
                'Preserve frozen sources, failures, original four CSVs and requested17 v10 delivery; never read inferred17 labels.'])
        runtime.setdefault('handles', {}).update(
            full_negative_dense_context_lgb_pipeline_session=9696,
            compose_history_dot_embedding_masks_pipeline_session=59427,
            compose_history_dot_embedding_masks_post_dense_refresh_pending=True,
            day17_current_best_export_v10_session=9949,
            day17_current_best_export_v10_terminal_exit_code=0,
            day17_current_best_delivery_v10_session=31690,
            day17_current_best_delivery_v10_terminal_exit_code=0)
        write(HERE / 'resumed_processes_v1.json', observed)
        write(HERE / 'resumed_goal_runtime_v1.json', runtime)
        state = read(HERE / 'STATUS.json')
        state.update(snapshot_utc=now.isoformat(), goal_status='active',
            last_goal_turn_classification='verified_wait', resumed_goal_runtime=runtime,
            current_full_negative_experiments=summaries,
            current_verified15_auc=scores['20191115'], current_verified_best_auc=scores['20191116'],
            target_verified=audit['goal_achieved'])
        write(HERE / 'STATUS.json', state)
        for name, filename in (
            ('full_negative_dense_context_lgb_v1', 'full_negative_dense_context_lgb_progress_v1.json'),
            ('compose_history_dot_embedding_masks_v1', 'compose_history_dot_embedding_masks_progress_v1.json')):
            write(HERE / filename, dict(timestamp_utc=now.isoformat(), **summaries[name]))
    print(json.dumps(dict(classification='verified_wait', scores=scores,
        goal_achieved=audit['goal_achieved'], latest_checkpoint=journal,
        saved_models=dense['saved_model_count'], preserved_snapshot=str(archive))), flush=True)


if __name__ == '__main__':
    main()
