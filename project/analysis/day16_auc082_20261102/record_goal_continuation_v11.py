"""Record the executed dense-cohort preregistration and live sequential queue."""
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from conditional_daily import read
from fit_stacker import HERE, ROOT, digest, write
from publish_progress import publication_lock
from record_goal_continuation_v9 import IDENTITIES
import record_goal_continuation_v3 as reporting

reporting.EXPERIMENTS.update(
    compose_semantic_embedding_masks_v1=(91774, 0),
    full_negative_history_dot_embeddings_v1=(59964, 12),
    full_negative_dense_context_lgb_v1=(9696, 12),
    compose_history_dot_embedding_masks_v1=(59427, 0),
    compose_dense_context_checkpoint_masks_v1=(65373, 0))
EXPECTED = dict(IDENTITIES)
EXPECTED[56348] = ('run_compose_dense_context_checkpoint_masks_v1.ps1',
                    '2026-11-03T16:41:49.313308-05:00')


def observed_processes():
    command = """$taskRecords = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @('python.exe','pwsh.exe') -and
        $_.CommandLine -match 'full_negative_class_balance_v1|full_negative_neural_embeddings_v2|compose_class_balance_masks_v1|compose_neural_embedding_masks_v1|full_negative_embedding_lgb_v[123]|full_negative_semantic_embeddings_v[12]|refresh_family_quantiles_v1|compose_semantic_embedding_masks_v1|full_negative_history_dot_embeddings_v1|full_negative_dense_context_lgb_v1|compose_history_dot_embedding_masks_v1|compose_dense_context_checkpoint_masks_v1' -and
        $_.CommandLine -notmatch 'Get-CimInstance Win32_Process' -and
        $_.CommandLine -notmatch ' -Command '
    } | ForEach-Object { [pscustomobject]@{
        process_id=$_.ProcessId; parent_process_id=$_.ParentProcessId;
        creation_date=$_.CreationDate.ToString('o'); command_line=$_.CommandLine
    } }); ConvertTo-Json -InputObject $taskRecords -Depth 3 -Compress"""
    result = subprocess.run([reporting.SHELL, '-NoProfile', '-Command', command], check=True,
        capture_output=True, text=True, encoding='utf-8-sig')
    observed = json.loads(result.stdout)
    for record in observed:
        identity = EXPECTED.get(record['process_id'])
        if identity:
            assert identity[0] in record['command_line']
            actual = datetime.fromisoformat(re.sub(r'([.][0-9]{6})[0-9]+', r'\1', record['creation_date']))
            assert abs((actual-datetime.fromisoformat(identity[1])).total_seconds()) < .01
            record['identity_and_creation_time_revalidated'] = True
    return observed


def main():
    observed = observed_processes()
    for pid in (74360, 29148, 61976, 56348):
        entry = next(p for p in observed if p['process_id'] == pid)
        assert entry['identity_and_creation_time_revalidated']
    summaries = {name: reporting.summary(name, observed) for name in reporting.EXPERIMENTS}
    assert summaries['full_negative_dense_context_lgb_v1']['status'] == 'running_fit'
    assert summaries['compose_dense_context_checkpoint_masks_v1']['status'] == 'waiting_predecessor_or_memory'
    plan_path = HERE / 'dense_context_checkpoint_cohort_transfer_queued_v1.json'
    plan = read(plan_path)
    assert plan['candidates_per_date'] == 1178 and plan['total_candidates'] == 2356
    assert plan['fixture']['status'] == 'passed'
    assert plan['fixture']['fixture_candidates'] == 1178
    assert plan['no_dense_held_margins_read_during_preregistration']
    assert plan['no_held_advertisement_labels_read_during_preregistration']
    for filename, expected in plan['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in plan['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    journal_path = HERE / 'runs/full_negative_dense_context_lgb_v1/checkpoint_save_journal.json'
    journal = read(journal_path)
    assert journal['stage'] == 'native_save_complete'
    assert digest(journal['path']) == journal['sha256']
    now = datetime.now(timezone.utc)
    with publication_lock():
        audit_path = ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json'
        audit = read(audit_path)
        assert audit['audit_status'] == 'passed' and audit['original_datasets_unchanged']
        runtime = read(HERE / 'resumed_goal_runtime_v1.json')
        delivery = runtime['latest_day17_delivery']
        assert 'current_best_20191117_20261103_v10' in delivery['submission']
        assert not delivery['inferred17_labels_read']
        assert digest(delivery['submission']) == delivery['submission_sha256']
        scores = {date: row['auc'] for date, row in audit['dates'].items()}
        archive = HERE / 'runtime_history' / now.strftime('%Y%m%dT%H%M%S%fZ_dense_checkpoint_cohort_queue')
        archive.mkdir(parents=True, exist_ok=False)
        for filename in ('STATUS.json', 'resumed_processes_v1.json', 'resumed_goal_runtime_v1.json',
                         'full_negative_dense_context_lgb_progress_v1.json',
                         'compose_history_dot_embedding_masks_progress_v1.json',
                         'dense_context_checkpoint_mask_composition_progress_v1.json'):
            path = HERE / filename
            if path.exists():
                shutil.copy2(path, archive / filename)
        evidence = dict(frozen_plan=str(plan_path), frozen_plan_sha256=digest(plan_path),
            total_candidates=2356, fixture_candidates=1178,
            maximum_fixture_replay_error=plan['fixture']['maximum_independent_prediction_error'],
            preregistration_session_id=56620, preregistration_exit_code=0,
            queued_wrapper_session_id=65373, queued_wrapper_process_id=56348,
            queued_wrapper_creation_date='2026-11-03T16:41:49.313308-05:00',
            queued_wrapper_identity_and_creation_time_revalidated=True,
            observed_main_fit_iteration=2000, checkpoint_save_journal=journal,
            all_source_margins_and_both_date_scores_unavailable_when_preregistered=True,
            no_competing_model_fit_launched=True,
            authoritative_audit=str(audit_path), authoritative_audit_sha256=digest(audit_path),
            scores_changed_since_previous_record=runtime['scores'] != scores,
            recorder_sha256=digest(Path(__file__)))
        runtime.update(timestamp_utc=now.isoformat(), goal_status='active',
            previous_goal_turn_classification=runtime.get('last_goal_turn_classification'),
            previous_goal_turn_evidence=runtime.get('last_goal_turn_evidence'),
            last_goal_turn_classification='progress',
            last_goal_turn_evidence=(
                'Previous goal turn was a verified wait on the existing dense fit. This turn '
                'implemented and executed2356-candidate dense checkpoint cohort preregistration; '
                'all1178 distinct fixture candidates passed independent arithmetic and untouched '
                'cohort complements, maximum error5.551115123125783e-17. Froze all six checkpoints '
                'plus late mean, three formulas, eight weights and seven masks before source '
                'margins or held scores existed. Started sequential queue65373/wrapper56348 '
                'creation16:41:49.313308-05:00; it waits for identity-checked publisher61976 '
                'after dense fitting, complete independent source replay and publication. '
                'New publisher v4 preserves the full prior inventory and adds this composition; '
                'all original publisher versions and frozen experiments remain unchanged. '
                'Main fit74360 advanced to day15 leaf31 tree2000. No new AUC is claimed.'),
            last_goal_turn_evidence_details=evidence,
            scores=scores, targets={date: row['target_auc'] for date, row in audit['dates'].items()},
            both_targets_met=audit['goal_achieved'], original_datasets_unchanged=True,
            advertisement_inferred17_labels_read=False, observed_processes=observed,
            current_experiments=summaries, preserved_runtime_snapshot=str(archive),
            latest_day17_delivery=delivery,
            queued_dense_context_checkpoint_cohort_transfer=evidence,
            next_actions=[
                'Observe existing dense-context session9696; finish12 checkpoints, lock both-date candidates, independently replay all saved models and audit.',
                'Keep existing watcher59427/wrapper61976 alive to republish complete v3 inventory after dense wrapper29148 exits.',
                'Observe sequential dense cohort session65373/wrapper56348 after61976 exits. It requires independently verified dense source, then freezes2356 both-date candidate hashes before held scoring and independently replays all masks and formulas.',
                'Publish only independently verified complete-cohort winners using the new v4 full inventory, followed by fresh target audit.',
                'Targets remain15>=.815 and16>=.82 on complete repeatedly reused development sets; preserve original CSVs and requested17 v10 submission and never read inferred17 labels.'])
        runtime.setdefault('handles', {}).update(
            dense_context_checkpoint_cohort_preregistration_session=56620,
            dense_context_checkpoint_cohort_preregistration_terminal_exit_code=0,
            compose_dense_context_checkpoint_masks_pipeline_session=65373)
        write(HERE / 'resumed_processes_v1.json', observed)
        write(HERE / 'resumed_goal_runtime_v1.json', runtime)
        state = read(HERE / 'STATUS.json')
        state.update(snapshot_utc=now.isoformat(), goal_status='active',
            last_goal_turn_classification='progress', resumed_goal_runtime=runtime,
            current_full_negative_experiments=summaries,
            current_verified15_auc=scores['20191115'], current_verified_best_auc=scores['20191116'],
            target_verified=audit['goal_achieved'])
        write(HERE / 'STATUS.json', state)
        for name, filename in (
            ('full_negative_dense_context_lgb_v1', 'full_negative_dense_context_lgb_progress_v1.json'),
            ('compose_history_dot_embedding_masks_v1', 'compose_history_dot_embedding_masks_progress_v1.json'),
            ('compose_dense_context_checkpoint_masks_v1', 'dense_context_checkpoint_mask_composition_progress_v1.json')):
            write(HERE / filename, dict(timestamp_utc=now.isoformat(), **summaries[name]))
    print(json.dumps(dict(classification='progress', scores=scores,
        target_met=audit['goal_achieved'], candidates=2356,
        queue=summaries['compose_dense_context_checkpoint_masks_v1']['status'],
        archive=str(archive))), flush=True)


if __name__ == '__main__':
    main()
