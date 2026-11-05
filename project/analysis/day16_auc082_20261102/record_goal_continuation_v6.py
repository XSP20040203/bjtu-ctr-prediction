"""Record completed semantic/cohort results and identity-checked production queues."""
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from conditional_daily import read
from fit_stacker import HERE, ROOT, digest, write
from publish_progress import publication_lock
import record_goal_continuation_v3 as reporting

reporting.EXPERIMENTS.update(compose_semantic_embedding_masks_v1=(91774, 0),
    full_negative_history_dot_embeddings_v1=(59964, 12))


def observed_processes():
    command = """$taskRecords = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @('python.exe','pwsh.exe') -and
        $_.CommandLine -match 'full_negative_class_balance_v1|full_negative_neural_embeddings_v2|compose_class_balance_masks_v1|compose_neural_embedding_masks_v1|full_negative_embedding_lgb_v[123]|full_negative_semantic_embeddings_v[12]|refresh_family_quantiles_v1|compose_semantic_embedding_masks_v1|full_negative_history_dot_embeddings_v1' -and
        $_.CommandLine -notmatch 'Get-CimInstance Win32_Process' -and
        $_.CommandLine -notmatch ' -Command '
    } | ForEach-Object { [pscustomobject]@{
        process_id=$_.ProcessId; parent_process_id=$_.ParentProcessId;
        creation_date=$_.CreationDate.ToString('o'); command_line=$_.CommandLine
    } }); ConvertTo-Json -InputObject $taskRecords -Depth 3 -Compress"""
    result = subprocess.run([reporting.SHELL, '-NoProfile', '-Command', command], check=True,
        capture_output=True, text=True, encoding='utf-8-sig')
    observed = json.loads(result.stdout)
    identities = {61136: ('run_full_negative_embedding_lgb_v3.ps1', '2026-11-03T15:09:29.328551-05:00'),
        57864: ('full_negative_embedding_lgb_v3.py --phase fit', '2026-11-03T15:24:36.725338-05:00'),
        64760: ('run_full_negative_history_dot_embeddings_v1.ps1', '2026-11-03T15:30:36.286956-05:00'),
        65428: ('full_negative_history_dot_embeddings_v1.py --phase prepare', '2026-11-03T15:30:39.169732-05:00')}
    for process in observed:
        identity = identities.get(process['process_id'])
        if identity:
            assert identity[0] in process['command_line']
            actual = datetime.fromisoformat(re.sub(r'([.][0-9]{6})[0-9]+', r'\1', process['creation_date']))
            expected = datetime.fromisoformat(identity[1])
            assert abs((actual-expected).total_seconds()) < .01
            process['identity_and_creation_time_revalidated'] = True
    return observed


def main():
    observed = observed_processes()
    reports = {name: reporting.summary(name, observed) for name in reporting.EXPERIMENTS}
    now = datetime.now(timezone.utc)
    with publication_lock():
        audit = read(ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json')
        assert audit['audit_status'] == 'passed' and audit['original_datasets_unchanged']
        runtime = read(HERE / 'resumed_goal_runtime_v1.json')
        delivery = runtime['latest_day17_delivery']
        assert 'current_best_20191117_20261103_v8' in delivery['submission']
        assert delivery['inferred17_labels_read'] is False
        assert digest(delivery['submission']) == delivery['submission_sha256']
        archive = HERE / 'runtime_history' / now.strftime('%Y%m%dT%H%M%S%fZ_semantic_masks_and_scalar_queue')
        archive.mkdir(parents=True, exist_ok=False)
        progress_files = ('resumed_processes_v1.json', 'resumed_goal_runtime_v1.json', 'STATUS.json',
            'full_negative_semantic_embeddings_progress_v1.json', 'full_negative_embedding_lgb_progress_v3.json')
        for filename in progress_files:
            path = HERE / filename
            if path.is_file():
                shutil.copy2(path, archive / filename)
        reporting.completion('full_negative_semantic_embeddings_v1',
            'full_negative_semantic_embeddings_completed_v1.json', audit, delivery)
        if reports['compose_semantic_embedding_masks_v1']['status'] == 'verified':
            reporting.completion('compose_semantic_embedding_masks_v1',
                'semantic_embedding_mask_composition_completed_v1.json', audit, delivery)
        scores = {key: value['auc'] for key, value in audit['dates'].items()}
        runtime.update(timestamp_utc=now.isoformat(), goal_status='active',
            previous_goal_turn_classification='progress',
            previous_goal_turn_evidence='Explicit user request completed: v8 day17 submission exported with best audited16 recipe .8139803604311606; all404182 rows, four frozen family models, compact feature hashes, independent17 replay, CSV and raw files verified. Legacy v7 int32/int64 log-ID hash mismatch preserved before any forecast; repaired in new v8 source.',
            last_goal_turn_classification='progress',
            last_goal_turn_evidence='Semantic ablation completed12 models and independent replay but no global gain. Completed1012 preregistered semantic cohort candidates against refreshed baseline; complete independently audited15/16 AUC improved to .8023891849526744/.8140553715131704. Saved frozen-model capacity/cohort diagnostics: direct learned coordinates still consume32-41 percent importance with standalone .765-.780 AUC. Completed401-column context plus six history-dot scalar preparation on all4,630,657 rows with independent bitwise projection and1088-row native CPU preflight. Session59964 now waits for identity-checked CPU-to-GPU memory handoff. Actual LightGBM v3 session77392 fitting15 leaf31 has saved the native400-tree production checkpoint,611800442 bytes. No extra GPU owner and no inferred17 labels.',
            scores=scores, targets={key: value['target_auc'] for key, value in audit['dates'].items()},
            both_targets_met=audit['goal_achieved'], original_datasets_unchanged=audit['original_datasets_unchanged'],
            advertisement_inferred17_labels_read=False, observed_processes=observed,
            current_experiments=reports, preserved_runtime_snapshot=str(archive),
            latest_day17_delivery=delivery, latest17_not_regenerated_during_goal_optimization=True,
            next_actions=[
                'Observe LightGBM v3 session77392, wrapper61136 creation15:09:29.328551-05:00, fit57864 creation15:24:36.725338-05:00. Native400-tree checkpoint saved successfully. Validate800/1600 saves and full checkpoint restore on actual full populations; do not restart due to observation timeout.',
                'Observe scalar history-dot ablation session59964, wrapper64760 creation15:30:36.286956-05:00, preparation65428 creation15:30:39.169732-05:00. All4,630,657 projected rows and1088-row native CPU fits passed; queue now waits for actual LightGBM v3 wrapper exit; new experiment does not consume LightGBM predictions.',
                'After scalar preparation, retain protocol and preflight; wait for predecessor exit and enough RAM, then a single GPU owner fits d4/d6,2400 rounds and12 checkpoints. Lock all both-date candidate hashes before evaluation and independent replay; publish only verified improvements.',
                'Preserve completed semantic12-model run and1012-candidate cohort locks. The new independently audited best is semantic_d6 cold_task rank .1 for15 and user_no_past_ad_click add_within .1 for16.',
                'Both original targets15>=.815 and16>=.82 remain unmet; the dates are repeatedly reused development sets.',
                'Preserve original four CSVs, prior failures and explicit-request v8 day17 delivery. Never read inferred17 labels.'])
        runtime.pop('latest17_not_regenerated_after_new_development_improvement', None)
        runtime.setdefault('handles', {}).update(full_negative_semantic_embeddings_pipeline_session=4745,
            full_negative_semantic_embeddings_pipeline_terminal_exit_code=0,
            compose_semantic_embedding_masks_pipeline_session=91774,
            compose_semantic_embedding_masks_pipeline_terminal_exit_code=0,
            full_negative_embedding_lgb_v3_pipeline_session=77392,
            full_negative_history_dot_embeddings_pipeline_session=59964)
        write(HERE / 'resumed_processes_v1.json', observed)
        write(HERE / 'resumed_goal_runtime_v1.json', runtime)
        state = read(HERE / 'STATUS.json')
        state.update(snapshot_utc=now.isoformat(), goal_status='active',
            last_goal_turn_classification='progress', resumed_goal_runtime=runtime,
            current_full_negative_experiments=reports, current_verified15_auc=scores['20191115'],
            current_verified_best_auc=scores['20191116'], target_verified=audit['goal_achieved'])
        write(HERE / 'STATUS.json', state)
        for name, filename in (
            ('full_negative_semantic_embeddings_v1', 'full_negative_semantic_embeddings_progress_v1.json'),
            ('compose_semantic_embedding_masks_v1', 'compose_semantic_embedding_masks_progress_v1.json'),
            ('full_negative_embedding_lgb_v3', 'full_negative_embedding_lgb_progress_v3.json'),
            ('full_negative_history_dot_embeddings_v1', 'full_negative_history_dot_embeddings_progress_v1.json')):
            write(HERE / filename, dict(timestamp_utc=now.isoformat(), **reports[name]))
    print(json.dumps(dict(classification='progress', scores=scores, targets_met=audit['goal_achieved'],
        experiments={key: dict(status=value['status'], saved_model_count=value['saved_model_count'],
            session_id=value['session_id']) for key, value in reports.items()},
        preserved_runtime=str(archive)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
