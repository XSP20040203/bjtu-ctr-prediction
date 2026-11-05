"""Record the terminal sparse-ID failure and the verified dense-context successor."""
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
    full_negative_history_dot_embeddings_v1=(59964, 12),
    full_negative_dense_context_lgb_v1=(9696, 12))


def observed_processes():
    command = """$taskRecords = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @('python.exe','pwsh.exe') -and
        $_.CommandLine -match 'full_negative_class_balance_v1|full_negative_neural_embeddings_v2|compose_class_balance_masks_v1|compose_neural_embedding_masks_v1|full_negative_embedding_lgb_v[123]|full_negative_semantic_embeddings_v[12]|refresh_family_quantiles_v1|compose_semantic_embedding_masks_v1|full_negative_history_dot_embeddings_v1|full_negative_dense_context_lgb_v1' -and
        $_.CommandLine -notmatch 'Get-CimInstance Win32_Process' -and
        $_.CommandLine -notmatch ' -Command '
    } | ForEach-Object { [pscustomobject]@{
        process_id=$_.ProcessId; parent_process_id=$_.ParentProcessId;
        creation_date=$_.CreationDate.ToString('o'); command_line=$_.CommandLine
    } }); ConvertTo-Json -InputObject $taskRecords -Depth 3 -Compress"""
    result = subprocess.run([reporting.SHELL, '-NoProfile', '-Command', command], check=True,
        capture_output=True, text=True, encoding='utf-8-sig')
    observed = json.loads(result.stdout)
    identities = {
        61136: ('run_full_negative_embedding_lgb_v3.ps1', '2026-11-03T15:09:29.328551-05:00'),
        57864: ('full_negative_embedding_lgb_v3.py --phase fit', '2026-11-03T15:24:36.725338-05:00'),
        64760: ('run_full_negative_history_dot_embeddings_v1.ps1', '2026-11-03T15:30:36.286956-05:00'),
        63696: ('full_negative_history_dot_embeddings_v1.py --phase fit', '2026-11-03T15:55:30.965111-05:00'),
        29148: ('run_full_negative_dense_context_lgb_v1.ps1', '2026-11-03T15:54:44.863567-05:00'),
        66316: ('full_negative_dense_context_lgb_v1.py --phase prepare', '2026-11-03T15:54:54.064768-05:00')}
    for process in observed:
        expected = identities.get(process['process_id'])
        if expected:
            assert expected[0] in process['command_line']
            actual = datetime.fromisoformat(re.sub(r'([.][0-9]{6})[0-9]+', r'\1', process['creation_date']))
            assert abs((actual-datetime.fromisoformat(expected[1])).total_seconds()) < .01
            process['identity_and_creation_time_revalidated'] = True
    return observed


def main():
    observed = observed_processes()
    reports = {name: reporting.summary(name, observed) for name in reporting.EXPERIMENTS}
    dense = HERE / 'runs/full_negative_dense_context_lgb_v1'
    preflight = read(dense / 'preflight.json')
    assert preflight['status'] == 'passed' and preflight['rows'] == 32768
    assert preflight['full_complete_training_and_held_encodings_independently_replayed']
    projection = read(dense / 'full_projection_verification.json')
    assert projection['status'] == 'passed'
    assert sum(entry['rows'] for date in projection['dates'].values() for entry in date.values()) == 4630657
    failure_path = HERE / 'runs/full_negative_embedding_lgb_v3/failure.json'
    failure = read(failure_path)
    assert failure['terminal_exit_code'] == -1073741819
    assert failure['windows_event_application_pid_start_time_and_lib_lightgbm_module_matched']
    assert reports['full_negative_embedding_lgb_v3']['status'] == 'failed'
    assert not any(record['process_id'] in (57864, 61136) for record in observed)
    probe_path = HERE / 'runs/dense_context_category_probe_v1/verification.json'
    probe = read(probe_path)
    assert probe['status'] == 'passed' and probe['raw_vs_dense_predictions_identical']
    now = datetime.now(timezone.utc)
    with publication_lock():
        audit = read(ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json')
        assert audit['audit_status'] == 'passed' and audit['original_datasets_unchanged']
        runtime = read(HERE / 'resumed_goal_runtime_v1.json')
        delivery = runtime['latest_day17_delivery']
        assert 'current_best_20191117_20261103_v9' in delivery['submission']
        assert delivery['inferred17_labels_read'] is False
        assert digest(delivery['submission']) == delivery['submission_sha256']
        archive = HERE / 'runtime_history' / now.strftime('%Y%m%dT%H%M%S%fZ_dense_context_and_native_restore_failure')
        archive.mkdir(parents=True, exist_ok=False)
        for filename in ('resumed_processes_v1.json', 'resumed_goal_runtime_v1.json', 'STATUS.json',
            'full_negative_embedding_lgb_progress_v3.json', 'full_negative_history_dot_embeddings_progress_v1.json'):
            path = HERE / filename
            if path.is_file():
                shutil.copy2(path, archive / filename)
        if reports['full_negative_history_dot_embeddings_v1']['status'] == 'verified':
            reporting.completion('full_negative_history_dot_embeddings_v1',
                'full_negative_history_dot_embeddings_completed_v1.json', audit, delivery)
        scores = {day: record['auc'] for day, record in audit['dates'].items()}
        runtime.update(timestamp_utc=now.isoformat(), goal_status='active',
            previous_goal_turn_classification='progress',
            previous_goal_turn_evidence='Explicit user request completed: v9 submission exported for all404182 day17 rows using the audited16 .8140553715131704 semantic/cohort winner. Full16 replay, independent17 raw encoding/feature/model/mask/arithmetic replay, original row order, CSV precision and original four datasets verified; no inferred17 labels.',
            last_goal_turn_classification='progress',
            last_goal_turn_evidence='Dense categorical1088-row probe passed exact raw/dense predictions and native file restore. LightGBM v3 session77392 terminal0xC0000005 occurred during native checkpoint file restore after1600 iterations; Windows event1000 matched PID57864, creation time andlib_lightgbm.dll. Preserved400/800/1600 files and hashes; the1600 file is2147483672 bytes with only1341/1600 tree headers and no end markers, while400/800 structural counts are complete. Root cause not asserted. Original wrapper/fit handles are absent and immutable failure.json recorded. History-dot session59964 automatically began actual GPU training under existing wrapper64760, fit63696. New395-column context-only dense-category LightGBM session9696 completed independently fitted strict-past dictionaries, full4,630,657-row encoding replay and32768-row native full-schema callbacks/save/restore. Heavy fit is queued behind actual history-dot wrapper64760 with identity and creation-time checks. No published AUC gain yet this turn.',
            scores=scores, targets={day: entry['target_auc'] for day, entry in audit['dates'].items()},
            both_targets_met=audit['goal_achieved'], original_datasets_unchanged=True,
            advertisement_inferred17_labels_read=False, observed_processes=observed,
            current_experiments=reports, preserved_runtime_snapshot=str(archive),
            latest_day17_delivery=delivery, latest17_not_regenerated_during_goal_optimization=True,
            dense_category_probe=dict(session_id=73481, terminal_exit_code=0,
                verification=str(probe_path), verification_sha256=digest(probe_path)),
            latest_native_failure=dict(path=str(failure_path), sha256=digest(failure_path),
                session_id=77392, terminal_exit_code=-1073741819),
            dense_context_encoding_comparison=preflight['representative_encoding_comparison'],
            next_actions=[
                'Observe history-dot session59964, wrapper64760 creation15:30:36.286956-05:00 and fit63696 creation15:55:30.965111-05:00. Complete d4/d6 full populations,12 checkpoints, both-date candidate locks, independent replay, publication and raw CSV audit.',
                'Observe dense-context session9696, wrapper29148 creation15:54:44.863567-05:00, preparation66316 creation15:54:54.064768-05:00. Full dictionary/encoding and32768-row native preflight passed. Heavy CPU fit waits for actual history-dot wrapper64760 exit and8GiB free RAM; do not restart live work.',
                'Dense-context experiment uses exactly395 verified context columns and every historical negative, no learned embedding coordinates. Test leaf31/weight5 andleaf63/weight25 at400/1200/2400 rounds,12 checkpoints total. Native paths are ASCII relative, categorical dictionaries strictly training-fitted; independent complete-cohort model replay required before promotion.',
                'Preserve terminal LightGBM v3 failure and all three native checkpoints.1600 text is structurally incomplete; exact restored checkpoint at crash was not logged and sparse IDs are not a proven crash root cause. Do not restart failed frozen versions.',
                'When history-dot margins finish, consider preregistered cohort transfer onto latest semantic-mask baseline if its older frozen baseline limits direct global gains.',
                'Both targets15>=.815 and16>=.82 remain unmet.15/16 are repeatedly reused development populations. Keep original four CSVs and requested17 v9 delivery unchanged; never read inferred17 labels.'])
        runtime.pop('latest17_not_regenerated_after_new_development_improvement', None)
        runtime.setdefault('handles', {}).update(full_negative_embedding_lgb_v3_pipeline_session=77392,
            full_negative_embedding_lgb_v3_terminal_exit_code=-1073741819,
            full_negative_history_dot_embeddings_pipeline_session=59964,
            full_negative_dense_context_lgb_pipeline_session=9696,
            dense_context_category_probe_session=73481, dense_context_category_probe_terminal_exit_code=0)
        write(HERE / 'resumed_processes_v1.json', observed)
        write(HERE / 'resumed_goal_runtime_v1.json', runtime)
        state = read(HERE / 'STATUS.json')
        state.update(snapshot_utc=now.isoformat(), goal_status='active', last_goal_turn_classification='progress',
            resumed_goal_runtime=runtime, current_full_negative_experiments=reports,
            current_verified15_auc=scores['20191115'], current_verified_best_auc=scores['20191116'],
            target_verified=audit['goal_achieved'])
        write(HERE / 'STATUS.json', state)
        for name, filename in (
            ('full_negative_embedding_lgb_v3', 'full_negative_embedding_lgb_progress_v3.json'),
            ('full_negative_history_dot_embeddings_v1', 'full_negative_history_dot_embeddings_progress_v1.json'),
            ('full_negative_dense_context_lgb_v1', 'full_negative_dense_context_lgb_progress_v1.json')):
            write(HERE / filename, dict(timestamp_utc=now.isoformat(), **reports[name]))
    print(json.dumps(dict(classification='progress', scores=scores, targets_met=audit['goal_achieved'],
        experiments={name: dict(status=entry['status'], saved_models=entry['saved_model_count'],
            session_id=entry['session_id']) for name, entry in reports.items()},
        preserved_runtime=str(archive)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
