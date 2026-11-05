"""Record completed history-dot models, audited cohort gains and the actual dense CPU fit."""
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
    full_negative_dense_context_lgb_v1=(9696, 12),
    compose_history_dot_embedding_masks_v1=(59427, 0))
IDENTITIES = {
    64760: ('run_full_negative_history_dot_embeddings_v1.ps1', '2026-11-03T15:30:36.286956-05:00'),
    63696: ('full_negative_history_dot_embeddings_v1.py --phase fit', '2026-11-03T15:55:30.965111-05:00'),
    29148: ('run_full_negative_dense_context_lgb_v1.ps1', '2026-11-03T15:54:44.863567-05:00'),
    61976: ('run_compose_history_dot_embedding_masks_v1.ps1', '2026-11-03T16:04:48.318746-05:00'),
    74360: ('full_negative_dense_context_lgb_v1.py --phase fit', '2026-11-03T16:16:21.343889-05:00')}


def observed_processes():
    command = """$taskRecords = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @('python.exe','pwsh.exe') -and
        $_.CommandLine -match 'full_negative_class_balance_v1|full_negative_neural_embeddings_v2|compose_class_balance_masks_v1|compose_neural_embedding_masks_v1|full_negative_embedding_lgb_v[123]|full_negative_semantic_embeddings_v[12]|refresh_family_quantiles_v1|compose_semantic_embedding_masks_v1|full_negative_history_dot_embeddings_v1|full_negative_dense_context_lgb_v1|compose_history_dot_embedding_masks_v1' -and
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
        identity = IDENTITIES.get(record['process_id'])
        if identity:
            assert identity[0] in record['command_line']
            actual = datetime.fromisoformat(re.sub(r'([.][0-9]{6})[0-9]+', r'\1', record['creation_date']))
            assert abs((actual-datetime.fromisoformat(identity[1])).total_seconds()) < .01
            record['identity_and_creation_time_revalidated'] = True
    return observed


def main():
    observed = observed_processes()
    summaries = {name: reporting.summary(name, observed) for name in reporting.EXPERIMENTS}
    plan_path = HERE / 'history_dot_cohort_transfer_queued_v1.json'
    plan = read(plan_path)
    assert plan['total_candidates'] == 1012 and plan['fixture']['status'] == 'passed'
    assert plan['no_held_advertisement_labels_read_during_preregistration']
    for filename, expected in plan['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in plan['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    historical = read(HERE / 'runs/full_negative_history_dot_embeddings_v1/verification.json')
    cohort = read(HERE / 'runs/compose_history_dot_embedding_masks_v1/verification.json')
    assert historical['status'] == cohort['status'] == 'passed'
    assert all(entry['independent_all_saved_models_and_candidates_replayed'] for entry in historical['dates'].values())
    assert all(entry['independent_temporal_masks_and_all_candidates_replayed'] for entry in cohort['dates'].values())
    now = datetime.now(timezone.utc)
    with publication_lock():
        audit_path = ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json'
        audit = read(audit_path)
        assert audit['audit_status'] == 'passed' and audit['original_datasets_unchanged']
        for day in (15, 16):
            published = audit['dates'][str(20191100 + day)]
            assert digest(HERE / f'runs/compose_history_dot_embedding_masks_v1/day{day}_prediction.npy') == published['prediction_sha256']
            assert abs(published['auc']-cohort['dates'][str(20191100 + day)]['auc']) < 1e-12
        runtime = read(HERE / 'resumed_goal_runtime_v1.json')
        previous_classification = runtime.get('last_goal_turn_classification')
        previous_evidence = runtime.get('last_goal_turn_evidence')
        previous_scores = runtime.get('scores')
        delivery = runtime['latest_day17_delivery']
        assert 'current_best_20191117_20261103_v9' in delivery['submission']
        assert delivery['inferred17_labels_read'] is False
        assert digest(delivery['submission']) == delivery['submission_sha256']
        scores = {date: value['auc'] for date, value in audit['dates'].items()}
        archive = HERE / 'runtime_history' / now.strftime('%Y%m%dT%H%M%S%fZ_history_dot_cohort_and_dense_context')
        archive.mkdir(parents=True, exist_ok=False)
        for filename in ('STATUS.json', 'resumed_processes_v1.json', 'resumed_goal_runtime_v1.json',
            'full_negative_history_dot_embeddings_progress_v1.json',
            'full_negative_dense_context_lgb_progress_v1.json',
            'compose_history_dot_embedding_masks_progress_v1.json'):
            path = HERE / filename
            if path.exists():
                shutil.copy2(path, archive / filename)
        for name, filename in (
            ('full_negative_history_dot_embeddings_v1', 'full_negative_history_dot_embeddings_completed_v1.json'),
            ('compose_history_dot_embedding_masks_v1', 'history_dot_embedding_mask_composition_completed_v1.json'),
            ('full_negative_dense_context_lgb_v1', 'full_negative_dense_context_lgb_completed_v1.json')):
            if summaries[name]['status'] == 'verified':
                reporting.completion(name, filename, audit, delivery)
        evidence = dict(frozen_cohort_plan=str(plan_path), frozen_cohort_plan_sha256=digest(plan_path),
            candidates=1012, fixture_candidates=plan['fixture']['fixture_candidates'],
            maximum_fixture_replay_error=plan['fixture']['maximum_independent_prediction_error'],
            scores_changed_since_previous_record=previous_scores != scores,
            authoritative_audit=str(audit_path), authoritative_audit_sha256=digest(audit_path),
            observed_status={name: dict(status=record['status'], saved_model_count=record['saved_model_count'])
                for name, record in summaries.items() if name in (
                    'full_negative_history_dot_embeddings_v1', 'full_negative_dense_context_lgb_v1',
                    'compose_history_dot_embedding_masks_v1')})
        runtime.update(timestamp_utc=now.isoformat(), goal_status='active',
            previous_goal_turn_classification=previous_classification,
            previous_goal_turn_evidence=previous_evidence,
            last_goal_turn_classification='progress',
            last_goal_turn_evidence='Completed history-dot session59964 exit0:12 full-population d4/d6 checkpoints and338 global candidates independently replayed; best global candidates remained the older baseline, so no direct promotion. Completed1012 preregistered history-dot cohort candidates against the semantic-mask winner with independent strict-past masks and all-formula replay. Published both improved complete-cohort predictions and fresh sklearn/CSV/raw-data audit:15=.8023895973270289,16=.8140787742362926.15 selected history_dot_d6/user_no_past_ad_click/add_within/.0125;16 selected history_dot_d6/task_at_most_100_prior_rows/replace_global/.1. Actual dense-context CPU fit now runs as PID74360/creation16:16:21.343889-05:00 under wrapper29148; do not restart. Cohort wrapper61976 remains live only to refresh the complete publisher inventory after dense wrapper29148 exits. Original four CSVs and requested17 v9 submission remain unchanged; no inferred17 labels.',
            last_goal_turn_evidence_details=evidence,
            scores=scores, targets={date: entry['target_auc'] for date, entry in audit['dates'].items()},
            both_targets_met=audit['goal_achieved'], original_datasets_unchanged=True,
            advertisement_inferred17_labels_read=False, observed_processes=observed,
            current_experiments=summaries, preserved_runtime_snapshot=str(archive),
            latest_day17_delivery=delivery, latest17_not_regenerated_during_goal_optimization=True,
            next_actions=[
                'Observe actual history-dot session59964. Preserve process identity/creation dates; complete12 checkpoints, both-date prediction locks, independent full-cohort replay and publisher/audit. Do not restart because observation timed out.',
                'Observe dense-context session9696, wrapper29148 creation15:54:44.863567-05:00.395-column complete4,630,657-row dictionary/encoding replay and32768-row native callback/file restore preflight passed. Full CPU leaf31/weight5 andleaf63/weight25 fits wait for actual history-dot wrapper64760 exit and8GiB available RAM.',
                'Observe cohort session59427, wrapper61976 creation16:04:48.318746-05:00. Frozen1012 candidates and170-case fixture passed. It requires verified history-dot source margins; freeze every candidate on both dates before scoring and independently rebuild all temporal masks and formulas.',
                'Cohort wrapper republishes the v3 complete inventory after actual dense-context wrapper29148 exit. Preserve v1/v2 publisher source versions and all frozen experiment protocols.',
                'Preserve LightGBM v3 terminal0xC0000005 failure and its three saved files.1600 file has only1341/1600 trees; exact checkpoint at restore crash was not logged. No native root cause claim or restart of frozen failed versions.',
                'Use only completed independently verified complete15/16 development results for promotion. Targets remain15>=.815 and16>=.82; repeatedly reused development sets are not blind tests. Preserve original four CSVs and requested17 v9 file; never read inferred17 labels.'])
        runtime.setdefault('handles', {}).update(full_negative_history_dot_embeddings_pipeline_session=59964,
            full_negative_history_dot_embeddings_pipeline_terminal_exit_code=0,
            full_negative_dense_context_lgb_pipeline_session=9696,
            compose_history_dot_embedding_masks_pipeline_session=59427,
            compose_history_dot_embedding_masks_initial_publication_complete=True,
            compose_history_dot_embedding_masks_post_dense_refresh_pending=True)
        write(HERE / 'resumed_processes_v1.json', observed)
        write(HERE / 'resumed_goal_runtime_v1.json', runtime)
        state = read(HERE / 'STATUS.json')
        state.update(snapshot_utc=now.isoformat(), goal_status='active', last_goal_turn_classification='progress',
            resumed_goal_runtime=runtime, current_full_negative_experiments=summaries,
            current_verified15_auc=scores['20191115'], current_verified_best_auc=scores['20191116'],
            target_verified=audit['goal_achieved'])
        write(HERE / 'STATUS.json', state)
        for name, filename in (
            ('full_negative_history_dot_embeddings_v1', 'full_negative_history_dot_embeddings_progress_v1.json'),
            ('full_negative_dense_context_lgb_v1', 'full_negative_dense_context_lgb_progress_v1.json'),
            ('compose_history_dot_embedding_masks_v1', 'compose_history_dot_embedding_masks_progress_v1.json')):
            write(HERE / filename, dict(timestamp_utc=now.isoformat(), **summaries[name]))
    print(json.dumps(dict(classification='progress', scores=scores, target_met=audit['goal_achieved'],
        tracked=evidence['observed_status'], archive=str(archive)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
