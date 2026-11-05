"""Record completed neural evidence and current live handles without losing history."""
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from conditional_daily import read
from fit_stacker import HERE, ROOT, digest, write
from publish_progress import publication_lock

SHELL = 'C:/Users/30272/AppData/Local/Programs/PowerShell/7/pwsh.exe'
EXPERIMENTS = {
    'full_negative_class_balance_v1': (3346, 18),
    'full_negative_neural_embeddings_v2': (65178, 12),
    'compose_class_balance_masks_v1': (19462, 0),
    'compose_neural_embedding_masks_v1': (93920, 0),
    'full_negative_embedding_lgb_v1': (52215, 12),
    'full_negative_semantic_embeddings_v1': (4745, 12),
    'refresh_family_quantiles_v1': (45474, 0),
    'full_negative_embedding_lgb_v2': (59382, 12),
    'full_negative_embedding_lgb_v3': (77392, 12),
}


def observed_processes():
    command = """$taskRecords = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @('python.exe','pwsh.exe') -and
        $_.CommandLine -match 'full_negative_class_balance_v1|full_negative_neural_embeddings_v2|compose_class_balance_masks_v1|compose_neural_embedding_masks_v1|full_negative_embedding_lgb_v[123]|full_negative_semantic_embeddings_v[12]|refresh_family_quantiles_v1' -and
        $_.CommandLine -notmatch 'Get-CimInstance Win32_Process' -and
        $_.CommandLine -notmatch ' -Command '
    } | ForEach-Object { [pscustomobject]@{
        process_id=$_.ProcessId; parent_process_id=$_.ParentProcessId;
        creation_date=$_.CreationDate.ToString('o'); command_line=$_.CommandLine
    } }); ConvertTo-Json -InputObject $taskRecords -Depth 3 -Compress"""
    result = subprocess.run([SHELL, '-NoProfile', '-Command', command], check=True,
        capture_output=True, text=True, encoding='utf-8-sig')
    return json.loads(result.stdout)


def summary(name, observed):
    folder = HERE / 'runs' / name
    active = [value for value in observed if f'{name}.py' in value['command_line']
              or f'run_{name}.ps1' in value['command_line']
              or (name == 'full_negative_semantic_embeddings_v1' and
                  'run_full_negative_semantic_embeddings_v2.ps1' in value['command_line'])]
    filenames = ('protocol.json', 'inputs_locked.json', 'preflight.json',
        'full_projection_verification.json', 'models_locked.json', 'predictions_locked.json',
        'verification.json', 'failure.json')
    hashes = {filename: digest(folder / filename) for filename in filenames if (folder / filename).is_file()}
    verification = read(folder / 'verification.json') if 'verification.json' in hashes else None
    status = ('verified' if verification and verification['status'] == 'passed'
        else 'failed' if 'failure.json' in hashes
        else 'running_fit' if any('--phase fit' in p['command_line'] for p in active)
        else 'running_evaluation' if any('--phase evaluate' in p['command_line'] for p in active)
        else 'running_verification' if any('--phase verify' in p['command_line'] for p in active)
        else 'preparing' if any('--phase prepare' in p['command_line'] for p in active)
        else 'prepared_waiting_predecessor' if active and 'preflight.json' in hashes
        else 'waiting_predecessor_or_memory' if active else 'not_observed_live')
    models = sorted(path.name for path in folder.glob('day*') if path.suffix in ('.cbm', '.txt'))
    preflight = read(folder / 'preflight.json') if 'preflight.json' in hashes else None
    verified = None
    if verification:
        verified = dict(status=verification['status'], dates={key: {
            field: value[field] for field in ('auc', 'best', 'rows', 'clicks',
                'maximum_independent_prediction_error', 'target_met') if field in value}
            for key, value in verification['dates'].items()})
    return dict(session_id=EXPERIMENTS[name][0], expected_models=EXPERIMENTS[name][1],
        status=status, run=str(folder), observed_processes=active, saved_model_count=len(models),
        saved_models=models, files_sha256=hashes, preflight=preflight, verification=verified)


def completion(name, filename, audit, delivery):
    path = HERE / filename
    if path.exists():
        return
    folder = HERE / 'runs' / name
    verified = read(folder / 'verification.json')
    assert verified['status'] == 'passed'
    candidates = read(folder / 'predictions_locked.json')['dates']
    models = read(folder / 'models_locked.json')['models'] if (folder / 'models_locked.json').exists() else []
    files = [folder / item for item in ('protocol.json', 'verification.json',
        'predictions_locked.json', 'inputs_locked.json')]
    if models:
        files.append(folder / 'models_locked.json')
    write(path, dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        status='independently_verified_complete', session_id=EXPERIMENTS[name][0],
        models=len(models), total_candidates=sum(len(value) for value in candidates.values()),
        scores={key: value['auc'] for key, value in verified['dates'].items()},
        best={key: value['best'] for key, value in verified['dates'].items()},
        maximum_prediction_replay_error=max(value['maximum_independent_prediction_error']
            for value in verified['dates'].values()),
        development_dates_repeatedly_reused=True, inferred17_labels_read=False,
        original_datasets_unchanged=verified['original_datasets_unchanged'],
        both_targets_met=False, latest_audited_scores={key: value['auc'] for key, value in audit['dates'].items()},
        current_day17_delivery=delivery, files_sha256={str(p): digest(p) for p in files}))


def main():
    observed = observed_processes()
    reports = {name: summary(name, observed) for name in EXPERIMENTS}
    now = datetime.now(timezone.utc)
    with publication_lock():
        audit = read(ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json')
        assert audit['audit_status'] == 'passed'
        delivery = read(ROOT / 'analysis/final_best_recipe_compact_dense_20261102/delivery_current_best_17_v6.json')
        assert delivery['request_complete'] and delivery['inferred17_labels_read'] is False
        assert digest(delivery['submission']) == delivery['submission_sha256']
        archive = HERE / 'runtime_history' / now.strftime('%Y%m%dT%H%M%S%fZ')
        archive.mkdir(parents=True, exist_ok=False)
        for filename in ('resumed_processes_v1.json', 'resumed_goal_runtime_v1.json', 'STATUS.json'):
            path = HERE / filename
            if path.is_file():
                shutil.copy2(path, archive / filename)
        completion('full_negative_neural_embeddings_v2',
            'full_negative_neural_embeddings_completed_v2.json', audit, delivery)
        completion('compose_neural_embedding_masks_v1',
            'neural_embedding_mask_composition_completed_v1.json', audit, delivery)
        runtime = read(HERE / 'resumed_goal_runtime_v1.json')
        scores = {key: value['auc'] for key, value in audit['dates'].items()}
        runtime.update(timestamp_utc=now.isoformat(), goal_status='active',
            previous_goal_turn_classification='progress',
            previous_goal_turn_evidence='Generated and independently replayed requested day17 v6 export using latest audited day16 recipe .8139543896488788;404182 CSV rows, ID order, probabilities and preserved raw-test hashes freshly checked; prior deliveries preserved.',
            last_goal_turn_classification='progress',
            last_goal_turn_evidence='Confirmed original LightGBM wrapper56696/creation13:55:05.133816-05:00 and fit58516/creation14:26:57.933321-05:00 still live; native output advanced to day15 leaf31 tree1000. Launched semantic session19435/wrapper55988/creation14:43:07.543410-05:00; all full training and held489-column projections passed independent bitwise replay on both dates, and real1088-row CPU fits and save/reload passed for d4/d6. GPU fits wait for identity-checked LightGBM memory handoff.',
            scores=scores, targets={key: value['target_auc'] for key, value in audit['dates'].items()},
            both_targets_met=audit['goal_achieved'], original_datasets_unchanged=audit['original_datasets_unchanged'],
            advertisement_inferred17_labels_read=False, observed_processes=observed,
            current_experiments=reports, preserved_runtime_snapshot=str(archive),
            latest_day17_delivery=delivery,
            next_actions=[
                'Observe existing LightGBM session52215 and original process identities; do not restart live work. Complete12 saved checkpoints, both-date candidate locks, independent full-cohort replay, publication and original CSV audit.',
                'Observe semantic ablation session19435/wrapper55988 after passed full primary/independent489-column projections and1088-row native CPU fits; wait for actual LightGBM wrapper56696 to exit and verification.json to pass before GPU fits.',
                'Preserve completed neural12-model/338-candidate and neural-mask1012-candidate verification evidence; newest audited winners are .8023220737217925/.8139543896488788.',
                'Select and audit further improvements using complete15/16 development populations only; these dates are repeatedly reused and are not blind tests.',
                'Keep all frozen sources, failed versions, original four CSVs and latest17 v6 delivery unchanged. Never read inferred17 labels.',
                'Goal remains active until independently verified complete15 AUC>=.815 and16 AUC>=.82.'])
        runtime.setdefault('handles', {}).update(full_negative_embedding_lgb_pipeline_session=52215,
            full_negative_semantic_embeddings_pipeline_session=19435)
        write(HERE / 'resumed_processes_v1.json', observed)
        write(HERE / 'resumed_goal_runtime_v1.json', runtime)
        state = read(HERE / 'STATUS.json')
        state.update(goal_status='active', last_goal_turn_classification='progress',
            resumed_goal_runtime=runtime, snapshot_utc=now.isoformat(),
            current_verified15_auc=scores['20191115'], current_verified_best_auc=scores['20191116'],
            target_verified=audit['goal_achieved'], current_full_negative_experiments=reports)
        write(HERE / 'STATUS.json', state)
        for name, filename in (
            ('full_negative_embedding_lgb_v1', 'full_negative_embedding_lgb_progress_v1.json'),
            ('full_negative_semantic_embeddings_v1', 'full_negative_semantic_embeddings_progress_v1.json'),
            ('full_negative_neural_embeddings_v2', 'full_negative_neural_embeddings_progress_v2.json'),
            ('compose_neural_embedding_masks_v1', 'neural_embedding_mask_composition_progress_v1.json')):
            write(HERE / filename, dict(timestamp_utc=now.isoformat(), **reports[name]))
    print(json.dumps(dict(timestamp_utc=now.isoformat(), scores=scores, both_targets_met=audit['goal_achieved'],
        experiments={key:dict(status=value['status'], saved_model_count=value['saved_model_count'],
            session_id=value['session_id']) for key, value in reports.items()},
        runtime_snapshot=str(archive)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
