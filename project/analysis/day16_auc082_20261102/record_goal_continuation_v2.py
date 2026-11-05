"""Refresh observed process identities while preserving prior runtime snapshots."""
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
    'full_negative_class_balance_v1': dict(session=3346, expected_models=18),
    'full_negative_neural_embeddings_v2': dict(session=65178, expected_models=12),
    'full_negative_embedding_lgb_v1': dict(session=52215, expected_models=12),
    'compose_class_balance_masks_v1': dict(session=19462, expected_models=0),
    'compose_neural_embedding_masks_v1': dict(session=93920, expected_models=0),
}


def processes():
    command = """$taskRecords = @(Get-CimInstance Win32_Process | Where-Object {
        ($_.Name -eq 'python.exe' -or $_.Name -eq 'pwsh.exe') -and
        ($_.CommandLine -match 'full_negative_class_balance_v1.py|full_negative_neural_embeddings_v2.py|full_negative_embedding_lgb_v1.py|run_full_negative_class_balance_v1.ps1|run_full_negative_neural_embeddings_v2.ps1|run_full_negative_embedding_lgb_v1.ps1|compose_class_balance_masks_v1.py|run_compose_class_balance_masks_v1.ps1|compose_neural_embedding_masks_v1.py|run_compose_neural_embedding_masks_v1.ps1')
    } | ForEach-Object { [pscustomobject]@{
        process_id=$_.ProcessId; parent_process_id=$_.ParentProcessId;
        creation_date=$_.CreationDate.ToString('o'); command_line=$_.CommandLine
    } }); ConvertTo-Json -InputObject $taskRecords -Depth 3 -Compress"""
    result = subprocess.run([SHELL, '-NoProfile', '-Command', command],
        check=True, capture_output=True, text=True, encoding='utf-8-sig')
    records = json.loads(result.stdout)
    # The inventory PowerShell command itself embeds all search strings.
    return [record for record in records if 'Get-CimInstance Win32_Process' not in record['command_line']]


def main():
    observed = processes()
    now = datetime.now(timezone.utc)
    reports = {}
    for name, setup in EXPERIMENTS.items():
        folder = HERE / 'runs' / name
        active = [p for p in observed if f'{name}.py' in p['command_line']
            or f'run_{name}.ps1' in p['command_line']]
        checkpoints = [path.name for path in folder.glob('day*') if path.suffix in ('.cbm', '.txt')]
        files = {filename: digest(folder / filename) for filename in
            ('protocol.json', 'inputs_locked.json', 'preflight.json', 'models_locked.json',
             'predictions_locked.json', 'verification.json', 'failure.json') if (folder / filename).is_file()}
        preflight = read(folder / 'preflight.json') if 'preflight.json' in files else None
        verification = read(folder / 'verification.json') if 'verification.json' in files else None
        status = ('verified' if verification else 'running_fit' if any('--phase fit' in p['command_line'] for p in active)
            else 'running_verification' if any('--phase verify' in p['command_line'] for p in active)
            else 'running_evaluation' if any('--phase evaluate' in p['command_line'] for p in active)
            else 'preparing' if any('--phase prepare' in p['command_line'] for p in active)
            else 'prepared_waiting_predecessor' if active and preflight
            else 'waiting_predecessor_or_memory_guard' if active else 'not_observed_live')
        reports[name] = dict(session_id=setup['session'], status=status, run=str(folder),
            expected_models=setup['expected_models'], saved_model_count=len(checkpoints),
            saved_models=sorted(checkpoints), observed_processes=active, files_sha256=files,
            preflight=preflight, verification=verification)
    with publication_lock():
        audit = read(ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json')
        assert audit['audit_status'] == 'passed'
        archive = HERE / 'runtime_history' / now.strftime('%Y%m%dT%H%M%S%fZ')
        archive.mkdir(parents=True, exist_ok=False)
        for name in ('resumed_processes_v1.json', 'resumed_goal_runtime_v1.json', 'STATUS.json',
                     'full_negative_class_balance_progress_v1.json', 'full_negative_neural_embeddings_progress_v2.json',
                     'full_negative_embedding_lgb_progress_v1.json'):
            path = HERE / name
            if path.is_file():
                shutil.copy2(path, archive / name)
        runtime = read(HERE / 'resumed_goal_runtime_v1.json')
        scores = {day: value['auc'] for day, value in audit['dates'].items()}
        runtime.update(timestamp_utc=now.isoformat(), goal_status='active',
            previous_goal_turn_classification='progress',
            previous_goal_turn_evidence='Completed class-weight cohort composition19462:1010 independent candidate replays, both-date gains and full CSV audit .8021345525485678/.8138425288439717; neural fits advanced to the final16 deep configuration.',
            last_goal_turn_classification='progress',
            last_goal_turn_evidence='Neural-feature session65178 completed all12 saved checkpoints for full1679557-row15 and2104337-row16 training populations; fit PID61700 exited and actual evaluate PID57296/creation14:22:40.485115-05:00 is live. Added transfer onto the stronger class-weight cohort baseline with both source winners as controls;170 native fixtures passed independent replay at maximum5.56e-17, and queued session93920/wrapper27144 waits for actual neural completion and independent verification before full-cohort candidate generation.',
            scores=scores, targets={day: value['target_auc'] for day, value in audit['dates'].items()},
            both_targets_met=audit['goal_achieved'], original_datasets_unchanged=audit['original_datasets_unchanged'],
            advertisement_inferred17_labels_read=False, observed_processes=observed,
            current_experiments=reports, preserved_runtime_snapshot=str(archive),
            neural_v1_failure=read(HERE / 'runs/full_negative_neural_embeddings_v1/failure.json'),
            next_actions=[
                'Class-balance session3346 exited0; preserve its18 models,482 independently replayed candidates and both-date published improvements.',
                'Observe existing neural-feature session65178 and its actual fit process without restarting live work; complete12 models and all independent candidate replays.',
                'Observe full-negative LightGBM session52215; its CPU preparation leaves the GPU untouched and production waits for neural wrapper exit to release RAM.',
                'Class-weight cohort composition session19462 exited0; preserve all1010 independently replayed candidates, both-date improvements and completion record.',
                'Observe neural-feature transfer session93920; it must wait for actual neural wrapper60244/creation13:42:18.899626-05:00 to exit and its independent report to pass, then freeze both-date candidates before scoring.',
                'After each successful publication inspect fresh independent full-cohort target audit; never use inferred17 labels.',
                'Keep frozen sources, failures and all four original CSVs unchanged; targets remain15 >= .815 and16 >= .82.'])
        handles = runtime.setdefault('handles', {})
        handles.update(full_negative_class_balance_pipeline_session=3346,
            full_negative_neural_embeddings_pipeline_session=65178,
            full_negative_embedding_lgb_pipeline_session=52215,
            class_balance_mask_composition_session=19462,
            neural_embedding_mask_composition_session=93920)
        write(HERE / 'resumed_processes_v1.json', observed)
        write(HERE / 'resumed_goal_runtime_v1.json', runtime)
        state = read(HERE / 'STATUS.json')
        state.update(goal_status='active', last_goal_turn_classification='progress',
            resumed_goal_runtime=runtime, snapshot_utc=now.isoformat(),
            current_verified15_auc=scores['20191115'], current_verified_best_auc=scores['20191116'],
            target_verified=audit['goal_achieved'], current_full_negative_experiments=reports)
        write(HERE / 'STATUS.json', state)
        for name, record in reports.items():
            suffix = {'full_negative_class_balance_v1': 'full_negative_class_balance_progress_v1.json',
                'full_negative_neural_embeddings_v2': 'full_negative_neural_embeddings_progress_v2.json',
                'full_negative_embedding_lgb_v1': 'full_negative_embedding_lgb_progress_v1.json',
                'compose_class_balance_masks_v1': 'class_balance_mask_composition_progress_v1.json',
                'compose_neural_embedding_masks_v1': 'neural_embedding_mask_composition_progress_v1.json'}[name]
            write(HERE / suffix, dict(timestamp_utc=now.isoformat(), **record))
    print(json.dumps(dict(timestamp_utc=now.isoformat(), scores=scores,
        targets_met=audit['goal_achieved'], experiments={name:dict(status=r['status'],
        saved_model_count=r['saved_model_count'], session_id=r['session_id']) for name, r in reports.items()},
        previous_runtime_preserved=str(archive)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
