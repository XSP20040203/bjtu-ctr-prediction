"""Preserve the terminal native restore failure and inspect checkpoint structure."""
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from conditional_daily import read
from fit_stacker import HERE, ROOT, digest, write

OUT = HERE / 'runs/full_negative_embedding_lgb_v3'
SHELL = 'C:/Users/30272/AppData/Local/Programs/PowerShell/7/pwsh.exe'


def main():
    assert not (OUT / 'failure.json').exists(), 'Preserve previous failure records'
    command = """$taskProcess = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -in @(57864,61136)
    } | Select-Object ProcessId,CreationDate,CommandLine);
    $taskEvents = @(Get-WinEvent -FilterHashtable @{LogName='Application';Id=1000;
        StartTime=[datetime]'2026-11-03T15:54:00-05:00'} -MaxEvents 16 -ErrorAction SilentlyContinue |
        Where-Object { $_.Message -match '0xE208' -and $_.Message -match 'lib_lightgbm.dll' } |
        ForEach-Object { [pscustomobject]@{created=$_.TimeCreated.ToString('o'); message=$_.Message} });
    [pscustomobject]@{processes=$taskProcess; events=$taskEvents} | ConvertTo-Json -Depth 5 -Compress"""
    result = subprocess.run([SHELL, '-NoProfile', '-Command', command], check=True,
        capture_output=True, text=True, encoding='utf-8-sig')
    observed = json.loads(result.stdout)
    assert not observed['processes'], 'Original process handles must be terminal'
    assert observed['events'], 'Require matching Windows native crash event'
    expected_start = datetime.fromisoformat('2026-11-03T15:24:36.725338-05:00')
    native_start = datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=int('1DD6BD158730787', 16)//10)
    assert abs((native_start-expected_start).total_seconds()) < .001
    checkpoints = []
    for path in sorted(OUT.glob('day*.txt')):
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
        checkpoints.append(dict(name=path.name, bytes=path.stat().st_size,
            sha256=digest(path), declared_tree_sizes_count=declared,
            actual_tree_header_count=trees, end_of_trees_present=end_trees,
            end_of_parameters_present=end_parameters,
            expected_tree_headers_complete=trees == declared))
    assert len(checkpoints) == 3 and not (OUT / 'models_locked.json').exists()
    protocol = read(OUT / 'protocol.json')
    for filename, expected in protocol['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), status='failed',
        session_id=77392, terminal_exit_code=-1073741819, terminal_exception='0xC0000005',
        fit_process_id=57864, wrapper_process_id=61136,
        fit_creation_date=expected_start.isoformat(), windows_native_event_start_time=native_start.isoformat(),
        process_handles_missing_after_terminal_session_exit=True,
        windows_event_application_pid_start_time_and_lib_lightgbm_module_matched=True,
        crash_location='full_negative_embedding_lgb_v3.py:246, lgb.Booster(model_file=native_path(path)); lightgbm/basic.py:3698',
        native_call='Saved checkpoint file restore after day15 leaf31 completed1600 iterations',
        exact_checkpoint_being_restored_not_logged=True,
        checkpoints=checkpoints, expected_models=12, models_locked=False,
        production_reliability_proven=False, independent_full_cohort_auc_available=False,
        sparse_category_size_is_observed_but_native_root_cause_not_proven=True,
        original_sources_checkpoints_and_failures_preserved=True,
        existing_history_dot_queue_proceeded_independently=True,
        new_dense_context_experiment_preparing_separately=True,
        original_datasets_unchanged=True, inferred17_labels_read=False,
        raw_datasets_sha256=protocol['raw_datasets_sha256'],
        protocol_sha256=digest(OUT / 'protocol.json'), windows_crash_events=observed['events'])
    write(OUT / 'failure.json', report)
    print(json.dumps(dict(status='failed', session_id=77392,
        checkpoint_structure=[{key: value for key, value in item.items() if key != 'sha256'} for item in checkpoints],
        native_root_cause_proven=False, original_datasets_unchanged=True), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
