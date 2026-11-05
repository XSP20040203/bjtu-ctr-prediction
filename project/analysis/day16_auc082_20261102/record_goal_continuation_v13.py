"""Extend generic live evidence to the frozen sequential identifier ablation."""
import json
import re
import subprocess
from datetime import datetime,timezone
from pathlib import Path
from conditional_daily import read
from fit_stacker import HERE,digest,write
from publish_progress import publication_lock
import record_goal_continuation_v12 as base

NAME='full_negative_context_no_ids_lgb_v1'
previous_observer=base.observed_processes


def observed_processes():
    observed=previous_observer()
    launch=read(HERE/'context_no_ids_lgb_launch_v1.json')
    expected=launch['wrapper']
    command="""$taskRecords = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -in @('python.exe','pwsh.exe') -and
        $_.CommandLine -match 'full_negative_context_no_ids_lgb_v1' -and
        $_.CommandLine -notmatch 'Get-CimInstance Win32_Process' -and
        $_.CommandLine -notmatch ' -Command '
    } | ForEach-Object { [pscustomobject]@{
        process_id=$_.ProcessId; parent_process_id=$_.ParentProcessId;
        creation_date=$_.CreationDate.ToString('o'); command_line=$_.CommandLine
    } }); ConvertTo-Json -InputObject $taskRecords -Depth 3 -Compress"""
    result=subprocess.run([base.reporting.SHELL,'-NoProfile','-Command',command],
        check=True,capture_output=True,text=True,encoding='utf-8-sig')
    for row in json.loads(result.stdout):
        if row['process_id']==expected['process_id']:
            assert row['command_line']==expected['command_line']
            def parsed(value):
                return datetime.fromisoformat(re.sub(r'([.][0-9]{6})[0-9]+',r'\1',value))
            assert abs((parsed(row['creation_date'])-parsed(expected['creation_date'])).total_seconds())<.01
            row['identity_and_creation_time_revalidated']=True
        observed.append(row)
    assert len({p['process_id'] for p in observed})==len(observed)
    return observed


def main():
    launch_path=HERE/'context_no_ids_lgb_launch_v1.json'
    launch=read(launch_path)
    plan_path=Path(launch['plan'])
    assert digest(plan_path)==launch['plan_sha256']
    plan=read(plan_path)
    for filename,expected in plan['sources_sha256'].items():
        assert digest(filename)==expected,filename
    base.observed_processes=observed_processes
    base.reporting.EXPERIMENTS[NAME]=(launch['session_id'],12)
    base.TRACKED=base.TRACKED+((NAME,'context_no_ids_lgb_progress_v1.json'),)
    base.main()
    with publication_lock():
        runtime=read(HERE/'resumed_goal_runtime_v1.json')
        previous=runtime.get('queued_context_no_ids_ablation',{})
        added=previous.get('plan_sha256')!=launch['plan_sha256']
        fixture=read(plan['native_fixture'])
        evidence=dict(plan=str(plan_path),plan_sha256=launch['plan_sha256'],launch=str(launch_path),
            launch_sha256=digest(launch_path),session_id=launch['session_id'],wrapper=launch['wrapper'],
            removed_features=plan['removed_features'],columns=389,expected_checkpoints=12,
            fixture_status=fixture['status'],fixture_projection_rows=plan['fixture_projection_rows'],
            fixture_native_models=len(fixture['actual_native_models']),
            raw_identifier_split_diagnostic=plan['raw_identifier_split_diagnostic'],
            newly_recorded_experiment=added,recorder_sha256=digest(Path(__file__)))
        runtime['queued_context_no_ids_ablation']=evidence
        runtime.setdefault('handles',{})['context_no_ids_lgb_pipeline_session']=launch['session_id']
        runtime['last_goal_turn_evidence_details']['context_no_ids_ablation']=evidence
        if added:
            runtime['last_goal_turn_classification']='progress'
            runtime['last_goal_turn_evidence'] += (
                ' This turn also inspected completed day15 native feature split counts, implemented '
                'a separate389-column six-identifier ablation, passed8000 independent feature fixture '
                'rows and six actual native save/reload checks, froze all sources and launched the '
                'identity-bound sequential wrapper after existing dense cohort queue56348. '
                'No concurrent full-population fit or new held-score claim.')
        state=read(HERE/'STATUS.json')
        state.update(last_goal_turn_classification=runtime['last_goal_turn_classification'],
            resumed_goal_runtime=runtime,queued_context_no_ids_ablation=evidence)
        write(HERE/'resumed_goal_runtime_v1.json',runtime)
        write(HERE/'STATUS.json',state)
    print(json.dumps(dict(classification=runtime['last_goal_turn_classification'],
        no_ids_status=runtime['current_experiments'][NAME]['status'],session_id=launch['session_id'],
        goal_status=runtime['goal_status'],scores=runtime['scores'])),flush=True)


if __name__=='__main__':
    main()
