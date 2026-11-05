"""Bind the real sequential wrapper identity and its existing exec handle."""
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path
from conditional_daily import read
from fit_stacker import HERE,digest,write


def main():
    target=HERE/'context_no_ids_lgb_launch_v1.json'
    assert not target.exists(), 'Preserve prior launch evidence'
    plan_path=HERE/'context_no_ids_lgb_queued_v1.json'
    plan=read(plan_path)
    for filename,expected in plan['sources_sha256'].items():
        assert digest(filename)==expected,filename
    command="""$taskRecords = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'pwsh.exe' -and
        $_.CommandLine -match 'run_full_negative_context_no_ids_lgb_v1.ps1' -and
        $_.CommandLine -notmatch ' -Command '
    } | ForEach-Object { [pscustomobject]@{
        process_id=$_.ProcessId; parent_process_id=$_.ParentProcessId;
        creation_date=$_.CreationDate.ToString('o'); command_line=$_.CommandLine
    } }); ConvertTo-Json -InputObject $taskRecords -Depth 3 -Compress"""
    result=subprocess.run(['C:/Users/30272/AppData/Local/Programs/PowerShell/7/pwsh.exe',
        '-NoProfile','-Command',command],check=True,capture_output=True,text=True,encoding='utf-8-sig')
    matches=json.loads(result.stdout)
    assert len(matches)==1,matches
    wrapper=matches[0]
    assert '-PredecessorId 56348' in wrapper['command_line']
    record=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),status='live_sequential_wait',
        session_id=66594,wrapper=wrapper,plan=str(plan_path),plan_sha256=digest(plan_path),
        previous_preregistration_session=27955,previous_preregistration_exit_code=0,
        fixture_session=92136,fixture_exit_code=0,queue_source_hashes_revalidated=True,
        no_additional_full_population_fit_started=True,inferred17_labels_read=False,
        recorder_sha256=digest(Path(__file__)))
    write(target,record)
    print(json.dumps(dict(status=record['status'],session_id=record['session_id'],wrapper=wrapper)),flush=True)


if __name__=='__main__':
    main()
