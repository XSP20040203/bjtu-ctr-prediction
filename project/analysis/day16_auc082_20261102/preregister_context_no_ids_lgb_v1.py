"""Freeze an identifier ablation before any new full-population fit or scores."""
from pathlib import Path
from datetime import datetime,timezone
from conditional_daily import read
from fit_stacker import HERE,ROOT,digest,write
from record_goal_continuation_v11 import observed_processes
import full_negative_context_no_ids_lgb_v1 as experiment


def main():
    target=HERE/'context_no_ids_lgb_queued_v1.json'
    assert not target.exists() and not experiment.OUT.exists()
    live=observed_processes()
    processes={}
    for pid in (29148,74360,61976,56348):
        row=next(p for p in live if p['process_id']==pid)
        assert row['identity_and_creation_time_revalidated']
        processes[str(pid)]=row
    fixture_path=HERE/'runs/context_no_ids_native_fixture_v1/verification.json'
    fixture=read(fixture_path)
    assert fixture['status']=='passed' and fixture['columns']==389
    assert len(fixture['actual_native_models'])==6
    assert not fixture['held_advertisement_labels_read'] and not fixture['auc_evaluated']
    for filename,expected in fixture['sources_sha256'].items():
        assert digest(filename)==expected,filename
    diag=read(experiment.DIAG)
    assert diag['status']=='checked' and not diag['auc_evaluated']
    hashes=read(experiment.PARENT/'protocol.json')['raw_datasets_sha256']
    assert hashes==read(experiment.BASE/'protocol.json')['raw_datasets_sha256']
    for filename,expected in hashes.items():
        assert digest(ROOT/'2026ml'/filename)==expected,filename
    paths=[Path(__file__),fixture_path,experiment.DIAG]
    paths += [HERE/name for name in ('full_negative_context_no_ids_lgb_v1.py',
        'audit_context_no_ids_projection_v1.py','run_full_negative_context_no_ids_lgb_v1.ps1',
        'diagnose_dense_context_feature_use_v1.py','publish_progress_v5.py','record_two_day_progress_v5.py')]
    paths += [experiment.PARENT/name for name in ('protocol.json','inputs_locked.json','preflight.json','full_projection_verification.json')]
    paths += [experiment.BASE/name for name in ('protocol.json','verification.json')]
    plan=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),status='preregistered_waiting_existing_dense_cohort_queue',
        experiment=str(experiment.OUT),baseline=str(experiment.BASE),removed_features=experiment.REMOVE,
        retained_columns=389,arms=experiment.ARMS,params=experiment.PARAMS,checkpoints=experiment.CHECKPOINTS,
        total_planned_model_checkpoints=12,training='All strictly earlier rows; unchanged class weights and tree parameters; no concurrent full fit',
        native_fixture=str(fixture_path),fixture_status='passed',fixture_projection_rows=8000,
        raw_identifier_split_diagnostic=str(experiment.DIAG),
        interpretation='Identifier split concentration motivates an ablation; it does not establish overfitting or a gain.',
        sequential_after_predecessor=processes['56348'],current_live_processes=processes,
        skip_further_fits_if_both_targets_already_audited=True,
        both_date_candidate_hashes_required_before_scoring=True,complete_independent_model_and_projection_replay_required=True,
        held_labels_read_during_preregistration=False,inferred17_labels_read=False,
        original_datasets_unchanged=True,raw_datasets_sha256=hashes,
        sources_sha256={str(path):digest(path) for path in paths})
    write(target,plan)
    print(dict(status=plan['status'],columns=389,checkpoints=12,predecessor=56348,fixture='passed'),flush=True)


if __name__=='__main__':
    main()
