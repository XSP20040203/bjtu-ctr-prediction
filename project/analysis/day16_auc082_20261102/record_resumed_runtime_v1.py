"""Record observed continuation handles without reviving stale optimization queues."""
from datetime import datetime, timezone
from pathlib import Path
from conditional_daily import read
from fit_stacker import HERE, ROOT, digest, write


def main():
    processes = read(HERE / 'resumed_processes_v1.json')
    audit = read(ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json')
    assert audit['audit_status'] == 'passed'
    names = ['compose_transition_priors_v1', 'categorical_interactions_v1',
             'compose_probability_quantiles_v1', 'transition_posterior_v1', 'row_deepfm_v2',
             'compact_lgb_rows_v1', 'compose_family_quantiles_v1', 'group_event_posterior_v1',
             'full_negative_context_v1', 'full_negative_context_v2', 'full_rich_negative_v1',
             'full_population_original_v1', 'full_rich_pair_ranker_v1',
             'full_rich_negative_v2', 'full_rich_pair_ranker_v2', 'compose_full_negative_masks_v1']
    experiments = {}
    for name in names:
        run = HERE / 'runs' / name
        files = {filename: digest(run / filename) for filename in
                 ['protocol.json', 'inputs_locked.json', 'pairs_locked.json', 'models_locked.json',
                  'predictions_locked.json', 'pair_verification.json', 'verification.json']
                 if (run / filename).is_file()}
        report = read(run / 'verification.json') if (run / 'verification.json').is_file() else None
        experiments[name] = dict(run=str(run), files_sha256=files,
            verification=report['status'] if report else 'pending',
            feature_only=report.get('feature_only', False) if report else name == 'full_population_original_v1',
            aucs={date: record['auc'] for date, record in report.get('dates', {}).items()
                  if 'auc' in record} if report else {})
    queued_sources = [HERE / filename for filename in ['start_followup_optimization_v1.ps1',
        'start_followup_deepfm_v2.ps1', 'run_row_deepfm_v2.ps1', 'row_deepfm_v2.py',
        'row_deepfm_core_v1.py', 'test_row_deepfm_core_v1.py', 'start_followup_deepfm_v3.ps1',
        'run_full_negative_context_v1.ps1', 'full_negative_context_v1.py',
        'full_negative_context_inputs_v1.py', 'full_negative_context_core_v1.py',
        'test_full_negative_context_core_v1.py', 'full_negative_context_v2.py',
        'run_full_negative_context_v2.ps1', 'prepare_full_negative_context_v2.py',
        'full_population_original_v1.py', 'run_full_population_original_v1.ps1',
        'full_rich_negative_v1.py', 'full_rich_negative_inputs_v1.py',
        'test_full_rich_negative_inputs_v1.py', 'run_full_rich_negative_v1.ps1',
        'full_rich_pair_ranker_v1.py', 'full_negative_pair_core_v1.py',
        'test_full_negative_pair_core_v1.py', 'run_full_rich_pair_ranker_v1.ps1',
        'audit_full_negative_pair_population_v1.py', 'prepare_full_rich_gpu_recovery_v2.py',
        'full_rich_negative_v2.py', 'full_rich_pair_ranker_v2.py',
        'run_full_rich_negative_v2.ps1', 'run_full_rich_pair_ranker_v2.ps1',
        'diagnose_full_negative_errors_v1.py', 'full_negative_mask_core_v1.py',
        'test_full_negative_mask_core_v1.py', 'compose_full_negative_masks_v1.py',
        'run_compose_full_negative_masks_v1.ps1']]
    record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        goal_status='active', last_goal_turn_classification='progress',
        last_goal_turn_evidence='Observed both day15 full576-column fits complete and day16 training live; completed exact full-cohort positive/negative AUC-error decomposition, finding cold-task fractions12.5%/7.4% and lower cold-task AUC; implemented a strict-past cohort-conditioned transfer experiment, passed temporal exclusion/unseen-key/duplicate-query/independent aggregation tests and queued its CPU execution behind the actual576-column fit process with a3GiB memory guard',
        scores={date: report['auc'] for date, report in audit['dates'].items()},
        targets={date: report['target_auc'] for date, report in audit['dates'].items()},
        both_targets_met=audit['goal_achieved'], original_datasets_unchanged=audit['original_datasets_unchanged'],
        advertisement_inferred17_labels_read=False,
        handles=dict(categorical_pipeline_session=71238, quantile_and_posterior_queue_session=32174,
                     row_deepfm_prepare_session=73514, row_deepfm_queue_session=85018,
                     compact_lgb_pipeline_session=24019, family_quantile_pipeline_session=37357,
                     group_event_pipeline_session=14988, row_deepfm_recovered_pipeline_session=98937,
                     failed_full_negative_pipeline_session=27513,
                     full_negative_recovered_pipeline_session=79056,
                     full_population_original_pipeline_session=22838,
                     full_rich_negative_pipeline_session=96564,
                     full_rich_pair_pipeline_session=78422,
                     full_negative_pair_population_audit_session=39735,
                     full_rich_negative_recovered_pipeline_session=59324,
                     full_rich_pair_recovered_pipeline_session=73587,
                     full_negative_error_diagnosis_session=29145,
                     full_negative_mask_composition_session=1340),
        observed_processes=processes, experiments=experiments,
        queued_sources_sha256={str(path): digest(path) for path in queued_sources},
        preserved_interrupted_categorical_preflight=str(HERE / 'runs/categorical_interactions_preflight_interrupted_v1'),
        preserved_failed_neural_source=str(HERE / 'row_deepfm_v1.py'),
        neural_recovery=str(HERE / 'row_deepfm_recovery_v2.json'),
        neural_queue_recovery=str(HERE / 'row_deepfm_queue_recovery_v3.json'),
        full_negative_inputs=read(HERE / 'runs/full_negative_context_v1/inputs_locked.json')
            if (HERE / 'runs/full_negative_context_v1/inputs_locked.json').exists() else None,
        full_negative_recovery=str(HERE / 'full_negative_recovery_v2.json'),
        full_negative_real_preflight=read(HERE / 'runs/full_negative_context_v2/preflight.json')
            if (HERE / 'runs/full_negative_context_v2/preflight.json').exists() else None,
        full_rich_real_preflight=read(HERE / 'runs/full_rich_negative_preflight_v1/verification.json')
            if (HERE / 'runs/full_rich_negative_preflight_v1/verification.json').exists() else None,
        full_rich_pair_real_preflight=read(HERE / 'runs/full_rich_pair_preflight_v1/verification.json')
            if (HERE / 'runs/full_rich_pair_preflight_v1/verification.json').exists() else None,
        full_negative_pair_population_audit=read(HERE / 'full_negative_pair_population_audit_v1.json')
            if (HERE / 'full_negative_pair_population_audit_v1.json').exists() else None,
        full_rich_gpu_recovery=str(HERE / 'full_rich_gpu_recovery_v2.json'),
        full_rich_gpu_native_preflight=read(HERE / 'full_rich_gpu_native_preflight_v2.json'),
        full_rich_v2_queue_handle=read(HERE / 'full_rich_v2_queue_handle.json'),
        full_negative_error_diagnosis=str(HERE / 'error_diagnosis_full_negative_v1.json'),
        recovered_full_rich_real_preflight=read(HERE / 'runs/full_rich_negative_preflight_v2/verification.json')
            if (HERE / 'runs/full_rich_negative_preflight_v2/verification.json').exists() else None,
        recovered_full_rich_pair_real_preflight=read(HERE / 'runs/full_rich_pair_preflight_v2/verification.json')
            if (HERE / 'runs/full_rich_pair_preflight_v2/verification.json').exists() else None,
        full_population_original_inputs=read(HERE / 'runs/full_population_original_v1/inputs_locked.json')
            if (HERE / 'runs/full_population_original_v1/inputs_locked.json').exists() else None,
        neural_real_preflight=read(HERE / 'row_deepfm_real_preflight_v2.json')
            if (HERE / 'row_deepfm_real_preflight_v2.json').exists() else None,
        next_actions=['Categorical session71238 finished successfully; original best was retained',
                      'Compact LightGBM session24019 completed both-date independent replay and scoring without improvement; allow publication to finish if wrapper11424 remains live',
                      'Family quantile pipeline37357 completed successfully and both-date improvement was audited',
                      'Group-event posterior14988 completed successfully and both-date improvement was independently audited',
                      'Row DeepFM v2 input preparation73514 and real-source preflight completed successfully',
                      'Quantile/posterior successor32174 completed successfully; current best retained',
                      'Old DeepFM queue85018 failed safely because preparer PID31532 was reused; preserve it',
                      'Recovered DeepFM session98937 completed successfully; full independent replay passed, current best retained',
                      'Full-negative pipeline27513 completed both primary populations but failed CPU preflight at the CatBoost base constructor; preserve it',
                      'Full-negative pipeline79056 completed all fits and full independent replay; both-date improvement is published and independently audited',
                      'Full-population original feature pipeline22838 completed both-date full independent rebuild; its feature-only verifier passed and has no AUC',
                      'Old576-column queues96564 and78422 were intentionally stopped before fits after reproducing the pinned-memory integer API failure; preserve all sources and CPU preflights',
                      'Poll recovered576-column pointwise session59324/wrapper41144 without restarting while live; v2 uses string1GB and native GPU options preflight passed',
                      'Poll recovered pair-ranking session73587; every earlier negative participates in date-isolated ranking, and GPU ownership follows actual recovered pointwise wrapper41144',
                      'Error diagnosis29145 completed on both full cohorts with no model changes; cold tasks have weaker ranking, motivating predefined historical masks',
                      'Poll CPU cohort-transfer composition1340; it waits for actual pointwise fit PID32748 to exit and3GiB free RAM, then locks all both-date predictions before scoring',
                      'After experiments pass, inspect latest audit for both complete-cohort targets'])
    write(HERE / 'resumed_goal_runtime_v1.json', record)
    state = read(HERE / 'STATUS.json')
    state.update(goal_status='active', last_goal_turn_classification='progress',
                 resumed_goal_runtime=record, snapshot_utc=record['timestamp_utc'],
                 current_verified15_auc=record['scores']['20191115'],
                 current_verified_best_auc=record['scores']['20191116'], target_verified=record['both_targets_met'])
    write(HERE / 'STATUS.json', state)
    print({key: record[key] for key in ['scores', 'both_targets_met', 'original_datasets_unchanged', 'handles']}, flush=True)


if __name__ == '__main__':
    main()
