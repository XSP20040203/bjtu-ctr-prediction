"""Capture the completed family gain, terminal CPU failure and real recovery handles."""
import json
import shutil
from datetime import datetime, timezone
from conditional_daily import read
from fit_stacker import HERE, ROOT, digest, write
from publish_progress import publication_lock
from record_goal_continuation_v3 import observed_processes, summary, EXPERIMENTS


def main():
    observed = observed_processes()
    reports = {name: summary(name, observed) for name in EXPERIMENTS}
    now = datetime.now(timezone.utc)
    family = HERE / 'runs/refresh_family_quantiles_v1'
    verified = read(family / 'verification.json')
    assert verified['status'] == 'passed'
    count = sum(len(value) for value in read(family / 'predictions_locked.json')['dates'].values())
    assert count == 730
    with publication_lock():
        audit = read(ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json')
        assert audit['audit_status'] == 'passed'
        scores = {key: value['auc'] for key, value in audit['dates'].items()}
        archive = HERE / 'runtime_history' / now.strftime('%Y%m%dT%H%M%S%fZ')
        archive.mkdir(parents=True, exist_ok=False)
        for filename in ('resumed_processes_v1.json', 'resumed_goal_runtime_v1.json', 'STATUS.json'):
            if (HERE / filename).exists():
                shutil.copy2(HERE / filename, archive / filename)
        completion = HERE / 'refresh_family_quantiles_completed_v1.json'
        if not completion.exists():
            previous = read(family / 'selection_audit_snapshot.json')
            write(completion, dict(timestamp_utc=now.isoformat(), session_id=45474, exit_code=0,
                status='independently_verified_complete', models_fitted=0, total_candidates=count,
                scores={key: value['auc'] for key, value in verified['dates'].items()},
                previous_scores={key: value['auc'] for key, value in previous['dates'].items()},
                gains={key: verified['dates'][key]['auc'] - previous['dates'][key]['auc'] for key in verified['dates']},
                best={key: value['best'] for key, value in verified['dates'].items()},
                maximum_prediction_replay_error=max(value['maximum_independent_prediction_error']
                    for value in verified['dates'].values()),
                raw_datasets_unchanged=audit['original_datasets_unchanged'], inferred17_labels_read=False,
                development_dates_repeatedly_reused=True, files_sha256={str(path): digest(path)
                    for path in [family / filename for filename in ('protocol.json', 'inputs_locked.json',
                        'predictions_locked.json', 'verification.json')]}))
        runtime = read(HERE / 'resumed_goal_runtime_v1.json')
        delivery = runtime['latest_day17_delivery']
        assert digest(delivery['submission']) == delivery['submission_sha256']
        runtime.update(timestamp_utc=now.isoformat(), goal_status='active',
            previous_goal_turn_classification=runtime.get('last_goal_turn_classification'),
            previous_goal_turn_evidence=runtime.get('last_goal_turn_evidence'),
            last_goal_turn_classification='progress',
            last_goal_turn_evidence='Completed730 refreshed family candidates, independent historic forecast/rank/mask replay and complete CSV audit; both-date AUC improved to .8023586867696388/.8139803604311606. Original CPU LightGBM terminated with native0xC0000005 after tree1600; Windows event1000 confirms exact PID58516/module. Preserved v1 native and v2 Unicode-absolute-path preflight failures. Actual semantic GPU recovery4745 has saved all6 day15 checkpoints and started complete2104337-row16 training. Native probe proves identical file save/load predictions through ASCII relative paths. LGB v3 session77392/wrapper61136 preflights direct native callback checkpoints and keep_training_booster, then waits for actual semantic wrapper54004 to exit successfully.',
            scores=scores, targets={key: value['target_auc'] for key, value in audit['dates'].items()},
            both_targets_met=audit['goal_achieved'], original_datasets_unchanged=audit['original_datasets_unchanged'],
            advertisement_inferred17_labels_read=False, observed_processes=observed,
            current_experiments=reports, preserved_runtime_snapshot=str(archive),
            native_lgb_v1_failure=read(HERE / 'runs/full_negative_embedding_lgb_v1/failure.json'),
            native_lgb_v2_failure=read(HERE / 'runs/full_negative_embedding_lgb_v2/failure.json'),
            native_lgb_ascii_path_probe=read(HERE / 'runs/native_lgb_ascii_path_probe_v1/verification.json'),
            semantic_original_queue_failure=read(HERE / 'semantic_embedding_queue_recovery_v2.json'),
            latest_day17_delivery=delivery,
            latest17_not_regenerated_after_new_development_improvement=True,
            next_actions=[
                'Observe actual semantic recovery session4745/wrapper54004 creation15:01:15.277938-05:00 and fit63104 creation15:01:39.880203-05:00. Complete12 models, all candidate locks and independent projection/model/candidate replay before publishing. One GPU owner only.',
                'Observe LightGBM repair session77392/wrapper61136 creation15:09:29.328551-05:00. Complete native callback/ASCII-relative-path save/reload preflight, then wait for actual semantic recovery wrapper54004 to exit with passed independent verification before full CPU training.',
                'Do not restart terminal LightGBM52215, failed v2 preflight59382 or original semantic queue19435. Preserve their frozen sources and failure evidence. V3 reliability repair must prove capacity and correctness in real production.',
                'Refreshed family composition session45474 exited0; preserve all730 candidate locks, independent verification, new both-date winners and completion record.',
                'Fresh target audit remains authoritative; targets15>=.815 and16>=.82 are both unmet. Development dates are repeatedly reused.',
                'Keep original four CSVs and latest17 v6 delivery unchanged; never read inferred17 labels.'])
        runtime.setdefault('handles', {}).update(full_negative_embedding_lgb_pipeline_session=52215,
            full_negative_embedding_lgb_pipeline_terminal_exit_code=-1073741819,
            full_negative_semantic_embeddings_original_pipeline_session=19435,
            full_negative_semantic_embeddings_pipeline_session=4745,
            full_negative_embedding_lgb_v2_pipeline_session=59382,
            full_negative_embedding_lgb_v2_pipeline_terminal_exit_code=1,
            full_negative_embedding_lgb_v3_pipeline_session=77392,
            refresh_family_quantiles_pipeline_session=45474)
        write(HERE / 'resumed_processes_v1.json', observed)
        write(HERE / 'resumed_goal_runtime_v1.json', runtime)
        state = read(HERE / 'STATUS.json')
        state.update(snapshot_utc=now.isoformat(), goal_status='active', last_goal_turn_classification='progress',
            resumed_goal_runtime=runtime, current_full_negative_experiments=reports,
            current_verified15_auc=scores['20191115'], current_verified_best_auc=scores['20191116'],
            target_verified=audit['goal_achieved'])
        write(HERE / 'STATUS.json', state)
        for name, filename in (
            ('full_negative_embedding_lgb_v1', 'full_negative_embedding_lgb_progress_v1.json'),
            ('full_negative_embedding_lgb_v2', 'full_negative_embedding_lgb_progress_v2.json'),
            ('full_negative_embedding_lgb_v3', 'full_negative_embedding_lgb_progress_v3.json'),
            ('full_negative_semantic_embeddings_v1', 'full_negative_semantic_embeddings_progress_v1.json'),
            ('refresh_family_quantiles_v1', 'refresh_family_quantiles_progress_v1.json')):
            write(HERE / filename, dict(timestamp_utc=now.isoformat(), **reports[name]))
    print(json.dumps(dict(timestamp_utc=now.isoformat(), classification='progress', scores=scores,
        targets_met=audit['goal_achieved'], experiments={key:dict(status=value['status'],
        saved_model_count=value['saved_model_count'], session_id=value['session_id'])
        for key, value in reports.items()}, preserved_runtime=str(archive)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
