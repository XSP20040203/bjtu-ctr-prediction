"""Freeze dense checkpoint cohort rules while its existing fit remains live."""
from datetime import datetime, timezone
from pathlib import Path
from conditional_daily import read
from fit_stacker import HERE, ROOT, digest, write
from record_goal_continuation_v9 import observed_processes
import compose_dense_context_checkpoint_masks_v1 as experiment


def main():
    path = HERE / 'dense_context_checkpoint_cohort_transfer_queued_v1.json'
    assert not path.exists() and not experiment.OUT.exists(), 'Preserve earlier plans and runs'
    observed = observed_processes()
    for pid in (74360, 29148, 61976):
        process = next(p for p in observed if p['process_id'] == pid)
        assert process['identity_and_creation_time_revalidated']
    assert not (experiment.DENSE / 'models_locked.json').exists()
    assert not (experiment.DENSE / 'predictions_locked.json').exists()
    assert not (experiment.DENSE / 'verification.json').exists()
    fixture = experiment.preflight()
    assert fixture['status'] == 'passed' and fixture['fixture_candidates'] == experiment.PER_DATE
    assert read(experiment.BASE / 'verification.json')['status'] == 'passed'
    assert read(experiment.DENSE / 'preflight.json')['status'] == 'passed'
    raw_hashes = read(experiment.BASE / 'protocol.json')['raw_datasets_sha256']
    assert raw_hashes == read(experiment.DENSE / 'protocol.json')['raw_datasets_sha256']
    for filename, expected in raw_hashes.items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    sources = [Path(__file__)] + [HERE / filename for filename in (
        'compose_dense_context_checkpoint_masks_v1.py', 'run_compose_dense_context_checkpoint_masks_v1.ps1',
        'compose_full_negative_masks_v1.py', 'full_negative_mask_core_v1.py',
        'cross_user_pair_ranker_v1.py', 'quantile_blend_core_v1.py',
        'publish_progress_v4.py', 'record_two_day_progress_v4.py')]
    sources += [experiment.DENSE / filename for filename in ('protocol.json', 'preflight.json')]
    sources += [experiment.BASE / filename for filename in ('protocol.json', 'verification.json')]
    sources += [experiment.BASE / f'day{day}_{name}.npy' for day in (15, 16) for name in ('indices', 'prediction')]
    record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        status='preregistered_waiting_live_dense_context_pipeline',
        baseline=str(experiment.BASE), source_forecasts=str(experiment.DENSE),
        arms=experiment.ARMS, modes=experiment.MODES, weights=experiment.WEIGHTS,
        masks=experiment.MASKS, candidates_per_date=experiment.PER_DATE,
        total_candidates=2 * experiment.PER_DATE, fixture=fixture, new_model_training=False,
        no_dense_held_margins_read_during_preregistration=True,
        no_held_advertisement_labels_read_during_preregistration=True,
        candidate_prediction_hashes_on_both_dates_required_before_scoring=True,
        independent_full_cohort_masks_and_all_candidates_required=True,
        source_complete12_model_replay_required=True,
        development_dates_repeatedly_reused=True, inferred17_labels_read=False,
        dense_fit=dict(process_id=74360, creation_date='2026-11-03T16:16:21.343889-05:00'),
        dense_wrapper=dict(process_id=29148, creation_date='2026-11-03T15:54:44.863567-05:00'),
        complete_publisher_predecessor=dict(process_id=61976, creation_date='2026-11-03T16:04:48.318746-05:00'),
        sequential_after_dense_fit_and_complete_inventory_publication=True,
        original_datasets_unchanged=True, raw_datasets_sha256=raw_hashes,
        sources_sha256={str(source): digest(source) for source in sources})
    write(path, record)
    print(dict(status=record['status'], candidates=record['total_candidates'],
        fixture_candidates=fixture['fixture_candidates'], fixture_error=fixture['maximum_independent_prediction_error']), flush=True)


if __name__ == '__main__':
    main()
