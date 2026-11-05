"""Freeze the next composition before its source forecasts or scores exist."""
from datetime import datetime, timezone
from pathlib import Path
from conditional_daily import read
from fit_stacker import HERE, ROOT, digest, write
import compose_history_dot_embedding_masks_v1 as experiment


def main():
    path = HERE / 'history_dot_cohort_transfer_queued_v1.json'
    assert not path.exists() and not experiment.OUT.exists(), 'Preserve earlier plans and runs'
    fixture = experiment.preflight()
    assert fixture['status'] == 'passed' and fixture['fixture_candidates'] == 170
    assert read(experiment.BASE / 'verification.json')['status'] == 'passed'
    assert read(experiment.NEURAL / 'preflight.json')['status'] == 'passed'
    raw_hashes = read(experiment.BASE / 'protocol.json')['raw_datasets_sha256']
    assert raw_hashes == read(experiment.NEURAL / 'protocol.json')['raw_datasets_sha256']
    for filename, expected in raw_hashes.items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    sources = [Path(__file__)] + [HERE / filename for filename in (
        'compose_history_dot_embedding_masks_v1.py', 'run_compose_history_dot_embedding_masks_v1.ps1',
        'compose_full_negative_masks_v1.py', 'full_negative_mask_core_v1.py',
        'cross_user_pair_ranker_v1.py', 'quantile_blend_core_v1.py',
        'publish_progress_v3.py', 'record_two_day_progress_v3.py')]
    sources += [experiment.NEURAL / filename for filename in ('protocol.json', 'preflight.json')]
    sources += [experiment.BASE / filename for filename in ('protocol.json', 'verification.json')]
    sources += [experiment.BASE / f'day{day}_{name}.npy' for day in (15, 16) for name in ('indices', 'prediction')]
    record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        status='preregistered_waiting_live_history_dot_pipeline',
        baseline=str(experiment.BASE), source_forecasts=str(experiment.NEURAL),
        arms=experiment.ARMS, modes=experiment.MODES, weights=experiment.WEIGHTS,
        masks=experiment.MASKS, candidates_per_date=506, total_candidates=1012,
        fixture=fixture, new_model_training=False,
        no_history_dot_held_margins_read_during_preregistration=True,
        no_held_advertisement_labels_read_during_preregistration=True,
        candidate_prediction_hashes_on_both_dates_required_before_scoring=True,
        independent_full_cohort_masks_and_all_candidates_required=True,
        development_dates_repeatedly_reused=True, inferred17_labels_read=False,
        history_dot_predecessor=dict(process_id=64760, creation_date='2026-11-03T15:30:36.286956-05:00'),
        dense_publisher_refresh_predecessor=dict(process_id=29148, creation_date='2026-11-03T15:54:44.863567-05:00'),
        refresh_complete_publisher_inventory_after_dense_wrapper_exit=True,
        original_datasets_unchanged=True, raw_datasets_sha256=raw_hashes,
        sources_sha256={str(source): digest(source) for source in sources})
    write(path, record)
    print(dict(status=record['status'], candidates=record['total_candidates'],
        fixture_candidates=fixture['fixture_candidates'], fixture_error=fixture['maximum_independent_prediction_error']), flush=True)


if __name__ == '__main__':
    main()
