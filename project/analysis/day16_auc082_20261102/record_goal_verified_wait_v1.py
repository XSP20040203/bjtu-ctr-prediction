"""Preserve a fresh identity-checked wait on the two existing pipelines."""
import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from conditional_daily import read
from fit_stacker import HERE, ROOT, digest, write
from publish_progress import publication_lock
from record_goal_continuation_v3 import observed_processes, summary, EXPERIMENTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observed-lgb-trees', type=int, required=True)
    parser.add_argument('--observed-day', type=int, choices=(15, 16), required=True)
    parser.add_argument('--observed-arm', required=True)
    args = parser.parse_args()
    observed = observed_processes()
    expected = {
        56696: ('2026-11-03T13:55:05.133816-05:00', 'run_full_negative_embedding_lgb_v1.ps1'),
        58516: ('2026-11-03T14:26:57.933321-05:00', 'full_negative_embedding_lgb_v1.py --phase fit'),
        55988: ('2026-11-03T14:43:07.543410-05:00', 'run_full_negative_semantic_embeddings_v1.ps1'),
    }
    for process_id, (creation, command) in expected.items():
        actual = next(value for value in observed if value['process_id'] == process_id)
        assert command in actual['command_line']
        actual_creation = re.sub(r'(\.\d{6})\d+(?=[+-]|Z|$)', r'\1', actual['creation_date'])
        difference = abs((datetime.fromisoformat(actual_creation) - datetime.fromisoformat(creation)).total_seconds())
        assert difference < .001, actual
    reports = {name: summary(name, observed) for name in EXPERIMENTS}
    assert reports['full_negative_embedding_lgb_v1']['status'] == 'running_fit'
    assert reports['full_negative_semantic_embeddings_v1']['status'] == 'prepared_waiting_predecessor'
    assert reports['full_negative_semantic_embeddings_v1']['preflight']['status'] == 'passed'
    now = datetime.now(timezone.utc)
    event = dict(event='full_negative_embedding_lgb_iteration', day=args.observed_day,
        arm=args.observed_arm, trees=args.observed_lgb_trees,
        source='Fresh native output from existing exec session52215; value is last emitted checkpoint, not an overall percentage')
    with publication_lock():
        audit = read(ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json')
        assert audit['audit_status'] == 'passed' and not audit['goal_achieved']
        archive = HERE / 'runtime_history' / now.strftime('%Y%m%dT%H%M%S%fZ_verified_wait')
        archive.mkdir(parents=True, exist_ok=False)
        for filename in ('resumed_processes_v1.json', 'resumed_goal_runtime_v1.json', 'STATUS.json'):
            if (HERE / filename).exists():
                shutil.copy2(HERE / filename, archive / filename)
        runtime = read(HERE / 'resumed_goal_runtime_v1.json')
        delivery = runtime['latest_day17_delivery']
        assert digest(delivery['submission']) == delivery['submission_sha256']
        runtime.update(timestamp_utc=now.isoformat(), goal_status='active',
            previous_goal_turn_classification=runtime.get('last_goal_turn_classification', 'progress'),
            previous_goal_turn_evidence=runtime.get('last_goal_turn_evidence'),
            last_goal_turn_classification='verified_wait',
            last_goal_turn_evidence='Fresh exec-session output and exact live process creation identities confirm original LightGBM fit continues. Semantic CPU/full-population preflight passed; existing wrapper waits for verified predecessor completion. No pipeline restarted and no new score claimed.',
            latest_observed_lightgbm_iteration=event, observed_processes=observed,
            current_experiments=reports, preserved_runtime_snapshot=str(archive),
            scores={day: value['auc'] for day, value in audit['dates'].items()},
            both_targets_met=False, advertisement_inferred17_labels_read=False)
        write(HERE / 'resumed_processes_v1.json', observed)
        write(HERE / 'resumed_goal_runtime_v1.json', runtime)
        state = read(HERE / 'STATUS.json')
        state.update(snapshot_utc=now.isoformat(), goal_status='active',
            last_goal_turn_classification='verified_wait', resumed_goal_runtime=runtime,
            current_full_negative_experiments=reports, latest_observed_lightgbm_iteration=event)
        write(HERE / 'STATUS.json', state)
        write(archive / 'verified_wait_evidence.json', dict(timestamp_utc=now.isoformat(),
            status='verified_wait', native_event=event, live_processes=observed,
            scores=runtime['scores'], original_goal_remains_active=True, inferred17_labels_read=False))
    print(json.dumps(dict(timestamp_utc=now.isoformat(), classification='verified_wait',
        event=event, scores=runtime['scores'], live_process_ids=[v['process_id'] for v in observed],
        evidence=str(archive / 'verified_wait_evidence.json')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
