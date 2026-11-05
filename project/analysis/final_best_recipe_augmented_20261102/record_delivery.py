"""Record the checked day17 delivery while retaining the active two-day objective."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'output/best_recipe_augmented_20191117_20261102'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


result = read(OUTPUT/'verification.json')
audit = read(ROOT/'output/two_day_auc_targets_20261102/independent_target_audit.json')
assert result['status'] == 'passed' and audit['audit_status'] == 'passed'
assert result['original_datasets_unchanged'] and audit['original_datasets_unchanged']
submission = OUTPUT/'submission_best_20191117.csv'
assert hashlib.sha256(submission.read_bytes()).hexdigest() == result['submission_sha256']
record = {'status': 'complete_and_independently_verified', 'rows': result['rows'],
          'submission': str(submission),
          'assignment_variant': str(OUTPUT/'submission_log_id_20191117.csv'),
          'verification': str(OUTPUT/'verification.json'),
          'inferred_label_auc': result['inferred_label_auc'],
          'inferred_labels_not_ground_truth': True, 'real_platform_auc': None,
          'inferred_labels_used_for_training': False,
          'source_development16_auc': result['development16_auc'],
          'all_original_datasets_unchanged': True}
now = datetime.now(timezone.utc).isoformat()
for relative in ['analysis/day16_auc082_20261102/STATUS.json', 'analysis/auc85_goal_state.json']:
    path = ROOT/relative
    state = read(path)
    prior = state.get('latest_best_recipe17_delivery')
    if prior and prior.get('submission') != record['submission']:
        state.setdefault('best_recipe17_delivery_history', []).append(prior)
    state['latest_best_recipe17_delivery'] = record
    state['snapshot_utc'] = now
    if path.name == 'STATUS.json':
        runtime = {'timestamp_utc': now, 'goal_status': 'active',
                   'last_goal_turn_classification': 'progress',
                   'verified15_auc': audit['dates']['20191115']['auc'],
                   'verified16_auc': audit['dates']['20191116']['auc'],
                   'targets_met': False, 'raw_datasets_unchanged': True,
                   'completed_sessions': {'augmented_rows': 40896, 'augmented_blend16': 6062,
                                          'bayesian_daily': 3405, 'day17_prediction': 55682,
                                          'day17_verification': 20144},
                   'live_training_sessions': [],
                   'day17_delivery': record,
                   'next_actions': ['Continue both real15/16 targets under the existing strict-past protocol',
                                    'Revalidate actual processes before starting new training',
                                    'Never use inferred17 outcomes to train, tune or select the real15/16 model']}
        previous = state.get('latest_goal_runtime')
        if previous:
            state.setdefault('goal_runtime_history', []).append(previous)
        state['latest_goal_runtime'] = runtime
        state.update(goal_status='active', last_goal_turn_classification='progress', target_verified=False,
                     current_verified15_auc=runtime['verified15_auc'],
                     current_verified_best_auc=runtime['verified16_auc'],
                     current_verified_best='runs/bayesian_daily_offset_v1/verification.json',
                     current_verified15_source=str(ROOT/'analysis/day16_auc082_20261102/runs/bayesian_daily_offset_v1/verification.json'),
                     training_status_at_snapshot='Prior queued training and audits completed; current17 export independently reconstructed and evaluated',
                     final17_submission_status='complete_and_independently_verified',
                     live_session_ids_to_revalidate=[])
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(record, ensure_ascii=False, indent=2))
