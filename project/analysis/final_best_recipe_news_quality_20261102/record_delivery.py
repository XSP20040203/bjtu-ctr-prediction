"""Record the verified17 export without changing the active two-day objective."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'output/best_recipe_news_quality_20191117_20261102'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


result = read(OUTPUT/'verification.json')
assert result['status'] == 'passed' and result['original_datasets_unchanged']
submission = OUTPUT/'submission_best_20191117.csv'
assert hashlib.sha256(submission.read_bytes()).hexdigest() == result['submission_sha256']
record = {'status': 'complete_and_independently_verified', 'rows': result['rows'],
    'submission': str(submission), 'assignment_variant': str(OUTPUT/'submission_log_id_20191117.csv'),
    'verification': str(OUTPUT/'verification.json'), 'inferred_label_auc': result['inferred_label_auc'],
    'inferred_labels_not_ground_truth': True, 'real_platform_auc': None,
    'inferred_labels_used_for_training': False, 'inferred_labels_used_for_selection': False,
    'source_development16_auc': result['development16_auc'], 'all_original_datasets_unchanged': True}
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
        state['final17_submission_status'] = record['status']
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(record, ensure_ascii=False, indent=2))
