"""Record the verified deliverable without changing the active goal."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
output=ROOT/'output/best_recipe_20191117_20261102'
result=json.loads((output/'verification.json').read_text(encoding='utf8'))
assert result['status']=='passed'
record={'status':'complete_and_independently_verified','rows':result['rows'],
        'submission':str(output/'submission_best_20191117.csv'),
        'assignment_variant':str(output/'submission_log_id_20191117.csv'),
        'verification':str(output/'verification.json'),
        'inferred_label_auc':result['inferred_label_auc'],'inferred_labels_not_ground_truth':True,
        'real_platform_auc':None,'inferred_labels_used_for_training':False,
        'source_development16_auc':result['development16_auc'],
        'all_original_datasets_unchanged':True}
for relative in ['analysis/day16_auc082_20261102/STATUS.json','analysis/auc85_goal_state.json']:
    path=ROOT/relative
    state=json.loads(path.read_text(encoding='utf8'))
    state['latest_best_recipe17_delivery']=record
    state['snapshot_utc']=datetime.now(timezone.utc).isoformat()
    if path.name=='STATUS.json':
        state.update(goal_status='active',last_goal_turn_classification='progress',
                     current_verified_best_auc=.8088122948138387,target_verified=False,
                     training_status_at_snapshot='Daily-pair ranking session96686 live; recurrent4 models independently verified, best0.8083177328; final17 export complete',
                     final17_submission_status='complete_and_independently_verified',
                     live_session_ids_to_revalidate=[96686],
                     latest_recurrent_verification='runs/recurrent_userday_v1/verification.json',
                     latest_recurrent_auc=.8083177328447492)
    path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(record,ensure_ascii=False,indent=2))
