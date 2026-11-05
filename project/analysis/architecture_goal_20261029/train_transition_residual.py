"""Conditional transition response priors fitted only on earlier advertisement days.

No held-day labels fit the corrections. The same-day snapshot transition inputs
are permitted by the complete offline input setting, not exposure-time online.
Correction strengths are first explored on day15 then fixed for day16.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    out = HERE / 'transition_residual'
    if out.exists():
        raise FileExistsError('Preserving transition residual experiment')
    meta = read(HERE / 'transition_features.json')
    if meta['advertisement_labels_loaded'] is not False:
        raise AssertionError('Transition inputs must be target-free')
    transitions = np.load(HERE / 'transition_features.npy', mmap_mode='r')
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label', 'slot_id', 'task_id', 'creat_type_cd', 'adv_prim_id'], dtype='int32')
    if len(raw) != len(transitions):
        raise AssertionError('Raw transition row mismatch')
    out.mkdir(parents=True)
    plan = {'method': 'Train only on earlier dates; conditional odds correction of transition boundary versus existing other-snapshot indicator',
            'strength_candidates': [0., .1, .25, .5, .75, 1.], 'day16_selection': 'strength picked on day15',
            'held_labels_used_for_fit': False, 'final_test_read': False,
            'availability': 'Complete supplied same-day ad snapshots, empirical log_id order; not online timestamps'}
    (out / 'protocol.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    results = {}
    chosen = None
    for day in [15, 16]:
        date = 20191100 + day
        train, valid = np.flatnonzero(raw.pt_d.to_numpy() < date), np.flatnonzero(raw.pt_d.to_numpy() == date)
        names = meta['features']
        # The baseline already knows history membership. Transfer only a
        # historical conditional contrast inside those observable matches.
        family = 'transition__ad_click_list_v001__'
        own = transitions[:, names.index(family + 'candidate_in_own')] > .5
        elsewhere = transitions[:, names.index(family + 'candidate_absent_own_in_union')] > .5
        nearest = ((transitions[:, names.index(family + 'candidate_last_before_addition')] > .5) |
                   (transitions[:, names.index(family + 'candidate_first_after_removal')] > .5))
        # Existing baseline models encode own/elsewhere. Only the conditional
        # nearest/non-nearest contrast inside elsewhere is transferred.
        ytrain = raw.label.to_numpy()[train]
        historical_mask = elsewhere[train]
        total_n = float(historical_mask.sum())
        total_c = float(ytrain[historical_mask].sum())
        parent = (total_c + 20 * .2) / (total_n + 20)
        rates = []
        offset = np.zeros(len(valid), dtype='float64')
        for state in [False, True]:
            mask = historical_mask & (nearest[train] == state)
            count = int(mask.sum())
            clicks = int(ytrain[mask].sum())
            rate = (clicks + 20 * parent) / (count + 20)
            correction = float(logit(rate) - logit(parent))
            apply = elsewhere[valid] & (nearest[valid] == state)
            offset[apply] = correction
            rates.append({'nearest': state, 'rows': count, 'clicks': clicks, 'rate': rate, 'relative_logit': correction})
        baseline = np.load(HERE / f'fixed_gated20_allocated10_core30_day{day}.npy')
        y = raw.label.to_numpy()[valid]
        candidates = []
        for strength in plan['strength_candidates']:
            p = 1 / (1 + np.exp(-np.clip(logit(baseline) + strength * offset, -40, 40)))
            candidates.append({'strength': strength, 'auc': float(roc_auc_score(y, p)), 'logloss': float(log_loss(y, p))})
        if day == 15:
            chosen = max(candidates, key=lambda item: item['auc'])['strength']
            (out / 'selection_frozen.json').write_text(json.dumps({'selected_strength': chosen, 'day15_candidates': candidates,
                'day16_labels_used_for_strength': False}, indent=2), encoding='utf-8')
        prediction = 1 / (1 + np.exp(-np.clip(logit(baseline) + chosen * offset, -40, 40)))
        np.save(out / f'day{day}_prediction.npy', prediction)
        np.save(out / f'day{day}_indices.npy', valid)
        results[str(date)] = {'baseline_auc': float(roc_auc_score(y, baseline)),
            'selected_strength': chosen, 'auc': float(roc_auc_score(y, prediction)),
            'logloss': float(log_loss(y, prediction)), 'affected_rows': int(elsewhere[valid].sum()),
            'historical_class_rates': rates, 'candidates': candidates if day == 15 else None}
        print('TRANSITION_RESIDUAL', date, results[str(date)]['auc'], chosen, flush=True)
    (out / 'metrics.json').write_text(json.dumps({'status': 'complete', 'results': results,
        'goal_met': all(r['auc'] > .85 for r in results.values()), 'final_auc': None}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
