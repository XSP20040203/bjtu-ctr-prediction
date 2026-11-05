"""Recompute all new architecture AUCs using complete, aligned held-out days."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / 'analysis/nextday_goal_20261029'


def main():
    records = []
    for directory in sorted((HERE / 'results').glob('*')):
        path = directory / 'metrics.json'
        if not path.exists():
            continue
        saved = json.loads(path.read_text(encoding='utf-8'))
        day = saved['day']
        cache = ROOT / ('analysis/user_personalization/cache_day15_stats' if day == 20191115
                        else 'analysis/auc085/cache_rich_stats_time16')
        valid = np.load(cache / 'valid_indices.npy')
        np.testing.assert_array_equal(valid, np.load(directory / 'valid_indices.npy'))
        dates = np.load(cache / 'date.npy', mmap_mode='r')
        if not np.all(dates[valid] == day):
            raise AssertionError('Evaluation date mismatch')
        labels = np.load(cache / 'label.npy', mmap_mode='r')[valid]
        prediction = np.load(directory / 'validation_prediction.npy')
        score = float(roc_auc_score(labels, prediction))
        auc_rounding_difference = abs(score - saved['auc'])
        allowed_auc_difference = 1e-7 if saved.get('model') in ['SetContextCTR', 'AllocatedSetContextCTR'] else 1e-12
        if auc_rounding_difference > allowed_auc_difference:
            raise AssertionError(f'Saved architecture AUC differs for {directory.name}: {auc_rounding_difference}')
        baseline = np.load(PREVIOUS / f'duplicate_multiplicity/day{day % 100}_prediction.npy')
        records.append({'day': day, 'name': directory.name, 'auc': score,
                        'logloss': float(log_loss(labels, prediction)), 'kind': 'single model',
                        'saved_vs_replay_auc_difference': auc_rounding_difference})
        for weight in [.05, .1, .2, .3]:
            p = (1 - weight) * baseline + weight * prediction
            records.append({'day': day, 'name': f'{directory.name}__weight{weight}',
                            'auc': float(roc_auc_score(labels, p)), 'logloss': float(log_loss(labels, p)),
                            'kind': 'development blend'})
        if directory.name == 'gated_oof_day16':
            fixed = .8 * baseline + .2 * prediction
            np.save(HERE / 'gated_fixed20_day16_prediction.npy', fixed)
            np.save(HERE / 'gated_fixed20_day16_indices.npy', valid)
    for day in [15, 16]:
        allocation = HERE / f'results/set_allocated_day{day}/validation_prediction.npy'
        gated = HERE / f'results/gated_oof_day{day}/validation_prediction.npy'
        if allocation.exists() and gated.exists() and (allocation.parent / 'metrics.json').exists():
            labels = np.load(ROOT / f'analysis/auc085_next/evaluation/day{day}_labels.npy')
            baseline = np.load(PREVIOUS / f'duplicate_multiplicity/day{day}_prediction.npy')
            prediction = .9 * (.8 * baseline + .2 * np.load(gated)) + .1 * np.load(allocation)
            records.append({'day': 20191100 + day, 'name': 'fixed_gated20_allocated10',
                            'auc': float(roc_auc_score(labels, prediction)),
                            'logloss': float(log_loss(labels, prediction)), 'kind': 'fixed development blend'})
            np.save(HERE / f'fixed_gated20_allocated10_day{day}.npy', prediction)
            reduced = HERE / f'results/gated166_day{day}/validation_prediction.npy'
            if reduced.exists() and (reduced.parent / 'metrics.json').exists():
                expanded = .7 * prediction + .3 * np.load(reduced)
                records.append({'day': 20191100 + day, 'name': 'fixed_gated20_allocated10_core30',
                                'auc': float(roc_auc_score(labels, expanded)),
                                'logloss': float(log_loss(labels, expanded)), 'kind': 'fixed development blend'})
                np.save(HERE / f'fixed_gated20_allocated10_core30_day{day}.npy', expanded)
    for day in [15, 16]:
        base_file = HERE / f'fixed_gated20_allocated10_core30_day{day}.npy'
        seed_file = HERE / f'results/gated166_seed2026_day{day}/validation_prediction.npy'
        if base_file.exists() and seed_file.exists() and (seed_file.parent / 'metrics.json').exists():
            labels = np.load(ROOT / f'analysis/auc085_next/evaluation/day{day}_labels.npy')
            p = .9 * np.load(base_file) + .1 * np.load(seed_file)
            records.append({'day': 20191100 + day, 'name': 'fixed_core30_seed10',
                            'auc': float(roc_auc_score(labels, p)), 'logloss': float(log_loss(labels, p)),
                            'kind': 'fixed development blend'})
    for day in [15, 16]:
        base_file = HERE / f'fixed_gated20_allocated10_core30_day{day}.npy'
        transition_file = HERE / f'results/gated166_transition_day{day}/validation_prediction.npy'
        if base_file.exists() and transition_file.exists() and (transition_file.parent / 'metrics.json').exists():
            labels = np.load(ROOT / f'analysis/auc085_next/evaluation/day{day}_labels.npy')
            p = .8 * np.load(base_file) + .2 * np.load(transition_file)
            records.append({'day': 20191100 + day, 'name': 'fixed_core30_transition20',
                            'auc': float(roc_auc_score(labels, p)), 'logloss': float(log_loss(labels, p)),
                            'kind': 'fixed development blend'})
    frame = pd.DataFrame(records).sort_values(['day', 'auc'], ascending=[True, False])
    frame.to_csv(HERE / 'architecture_comparison.csv', index=False, encoding='utf-8-sig')
    report = {'goal_auc': .85, 'goal_met': bool((frame.auc > .85).any()),
              'final_auc': None, 'scope': 'Repeatedly used whole future-day development; no independent final claim',
              'validation_labels_used_in_training_features_or_optimizer': False,
              'old_v6_submission_preserved': True, 'experiments': records}
    (HERE / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(frame.to_string(index=False))
    print('GOAL_MET', report['goal_met'])


if __name__ == '__main__':
    main()
