"""Recompute chronological development AUCs from saved, row-aligned predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DAILY = ROOT / 'analysis/daily_interest_din'


def evaluate(day):
    cache = ROOT / ('analysis/user_personalization/cache_day15_stats' if day == 15 else 'analysis/auc085/cache_rich_stats_time16')
    rows = np.load(cache / 'valid_indices.npy')
    dates = np.load(cache / 'date.npy', mmap_mode='r')
    y = np.load(cache / 'label.npy', mmap_mode='r')[rows]
    assert np.all(dates[rows] == 20191100 + day)
    records = []
    predictions = {}
    candidates = [
        ('dcn', DAILY / f'dense_results/day{day}_dcn_exposure_snapshot/validation_prediction.npy'),
        ('tree', DAILY / f'tree_results/day{day}_snapshot_exposure_d6/validation_prediction.npy'),
        ('lgb', DAILY / ('lgb_results/goal085_day15_lgb31_utf8/validation_prediction.npy' if day == 15 else 'lgb_results/goal085_day16_lgb31/validation_prediction.npy')),
    ]
    for folder in ['tree_results', 'dense_results', 'lgb_results']:
        for directory in sorted((DAILY / folder).glob(f'goal085_day{day}_*')):
            path = directory / 'validation_prediction.npy'
            metric_path = directory / 'metrics.json'
            if path.exists() and metric_path.exists() and json.loads(metric_path.read_text(encoding='utf-8')).get('status') == 'complete_and_verified':
                candidates.append((directory.name, path))
    for name, path in candidates:
        if not path.exists():
            continue
        p = np.load(path)
        if len(p) != len(y) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Invalid prediction: ' + str(path))
        predictions[name] = p
    if {'dcn', 'tree'} <= predictions.keys():
        predictions['existing_equal_blend'] = .5 * predictions['dcn'] + .5 * predictions['tree']
        if 'lgb' in predictions:
            predictions['fixed_40_40_20_blend'] = .4 * predictions['dcn'] + .4 * predictions['tree'] + .2 * predictions['lgb']
        multi_name = f'goal085_day{day}_multiscale_dcn'
        if multi_name in predictions:
            predictions['fixed_multiscale25_blend'] = .75 * predictions['existing_equal_blend'] + .25 * predictions[multi_name]
            if 'fixed_40_40_20_blend' in predictions:
                predictions['fixed_lgb_multiscale25_blend'] = .75 * predictions['fixed_40_40_20_blend'] + .25 * predictions[multi_name]
        shared_name = f'goal085_day{day}_lgb_shared_multi'
        if shared_name in predictions and 'fixed_lgb_multiscale25_blend' in predictions:
            predictions['fixed_shared20_multi_lgb_blend'] = .8 * predictions['fixed_lgb_multiscale25_blend'] + .2 * predictions[shared_name]
        oof_name = f'goal085_day{day}_oof_exposure'
        if oof_name in predictions and 'fixed_shared20_multi_lgb_blend' in predictions:
            predictions['fixed_oof25_shared_blend'] = .75 * predictions['fixed_shared20_multi_lgb_blend'] + .25 * predictions[oof_name]
        source_dcn = f'goal085_day{day}_source_context_dcn'
        source_tree = f'goal085_day{day}_source_context_tree'
        if source_dcn in predictions and source_tree in predictions and 'fixed_shared20_multi_lgb_blend' in predictions:
            predictions['fixed_source10_each_shared_blend'] = .8 * predictions['fixed_shared20_multi_lgb_blend'] + .1 * predictions[source_dcn] + .1 * predictions[source_tree]
        seed_name = f'goal085_day{day}_seed2026'
        if seed_name in predictions and 'fixed_lgb_multiscale25_blend' in predictions:
            predictions['fixed_seed25_multi_lgb_blend'] = .75 * predictions['fixed_lgb_multiscale25_blend'] + .25 * predictions[seed_name]
        for name in list(predictions):
            if name.startswith('goal085_'):
                predictions['dcn50_' + name] = .5 * predictions['dcn'] + .5 * predictions[name]
                predictions['baseline75_' + name] = .75 * predictions['existing_equal_blend'] + .25 * predictions[name]
    out = HERE / 'evaluation'
    out.mkdir(exist_ok=True)
    np.save(out / f'day{day}_labels.npy', y)
    np.save(out / f'day{day}_valid_indices.npy', rows)
    for name, prediction in predictions.items():
        records.append({'day': 20191100 + day, 'model': name,
                        'auc': float(roc_auc_score(y, prediction)),
                        'logloss': float(log_loss(y, prediction, labels=[0, 1])),
                        'average_precision': float(average_precision_score(y, prediction)),
                        'rows': len(y), 'clicks': int(y.sum())})
        if name == 'fixed_40_40_20_blend':
            np.save(out / f'day{day}_fixed_blend.npy', prediction)
        if name in ['fixed_multiscale25_blend', 'fixed_lgb_multiscale25_blend', 'fixed_shared20_multi_lgb_blend', 'fixed_seed25_multi_lgb_blend', 'fixed_oof25_shared_blend', 'fixed_source10_each_shared_blend']:
            np.save(out / f'day{day}_{name}.npy', prediction)
    return records


def main():
    records = evaluate(15) + evaluate(16)
    frame = pd.DataFrame(records).sort_values(['day', 'auc'], ascending=[True, False])
    frame.to_csv(HERE / 'comparison.csv', index=False, encoding='utf-8-sig')
    report = {'goal_auc': .85, 'goal_met_on_chronological_development': bool((frame.auc > .85).any()),
              'final_auc': None, 'final_labels_available': False,
              'protocol': 'Whole future days 15/16; repeatedly used development sets, not untouched test; target-derived inputs strictly earlier dates',
              'same_day_input_assumption': 'Offline supplied full-day unlabeled exposures and history snapshots',
              'experiments': records,
              'prediction_note': '40/40/20 weights chosen on day15; day16 confirms direction, both are development dates',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE / 'round_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(frame[['day', 'model', 'auc', 'logloss']].to_string(index=False))


if __name__ == '__main__':
    main()
