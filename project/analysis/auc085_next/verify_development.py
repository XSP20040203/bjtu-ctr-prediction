"""Independent raw-label/date alignment and bootstrap uncertainty for saved blends."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['label', 'pt_d', 'log_id'], dtype='int32')
    summary = []
    for day in [15, 16]:
        cache = ROOT / ('analysis/user_personalization/cache_day15_stats' if day == 15 else 'analysis/auc085/cache_rich_stats_time16')
        rows = np.load(cache / 'valid_indices.npy')
        train = np.load(cache / 'train_indices.npy')
        expected = np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day)
        np.testing.assert_array_equal(rows, expected)
        assert raw.iloc[train].pt_d.max() < 20191100 + day
        y = raw.label.to_numpy()[rows]
        np.testing.assert_array_equal(y, np.load(HERE / f'evaluation/day{day}_labels.npy'))
        base = np.load(ROOT / f'analysis/daily_interest_din/dense_results/day{day}_dcn_exposure_snapshot/validation_prediction.npy').astype(np.float64) * .5
        base += np.load(ROOT / f'analysis/daily_interest_din/tree_results/day{day}_snapshot_exposure_d6/validation_prediction.npy') * .5
        probability = np.load(HERE / f'evaluation/day{day}_fixed_shared20_multi_lgb_blend.npy')
        score = float(roc_auc_score(y, probability))
        positive = np.flatnonzero(y == 1)
        negative = np.flatnonzero(y == 0)
        sorted_negative = np.sort(probability[negative])
        contributions = (np.searchsorted(sorted_negative, probability[positive], side='left') +
                         np.searchsorted(sorted_negative, probability[positive], side='right')) / (2. * len(negative))
        np.testing.assert_allclose(contributions.mean(), score, atol=1e-12)
        rng = np.random.default_rng(20261029 + day)
        draws = rng.integers(0, len(positive), size=(2000, len(positive)))
        bootstrap = contributions[draws].mean(axis=1)
        summary.append({'day': 20191100 + day, 'rows': len(y), 'positive': int(y.sum()),
                        'raw_labels_and_day_indices_match': True, 'strict_train_validation_date_boundary': True,
                        'baseline_auc': float(roc_auc_score(y, base)), 'new_blend_auc': score,
                        'auc_gain': score - float(roc_auc_score(y, base)),
                        'positive_resample_95_percent_interval': np.quantile(bootstrap, [.025, .975]).tolist(),
                        'uncertainty_scope': 'Only positives resampled; negatives fixed; exploratory development uncertainty, not independent test guarantee',
                        'goal_met': score > .85,
                        'prediction_sha256': hashlib.sha256(probability.tobytes()).hexdigest()})
    report = {'status': 'passed', 'goal_auc': .85, 'goal_met': all(item['goal_met'] for item in summary),
              'final_auc': None, 'final_labels_available': False, 'days': summary,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE / 'development_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
