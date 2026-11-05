"""Current full-cohort AUC decomposition and metric-only oracle diagnostics.

Oracle statistics use development labels solely to describe remaining errors.
No oracle scores, offsets, labels, or row-level predictions are written.
"""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write

SOURCE = HERE / 'runs/compose_full_negative_masks_v1'
OUT = HERE / 'current_user_ranking_diagnosis_v2.json'


def pair_wins(labels, scores):
    positives = int(np.sum(labels))
    pairs = positives * (len(labels) - positives)
    wins = float(rankdata(scores)[labels == 1].sum() - positives * (positives + 1) / 2)
    return wins, pairs


def diagnose(labels, scores, users):
    group, unique = pd.factorize(users, sort=True)
    table = pd.DataFrame(dict(label=labels, score=scores, group=group))
    counts = table.groupby('group', sort=True).label.agg(['sum', 'size'])
    clicks = counts['sum'].to_numpy(dtype='int64')
    size = counts['size'].to_numpy(dtype='int64')
    within_pairs = int(np.sum(clicks * (size - clicks)))
    within_ranks = table.groupby('group', sort=False).score.rank(method='average').to_numpy()
    within_wins = float(within_ranks[labels == 1].sum() - np.sum(clicks * (clicks + 1) / 2))
    # Independent score searches check the group-rank implementation, including ties.
    independent_wins, independent_pairs = 0., 0
    global_ranks = rankdata(scores)
    oracle_rank_sum = 0.
    for positions in table.groupby('group', sort=False).indices.values():
        local_labels = labels[positions]
        local_scores = scores[positions]
        positive = local_scores[local_labels == 1]
        negative = np.sort(local_scores[local_labels == 0])
        lower = np.searchsorted(negative, positive, side='left')
        upper = np.searchsorted(negative, positive, side='right')
        independent_wins += float((lower + .5 * (upper - lower)).sum())
        independent_pairs += len(positive) * len(negative)
        count = len(positive)
        if count:
            # Largest k ranks maximize global AUC for a fixed score multiset per user.
            oracle_rank_sum += float(np.sort(global_ranks[positions])[-count:].sum())
    assert independent_wins == within_wins and independent_pairs == within_pairs
    wins, pairs = pair_wins(labels, scores)
    auc = wins / pairs
    assert abs(auc - roc_auc_score(labels, scores)) < 1e-12
    positive_count = int(labels.sum())
    oracle_permutation_auc = (oracle_rank_sum - positive_count * (positive_count + 1) / 2) / pairs
    assert oracle_permutation_auc >= auc - 1e-12
    means = table.groupby('group', sort=False).score.transform('mean').to_numpy()
    # A separate metric-only calibration oracle changes one intercept per user,
    # preserves every within-user ordering, and matches observed group click sums.
    margin = logit(np.clip(scores, 1e-12, 1 - 1e-12))
    lower = np.full(len(unique), -40.)
    upper = np.full(len(unique), 40.)
    for _ in range(64):
        middle = (lower + upper) / 2
        sums = np.bincount(group, weights=expit(margin + middle[group]), minlength=len(unique))
        below = sums < clicks
        lower[below], upper[~below] = middle[below], middle[~below]
    calibrated = expit(margin + ((lower + upper) / 2)[group])
    matched = np.bincount(group, weights=calibrated, minlength=len(unique))
    assert np.max(np.abs(matched - clicks)) < 1e-9
    calibration_auc = float(roc_auc_score(labels, calibrated))
    return dict(auc=auc, rows=len(labels), clicks=positive_count, users=len(unique),
        users_with_clicks=int(np.sum(clicks > 0)),
        within_user_pairs=within_pairs, all_positive_negative_pairs=pairs,
        within_user_pair_fraction=within_pairs / pairs,
        within_user_auc=within_wins / within_pairs,
        between_user_auc=(wins - within_wins) / (pairs - within_pairs),
        user_mean_only_auc=float(roc_auc_score(labels, means)),
        metric_only_oracle_within_user_permutation_auc=oracle_permutation_auc,
        metric_only_oracle_user_intercept_auc=calibration_auc,
        independent_group_pair_search_matches=True,
        caveat='Pair share does not bound global gains from within-user reorderings; positives also move relative to other users.')


def main():
    assert not OUT.exists(), 'Preserve earlier diagnostic evidence'
    verified = read(SOURCE / 'verification.json')
    assert verified['status'] == 'passed'
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label', 'user_id'], dtype='int32')
    reports = {}
    sources = [Path(__file__), SOURCE / 'verification.json']
    for day in (15, 16):
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day)
        np.testing.assert_array_equal(rows, np.load(SOURCE / f'day{day}_indices.npy'))
        probability = np.load(SOURCE / f'day{day}_prediction.npy')
        report = diagnose(raw.label.to_numpy()[rows], probability, raw.user_id.to_numpy()[rows])
        assert abs(report['auc'] - verified['dates'][str(20191100 + day)]['auc']) < 1e-12
        reports[str(20191100 + day)] = report
        sources += [SOURCE / f'day{day}_{name}.npy' for name in ('indices', 'prediction')]
        emit('current_user_ranking_diagnosis', day=day, **report)
    expected = read(SOURCE / 'protocol.json')['raw_datasets_sha256']
    assert {name: digest(ROOT / '2026ml' / name) for name in expected} == expected
    write(OUT, dict(status='passed', dates=reports, diagnostic_only=True,
        held15_16_labels_for_aggregate_diagnostic_metrics_only=True,
        development_dates_repeatedly_reused=True, new_models_fitted=False,
        oracle_row_predictions_or_offsets_saved=False, inferred17_labels_read=False,
        original_datasets_unchanged=True, raw_datasets_sha256=expected,
        sources_sha256={str(path): digest(path) for path in sources}))


if __name__ == '__main__':
    main()
