"""Exact positive/negative AUC error decomposition for the audited full cohorts."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write

RUN = HERE / 'runs/full_negative_context_v2'
ANCHOR = HERE / 'runs/group_event_posterior_v1'
OUTPUT = HERE / 'error_diagnosis_full_negative_v1.json'


def errors(labels, probability):
    positive, negative = probability[labels == 1], probability[labels == 0]
    neg, pos = np.sort(negative), np.sort(positive)
    lower, upper = np.searchsorted(neg, positive, 'left'), np.searchsorted(neg, positive, 'right')
    pos_error = (len(neg) - upper + .5 * (upper - lower)) / len(neg)
    lower, upper = np.searchsorted(pos, negative, 'left'), np.searchsorted(pos, negative, 'right')
    neg_error = (lower + .5 * (upper - lower)) / len(pos)
    result = np.empty(len(labels), dtype='float64')
    result[labels == 1], result[labels == 0] = pos_error, neg_error
    auc = float(roc_auc_score(labels, probability))
    assert abs(pos_error.mean() - (1 - auc)) < 1e-12
    assert abs(neg_error.mean() - (1 - auc)) < 1e-12
    return result, auc


def main():
    assert not OUTPUT.exists(), 'Preserve earlier diagnostics'
    verified = read(RUN / 'verification.json')
    assert verified['status'] == 'passed'
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label', 'user_id', 'task_id', 'spread_app_id'], dtype='int32')
    dates = {}
    for day in (15, 16):
        held = raw.pt_d.eq(20191100 + day).to_numpy()
        previous = raw.loc[raw.pt_d.lt(20191100 + day)]
        query = raw.loc[held].reset_index(drop=True)
        np.testing.assert_array_equal(np.flatnonzero(held), np.load(RUN / f'day{day}_indices.npy'))
        labels = query.label.to_numpy()
        prediction = np.load(RUN / f'day{day}_prediction.npy')
        old = np.load(ANCHOR / f'day{day}_prediction.npy')
        current_errors, current_auc = errors(labels, prediction)
        old_errors, old_auc = errors(labels, old)
        assert abs(current_auc - verified['dates'][str(20191100 + day)]['auc']) < 1e-12
        user = previous.groupby('user_id', observed=True).label.agg(['size', 'sum'])
        user_count = query.user_id.map(user['size']).fillna(0).to_numpy()
        user_click = query.user_id.map(user['sum']).fillna(0).to_numpy()
        task_count = query.task_id.map(previous.groupby('task_id', observed=True).size()).fillna(0).to_numpy()
        app_count = query.spread_app_id.map(previous.groupby('spread_app_id', observed=True).size()).fillna(0).to_numpy()
        masks = dict(cold_user=user_count == 0, seen_user=user_count > 0,
                     user_no_past_ad_click=user_click == 0, user_with_past_ad_click=user_click > 0,
                     user_one_to_three_past_clicks=(user_click > 0) & (user_click <= 3),
                     user_over_three_past_clicks=user_click > 3,
                     cold_task=task_count == 0, seen_task=task_count > 0,
                     task_at_most_100_prior_rows=task_count <= 100, task_over_100_prior_rows=task_count > 100,
                     cold_app=app_count == 0, seen_app=app_count > 0)
        slices = {}
        for name, mask in masks.items():
            positives, negatives = mask & (labels == 1), mask & (labels == 0)
            clicks = int(positives.sum())
            count = int(mask.sum())
            positive_mass = float(current_errors[positives].sum() / labels.sum())
            negative_mass = float(current_errors[negatives].sum() / (len(labels) - labels.sum()))
            slices[name] = dict(rows=count, clicks=clicks,
                row_fraction=float(mask.mean()), click_fraction=float(clicks / labels.sum()),
                within_slice_auc=float(roc_auc_score(labels[mask], prediction[mask])) if clicks and negatives.any() else None,
                global_positive_pair_error_mass=positive_mass,
                global_negative_pair_error_mass=negative_mass,
                global_positive_error_share=positive_mass / (1 - current_auc),
                global_negative_error_share=negative_mass / (1 - current_auc),
                positive_pair_auc_gain=float((old_errors[positives] - current_errors[positives]).sum() / labels.sum()),
                negative_pair_auc_gain=float((old_errors[negatives] - current_errors[negatives]).sum() / (len(labels) - labels.sum())))
        dates[str(20191100 + day)] = dict(auc=current_auc, anchor_auc=old_auc,
            rows=len(labels), clicks=int(labels.sum()), max_mask_label_date=int(previous.pt_d.max()), slices=slices)
        emit('full_negative_error_diagnosis', day=day, auc=current_auc,
             slices={name: slices[name] for name in ['user_no_past_ad_click', 'cold_task', 'task_at_most_100_prior_rows']})
    hashes = read(RUN / 'protocol.json')['raw_datasets_sha256']
    assert {name: digest(ROOT / '2026ml' / name) for name in hashes} == hashes
    paths = [Path(__file__), RUN / 'verification.json', ANCHOR / 'verification.json']
    for day in (15, 16):
        paths += [RUN / f'day{day}_prediction.npy', RUN / f'day{day}_indices.npy', ANCHOR / f'day{day}_prediction.npy']
    write(OUTPUT, dict(status='passed', diagnostic_only=True, dates=dates,
        masks_use_only_strictly_earlier_advertisement_labels=True,
        held_labels_for_diagnostics_only=True, development_dates_repeatedly_reused=True,
        fitted_model_or_submission_changed=False, inferred17_labels_read=False,
        original_datasets_unchanged=True, raw_datasets_sha256=hashes,
        sources_sha256={str(path): digest(path) for path in paths}))


if __name__ == '__main__':
    main()
