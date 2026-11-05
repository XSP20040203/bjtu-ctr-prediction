"""Attribute full-development AUC errors to strictly historical user/task cohorts."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from fit_stacker import ROOT, HERE, digest, write
from conditional_daily import read


def summarize(y, prediction, mask, positive_win, negative_win):
    positive = mask & (y == 1)
    negative = mask & (y == 0)
    npos, nneg = int(positive.sum()), int(negative.sum())
    return dict(rows=int(mask.sum()), positives=npos,
        cohort_auc=float(roc_auc_score(y[mask], prediction[mask])) if npos and nneg else None,
        positive_against_all_negatives=float(positive_win[positive].mean()) if npos else None,
        negative_against_all_positives=float(negative_win[negative].mean()) if nneg else None,
        share_of_all_positive_errors=float((1-positive_win[positive]).sum()/(1-positive_win[y == 1]).sum()),
        share_of_all_negative_errors=float((1-negative_win[negative]).sum()/(1-negative_win[y == 0]).sum()))


def main():
    out = HERE/'row_error_cohorts_v1.json'
    assert not out.exists(), 'Preserve earlier diagnostics'
    raw = pd.read_csv(ROOT/'2026ml/train.csv',
        usecols=['pt_d', 'label', 'user_id', 'task_id', 'spread_app_id', 'slot_id'], dtype='int32')
    result = dict(purpose='Diagnostic only; repeatedly used development dates; no model fit',
                  inferred17_labels_read=False, dates={})
    for day, source_name in [(15, 'task_audience_row_ranker_v1'), (16, 'task_audience_rate_v1')]:
        source = HERE/'runs'/source_name
        verified = read(source/'verification.json')
        assert verified['status'] == 'passed'
        date = 20191100+day
        positions = np.flatnonzero(raw.pt_d.to_numpy() == date)
        np.testing.assert_array_equal(positions, np.load(source/f'day{day}_indices.npy'))
        query = raw.iloc[positions]
        prior = raw[raw.pt_d < date]
        prediction = np.load(source/f'day{day}_prediction.npy')
        y = query.label.to_numpy()
        auc = roc_auc_score(y, prediction)
        assert abs(auc-verified['dates'][str(date)]['auc']) < 1e-12
        sorted_positive = np.sort(prediction[y == 1])
        sorted_negative = np.sort(prediction[y == 0])
        positive_win = (np.searchsorted(sorted_negative, prediction, 'left')+
                        np.searchsorted(sorted_negative, prediction, 'right'))/(2*len(sorted_negative))
        negative_win = 1-(np.searchsorted(sorted_positive, prediction, 'left')+
                          np.searchsorted(sorted_positive, prediction, 'right'))/(2*len(sorted_positive))
        assert abs(positive_win[y == 1].mean()-auc) < 1e-12
        assert abs(negative_win[y == 0].mean()-auc) < 1e-12
        count = prior.groupby('user_id').label.sum()
        clicks = query.user_id.map(count).fillna(-1).to_numpy()
        cohorts = {'user_unseen': clicks < 0, 'user_zero_clicks': clicks == 0,
                   'user_one_click': clicks == 1, 'user_two_to_four_clicks': (clicks >= 2) & (clicks <= 4),
                   'user_five_plus_clicks': clicks >= 5}
        for field in ['task_id', 'spread_app_id']:
            known = query[field].isin(prior[field].unique()).to_numpy()
            cohorts[field+'_known'] = known
            cohorts[field+'_new'] = ~known
        for slot in sorted(query.slot_id.unique()):
            cohorts[f'slot_{slot}'] = query.slot_id.to_numpy() == slot
        result['dates'][str(date)] = dict(auc=float(auc), source=source_name,
            verification_sha256=digest(source/'verification.json'),
            cohorts={name:summarize(y,prediction,mask,positive_win,negative_win)
                     for name,mask in cohorts.items()})
    write(out, result)
    for date, data in result['dates'].items():
        print(date, 'auc', data['auc'])
        for name, record in data['cohorts'].items():
            print(name, 'n',record['rows'],'positive_errors',round(record['share_of_all_positive_errors'],4),
                  'positive_pair_auc',record['positive_against_all_negatives'])


if __name__ == '__main__':
    main()
