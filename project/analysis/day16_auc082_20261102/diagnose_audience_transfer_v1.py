"""Compare the frozen task-audience correction on existing and new tasks."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from conditional_daily import read
from fit_stacker import ROOT, HERE, digest, write


def main():
    destination = HERE / 'task_audience_effect_v1.json'
    assert not destination.exists(), 'Preserve earlier diagnostics'
    source = HERE / 'runs/task_audience_rate_v1'
    report = read(source / 'verification.json')
    assert report['status'] == 'passed'
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'task_id', 'label'], dtype='int32')
    result = dict(purpose='Diagnostic only on reused real development dates',
                  source_verification_sha256=digest(source / 'verification.json'),
                  inferred17_labels_read=False, models_fitted=False, dates={})
    for day, parent in [(15, 'firstday_task_ranker_v1'), (16, 'cold_task_ranker_v1')]:
        date = 20191100 + day
        rows = np.flatnonzero(raw.pt_d.to_numpy() == date)
        np.testing.assert_array_equal(rows, np.load(source / f'day{day}_indices.npy'))
        y = raw.label.to_numpy()[rows]
        before = np.load(HERE / 'runs' / parent / f'day{day}_prediction.npy')
        after = np.load(source / f'day{day}_prediction.npy')
        assert abs(roc_auc_score(y, after) - report['dates'][str(date)]['auc']) < 1e-12
        known_tasks = raw.loc[raw.pt_d < date, 'task_id'].unique()
        known = raw.task_id.iloc[rows].isin(known_tasks).to_numpy()
        groups = {}
        for name, mask in [('all', np.ones(len(rows), dtype=bool)), ('known_tasks', known), ('new_tasks', ~known)]:
            old_auc = float(roc_auc_score(y[mask], before[mask]))
            new_auc = float(roc_auc_score(y[mask], after[mask]))
            groups[name] = dict(rows=int(mask.sum()), clicks=int(y[mask].sum()),
                                parent_auc=old_auc, audience_auc=new_auc, delta=new_auc - old_auc)
        result['dates'][str(date)] = groups
    write(destination, result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
