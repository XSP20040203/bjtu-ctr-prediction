"""Inspect the verified cold-task effect; never fit or select a model here."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from fit_stacker import HERE, ROOT, digest, write
from conditional_daily import read
from cold_task_core import cold_mask


def main():
    output=HERE/'cold_task_effect_v1.json'
    assert not output.exists(), 'Preserve diagnostics'
    source=HERE/'runs/cross_user_pair_ranker_v1'
    candidate=HERE/'runs/cold_task_ranker_v1'
    assert read(source/'verification.json')['status']=='passed'
    assert read(candidate/'verification.json')['status']=='passed'
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','task_id','label'],dtype='int32')
    reports={}
    for day in [15,16]:
        rows=np.flatnonzero(raw.pt_d.to_numpy()==20191100+day)
        np.testing.assert_array_equal(rows,np.load(candidate/f'day{day}_indices.npy'))
        a=np.load(source/f'day{day}_prediction.npy')
        b=np.load(candidate/f'day{day}_prediction.npy')
        query=raw.iloc[rows][['pt_d','task_id']]
        mask=cold_mask(raw[['pt_d','task_id']],query,20191100+day)
        np.testing.assert_array_equal(a[~mask],b[~mask])
        labels=raw.label.to_numpy()[rows]
        reports[str(20191100+day)]={}
        for name,part in [('all',np.ones(len(rows),dtype=bool)),('known_tasks',~mask),('new_tasks',mask)]:
            reports[str(20191100+day)][name]=dict(rows=int(part.sum()),clicks=int(labels[part].sum()),
                before_auc=float(roc_auc_score(labels[part],a[part])),
                after_auc=float(roc_auc_score(labels[part],b[part])))
    report=dict(status='passed',dates=reports,no_new_model_fitted=True,
        no_model_selected=True,inferred17_labels_read=False,
        non_cold_predictions_unchanged_exactly=True,
        candidate_verification_sha256=digest(candidate/'verification.json'),
        source_verification_sha256=digest(source/'verification.json'),source_sha256=digest(__file__))
    write(output,report)
    print(report,flush=True)


if __name__=='__main__':
    main()
