"""Decompose current complete development-cohort ranking, without fitting anything."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from fit_stacker import ROOT,HERE,write,digest
from conditional_daily import read,emit

SOURCE=HERE/'runs/cross_user_pair_ranker_v1'
OUT=HERE/'remaining_auc_gap_v1.json'


def concordance(y,p):
    n=int(y.sum())
    pairs=n*(len(y)-n)
    wins=rankdata(p)[y==1].sum()-n*(n+1)/2
    return float(wins),int(pairs)


def decomposition(y,p,groups):
    table=pd.DataFrame(dict(y=y,p=p,group=groups))
    ranks=table.groupby('group',sort=False).p.rank(method='average').to_numpy()
    counts=table.groupby('group',sort=False).y.agg(['sum','count'])
    within_pairs=float((counts['sum']*(counts['count']-counts['sum'])).sum())
    within_wins=float(ranks[y==1].sum()-(counts['sum']*(counts['sum']+1)/2).sum())
    total_wins,total_pairs=concordance(y,p)
    means=table.groupby('group',sort=False).p.transform('mean').to_numpy()
    mean_wins,_=concordance(y,means)
    return dict(within_group_auc=within_wins/within_pairs,within_group_pair_fraction=within_pairs/total_pairs,
        between_group_auc=(total_wins-within_wins)/(total_pairs-within_pairs),
        group_mean_only_auc=mean_wins/total_pairs,groups=len(counts),
        groups_with_clicks=int((counts['sum']>0).sum()))


if __name__=='__main__':
    assert not OUT.exists(),'Preserve earlier diagnostics'
    assert read(SOURCE/'verification.json')['status']=='passed'
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label','user_id','task_id'],dtype='int32')
    records={}
    for day in [15,16]:
        rows=np.flatnonzero(raw.pt_d.to_numpy()==20191100+day)
        np.testing.assert_array_equal(rows,np.load(SOURCE/f'day{day}_indices.npy'))
        frame=raw.iloc[rows]
        y=frame.label.to_numpy()
        p=np.load(SOURCE/f'day{day}_prediction.npy')
        wins,pairs=concordance(y,p)
        value=dict(auc=wins/pairs,user=decomposition(y,p,frame.user_id.to_numpy()),
            task=decomposition(y,p,frame.task_id.to_numpy()))
        for title,group in [('user',frame.user_id.to_numpy()),('task',frame.task_id.to_numpy())]:
            prior=raw.loc[raw.pt_d.lt(20191100+day),title+'_id'].unique()
            known=np.isin(group,prior)
            value[title+'_history']={}
            for label,mask in [('known',known),('new',~known)]:
                if mask.sum() and y[mask].sum() and y[mask].sum()<mask.sum():
                    win,count=concordance(y[mask],p[mask])
                    value[title+'_history'][label]=dict(rows=int(mask.sum()),clicks=int(y[mask].sum()),auc=win/count)
        records[str(20191100+day)]=value
    write(OUT,dict(status='passed',purpose='Development ranking diagnosis only; no new model selection',
        inferred17_labels_read=False,new_models_fitted=False,source_verification_sha256=digest(SOURCE/'verification.json'),dates=records))
    emit('remaining_auc_gap',dates=records)
