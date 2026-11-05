"""Full-cohort task/day audience features with strictly earlier ad feedback."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

FIELDS = ['pt_d','task_id','user_id','adv_prim_id','spread_app_id','app_second_class',
          'creat_type_cd','inter_type_cd','hispace_app_tags','slot_id','age','sex','city_rank',
          'net_type','u_refreshTimes','u_feedLifeCycle','app_score']
CATEGORIES = ['adv_prim_id','spread_app_id','app_second_class','creat_type_cd',
              'inter_type_cd','hispace_app_tags']
HISTOGRAMS = ['age','sex','city_rank','net_type','creat_type_cd','inter_type_cd',
              'app_second_class','slot_id']


def aggregate(values, groups, weights, number, engine='bincount'):
    values=np.asarray(values,dtype='float64')
    groups=np.asarray(groups,dtype='int64')
    weights=np.asarray(weights,dtype='float64')
    if engine=='bincount':
        return np.bincount(groups,weights=values*weights,minlength=number)
    if engine=='sparse':
        matrix=csr_matrix((weights,(groups,np.arange(len(groups)))),shape=(number,len(groups)))
        return np.asarray(matrix@values).reshape(-1)
    raise ValueError('Unknown aggregation engine')


def build(advertisements, labels, quality_keys, quality, cutoff, engine='bincount'):
    if list(advertisements)!=FIELDS:
        raise ValueError('Only the declared unlabeled advertisement fields are allowed')
    labels=np.asarray(labels)
    if labels.shape!=(len(advertisements),):
        raise ValueError('Labels must align with the original rows')
    all_dates=advertisements.pt_d.to_numpy()
    if np.any(labels[all_dates>=cutoff]!=-1):
        raise ValueError('Held and future advertisement labels must be masked')
    if not np.isin(labels[all_dates<cutoff],[0,1]).all():
        raise ValueError('Earlier advertisement labels must be binary')
    selected=np.flatnonzero(all_dates<=cutoff)
    ads=advertisements.iloc[selected].reset_index(drop=True)
    y=labels[selected]
    groups,index=pd.factorize(pd.MultiIndex.from_frame(ads[['pt_d','task_id']]),sort=True)
    keys=index.to_frame(index=False).to_numpy(dtype='int64')
    ng=len(keys)
    sizes=np.bincount(groups,minlength=ng).astype('float64')
    counts=np.bincount(groups,weights=np.maximum(y,0),minlength=ng)
    counts[keys[:,0]>=cutoff]=-1
    columns={}
    cats=[]
    grouped=ads.groupby(['pt_d','task_id'],sort=True)
    first=grouped[CATEGORIES].first()
    for name in CATEGORIES:
        columns['category__'+name]=first[name].to_numpy(dtype='int64')
        cats.append('category__'+name)
        columns['log_unique__'+name]=np.log1p(grouped[name].nunique().to_numpy())
    pair=ads.groupby(['pt_d','task_id','user_id'],sort=True).size().rename('weight').reset_index()
    pg=index.get_indexer(pd.MultiIndex.from_frame(pair[['pt_d','task_id']]))
    unique=np.bincount(pg,minlength=ng).astype('float64')
    columns['log_exposures']=np.log1p(sizes)
    columns['log_users']=np.log1p(unique)
    columns['exposures_per_user']=sizes/unique
    dates=ads.pt_d.to_numpy()
    for name in HISTOGRAMS:
        vocabulary=np.sort(ads.loc[dates<cutoff,name].unique())
        for value in vocabulary:
            columns[f'fraction__{name}__{value}']=aggregate(
                ads[name].to_numpy()==value,groups,np.ones(len(ads)),ng,engine)/sizes
        columns['fraction__'+name+'__unknown']=aggregate(
            ~ads[name].isin(vocabulary).to_numpy(),groups,np.ones(len(ads)),ng,engine)/sizes
    for name in ['u_refreshTimes','u_feedLifeCycle','app_score']:
        value=ads[name].to_numpy(dtype='float64')
        mean=aggregate(value,groups,np.ones(len(ads)),ng,engine)/sizes
        second=aggregate(value**2,groups,np.ones(len(ads)),ng,engine)/sizes
        columns[name+'__mean']=mean
        columns[name+'__std']=np.sqrt(np.maximum(second-mean**2,0))
    # Historical user and item response is recomputed separately for each day.
    for field in ['user_id','task_id','adv_prim_id','spread_app_id']:
        previous_n=pd.Series(dtype='float64')
        previous_c=pd.Series(dtype='float64')
        values=np.zeros((len(ads),5),dtype='float64')
        for date in np.unique(dates):
            rows=np.flatnonzero(dates==date)
            ids=ads[field].iloc[rows]
            n=ids.map(previous_n).fillna(0).to_numpy()
            c=ids.map(previous_c).fillna(0).to_numpy()
            prior=(previous_c.sum()+.5)/(previous_n.sum()+50.)
            values[rows]=np.column_stack([np.log1p(n),np.log1p(c),
                (c+20*prior)/(n+20),(c+200*prior)/(n+200),n==0])
            if date<cutoff:
                current=pd.DataFrame({'id':ids.to_numpy(),'label':y[rows]}).groupby('id').label.agg(['size','sum'])
                previous_n=previous_n.add(current['size'],fill_value=0)
                previous_c=previous_c.add(current['sum'],fill_value=0)
        for i,title in enumerate(['log_n','log_c','ctr20','ctr200','new']):
            mean=aggregate(values[:,i],groups,np.ones(len(ads)),ng,engine)/sizes
            second=aggregate(values[:,i]**2,groups,np.ones(len(ads)),ng,engine)/sizes
            columns[f'past__{field}__{title}__mean']=mean
            columns[f'past__{field}__{title}__std']=np.sqrt(np.maximum(second-mean**2,0))
    quality_keys=np.asarray(quality_keys)
    quality=np.asarray(quality,dtype='float64')
    if quality_keys.shape!=(len(quality),2) or np.any(quality_keys[:,0]>cutoff):
        raise ValueError('Quality inputs must be aligned and not contain future dates')
    qindex=pd.MultiIndex.from_arrays(quality_keys.T)
    if not qindex.is_unique:
        raise ValueError('Quality user/day keys must be unique')
    qi=qindex.get_indexer(pd.MultiIndex.from_frame(pair[['pt_d','user_id']]))
    if np.any(qi<0):
        raise ValueError('Missing quality user/day key')
    weights=pair.weight.to_numpy(dtype='float64')
    if engine=='sparse':
        weighted=csr_matrix((weights,(pg,qi)),shape=(ng,len(quality)))
        unweighted=csr_matrix((np.ones(len(pg)),(pg,qi)),shape=(ng,len(quality)))
        means=np.asarray(weighted@quality)/sizes[:,None]
        second=np.asarray(weighted@(quality**2))/sizes[:,None]
        unique_means=np.asarray(unweighted@quality)/unique[:,None]
    else:
        means=np.empty((ng,quality.shape[1]))
        second=np.empty_like(means)
        unique_means=np.empty_like(means)
        for i in range(quality.shape[1]):
            q=quality[qi,i]
            means[:,i]=aggregate(q,pg,weights,ng,engine)/sizes
            second[:,i]=aggregate(q*q,pg,weights,ng,engine)/sizes
            unique_means[:,i]=aggregate(q,pg,np.ones(len(pg)),ng,engine)/unique
    std=np.sqrt(np.maximum(second-means**2,0))
    for i in range(quality.shape[1]):
        columns[f'quality_{i}__exposure_mean']=means[:,i]
        columns[f'quality_{i}__user_mean']=unique_means[:,i]
        columns[f'quality_{i}__exposure_std']=std[:,i]
    frame=pd.DataFrame(columns)
    for name in frame:
        if name not in cats:
            # Variance near zero can differ below float precision between engines.
            frame[name]=np.round(frame[name],5).astype('float32')
    assert np.isfinite(frame.to_numpy()).all()
    held=dates==cutoff
    held_groups=np.flatnonzero(keys[:,0]==cutoff)
    local=np.full(ng,-1,dtype='int64')
    local[held_groups]=np.arange(len(held_groups))
    meta=dict(keys=keys,sizes=sizes,counts=counts,rows=selected[held],
              row_groups=local[groups[held]],held_groups=held_groups)
    return frame,cats,meta
