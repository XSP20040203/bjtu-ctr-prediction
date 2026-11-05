"""Content-adjusted feedback at exact unlabeled advertisement/news snapshot keys."""
import numpy as np
import pandas as pd
from news_quality_features import leave_user_out

AD_FIELDS=['pt_d','user_id','u_newsCatInterestsST','u_refreshTimes','u_feedLifeCycle']
NEWS_FIELDS=['e_et','u_userId','u_newsCatInterestsST','u_refreshTimes','u_feedLifeCycle',
    'i_docId','i_cat','e_po','e_ch','label','pro','cillabel']
CONTEXTS=[('exact',AD_FIELDS),('interest_refresh',AD_FIELDS[:-1]),
    ('refresh_lifecycle',['pt_d','user_id','u_refreshTimes','u_feedLifeCycle'])]
ITEMS=[['i_docId'],['i_cat'],['i_cat','e_po'],['i_cat','e_ch']]
METRICS=['log_click_oe','click_pearson','expected_ctr','log_read_oe','read_residual',
    'log_like_oe','click_surprise','other_user_confidence','log_size','missing']


def normalized(frame):
    result=frame.copy()
    result['u_newsCatInterestsST']=result.u_newsCatInterestsST.fillna('').astype(str)
    for name in ['pt_d','user_id','u_refreshTimes','u_feedLifeCycle']:
        result[name]=result[name].astype('int64')
    return result


def build(advertisements,source,cutoff):
    if list(advertisements)!=AD_FIELDS or list(source)!=NEWS_FIELDS:
        raise ValueError('Only declared advertisement keys and separate news fields are allowed')
    ads=normalized(advertisements)
    if np.any(ads.pt_d.to_numpy()>cutoff):
        raise ValueError('Future advertisement dates are excluded')
    news=source.loc[source.e_et.le(cutoff)].copy()
    if not news.label.isin([-1,1]).all():
        raise ValueError('News feedback must use the declared -1/1 labels')
    users,_=pd.factorize(pd.MultiIndex.from_frame(news[['e_et','u_userId']]),sort=True)
    clicked=news.label.eq(1).to_numpy(dtype='float64')
    target=np.column_stack([clicked,clicked*np.clip(news.pro.to_numpy(dtype='float64'),0,100)/100,
        news.cillabel.eq(1).to_numpy(dtype='float64')])
    requests=normalized(news.rename(columns={'e_et':'pt_d','u_userId':'user_id'})[AD_FIELDS])
    contexts=[]
    for title,keys in CONTEXTS:
        codes,index=pd.factorize(pd.MultiIndex.from_frame(requests[keys]),sort=True)
        positions=index.get_indexer(pd.MultiIndex.from_frame(ads[keys]))
        size=np.bincount(codes,minlength=len(index)).astype('float64')
        observed=np.column_stack([np.bincount(codes,weights=target[:,i],minlength=len(index)) for i in range(3)])
        contexts.append((title,codes,positions,size,observed))
    blocks,names=[],[]
    for fields in ITEMS:
        for alpha in [5.,50.]:
            expected,others=leave_user_out(news,target,users,fields,alpha)
            probability=expected[:,0].clip(1e-7,1-1e-7)
            surprise=-(clicked*np.log(probability)+(1-clicked)*np.log1p(-probability))
            confidence=others/(others+alpha)
            for title,codes,positions,size,observed in contexts:
                expectation=np.column_stack([np.bincount(codes,weights=expected[:,i],minlength=len(size)) for i in range(3)])
                variance=np.bincount(codes,weights=probability*(1-probability),minlength=len(size))
                columns=np.column_stack([
                    np.log((observed[:,0]+.5)/(expectation[:,0]+.5)),
                    (observed[:,0]-expectation[:,0])/np.sqrt(variance+1),
                    expectation[:,0]/np.maximum(size,1),
                    np.log((observed[:,1]+.25)/(expectation[:,1]+.25)),
                    (observed[:,1]-expectation[:,1])/np.sqrt(expectation[:,1]+1),
                    np.log((observed[:,2]+.5)/(expectation[:,2]+.5)),
                    np.bincount(codes,weights=surprise,minlength=len(size))/np.maximum(size,1),
                    np.bincount(codes,weights=confidence,minlength=len(size))/np.maximum(size,1),
                    np.log1p(size),np.zeros(len(size))])
                block=np.zeros((len(ads),len(METRICS)),dtype='float32')
                valid=positions>=0
                block[valid]=columns[positions[valid]]
                block[~valid,-1]=1
                assert np.isfinite(block).all()
                blocks.append(block)
                names.extend('snapshot_quality__'+title+'__'+'_'.join(fields)+f'__a{alpha:g}__'+metric for metric in METRICS)
    values=np.column_stack(blocks)
    assert values.shape==(len(ads),240) and len(set(names))==240
    return values,names

