"""Rate baselines and group-constant corrections for chronological daily ranking."""
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from rich_userday_activity import correct


def rate_baseline(mass,sizes):
    mass=np.asarray(mass,dtype='float64')
    sizes=np.asarray(sizes,dtype='float64')
    if mass.shape!=sizes.shape or not np.isfinite(mass).all() or not np.isfinite(sizes).all():
        raise ValueError('Finite aligned rate mass and exposures are required')
    if np.any(mass<=0) or np.any(sizes<=0):
        raise ValueError('Positive rate mass and exposures are required')
    # Match the float32 representation stored by CatBoost Pool during fitting.
    return logit(np.clip(mass/sizes,1e-5,.25)).astype('float32').astype('float64')


def daily_prediction(base,residual,total,sizes,groups,weight,mode,independent=False):
    base=np.asarray(base,dtype='float64')
    groups=np.asarray(groups,dtype='int64')
    residual,total,sizes=map(np.asarray,(residual,total,sizes))
    if base.shape!=groups.shape or residual.shape!=total.shape or total.shape!=sizes.shape:
        raise ValueError('Daily forecasts and row groups must align')
    if np.any(groups<0) or np.any(groups>=len(total)) or not np.isfinite(total).all():
        raise ValueError('Known groups and finite forecasts are required')
    if weight==0:
        return base.copy()
    if mode=='count_mass':
        response=expit(total)*sizes
        if not independent:
            return correct(base,response,groups,'poisson',weight)
        p=np.clip(base,1e-8,1-1e-8)
        table=pd.DataFrame(dict(group=groups,p=p))
        group_mass=table.groupby('group',sort=False).p.transform('sum').to_numpy()
        ratio=np.clip(np.maximum(response[groups],1e-6)/np.maximum(group_mass,1e-8),.05,20.)**weight
        odds=p/(1-p)*ratio
        return odds/(1+odds)
    if mode not in ['residual_add','rate_replace']:
        raise ValueError('Only declared between-user correction formulas are permitted')
    z=logit(np.clip(base,1e-8,1-1e-8))
    if independent:
        table=pd.DataFrame(dict(group=groups,z=z))
        existing=table.groupby('group',sort=False).z.transform('mean').to_numpy()
    else:
        n=np.bincount(groups).clip(1)
        existing=(np.bincount(groups,weights=z)/n)[groups]
    forecast=(residual if mode=='residual_add' else total)[groups]
    normalized=(forecast-forecast.mean())*existing.std()/max(forecast.std(),1e-6)
    delta=normalized if mode=='residual_add' else normalized+existing.mean()-existing
    return expit(np.clip(z+weight*delta,-35,20))
