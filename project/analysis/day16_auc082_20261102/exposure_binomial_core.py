"""Exposure-weighted binomial targets with strictly earlier advertisement counts."""
import numpy as np
import pandas as pd


def training_targets(keys,counts,sizes,cutoff,half_life=None):
    keys=np.asarray(keys)
    counts=np.asarray(counts)
    sizes=np.asarray(sizes)
    if keys.ndim!=2 or keys.shape[1]!=2 or counts.shape!=(len(keys),) or sizes.shape!=(len(keys),):
        raise ValueError('Aligned daily keys, counts, and exposures are required')
    if len(np.unique(keys,axis=0))!=len(keys):
        raise ValueError('Every user/date target must be unique')
    if np.any(counts[keys[:,0]>=cutoff]!=-1):
        raise ValueError('Held and future advertisement counts must be masked')
    if not np.isfinite(sizes).all() or np.any(sizes<=0) or not np.array_equal(sizes,sizes.astype('int64')):
        raise ValueError('Positive integer exposures are required')
    training=np.flatnonzero(keys[:,0]<cutoff)
    observed=counts[training]
    exposures=sizes[training].astype('float64')
    if not len(training) or not np.isfinite(observed).all() or np.any(observed<0) or np.any(observed>exposures):
        raise ValueError('Valid strictly-prior click counts are required')
    if not np.array_equal(observed,observed.astype('int64')):
        raise ValueError('Observed click counts must be integers')
    weights=exposures.copy()
    if half_life is not None:
        if half_life<=0:
            raise ValueError('Recency half life must be positive')
        age=(pd.Timestamp(str(cutoff))-pd.to_datetime(keys[training,0].astype(str))).days.to_numpy()-1
        weights*=np.exp2(-age/half_life)
    weights/=weights.mean()
    return training,observed/exposures,weights
