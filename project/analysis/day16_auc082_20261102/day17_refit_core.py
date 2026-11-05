"""Restore genuine earlier-day counts for a final next-day refit."""
import numpy as np
import pandas as pd


def restore_prior_counts(keys,sizes,masked_counts,raw,cutoff):
    keys=np.asarray(keys)
    sizes=np.asarray(sizes)
    masked_counts=np.asarray(masked_counts)
    if keys.ndim!=2 or keys.shape[1]!=2 or sizes.shape!=(len(keys),) or masked_counts.shape!=(len(keys),):
        raise ValueError('Aligned date/user keys, exposures, and masked counts are required')
    if np.any(keys[:,0]>=cutoff):
        raise ValueError('A refit cannot contain its prediction date or future targets')
    index=pd.MultiIndex.from_arrays(keys.T,names=['pt_d','user_id'])
    if not index.is_unique:
        raise ValueError('Every refit user/date record must be unique')
    earlier=raw.loc[raw.pt_d<cutoff,['pt_d','user_id','label']]
    if not earlier.label.isin([0,1]).all():
        raise ValueError('Only observed genuine earlier binary advertisement labels are allowed')
    observed=earlier.groupby(['pt_d','user_id'],sort=False).label.agg(['sum','size']).reindex(index)
    if observed.isna().any().any() or not np.array_equal(observed['size'].to_numpy(),sizes):
        raise ValueError('Original exposure counts and daily refit keys must agree exactly')
    counts=observed['sum'].to_numpy(dtype='int64')
    known=masked_counts>=0
    if not np.array_equal(counts[known],masked_counts[known]):
        raise ValueError('Previously observed prior counts cannot change')
    if np.any(masked_counts < -1) or np.any(counts>sizes):
        raise ValueError('Invalid daily counts')
    return counts
