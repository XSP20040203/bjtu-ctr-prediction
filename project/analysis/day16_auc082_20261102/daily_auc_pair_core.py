"""Strict-prior explicit pairs for advertisement-row AUC between daily users."""
import numpy as np


def sample_pairs(keys,sizes,counts,cutoff,negatives_per_positive=48,seed=151646):
    keys=np.asarray(keys)
    sizes=np.asarray(sizes)
    counts=np.asarray(counts)
    if keys.ndim!=2 or keys.shape[1]!=2 or sizes.shape!=(len(keys),) or counts.shape!=(len(keys),):
        raise ValueError('Aligned date/user keys, sizes, and counts are required')
    if len(np.unique(keys,axis=0))!=len(keys):
        raise ValueError('Every daily user key must be unique')
    if np.any(counts[keys[:,0]>=cutoff]!=-1):
        raise ValueError('All held and future advertisement counts must be masked')
    if not np.isfinite(sizes).all() or np.any(sizes<=0) or not np.array_equal(sizes,sizes.astype('int64')):
        raise ValueError('Positive integer exposure counts are required')
    training=np.flatnonzero(keys[:,0]<cutoff)
    training=training[np.lexsort((keys[training,1],keys[training,0]))]
    observed=counts[training]
    if not np.isfinite(observed).all() or np.any(observed<0) or np.any(observed>sizes[training]) or not np.array_equal(observed,observed.astype('int64')):
        raise ValueError('Strictly-prior observed integer click counts are required')
    if not isinstance(negatives_per_positive,int) or negatives_per_positive<=0:
        raise ValueError('A positive integer pair-sampling count is required')
    dates,groups=np.unique(keys[training,0],return_inverse=True)
    rng=np.random.default_rng(seed)
    pairs=[]
    for index,date in enumerate(dates):
        local=np.flatnonzero(groups==index)
        click=observed[local].astype('float64')
        nonclick=(sizes[training[local]]-observed[local]).astype('float64')
        if not click.sum() or not nonclick.sum():
            continue
        number=int(click.sum())*negatives_per_positive
        winner=rng.choice(local,size=number,p=click/click.sum())
        loser=rng.choice(local,size=number,p=nonclick/nonclick.sum())
        keep=winner!=loser
        pairs.append(np.column_stack([winner[keep],loser[keep]]))
    if not pairs or not sum(len(pair) for pair in pairs):
        raise ValueError('Nonconstant positive/nonclick pairs from earlier dates are required')
    pairs=np.concatenate(pairs).astype('int64')
    assert np.all(groups[pairs[:,0]]==groups[pairs[:,1]])
    assert np.all(pairs[:,0]!=pairs[:,1]) and np.all(keys[training,0]<cutoff)
    return training,pairs,groups.astype('int64')
