"""Select prior-trained numeric audience inputs and align them to raw ad rows."""
import numpy as np
import pandas as pd


def select_columns(names,categories,importance,number=192):
    values=np.asarray(importance,dtype='float64')
    if values.shape!=(len(names),) or not np.isfinite(values).all():
        raise ValueError('A finite importance for every feature is required')
    numeric=np.array([i for i,name in enumerate(names) if name not in categories],dtype='int64')
    order=numeric[np.lexsort((numeric,-values[numeric]))]
    return [names[i] for i in order[:number]]


def aligned(keys,values,query,cutoff):
    keys=np.asarray(keys)
    query=np.asarray(query)
    values=np.asarray(values,dtype='float32')
    if keys.ndim!=2 or keys.shape[1]!=2 or query.ndim!=2 or query.shape[1]!=2:
        raise ValueError('Only date/task keys are accepted')
    if len(values)!=len(keys) or np.any(keys[:,0]>cutoff) or np.any(query[:,0]>cutoff):
        raise ValueError('Aligned audience inputs must not contain future dates')
    index=pd.MultiIndex.from_arrays(keys.T)
    if not index.is_unique:
        raise ValueError('Audience date/task keys must be unique')
    positions=index.get_indexer(pd.MultiIndex.from_arrays(query.T))
    if np.any(positions<0):
        raise ValueError('Audience features missing for an advertisement task')
    output=values[positions]
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite audience features')
    return output
