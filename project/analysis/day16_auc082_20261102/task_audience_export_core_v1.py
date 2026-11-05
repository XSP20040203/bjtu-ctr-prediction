"""Apply a fitted audience schema without changing its unknown-category meaning."""
import numpy as np
import pandas as pd
from task_audience_core import HISTOGRAMS


def frozen_schema(frame, advertisements, meta, feature_names, independent=False):
    missing = set(feature_names) - set(frame)
    if missing:
        raise ValueError(f'Missing fitted audience features: {sorted(missing)}')
    output = frame.loc[:, feature_names].copy()
    keys = pd.MultiIndex.from_arrays(meta['keys'].T)
    groups = keys.get_indexer(pd.MultiIndex.from_frame(advertisements[['pt_d', 'task_id']]))
    if np.any(groups < 0):
        raise ValueError('Advertisement rows do not match the audience groups')
    for field in HISTOGRAMS:
        prefix = f'fraction__{field}__'
        known = [int(name[len(prefix):]) for name in feature_names
                 if name.startswith(prefix) and not name.endswith('__unknown')]
        unknown = ~advertisements[field].isin(known).to_numpy()
        if independent:
            counts = pd.Series(unknown.astype('int64')).groupby(groups).sum()
            counts = counts.reindex(np.arange(len(keys)), fill_value=0).to_numpy(dtype='float64')
        else:
            counts = np.bincount(groups, weights=unknown, minlength=len(keys))
        output[prefix + 'unknown'] = np.round(counts / meta['sizes'], 5).astype('float32')
    if list(output) != list(feature_names) or not np.isfinite(output.to_numpy()).all():
        raise ValueError('Invalid fitted audience schema')
    return output
