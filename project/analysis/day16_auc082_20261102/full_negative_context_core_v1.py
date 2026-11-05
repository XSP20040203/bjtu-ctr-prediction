"""Complete-population temporal row selection and independently checked joins."""
import numpy as np
import pandas as pd


def population(dates, labels, cutoff):
    dates = np.asarray(dates)
    safe = np.asarray(labels, dtype='int32').copy()
    if dates.ndim != 1 or dates.shape != safe.shape:
        raise ValueError('Aligned one-dimensional dates and labels required')
    safe[dates >= cutoff] = -1
    rows = np.flatnonzero(dates <= cutoff)
    train = np.flatnonzero(dates[rows] < cutoff)
    held = np.flatnonzero(dates[rows] == cutoff)
    if not len(train) or not len(held) or not np.isin(safe[rows[train]], [0, 1]).all():
        raise ValueError('Observed earlier binary outcomes and a complete held date required')
    return rows, train, held, safe


def positions(keys, query, independent=False):
    keys, query = np.asarray(keys), np.asarray(query)
    if keys.ndim != 2 or query.ndim != 2 or keys.shape[1] != 2 or query.shape[1] != 2:
        raise ValueError('Date/user key pairs required')
    if independent:
        lookup = {tuple(key): i for i, key in enumerate(keys)}
        if len(lookup) != len(keys):
            raise ValueError('Duplicate context key')
        result = np.fromiter((lookup.get(tuple(key), -1) for key in query), dtype='int64', count=len(query))
    else:
        index = pd.MultiIndex.from_arrays(keys.T)
        if not index.is_unique:
            raise ValueError('Duplicate context key')
        result = index.get_indexer(pd.MultiIndex.from_arrays(query.T))
    if np.any(result < 0):
        raise ValueError('Missing context key')
    return result
