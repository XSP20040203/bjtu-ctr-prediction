"""Per-exposure hierarchical priors using only strictly earlier ad outcomes."""
import numpy as np
import pandas as pd

FIELDS = ['pt_d', 'user_id', 'spread_app_id', 'slot_id']


def aggregate(past, query, fields, independent=False):
    if not len(past):
        return np.zeros(len(query)), np.zeros(len(query))
    if independent:
        # Sorted structured keys and integer reductions independently check groupby.
        dtype = np.dtype([(name, '<i8') for name in fields])
        before = np.empty(len(past), dtype=dtype)
        after = np.empty(len(query), dtype=dtype)
        for name in fields:
            before[name] = past[name].to_numpy(dtype='int64')
            after[name] = query[name].to_numpy(dtype='int64')
        keys, inverse, count = np.unique(before, return_inverse=True, return_counts=True)
        click = np.bincount(inverse, weights=past.label.to_numpy(), minlength=len(keys))
        position = np.searchsorted(keys, after)
        seen = position < len(keys)
        seen[seen] &= keys[position[seen]] == after[seen]
        clicks, counts = np.zeros(len(query)), np.zeros(len(query))
        clicks[seen], counts[seen] = click[position[seen]], count[position[seen]]
        return clicks, counts
    grouped = past.groupby(fields, observed=True, sort=False).label.agg(['sum', 'size'])
    index = pd.Index(query[fields[0]]) if len(fields) == 1 else pd.MultiIndex.from_frame(query[fields])
    aligned = grouped.reindex(index, fill_value=0)
    return aligned['sum'].to_numpy(), aligned['size'].to_numpy()


def build(metadata, labels, cutoff, independent=False, positive_weight=5.):
    if list(metadata) != FIELDS:
        raise ValueError('Only declared date/user/app/slot metadata is permitted')
    labels = np.asarray(labels)
    dates = metadata.pt_d.to_numpy()
    if len(labels) != len(metadata) or np.any(dates > cutoff):
        raise ValueError('Aligned current-or-earlier populations are required')
    if np.any(labels[dates >= cutoff] != -1) or not np.isin(labels[dates < cutoff], [0, 1]).all():
        raise ValueError('Current labels must be unavailable; earlier labels must be binary')
    if positive_weight <= 0:
        raise ValueError('Positive class weight must be positive')
    frame = metadata.copy()
    frame['label'] = labels
    probability = np.empty(len(frame), dtype='float64')
    audit = []
    for date in sorted(np.unique(dates)):
        past = frame.loc[frame.pt_d < date]
        query = frame.loc[frame.pt_d == date]
        assert len(past) == 0 or int(past.pt_d.max()) < int(date)
        assert np.isin(past.label, [0, 1]).all()
        global_rate = (float(past.label.sum()) + 8.4) / (len(past) + 1000.)
        user_click, user_count = aggregate(past, query, ['user_id'], independent)
        app_click, app_count = aggregate(past, query, ['spread_app_id'], independent)
        slot_click, slot_count = aggregate(past, query, ['slot_id'], independent)
        pair_click, pair_count = aggregate(past, query, ['user_id', 'spread_app_id'], independent)
        user = (user_click + 80. * global_rate) / (user_count + 80.)
        app = (app_click + 100. * global_rate) / (app_count + 100.)
        slot = (slot_click + 100. * global_rate) / (slot_count + 100.)
        pair_prior = (user * app / global_rate).clip(1e-5, .25)
        pair = (pair_click + 40. * pair_prior) / (pair_count + 40.)
        rate = (pair * slot / global_rate).clip(1e-5, .25)
        probability[np.flatnonzero(dates == date)] = rate
        audit.append(dict(query_date=int(date), max_label_date=int(past.pt_d.max()) if len(past) else None,
                          training_rows=len(past), query_rows=len(query), global_rate=global_rate,
                          current_date_labels_used=False))
    offset = (np.log(probability) - np.log1p(-probability) + np.log(positive_weight)).astype('float32')
    assert np.isfinite(offset).all()
    return offset, audit
