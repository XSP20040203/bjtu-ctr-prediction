"""Strict-past, label-free query masks for complete-negative model transfer."""
import numpy as np
import pandas as pd

NAMES = ['cold_task', 'seen_task', 'task_at_most_100_prior_rows',
         'task_over_100_prior_rows', 'user_no_past_ad_click', 'user_with_past_ad_click']


def lookup(keys, values, query):
    positions = np.searchsorted(keys, query)
    valid = positions < len(keys)
    valid[valid] &= keys[positions[valid]] == query[valid]
    result = np.zeros(len(query), dtype='float64')
    result[valid] = values[positions[valid]]
    return result


def masks(dates, users, tasks, training_rows, training_labels, query_rows, cutoff, independent=False):
    dates, users, tasks = map(np.asarray, (dates, users, tasks))
    training_rows, training_labels, query_rows = map(np.asarray, (training_rows, training_labels, query_rows))
    if dates.ndim != 1 or dates.shape != users.shape or dates.shape != tasks.shape:
        raise ValueError('Aligned raw metadata is required')
    if not np.array_equal(training_rows, np.flatnonzero(dates < cutoff)):
        raise ValueError('Every strictly earlier row is required')
    if len(training_rows) != len(training_labels) or not np.isin(training_labels, [0, 1]).all():
        raise ValueError('Only earlier binary labels are permitted')
    if np.any(query_rows < 0) or np.any(query_rows >= len(dates)) or np.any(dates[query_rows] != cutoff):
        raise ValueError('Only the complete requested date is queryable')
    if independent:
        task_keys, counts = np.unique(tasks[training_rows], return_counts=True)
        task_count = lookup(task_keys, counts, tasks[query_rows])
        user_keys, inverse = np.unique(users[training_rows], return_inverse=True)
        clicks = np.bincount(inverse, weights=training_labels)
        user_clicks = lookup(user_keys, clicks, users[query_rows])
    else:
        prior = pd.DataFrame({'task': tasks[training_rows], 'user': users[training_rows], 'label': training_labels})
        task_count = pd.Series(tasks[query_rows]).map(prior.groupby('task', observed=True).size()).fillna(0).to_numpy()
        user_clicks = pd.Series(users[query_rows]).map(prior.groupby('user', observed=True).label.sum()).fillna(0).to_numpy()
    result = dict(zip(NAMES, [task_count == 0, task_count > 0, task_count <= 100,
                             task_count > 100, user_clicks == 0, user_clicks > 0]))
    assert all(value.dtype == bool and value.shape == query_rows.shape for value in result.values())
    return result
