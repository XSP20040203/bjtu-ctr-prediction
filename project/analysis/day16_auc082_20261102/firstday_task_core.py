"""Select genuinely newly observed tasks from strictly earlier advertisement dates."""
import numpy as np
import pandas as pd
from cross_user_pair_core import design


def firstday_design(history, training_rows, targets, cutoff):
    if list(history) != ['pt_d', 'task_id']:
        raise ValueError('Only unlabeled date/task keys are allowed')
    rows = np.asarray(training_rows)
    labels = np.asarray(targets)
    if rows.ndim != 1 or rows.shape != labels.shape:
        raise ValueError('Training rows and targets must align')
    dates = history.pt_d.to_numpy()[rows]
    if np.any(dates >= cutoff) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Only observed strictly earlier training labels are allowed')
    prior = history.loc[history.pt_d.lt(cutoff)]
    first = prior.groupby('task_id', sort=False).pt_d.min()
    first_dates = history.task_id.iloc[rows].map(first).to_numpy()
    # Nov10 is partial and Nov11 is the first full logged day. Their first
    # appearances are left-censored; use observed launches from Nov12 onward.
    keep = np.flatnonzero((dates == first_dates) & (dates >= 20191112))
    if not len(keep):
        raise ValueError('No uncensored task launches before the cutoff')
    order, groups, pairs = design(dates[keep], labels[keep], cutoff)
    return keep[order], groups, pairs
