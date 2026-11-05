"""Random, date-isolated row queries for cross-user advertisement AUC training."""
import numpy as np


def design(dates, labels, cutoff, seed=151648, block_size=1024, draws=48):
    dates = np.asarray(dates)
    labels = np.asarray(labels)
    if dates.ndim != 1 or dates.shape != labels.shape:
        raise ValueError('Aligned dates and labels are required')
    if np.any(labels[dates >= cutoff] != -1):
        raise ValueError('Held and future advertisement labels must be masked')
    prior = np.flatnonzero(dates < cutoff)
    if not len(prior) or not np.isin(labels[prior], [0, 1]).all():
        raise ValueError('Observed earlier binary labels are required')
    if block_size < 2 or draws < 1:
        raise ValueError('Nontrivial query sizes and positive draws are required')
    rng = np.random.default_rng(seed)
    order, groups, pairs = [], [], []
    offset, group_number = 0, 0
    for date in np.unique(dates[prior]):
        shuffled = rng.permutation(prior[dates[prior] == date])
        for start in range(0, len(shuffled), block_size):
            block = shuffled[start:start + block_size]
            target = labels[block]
            positive = np.flatnonzero(target == 1) + offset
            negative = np.flatnonzero(target == 0) + offset
            order.append(block)
            groups.append(np.full(len(block), group_number, dtype='int64'))
            if len(positive) and len(negative):
                winner = np.repeat(positive, draws)
                loser = rng.choice(negative, len(winner), replace=True)
                pairs.append(np.column_stack([winner, loser]))
            offset += len(block)
            group_number += 1
    if not pairs:
        raise ValueError('At least one mixed-class query is required')
    return np.concatenate(order), np.concatenate(groups), np.concatenate(pairs)

