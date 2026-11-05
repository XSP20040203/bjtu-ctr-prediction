"""Date-isolated stratified queries covering every historical row and negative."""
import numpy as np


def design(dates, labels, cutoff, seed=151681, negative_block=512, positive_draws=64):
    dates, labels = np.asarray(dates), np.asarray(labels)
    if dates.ndim != 1 or dates.shape != labels.shape:
        raise ValueError('Aligned dates and labels are required')
    if np.any(labels[dates >= cutoff] != -1):
        raise ValueError('Held and future labels must be masked')
    prior = np.flatnonzero(dates < cutoff)
    if not len(prior) or not np.isin(labels[prior], [0, 1]).all():
        raise ValueError('Earlier binary training labels are required')
    if negative_block < 1 or positive_draws < 1:
        raise ValueError('Positive query limits are required')
    rng = np.random.default_rng(seed)
    orders, groups, pairs = [], [], []
    offset, number = 0, 0
    for date in np.unique(dates[prior]):
        rows = prior[dates[prior] == date]
        positive = rng.permutation(rows[labels[rows] == 1])
        negative = rng.permutation(rows[labels[rows] == 0])
        if not len(positive) or not len(negative):
            raise ValueError('Every training date must contain both classes')
        queries = min(len(positive), max(1, int(np.ceil(len(negative) / negative_block))))
        for pos, neg in zip(np.array_split(positive, queries), np.array_split(negative, queries)):
            block = rng.permutation(np.r_[pos, neg])
            win, lose = np.flatnonzero(labels[block] == 1) + offset, np.flatnonzero(labels[block] == 0) + offset
            # The first component covers every negative; the second covers every positive.
            cover = np.column_stack([rng.choice(win, len(lose), replace=True), lose])
            winners = np.repeat(win, positive_draws)
            extra = np.column_stack([winners, rng.choice(lose, len(winners), replace=True)])
            pairs.append(np.unique(np.r_[cover, extra], axis=0).astype('int32'))
            orders.append(block)
            groups.append(np.full(len(block), number, dtype='int64'))
            offset += len(block)
            number += 1
    order, group, pair = np.concatenate(orders), np.concatenate(groups), np.concatenate(pairs)
    np.testing.assert_array_equal(np.sort(order), prior)
    np.testing.assert_array_equal(np.unique(pair[:, 1]), np.flatnonzero(labels[order] == 0))
    np.testing.assert_array_equal(np.unique(pair[:, 0]), np.flatnonzero(labels[order] == 1))
    assert np.all(labels[order[pair[:, 0]]] == 1) and np.all(labels[order[pair[:, 1]]] == 0)
    np.testing.assert_array_equal(group[pair[:, 0]], group[pair[:, 1]])
    np.testing.assert_array_equal(dates[order[pair[:, 0]]], dates[order[pair[:, 1]]])
    return order, group, pair
