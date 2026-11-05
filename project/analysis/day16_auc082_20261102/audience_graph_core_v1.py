"""Rotation-invariant collaborative response features from chronological ad graphs."""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

RANKS = [8, 16, 32, 64]
METRICS = ['click_projection', 'exposure_projection', 'residual_projection',
           'click_cosine', 'neighbor_click_projection', 'neighbor_residual_projection']
NAMES = [f'graph_r{rank}__{metric}' for rank in RANKS for metric in METRICS]


def counts(users, tasks, weights, shape, independent=False):
    if independent:
        table = pd.DataFrame(dict(user=users, task=tasks, value=weights))
        grouped = table.groupby(['user', 'task'], sort=True).value.sum()
        index = grouped.index
        result = coo_matrix((grouped.to_numpy(dtype='float64'),
                             (index.get_level_values(0), index.get_level_values(1))), shape=shape).tocsr()
    else:
        result = coo_matrix((weights, (users, tasks)), shape=shape, dtype='float64').tocsr()
    result.sum_duplicates()
    result.sort_indices()
    return result


def build_day(ads, labels, day, query_positions, independent=False):
    dates = ads.pt_d.to_numpy()
    labels = np.asarray(labels)
    query_positions = np.asarray(query_positions, dtype='int64')
    if labels.shape != dates.shape or np.any(labels[dates >= day] != -1):
        raise ValueError('Current and future advertisement labels must be masked')
    if not np.isin(labels[dates < day], [0, 1]).all():
        raise ValueError('Historical labels must be binary')
    if np.any(dates[query_positions] != day):
        raise ValueError('Query rows must belong to the requested date')
    available = np.flatnonzero(dates <= day)
    current = ads.iloc[available]
    user_values = np.sort(current.user_id.unique())
    task_values = np.sort(current.task_id.unique())
    users = np.searchsorted(user_values, current.user_id.to_numpy())
    tasks = np.searchsorted(task_values, current.task_id.to_numpy())
    shape = (len(user_values), len(task_values))
    exposure = counts(users, tasks, np.ones(len(current)), shape, independent)
    exposure.data = np.log1p(exposure.data)
    exposure = normalize(exposure, norm='l2', axis=1, copy=False)
    past = dates[available] < day
    if not past.any():
        return np.zeros((len(query_positions), len(NAMES)), dtype='float32')
    history_n = counts(users[past], tasks[past], np.ones(int(past.sum())), shape, independent)
    history_c = counts(users[past], tasks[past], labels[available[past]].astype('float64'), shape, independent)
    rates = (np.asarray(history_c.sum(axis=1)).ravel() + .5) / (np.asarray(history_n.sum(axis=1)).ravel() + 50.)
    residual = history_c - history_n.multiply(rates[:, None])
    effective_rank = min(max(RANKS), min(shape) - 1)
    if effective_rank < 1:
        return np.zeros((len(query_positions), len(NAMES)), dtype='float32')
    svd = TruncatedSVD(n_components=effective_rank, n_iter=5, random_state=152601)
    scores = svd.fit_transform(exposure)
    user_basis = scores / np.maximum(svd.singular_values_, 1e-12)
    item_basis = svd.components_.T
    click_latent = history_c @ item_basis
    exposure_latent = history_n @ item_basis
    residual_latent = residual @ item_basis
    query = ads.iloc[query_positions]
    qu = np.searchsorted(user_values, query.user_id.to_numpy())
    qt = np.searchsorted(task_values, query.task_id.to_numpy())
    blocks = []
    for requested in RANKS:
        rank = min(requested, effective_rank)
        vb = item_basis[:, :rank]
        ub = user_basis[:, :rank]
        c = click_latent[:, :rank]
        n = exposure_latent[:, :rank]
        r = residual_latent[:, :rank]
        neighbor_c = ub @ (ub.T @ c)
        neighbor_r = ub @ (ub.T @ r)
        task = vb[qt]
        click_dot = np.einsum('ij,ij->i', c[qu], task)
        cosine = click_dot / np.maximum(np.linalg.norm(c[qu], axis=1) * np.linalg.norm(task, axis=1), 1e-12)
        values = [click_dot, np.einsum('ij,ij->i', n[qu], task),
                  np.einsum('ij,ij->i', r[qu], task), cosine,
                  np.einsum('ij,ij->i', neighbor_c[qu], task),
                  np.einsum('ij,ij->i', neighbor_r[qu], task)]
        block = np.column_stack(values)
        block = np.sign(block) * np.log1p(np.abs(block))
        blocks.append(block)
    result = np.round(np.column_stack(blocks), 5).astype('float32')
    if not np.isfinite(result).all():
        raise ValueError('Collaborative features must be finite')
    return result
