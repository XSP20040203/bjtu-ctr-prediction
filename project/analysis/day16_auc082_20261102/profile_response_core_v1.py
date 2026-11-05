"""Strict-past response of profile tokens crossed with advertised app/owner/slot."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from scipy.special import logit

CHANNELS = [f'ad_{action}_list_v00{i}' for action in ['click', 'close'] for i in [1, 2, 3]] + ['u_newsCatInterestsST']
TARGETS = ['spread_app_id', 'adv_prim_id', 'slot_id']
METRICS = ['mean_lift', 'max_lift', 'min_lift', 'std_lift', 'mean_confidence',
           'log_max_support', 'unknown_fraction', 'empty']
NAMES = [f'profile__{channel}__{target}__{metric}' for channel in CHANNELS for target in TARGETS for metric in METRICS]
FIELDS = ['pt_d'] + CHANNELS + TARGETS


def tokenize(value):
    return sorted(set(int(token) for token in str(value).split('^') if token and token != '-1'))


def token_matrix(series):
    codes, sequences = pd.factorize(series.astype(str), sort=True)
    bags = [tokenize(value) for value in sequences]
    vocabulary = np.array(sorted({token for bag in bags for token in bag}), dtype='int64')
    lengths = np.array([len(bag) for bag in bags], dtype='int64')
    if len(vocabulary) == 0:
        return csr_matrix((len(series), 0), dtype='float64')
    token_columns = np.searchsorted(vocabulary, np.concatenate([np.asarray(bag, dtype='int64') for bag in bags]))
    sequence_rows = np.repeat(np.arange(len(bags)), lengths)
    unique = coo_matrix((np.ones(len(token_columns)), (sequence_rows, token_columns)),
                        shape=(len(bags), len(vocabulary))).tocsr()
    return unique[codes]


def pair_statistics(history, targets, y, independent):
    if independent:
        exposure = (targets.T @ history).T.tocsr()
        clicks = (targets[y == 1].T @ history[y == 1]).T.tocsr()
    else:
        exposure = (history.T @ targets).tocsr()
        clicks = (history.multiply(y[:, None]).T @ targets).tocsr()
    exposure.sum_duplicates()
    clicks.sum_duplicates()
    return exposure, clicks


def pooled_features(history, target_codes, exposure, clicks, target_n, target_c, global_rate):
    size = len(target_codes)
    result = np.zeros((size, len(METRICS)), dtype='float64')
    counts = np.diff(history.indptr)
    result[:, -1] = counts == 0
    if history.nnz == 0:
        return result
    rows = np.repeat(np.arange(size), counts)
    tokens = history.indices
    target = target_codes[rows]
    support = np.asarray(exposure[tokens, target]).ravel()
    success = np.asarray(clicks[tokens, target]).ravel()
    token_n = np.asarray(exposure.sum(axis=1)).ravel()
    token_c = np.asarray(clicks.sum(axis=1)).ravel()
    token_prior = (token_c[tokens] + 20 * global_rate) / (token_n[tokens] + 20)
    target_prior = (target_c[target] + 50 * global_rate) / (target_n[target] + 50)
    prior = .5 * (token_prior + target_prior)
    probability = (success + 50 * prior) / (support + 50)
    lift = logit(np.clip(probability, 1e-8, 1 - 1e-8)) - logit(np.clip(target_prior, 1e-8, 1 - 1e-8))
    denominator = np.maximum(counts, 1)
    mean = np.bincount(rows, weights=lift, minlength=size) / denominator
    second = np.bincount(rows, weights=lift * lift, minlength=size) / denominator
    maximum = np.full(size, -np.inf)
    minimum = np.full(size, np.inf)
    max_support = np.zeros(size)
    np.maximum.at(maximum, rows, lift)
    np.minimum.at(minimum, rows, lift)
    np.maximum.at(max_support, rows, support)
    maximum[counts == 0] = minimum[counts == 0] = 0
    result[:, 0] = mean
    result[:, 1] = maximum
    result[:, 2] = minimum
    result[:, 3] = np.sqrt(np.maximum(second - mean * mean, 0))
    result[:, 4] = np.bincount(rows, weights=support / (support + 50), minlength=size) / denominator
    result[:, 5] = np.log1p(max_support)
    result[:, 6] = np.bincount(rows, weights=support == 0, minlength=size) / denominator
    return result


def build_day(ads, labels, day, query_positions, independent=False):
    dates = ads.pt_d.to_numpy()
    labels = np.asarray(labels)
    query_positions = np.asarray(query_positions, dtype='int64')
    if labels.shape != dates.shape or np.any(labels[dates >= day] != -1):
        raise ValueError('Current and future labels must be masked')
    if not np.isin(labels[dates < day], [0, 1]).all():
        raise ValueError('Earlier labels must be binary')
    if np.any(dates[query_positions] != day):
        raise ValueError('Query rows must belong to the requested date')
    train = np.flatnonzero(dates < day)
    if not len(train):
        return np.zeros((len(query_positions), len(NAMES)), dtype='float32')
    combined = np.r_[train, query_positions]
    y = labels[train]
    global_rate = (float(y.sum()) + .5) / (len(y) + 50)
    frame = ads.iloc[combined].reset_index(drop=True)
    target_data = {}
    for target in TARGETS:
        codes, vocabulary = pd.factorize(frame[target], sort=True)
        matrix = coo_matrix((np.ones(len(train)), (np.arange(len(train)), codes[:len(train)])),
                            shape=(len(train), len(vocabulary))).tocsr()
        n = np.bincount(codes[:len(train)], minlength=len(vocabulary))
        c = np.bincount(codes[:len(train)], weights=y, minlength=len(vocabulary))
        target_data[target] = matrix, codes[len(train):], n, c
    blocks = []
    for channel in CHANNELS:
        matrix = token_matrix(frame[channel])
        history, query = matrix[:len(train)], matrix[len(train):]
        for target in TARGETS:
            targets, codes, n, c = target_data[target]
            exposure, clicks = pair_statistics(history, targets, y, independent)
            blocks.append(pooled_features(query, codes, exposure, clicks, n, c, global_rate))
    result = np.round(np.column_stack(blocks), 5).astype('float32')
    if not np.isfinite(result).all():
        raise ValueError('Profile response features must be finite')
    return result
