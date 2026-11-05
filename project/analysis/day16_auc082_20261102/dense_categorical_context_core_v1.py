"""Train-fitted categorical dictionaries for verified complete context matrices."""
import numpy as np
import pandas as pd
from conditional_daily import read


def fit_dictionaries(directory):
    contract = read(directory / 'contract.json')
    train = np.load(directory / 'train.npy')
    dates = np.load(directory / 'dates.npy')
    cutoff = 20191100 + int(directory.name[-2:])
    assert np.all(dates[train] < cutoff)
    matrix = np.load(directory / 'features.npy', mmap_mode='r')
    names = contract['schema']['columns']
    tables = {}
    for name in contract['schema']['categories']:
        values = np.asarray(matrix[train, names.index(name)], dtype='float32')
        integer = values.astype('int64')
        np.testing.assert_array_equal(values, integer)
        tables[name] = np.unique(integer[integer >= 0])
    return tables


def encode(values, keys, independent=False):
    values = np.asarray(values)
    integer = values.astype('int64')
    np.testing.assert_array_equal(values, integer)
    keys = np.asarray(keys, dtype='int64')
    assert keys.ndim == 1 and np.all(keys >= 0) and np.all(keys[1:] > keys[:-1])
    if independent:
        mapping = {int(value): index for index, value in enumerate(keys)}
        result = pd.Series(integer).map(mapping).fillna(len(keys)).to_numpy(dtype='int32')
    else:
        positions = np.searchsorted(keys, integer)
        known = positions < len(keys)
        known[known] &= keys[positions[known]] == integer[known]
        result = np.full(len(integer), len(keys), dtype='int32')
        result[known] = positions[known]
    result[integer < 0] = -1
    assert np.all((result >= -1) & (result <= len(keys)))
    return result


def frame(directory, partition, tables, selected=None, independent=False, recode=True):
    contract = read(directory / 'contract.json')
    positions = np.load(directory / ('train.npy' if partition == 'train' else 'held.npy'))
    if selected is not None:
        positions = positions[np.asarray(selected)]
    matrix = np.load(directory / 'features.npy', mmap_mode='r')
    names, categories = contract['schema']['columns'], contract['schema']['categories']
    if independent:
        data = {name: np.asarray(matrix[positions, index], dtype='float32') for index, name in enumerate(names)}
        result = pd.DataFrame(data, copy=False)
    else:
        values = np.array(matrix[positions], dtype='float32', copy=True)
        result = pd.DataFrame(values, columns=names, copy=False)
    for name in categories:
        values = result[name].to_numpy()
        if recode:
            result[name] = encode(values, tables[name], independent)
        else:
            result[name] = values.astype('int32')
            np.testing.assert_array_equal(values, result[name])
    assert len(names) == 395 and np.isfinite(result.to_numpy()).all()
    return result, categories
