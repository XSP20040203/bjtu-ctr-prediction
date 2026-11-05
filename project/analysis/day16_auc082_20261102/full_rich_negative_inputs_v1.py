"""Join the two complete historical populations without sampling or label access."""
import numpy as np
import pandas as pd


def validate_alignment(left, right):
    for name in ('rows', 'train', 'held', 'labels'):
        if not np.array_equal(left[name], right[name]):
            raise ValueError('Population alignment differs: ' + name)
    rows = np.asarray(left['rows'])
    train, held = (np.asarray(left[name]) for name in ('train', 'held'))
    if rows.ndim != 1 or len(rows) == 0 or np.any(np.diff(rows) <= 0):
        raise ValueError('Global rows must be unique and increasing')
    if any(p.ndim != 1 or np.any(p < 0) or np.any(p >= len(rows)) for p in (train, held)):
        raise ValueError('Invalid partition positions')
    if not np.array_equal(np.sort(np.r_[train, held]), np.arange(len(rows))):
        raise ValueError('Train and held must partition the entire population')
    if len(left['labels']) != len(train) or not np.isin(left['labels'], [0, 1]).all():
        raise ValueError('Only training labels are permitted')


def compose(left, right, positions, names, categories, independent=False):
    positions = np.asarray(positions)
    if left.ndim != 2 or right.ndim != 2 or len(left) != len(right):
        raise ValueError('Component matrices must share the complete row axis')
    width = left.shape[1] + right.shape[1]
    if len(names) != width or len(set(names)) != width or not set(categories) <= set(names):
        raise ValueError('Feature schema must be unique and complete')
    if positions.ndim != 1 or positions.dtype.kind not in 'iu' or np.any(positions < 0) or np.any(positions >= len(left)):
        raise ValueError('Invalid query positions')
    if independent:
        # Column gathering is separate from the row-block primary assembly.
        columns = {}
        for index, name in enumerate(names):
            matrix, column = (left, index) if index < left.shape[1] else (right, index - left.shape[1])
            values = np.asarray(matrix[:, column][positions], dtype='float32')
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite component feature')
            if name in categories:
                integer = values.astype('int64')
                if not np.array_equal(values, integer):
                    raise ValueError('Fractional category code: ' + name)
                values = integer
            columns[name] = values
        return pd.DataFrame(columns, copy=False)
    values = np.empty((len(positions), width), dtype='float32')
    for start in range(0, len(positions), 10000):
        stop = min(start + 10000, len(positions))
        query = positions[start:stop]
        values[start:stop, :left.shape[1]] = left[query]
        values[start:stop, left.shape[1]:] = right[query]
        if not np.isfinite(values[start:stop]).all():
            raise ValueError('Nonfinite component feature')
    frame = pd.DataFrame(values, columns=names, copy=False)
    for name in categories:
        values_ = frame[name].to_numpy()
        integer = values_.astype('int64')
        if not np.array_equal(values_, integer):
            raise ValueError('Fractional category code: ' + name)
        frame[name] = integer
    return frame
