"""Deterministic feature selection and complete source-sample row extraction."""
import numpy as np
import pandas as pd


def select_indices(names, categories, importance, budget=384):
    importance = np.asarray(importance, dtype='float64')
    if importance.shape != (len(names),) or not np.isfinite(importance).all():
        raise ValueError('One finite importance per named feature is required')
    if len(set(names)) != len(names) or not set(categories).issubset(names):
        raise ValueError('Feature names must be unique and categories known')
    cat = [i for i, name in enumerate(names) if name in categories]
    numeric = [i for i, name in enumerate(names) if name not in categories]
    if budget < 1 or budget > len(numeric):
        raise ValueError('Numeric feature budget is invalid')
    ranked = sorted(numeric, key=lambda i: (-importance[i], i))[:budget]
    return np.asarray(sorted(cat + ranked), dtype='int64')


def compact_frame(matrix, positions, native_keep, quality, auxiliary,
                  selected, names, categories, independent=False):
    positions = np.asarray(positions, dtype='int64')
    selected = np.asarray(selected, dtype='int64')
    native_keep = np.asarray(native_keep, dtype='int64')
    quality = np.asarray(quality)
    auxiliary = np.asarray(auxiliary)
    if len(quality) != len(positions) or len(auxiliary) != len(positions):
        raise ValueError('All row features must be aligned')
    if len(names) != len(native_keep) + quality.shape[1] + auxiliary.shape[1]:
        raise ValueError('Schema width differs from input blocks')
    if np.any(positions < 0) or np.any(positions >= len(matrix)):
        raise ValueError('Raw matrix positions are out of range')
    if len(np.unique(selected)) != len(selected) or np.any(selected < 0) or np.any(selected >= len(names)):
        raise ValueError('Selected column indices must be valid and unique')
    output = np.empty((len(positions), len(selected)), dtype='float32')
    n_native = len(native_keep)
    n_quality = quality.shape[1]
    if independent:
        # Independent column-wise gather checks the primary block indexing.
        for j, column in enumerate(selected):
            if column < n_native:
                output[:, j] = matrix[positions, native_keep[column]]
            elif column < n_native + n_quality:
                output[:, j] = quality[:, column - n_native]
            else:
                output[:, j] = auxiliary[:, column - n_native - n_quality]
    else:
        native = selected < n_native
        news = (selected >= n_native) & (selected < n_native + n_quality)
        audience = selected >= n_native + n_quality
        for start in range(0, len(positions), 10000):
            stop = min(len(positions), start + 10000)
            output[start:stop, native] = matrix[np.ix_(positions[start:stop], native_keep[selected[native]])]
            output[start:stop, news] = quality[start:stop, selected[news] - n_native]
            output[start:stop, audience] = auxiliary[start:stop, selected[audience] - n_native - n_quality]
    if not np.isfinite(output).all():
        raise ValueError('Compact features must all be finite')
    columns = [names[i] for i in selected]
    frame = pd.DataFrame(output, columns=columns)
    cats = [name for name in columns if name in categories]
    for name in cats:
        values = frame[name].to_numpy()
        if not np.array_equal(values, values.astype('int64')):
            raise ValueError('Categorical codes must be integers')
        frame[name] = values.astype('int64')
    return frame, cats
