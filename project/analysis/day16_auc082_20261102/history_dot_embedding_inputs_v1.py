"""Keep verified395 context features and six scalar history/candidate affinities."""
import numpy as np
import pandas as pd
from conditional_daily import read
import frozen_neural_embedding_inputs_v2 as source


def selected_columns(columns):
    added = [name for name in columns if name.startswith('neural_history_dot__')]
    kept = list(columns[:395]) + added
    omitted = [name for name in columns if name not in kept]
    if len(columns) != 533 or len(set(columns)) != 533 or len(added) != 6 or len(omitted) != 132:
        raise ValueError('Verified533-column source and exactly six history affinity scalars required')
    if any(name.startswith('neural_') for name in columns[:395]) or len(kept) != 401:
        raise ValueError('The context block must remain unchanged')
    return kept, omitted


def frame(directory, partition, selected=None):
    day = int(directory.name[-2:])
    independent = directory.name.startswith('independent')
    contract = read(directory / 'contract.json')
    columns, _ = selected_columns(contract['schema']['columns'])
    categories = contract['schema']['categories']
    context_directory = source.context_folder(day, independent)
    np.testing.assert_array_equal(np.load(directory / 'rows.npy'), np.load(context_directory / 'rows.npy'))
    positions = np.load(directory / ('train.npy' if partition == 'train' else 'held.npy'))
    if selected is not None:
        positions = positions[selected]
    context = np.load(context_directory / 'features.npy', mmap_mode='r')
    embeddings = np.load(directory / 'features.npy', mmap_mode='r')
    embedding_positions = [contract['embedding_columns'].index(name) for name in columns[395:]]
    assert columns[:395] == contract['context_contract']['schema']['columns']
    assert len(embedding_positions) == 6
    if independent:
        data = {}
        for i, name in enumerate(columns):
            matrix, column = (context, i) if i < 395 else (embeddings, embedding_positions[i - 395])
            values = np.asarray(matrix[:, column][positions], dtype='float32')
            if name in categories:
                integer = values.astype('int64')
                np.testing.assert_array_equal(values, integer)
                values = integer
            assert np.isfinite(values).all()
            data[name] = values
        return pd.DataFrame(data, copy=False), categories
    values = np.empty((len(positions), 401), dtype='float32')
    for start in range(0, len(positions), 10000):
        stop = min(start + 10000, len(positions))
        rows = positions[start:stop]
        values[start:stop, :395] = context[rows]
        values[start:stop, 395:] = embeddings[np.ix_(rows, embedding_positions)]
        assert np.isfinite(values[start:stop]).all()
    result = pd.DataFrame(values, columns=columns, copy=False)
    for name in categories:
        numeric = result[name].to_numpy()
        integer = numeric.astype('int64')
        np.testing.assert_array_equal(numeric, integer)
        result[name] = integer
    return result, categories
