"""Outcome-free encodings from fixed, outer-training-only neural ID tables."""
import numpy as np
import pandas as pd

CATEGORIES = ['user_id', 'task_id', 'adv_id', 'adv_prim_id', 'spread_app_id', 'slot_id']
SEQUENCES = [f'ad_{action}_list_v00{i}' for action in ('click', 'close') for i in (1, 2, 3)]
FIELDS = ['pt_d'] + CATEGORIES + SEQUENCES
CANDIDATES = ['task_id', 'adv_prim_id', 'spread_app_id', 'slot_id']
HISTORY_TARGETS = ['task_id', 'adv_prim_id', 'spread_app_id'] * 2
DIM = 8
NAMES = ([f'neural_embedding__{name}__{i}' for name in CATEGORIES + SEQUENCES for i in range(DIM)] +
    [f'neural_user_product__{name}__{i}' for name in CANDIDATES for i in range(DIM)] +
    [f'neural_history_dot__{name}' for name in SEQUENCES] +
    [f'neural_user_dot__{name}' for name in CANDIDATES])
assert len(NAMES) == len(set(NAMES)) == 138


def normalized_key(value):
    text = str(value).strip()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def encode(frame, meta, vocabularies, cutoff, independent=False):
    if list(frame) != FIELDS:
        raise ValueError('Only declared unlabeled date, ID and history fields are allowed')
    if np.any(frame.pt_d.to_numpy() > cutoff):
        raise ValueError('Future metadata is not permitted')
    result = {}
    for name in CATEGORIES:
        vocabulary = vocabularies[meta['group_for_field'][name]]
        lookup = {int(raw): int(code) for raw, code in vocabulary.items()}
        values = frame[name].to_numpy()
        if independent:
            keys = np.array(sorted(lookup), dtype='int64')
            position = np.searchsorted(keys, values)
            known = position < len(keys)
            known[known] &= keys[position[known]] == values[known]
            encoded = np.ones(len(values), dtype='int64')
            codes = np.array([lookup[int(key)] for key in keys], dtype='int64')
            encoded[known] = codes[position[known]]
        else:
            encoded = frame[name].map(lookup).fillna(1).to_numpy(dtype='int64')
        result[name] = encoded
    for name in SEQUENCES:
        vocabulary = vocabularies[meta['group_for_field'][name]]
        texts = frame[name].fillna('').astype(str)
        if independent:
            # Original source encoder checks the separate local implementation.
            from benchmark_core import encode_sequence
            cache = {text: encode_sequence(text, vocabulary, 5) for text in texts.unique()}
        else:
            cache = {}
            for text in texts.unique():
                values = np.zeros(5, dtype='int64')
                if text.strip() not in ('', 'nan', 'None'):
                    for i, token in enumerate(text.split('^')[:5]):
                        values[i] = vocabulary.get(normalized_key(token), 1)
                cache[text] = values
        result[name] = np.stack([cache[text] for text in texts]).astype('int64', copy=False)
    for name, values in result.items():
        size = meta['vocabulary_sizes'][meta['group_for_field'][name]]
        if np.any(values < 0) or np.any(values >= size):
            raise ValueError('Encoded ID outside the frozen vocabulary')
    return result


def design(encoded, tables, meta, independent=False):
    if independent:
        import torch
        values = {}
        for name in CATEGORIES:
            table = torch.from_numpy(tables[meta['group_for_field'][name]])
            values[name] = table[torch.from_numpy(encoded[name])].to(torch.float64)
        for name in SEQUENCES:
            table = torch.from_numpy(tables[meta['group_for_field'][name]])
            ids = torch.from_numpy(encoded[name])
            selected = table[ids].to(torch.float64) * ids.ne(0).unsqueeze(-1)
            values[name] = selected.sum(1) / ids.ne(0).sum(1).clamp_min(1).unsqueeze(-1)
        blocks = [values[name] for name in CATEGORIES + SEQUENCES]
        blocks += [values['user_id'] * values[name] for name in CANDIDATES]
        blocks += [(values[name] * values[target]).sum(-1, keepdim=True)
                   for name, target in zip(SEQUENCES, HISTORY_TARGETS)]
        blocks += [(values['user_id'] * values[name]).sum(-1, keepdim=True) for name in CANDIDATES]
        result = torch.cat(blocks, dim=1).to(torch.float32).numpy()
    else:
        values = {name: tables[meta['group_for_field'][name]][encoded[name]].astype('float64')
                  for name in CATEGORIES}
        for name in SEQUENCES:
            ids = encoded[name]
            selected = tables[meta['group_for_field'][name]][ids].astype('float64')
            selected *= (ids != 0)[..., None]
            values[name] = selected.sum(1) / np.maximum((ids != 0).sum(1), 1)[:, None]
        blocks = [values[name] for name in CATEGORIES + SEQUENCES]
        blocks += [values['user_id'] * values[name] for name in CANDIDATES]
        blocks += [(values[name] * values[target]).sum(-1, keepdims=True)
                   for name, target in zip(SEQUENCES, HISTORY_TARGETS)]
        blocks += [(values['user_id'] * values[name]).sum(-1, keepdims=True) for name in CANDIDATES]
        result = np.column_stack(blocks).astype('float32')
    if result.shape != (len(encoded['user_id']), len(NAMES)) or not np.isfinite(result).all():
        raise ValueError('Finite aligned embedding features required')
    return result
