"""Stream both complete cohorts through independent frozen neural encodings."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import gc
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from fit_stacker import HERE, ROOT, digest, write
from conditional_daily import read, emit
from rich_userday_activity import array_hash
from frozen_neural_embedding_core_v1 import CATEGORIES, SEQUENCES, FIELDS, NAMES, encode, design

OUT = HERE / 'runs/full_negative_neural_embeddings_v1'
CONTEXT = HERE / 'runs/full_negative_context_v1'
INDEPENDENT = HERE / 'runs/full_negative_context_v2'
TEACHER = ROOT / 'analysis/userday_aux_20261101'
sys.path.insert(0, str(ROOT / 'analysis/architecture_goal_20261029'))
from interaction_models import CappedInteractionEncoder


def folder(day, independent=False):
    return OUT / f'{"independent" if independent else "input"}_day{day}'


def context_folder(day, independent=False):
    return (INDEPENDENT if independent else CONTEXT) / f'{"independent" if independent else "input"}_day{day}'


def teacher_paths(day):
    root = TEACHER / f'runs/protected_day_aux_v1_day{day}_seed42'
    cache = ROOT / ('analysis/user_personalization/cache_day15_stats' if day == 15
                    else 'analysis/auc085/cache_rich_stats_time16')
    return root, cache


def export_tables(day):
    assert not torch.cuda.is_initialized(), 'CPU preparation must leave the GPU owner alone'
    torch.set_num_threads(2)
    root, cache = teacher_paths(day)
    feature = read(root / 'feature_manifest.json')
    metrics = read(root / 'both/metrics.json')
    assert read(TEACHER / 'verification.json')['status'] == 'passed'
    assert metrics['status'] == 'complete_and_verified'
    for filename, expected in feature['metadata_sha256'].items():
        assert digest(filename) == expected, filename
    checkpoint_path = root / 'both/final_model.pt'
    assert digest(checkpoint_path) == metrics['files_sha256']['final_model.pt']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    meta = checkpoint['meta']
    assert meta['validation_day'] == 20191100 + day
    assert meta['target_stat_validation_labels_used'] is False
    assert feature['train_rows'] == read(context_folder(day) / 'contract.json')['training_rows']
    assert feature['valid_rows'] == read(context_folder(day) / 'contract.json')['held_rows']
    saved_meta = read(cache / 'manifest.json')
    for key in ('group_for_field', 'vocabulary_sizes', 'categorical_fields', 'sequence_fields'):
        assert meta[key] == saved_meta[key]
    encoder = CappedInteractionEncoder(meta['categorical_fields'], meta['sequence_fields'],
        meta['group_for_field'], meta['vocabulary_sizes']).cpu().eval()
    prefix = 'backbone.encoder.'
    encoder.load_state_dict({key[len(prefix):]: value for key, value in checkpoint['state_dict'].items()
                             if key.startswith(prefix)}, strict=True)
    tables, maximum = {}, 0.
    with torch.no_grad():
        for group in sorted({meta['group_for_field'][name] for name in CATEGORIES + SEQUENCES}):
            name = next(name for name in CATEGORIES if meta['group_for_field'][name] == group)
            ids = torch.arange(meta['vocabulary_sizes'][group], dtype=torch.int64)
            native, actual_ids = encoder.read(name, ids)
            np.testing.assert_array_equal(actual_ids.numpy(), ids.numpy())
            raw = checkpoint['state_dict'][prefix + 'embeddings.' + group + '.weight'].numpy().astype('float64')
            if group in encoder.group_norm_caps:
                norm = np.maximum(np.sqrt((raw * raw).sum(1)), 1e-12)
                raw *= np.minimum(encoder.group_norm_caps[group] / norm, 1.)[:, None]
            raw[0] = 0
            native = native.numpy().copy()
            error = float(np.max(np.abs(native - raw)))
            maximum = max(maximum, error)
            np.testing.assert_allclose(native, raw, rtol=3e-7, atol=2e-8)
            # Canonical capped tables are frozen model constants shared by the
            # two independent raw-ID/sequence and aggregation implementations.
            tables[group] = native
    np.savez(OUT / f'neural_tables_day{day}.npz', **tables)
    vocabularies = read(cache / 'vocabs.json')
    document = dict(day=20191100 + day, meta=meta,
        checkpoint=str(checkpoint_path), checkpoint_sha256=digest(checkpoint_path),
        checkpoint_endpoint='fixed_final_epoch1', checkpoint_selected_by_new_held_labels=False,
        encoder_supervised_fit_scope='All original outer-training dates strictly earlier than prediction date',
        training_row_embeddings_are_outer_training_fitted=True,
        per_historical_date_out_of_time_embeddings=False,
        held_day_labels_fit=False, inferred17_labels_read=False,
        tables_sha256={key: array_hash(value) for key, value in tables.items()},
        maximum_independent_norm_cap_error=maximum,
        every_vocabulary_id_checked_against_native_encoder=True,
        gpu_memory_allocated=False)
    write(OUT / f'neural_teacher_day{day}.json', document)
    del checkpoint, encoder
    gc.collect()
    return meta, vocabularies, tables, document


def build(day):
    meta, vocabularies, tables, teacher = export_tables(day)
    context = read(context_folder(day) / 'contract.json')
    assert context == read(context_folder(day, True) / 'contract.json')
    rows = np.load(context_folder(day) / 'rows.npy')
    outputs = []
    copied_files = ('rows.npy', 'train.npy', 'held.npy', 'labels.npy', 'dates.npy')
    for independent in (False, True):
        destination = folder(day, independent)
        destination.mkdir()
        source = context_folder(day, independent)
        assert digest(source / 'features.npy') == context['matrix_sha256']
        for filename in copied_files:
            assert digest(source / filename) == context['files_sha256'][filename]
            shutil.copy2(source / filename, destination / filename)
        outputs.append(np.lib.format.open_memmap(destination / 'features.npy', mode='w+',
            dtype='float32', shape=(len(rows), len(NAMES))))
    offset, written, last_emit = 0, 0, 0
    dtype = {name: 'str' if name in SEQUENCES else 'int32' for name in FIELDS}
    for chunk in pd.read_csv(ROOT / '2026ml/train.csv', usecols=FIELDS, dtype=dtype, chunksize=30000):
        keep = chunk.pt_d.to_numpy() <= 20191100 + day
        indices = offset + np.flatnonzero(keep)
        offset += len(chunk)
        if not len(indices):
            continue
        selected = chunk.loc[keep, FIELDS].reset_index(drop=True)
        stop = written + len(selected)
        np.testing.assert_array_equal(rows[written:stop], indices)
        primary = encode(selected, meta, vocabularies, 20191100 + day)
        independent = encode(selected, meta, vocabularies, 20191100 + day, True)
        for name in primary:
            np.testing.assert_array_equal(primary[name], independent[name])
        left = design(primary, tables, meta)
        right = design(independent, tables, meta, True)
        np.testing.assert_array_equal(left, right)
        outputs[0][written:stop], outputs[1][written:stop] = left, right
        written = stop
        if written - last_emit >= 600000 or written == len(rows):
            emit('frozen_neural_embeddings_rebuilt', day=day, rows=written, total=len(rows),
                independent_numpy_torch_exact=True)
            last_emit = written
    assert written == len(rows)
    for matrix in outputs:
        matrix.flush()
    del outputs
    gc.collect()
    assert digest(folder(day) / 'features.npy') == digest(folder(day, True) / 'features.npy')
    columns = context['schema']['columns'] + NAMES
    assert len(columns) == len(set(columns)) == 533
    contract = dict(shape=[len(rows), 138], fused_shape=[len(rows), 533], embedding_columns=NAMES,
        schema=dict(columns=columns, categories=context['schema']['categories']),
        stored_matrix='features.npy stores138 embeddings; the unchanged395-column context is fused in RAM',
        matrix_sha256=digest(folder(day) / 'features.npy'), context_matrix_sha256=context['matrix_sha256'],
        context_contract=context, teacher=teacher,
        rows_sha256=context['rows_sha256'], training_rows_sha256=context['training_rows_sha256'],
        training_labels_sha256=context['training_labels_sha256'], held_rows_sha256=context['held_rows_sha256'],
        training_rows=context['training_rows'], held_rows=context['held_rows'],
        all_historical_negatives_retained=True, own_held_labels_masked=True,
        max_training_date=context['max_training_date'], inferred17_labels_read=False,
        independent_raw_encoding_and_numpy_torch_features_exact=True,
        files_sha256={name: digest(folder(day) / name) for name in copied_files})
    for independent in (False, True):
        write(folder(day, independent) / 'contract.json', contract)
    return contract


def frame(directory, partition, selected=None):
    day = int(directory.name[-2:])
    independent = directory.name.startswith('independent')
    contract = read(directory / 'contract.json')
    source = context_folder(day, independent)
    assert digest(source / 'features.npy') == contract['context_matrix_sha256']
    np.testing.assert_array_equal(np.load(directory / 'rows.npy'), np.load(source / 'rows.npy'))
    positions = np.load(directory / ('train.npy' if partition == 'train' else 'held.npy'))
    if selected is not None:
        positions = positions[selected]
    context = np.load(source / 'features.npy', mmap_mode='r')
    embedding = np.load(directory / 'features.npy', mmap_mode='r')
    values = np.empty((len(positions), 533), dtype='float32')
    for start in range(0, len(positions), 10000):
        stop = min(start + 10000, len(positions))
        chosen = positions[start:stop]
        values[start:stop, :395], values[start:stop, 395:] = context[chosen], embedding[chosen]
    result = pd.DataFrame(values, columns=contract['schema']['columns'], copy=False)
    for name in contract['schema']['categories']:
        numeric = result[name].to_numpy()
        np.testing.assert_array_equal(numeric, numeric.astype('int64'))
        result[name] = numeric.astype('int64')
    assert np.isfinite(values).all()
    return result, contract['schema']['categories']
