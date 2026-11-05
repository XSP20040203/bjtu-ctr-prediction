"""Real1088-row raw/dense LightGBM comparison, without any final17 label access."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import gc
from pathlib import Path
import numpy as np
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash, lgb
from dense_categorical_context_core_v1 import encode, fit_dictionaries, frame

OUT = HERE / 'runs/dense_context_category_probe_v1'
INPUT = HERE / 'runs/full_negative_context_v1/input_day15'
PARAMS = dict(objective='binary', learning_rate=.04, num_leaves=15, min_data_in_leaf=16,
    scale_pos_weight=5., lambda_l2=40., max_bin=63, feature_fraction=.8,
    bagging_fraction=.8, bagging_freq=1, cat_l2=40., cat_smooth=40.,
    max_cat_threshold=64, min_data_per_group=10, num_threads=2, seed=151741,
    verbosity=-1, deterministic=True, force_col_wise=True, feature_pre_filter=False,
    metric='None', device_type='cpu')


def main():
    assert not OUT.exists(), 'Preserve previous probes and failures'
    contract = read(INPUT / 'contract.json')
    assert contract['max_training_date'] == 20191114 and contract['own_held_labels_masked']
    assert digest(INPUT / 'features.npy') == contract['matrix_sha256']
    labels = np.load(INPUT / 'labels.npy')
    selected = np.sort(np.r_[np.flatnonzero(labels == 1)[-64:],
        np.flatnonzero(labels == 0)[np.linspace(0, int((labels == 0).sum())-1, 1024, dtype='int64')]])
    tables = fit_dictionaries(INPUT)
    fixture_keys = np.array([0, 100, 1524772], dtype='int64')
    fixture = np.array([-7, -1, 0, 100, 100, 200, 1524772, 2000000], dtype='int64')
    expected = np.array([-1, -1, 0, 1, 1, 3, 2, 3], dtype='int32')
    for independent in (False, True):
        np.testing.assert_array_equal(encode(fixture, fixture_keys, independent), expected)
    OUT.mkdir()
    np.savez(OUT / 'category_dictionaries_day15.npz', **tables)
    write(OUT / 'protocol.json', dict(params=PARAMS, rows=len(selected), trees=20,
        fit_max_training_date=20191114, held_labels_used=False, inferred17_labels_read=False,
        dictionaries_fit_on_all_strictly_earlier_feature_rows=True,
        negative_values_preserved_as_missing=True, unknown_nonnegative_code='number_of_fitted_categories',
        raw_and_dense_parameters_identical=True,
        sources_sha256={str(path): digest(path) for path in (Path(__file__),
            HERE / 'dense_categorical_context_core_v1.py', INPUT / 'contract.json', INPUT / 'features.npy')},
        raw_datasets_sha256=read(HERE / 'runs/full_negative_context_v1/protocol.json')['raw_datasets_sha256']))
    raw, cats = frame(INPUT, 'train', tables, selected, recode=False)
    dense, other = frame(INPUT, 'train', tables, selected)
    replay, replay_cats = frame(INPUT, 'train', tables, selected, independent=True)
    assert cats == other == replay_cats
    np.testing.assert_array_equal(dense.to_numpy(), replay.to_numpy())
    query = np.arange(1500)
    raw_query, _ = frame(INPUT, 'held', tables, query, recode=False)
    dense_query, _ = frame(INPUT, 'held', tables, query)
    results, predictions = {}, {}
    for name, data, held in (('raw', raw, raw_query), ('dense', dense, dense_query)):
        dataset = lgb.Dataset(data, label=labels[selected], categorical_feature=cats, params=PARAMS)
        model = lgb.train(PARAMS, dataset, num_boost_round=20, keep_training_booster=True)
        path = OUT / (name + '.txt')
        relative = path.relative_to(ROOT).as_posix()
        relative.encode('ascii')
        model.save_model(relative)
        restored = lgb.Booster(model_file=relative)
        before = model.predict(held, raw_score=True, num_threads=2)
        after = restored.predict(held, raw_score=True, num_threads=2)
        np.testing.assert_array_equal(before, after)
        assert np.isfinite(before).all()
        predictions[name] = dict(train=model.predict(data, raw_score=True, num_threads=2), held=before)
        results[name] = dict(bytes=path.stat().st_size, model_sha256=digest(path),
            native_file_save_restore_prediction_error=0., train_rows=len(data), held_label_free_probe_rows=len(held))
        del model, restored, dataset
        gc.collect()
    maximum_train = float(np.max(np.abs(predictions['raw']['train']-predictions['dense']['train'])))
    maximum_held = float(np.max(np.abs(predictions['raw']['held']-predictions['dense']['held'])))
    report = dict(status='passed', models=results, fixture_missing_known_and_positive_unknown_codes_exact=True,
        independent_all395_real_sample_columns_exact=True,
        training_prediction_raw_vs_dense_maximum_error=maximum_train,
        held_prediction_raw_vs_dense_maximum_error=maximum_held,
        raw_vs_dense_predictions_identical=maximum_train == maximum_held == 0.,
        dense_to_raw_file_size_ratio=results['dense']['bytes']/results['raw']['bytes'],
        category_dictionary_sha256=digest(OUT / 'category_dictionaries_day15.npz'),
        all_dictionary_values_sha256={name: array_hash(value) for name, value in tables.items()},
        no_held_labels_used=True, inferred17_labels_read=False,
        production_capacity_proven=False, protocol_sha256=digest(OUT / 'protocol.json'))
    write(OUT / 'verification.json', report)
    emit('dense_context_category_probe_verified', **report)


if __name__ == '__main__':
    main()
