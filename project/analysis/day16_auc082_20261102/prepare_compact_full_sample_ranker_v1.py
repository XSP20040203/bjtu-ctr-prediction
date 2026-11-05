"""Generate a preserved full-source-sample compact row ranking experiment."""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


METADATA = '''def metadata(day):
    cutoff = 20191100 + day
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label', 'user_id', 'task_id'], dtype='int32')
    dates = raw.pt_d.to_numpy()
    labels = raw.label.to_numpy(copy=True)
    labels[dates >= cutoff] = -1
    chosen, weights, held_rows, train, held, y = selection(day, labels, dates)
    source = read(SOURCE/f'input_contract{day}.json')
    assert array_hash(chosen) == source['training_rows_sha256']
    assert array_hash(y) == source['training_labels_sha256']
    assert array_hash(weights) == source['weights_sha256']
    rows = np.sort(np.r_[chosen, held_rows])
    order, groups, pairs = design(dates[chosen], y, cutoff, seed=151653, draws=64)
    positions = train[order]
    target = labels[rows[positions]]
    np.testing.assert_array_equal(np.sort(rows[positions]), chosen)
    np.testing.assert_array_equal(held_rows, np.load(BASES[day]/f'day{day}_indices.npy'))
    assert np.all(dates[rows[positions]] < cutoff)
    held_users = raw.user_id.to_numpy()[held_rows]
    row_groups, _ = pd.factorize(held_users, sort=True)
    return dict(raw=raw, rows=rows, positions=positions, held=held, held_rows=held_rows,
        target=target, groups=groups, pairs=pairs, row_groups=row_groups,
        source=source, max_training_date=int(dates[rows[positions]].max()))
'''


INPUTS = '''def inputs(day, partition, independent=False):
    info = metadata(day)
    source = info['source']
    positions = info['positions'] if partition == 'train' else info['held']
    query = info['raw'].iloc[info['rows'][positions]][['pt_d', 'user_id']].to_numpy()
    task_query = info['raw'].iloc[info['rows'][positions]][['pt_d', 'task_id']].to_numpy()
    cached = np.load(QUALITY/f'quality{day}.npz')
    keys, values = cached['keys'], cached['values']
    native = read(QUALITY/f'input_contract{day}.json')
    assert array_hash(values) == native['quality_sha256']
    assert array_hash(keys) == native['native_contract']['native_contract']['keys_sha256']
    if independent:
        news = pd.read_csv(ROOT/'2026ml/train_feeds.csv', usecols=READ,
            dtype={'i_docId': str, 'i_s_sourceId': str})
        rebuilt, quality_names = build_quality(news, keys, 20191100+day)
        np.testing.assert_array_equal(rebuilt, values)
        assert quality_names == native['quality_names']
        values = rebuilt
        del news
        gc.collect()
    quality = quality_rows(keys, values, query)
    original_selector = read(SELECTOR/f'input_contract{day}.json')
    if independent:
        auxiliary, meta, contract = audience.inputs(day, True)
        assert contract == read(AUDIENCE/f'input_day{day}/contract.json')
    else:
        auxiliary, meta, contract = audience.load_inputs(AUDIENCE/f'input_day{day}')
    audience_columns = original_selector['audience_columns']
    extra = aligned(meta['keys'], auxiliary[audience_columns].to_numpy(dtype='float32'),
                    task_query, 20191100+day)
    native_keep = [i for i, name in enumerate(source['feature_names'])
                   if not name.startswith('rich_daily__quality__')]
    assert len(native_keep) == 680
    names = [f'f{i}' for i in range(1017)] + ['task_audience__'+name for name in audience_columns]
    categories = [f'f{native_keep.index(i)}' for i in source['categorical_columns']]
    model_path = SELECTOR/f'day{day}_audience_row_d6_trees1600.cbm'
    selector = load_model(model_path)
    assert selector.feature_names_ == names
    importance = selector.get_feature_importance(type='PredictionValuesChange', thread_count=2)
    selected = select_indices(names, categories, importance, 384)
    directory = SOURCE/(f'independent_day{day}' if independent else f'input_day{day}')
    path = directory/'features.npy'
    assert digest(path) == source['matrix_sha256']
    matrix = np.load(path, mmap_mode='r')
    frame, cats = compact_frame(matrix, positions, native_keep, quality, extra,
                                 selected, names, categories, independent=independent)
    # A conventional dense frame independently checks all selected blocks on actual rows.
    probe_size = min(1000, len(positions))
    dense, _, _ = full_frame(matrix, positions[:probe_size], source['feature_names'],
                            source['categorical_columns'], quality[:probe_size])
    dense = pd.concat([dense, pd.DataFrame(extra[:probe_size], columns=names[1017:])], axis=1)
    np.testing.assert_array_equal(frame.iloc[:probe_size].to_numpy(dtype='float32'),
                                  dense.iloc[:, selected].to_numpy(dtype='float32'))
    audit = dict(shape=list(frame.shape), frame_sha256=frame_hash(frame), categories=cats,
        selected_columns=list(frame), selector_model_sha256=digest(model_path),
        selector_importance_sha256=array_hash(np.round(importance, 10)),
        quality_sha256=array_hash(quality), query_keys_sha256=array_hash(query),
        audience_features_sha256=array_hash(extra), audience_columns=audience_columns,
        source_matrix_sha256=digest(path), training_rows_sha256=array_hash(info['rows'][info['positions']]),
        training_targets_sha256=array_hash(info['target']), pairs_sha256=array_hash(info['pairs']),
        groups_sha256=array_hash(info['groups']), max_training_date=info['max_training_date'],
        selector_max_training_date=20191100+day-1, evaluation_day_labels_fit=False,
        all_fixed_source_training_rows_retained=True, dense_reference_probe_rows=probe_size)
    del matrix, quality, cached, values, extra, auxiliary, dense, info['raw']
    gc.collect()
    emit('compact_full_sample_input_ready', day=day, partition=partition, independent=independent,
         shape=audit['shape'], pairs=len(info['pairs']))
    return frame, cats, info, audit
'''


DEPENDENCIES = '''def dependencies():
    paths = [Path(__file__), HERE/'prepare_compact_full_sample_ranker_v1.py',
        HERE/'compact_full_sample_core_v1.py', HERE/'test_compact_full_sample_core_v1.py',
        HERE/'run_compact_full_sample_ranker_v1.ps1', HERE/'cross_user_pair_core.py',
        HERE/'fit_stacker.py', HERE/'conditional_daily.py', HERE/'rich_userday_activity.py',
        HERE/'rich_row_catboost.py', HERE/'full_quality_row_catboost.py', HERE/'news_quality_features.py',
        HERE/'catboost_userday_residual.py', HERE/'verify_stacker.py', HERE/'within_user_ranking_core.py',
        HERE/'task_audience_rate_v1.py', HERE/'task_audience_core.py', HERE/'task_audience_row_core.py',
        HERE/'global_row_context.py', HERE/'global_row_context15.py']
    for folder in [SOURCE, QUALITY, SELECTOR, AUDIENCE, *BASES.values()]:
        assert read(folder/'verification.json')['status'] == 'passed'
        parent = read(folder/'protocol.json')
        for filename, expected in parent['sources_sha256'].items():
            assert digest(filename) == expected, filename
        paths += [folder/'protocol.json', folder/'verification.json']
    for day in [15, 16]:
        paths += [SOURCE/f'input_contract{day}.json', SOURCE/f'input_day{day}/features.npy',
            SOURCE/f'independent_day{day}/features.npy', QUALITY/f'quality{day}.npz',
            QUALITY/f'input_contract{day}.json', BASES[day]/f'day{day}_prediction.npy',
            BASES[day]/f'day{day}_indices.npy', SELECTOR/f'input_contract{day}.json',
            SELECTOR/f'day{day}_audience_row_d6_trees1600.cbm',
            AUDIENCE/f'input_day{day}/frame.pkl', AUDIENCE/f'input_day{day}/metadata.npz',
            AUDIENCE/f'input_day{day}/contract.json', AUDIENCE/f'input_day{day}/files_locked.json']
    return list(dict.fromkeys(paths))
'''


def replace_function(source, name, replacement):
    pattern = rf'^def {name}\([^\n]*\):.*?(?=^def |^if __name__)'
    source, count = re.subn(pattern, lambda _: replacement+'\n\n', source, count=1, flags=re.M | re.S)
    assert count == 1, name
    return source


def main():
    target = HERE/'compact_full_sample_ranker_v1.py'
    assert not target.exists(), 'Preserve prior experiment sources'
    source = (HERE/'task_audience_row_ranker_v1.py').read_text(encoding='utf8')
    changes = [
        ('"""Strict-prior cross-user row ranking with all native and337 news-quality features."""',
         '"""Compact historical row ranking retaining the complete fixed source sample."""'),
        ("OUT = HERE / 'runs/task_audience_row_ranker_v1'", "OUT = HERE / 'runs/compact_full_sample_ranker_v1'"),
        ("PREFLIGHT = HERE / 'runs/task_audience_row_preflight_v1'", "PREFLIGHT = HERE / 'runs/compact_full_sample_preflight_v1'"),
        ("BASES={day:AUDIENCE for day in [15,16]}",
         "SELECTOR=HERE/'runs/task_audience_row_ranker_v1'\nBASES={day:HERE/'runs/profile_response_ranker_v1' for day in [15,16]}"),
        ("name='audience_row_d6'", "name='full_sample_d6'"),
        ("name='audience_row_d8'", "name='full_sample_d8'"),
        ('CHECKPOINTS = [200, 600, 1600]', 'CHECKPOINTS = [600, 1600, 2400]'),
        ('WEIGHTS = [.025, .05, .1, .2, .35, .5, .75]', 'WEIGHTS = [.05, .1, .2, .35, .5, .75, 1.]'),
        ("learning_rate=.025, l2_leaf_reg=100.", "learning_rate=.035, l2_leaf_reg=40."),
        ('border_count=32', 'border_count=64'),
        ('random_seed=151648', 'random_seed=151653'),
        ('gpu_ram_part=.5', 'gpu_ram_part=.45'),
        ('from task_audience_row_core import select_columns,aligned',
         'from task_audience_row_core import aligned\nfrom compact_full_sample_core_v1 import select_indices, compact_frame'),
        ('All earlier positive rows and one quarter of fixed sampled negatives; random1024-row date queries;48 negative pairs per positive',
         'All earlier positives and all fixed one-third source negatives; no second downsample;1024-row date queries;64 pairs per positive'),
        ('1017 row features plus192 task audience features selected only by a fixed prior-trained600-tree depth6 task model',
         'All native categorical features plus384 numeric features selected by a fixed strictly historical model'),
        ('task_audience_row_', 'compact_full_sample_'),
    ]
    for old, new in changes:
        assert old in source, old
        source = source.replace(old, new)
    # The broad event-name rename must not change the frozen helper or selector paths.
    source = source.replace('from compact_full_sample_core import aligned', 'from task_audience_row_core import aligned')
    source = source.replace("SELECTOR=HERE/'runs/compact_full_sample_ranker_v1'", "SELECTOR=HERE/'runs/task_audience_row_ranker_v1'")
    source = replace_function(source, 'metadata', METADATA)
    source = replace_function(source, 'inputs', INPUTS)
    source = replace_function(source, 'dependencies', DEPENDENCIES)
    compile(source, str(target), 'exec')
    target.write_text(source, encoding='utf8')
    print(target)


if __name__ == '__main__':
    main()
