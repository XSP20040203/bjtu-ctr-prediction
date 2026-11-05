"""Extend the compact full-sample recipe with label-free state-set transitions."""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent

INPUTS = '''def inputs(day, partition, independent=False):
    frame, cats, info, parent = full_samples.inputs(day, partition, independent)
    key = (day, independent)
    if key not in SNAPSHOT_CACHE:
        raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=STATE_FIELDS,
            dtype={name:str for name in STATE_FIELDS if 'list' in name})
        allowed = np.flatnonzero(raw.pt_d.to_numpy() <= 20191100+day)
        values, names = build_states(raw.iloc[allowed], independent=independent)
        assert names == STATE_NAMES and values.shape == (len(allowed),72)
        SNAPSHOT_CACHE[key] = (allowed, values)
        del raw
    allowed, values = SNAPSHOT_CACHE[key]
    positions = info['positions'] if partition == 'train' else info['held']
    original_rows = info['rows'][positions]
    local = np.searchsorted(allowed, original_rows)
    np.testing.assert_array_equal(allowed[local], original_rows)
    extra = values[local]
    frame = pd.concat([frame.reset_index(drop=True),pd.DataFrame(extra,columns=STATE_NAMES)],axis=1)
    assert frame.shape[1] == 495
    audit = dict(parent, shape=list(frame.shape), frame_sha256=frame_hash(frame),
        unordered_features_sha256=array_hash(extra), unordered_columns=STATE_NAMES,
        unordered_features_read_advertisement_labels=False, unordered_features_ignore_row_order=True,
        unordered_features_group_only_same_user_and_date=True)
    if partition == 'held':
        SNAPSHOT_CACHE.pop(key,None)
    gc.collect()
    emit('unordered_snapshot_input_ready',day=day,partition=partition,
         independent=independent,shape=list(frame.shape))
    return frame,cats,info,audit
'''

DEPENDENCIES = '''def dependencies():
    checked = read(HERE/'unordered_snapshot_core_preflight_v1.json')
    assert checked['status'] == 'passed'
    assert checked['source_sha256'] == digest(HERE/'unordered_snapshot_core_v1.py')
    paths = full_samples.dependencies() + [Path(__file__),
        HERE/'prepare_unordered_snapshot_ranker_v1.py', HERE/'unordered_snapshot_core_v1.py',
        HERE/'test_unordered_snapshot_core_v1.py',HERE/'audit_unordered_snapshot_core_v1.py',
        HERE/'unordered_snapshot_core_preflight_v1.json',HERE/'run_unordered_snapshot_ranker_v1.ps1']
    for folder in BASES.values():
        assert read(folder/'verification.json')['status'] == 'passed'
        for path, expected in read(folder/'protocol.json')['sources_sha256'].items():
            assert digest(path) == expected, path
        paths += [folder/'verification.json',folder/'protocol.json']
        for day in [15,16]:
            paths += [folder/f'day{day}_prediction.npy',folder/f'day{day}_indices.npy']
    return list(dict.fromkeys(paths))
'''


def replace_function(source, name, replacement):
    pattern = rf'^def {name}\([^\n]*\):.*?(?=^def |^if __name__)'
    result, count = re.subn(pattern, lambda _: replacement+'\n\n', source,
                            count=1, flags=re.M | re.S)
    assert count == 1, name
    return result


def main():
    target = HERE/'unordered_snapshot_ranker_v1.py'
    assert not target.exists(), 'Preserve earlier sources'
    source = (HERE/'compact_full_sample_ranker_v1.py').read_text(encoding='utf8')
    changes = [
        ('Compact historical row ranking retaining the complete fixed source sample.',
         'Full-sample row ranking with label-free unordered click and close state changes.'),
        ("OUT = HERE / 'runs/compact_full_sample_ranker_v1'", "OUT = HERE / 'runs/unordered_snapshot_ranker_v1'"),
        ("PREFLIGHT = HERE / 'runs/compact_full_sample_preflight_v1'", "PREFLIGHT = HERE / 'runs/unordered_snapshot_preflight_v1'"),
        ("BASES={day:HERE/'runs/profile_response_ranker_v1' for day in [15,16]}",
         "BASES={day:HERE/'runs/compact_full_sample_ranker_v1' for day in [15,16]}"),
        ("dict(name='full_sample_d6'", "dict(name='unordered_d6'"),
        ("dict(name='full_sample_d8'", "dict(name='unordered_d8'"),
        ('random_seed=151653', 'random_seed=151654'),
        ("emit('compact_full_sample_", "emit('unordered_snapshot_"),
        ('All native categorical features plus384 numeric features selected by a fixed strictly historical model',
         '423 frozen compact features plus72 label-free same-user/date unordered snapshot transition features'),
        ('AUDIENCE_CACHE = {}', 'AUDIENCE_CACHE = {}\nSNAPSHOT_CACHE = {}'),
    ]
    for old,new in changes:
        assert old in source,old
        source = source.replace(old,new)
    source = source.replace('import task_audience_rate_v1 as audience',
        'import task_audience_rate_v1 as audience\nimport compact_full_sample_ranker_v1 as full_samples\n'
        'from unordered_snapshot_core_v1 import build as build_states, FIELDS as STATE_FIELDS, NAMES as STATE_NAMES')
    source = replace_function(source,'inputs',INPUTS)
    source = replace_function(source,'dependencies',DEPENDENCIES)
    compile(source,str(target),'exec')
    target.write_text(source,encoding='utf8')
    print(target)


if __name__ == '__main__':
    main()
