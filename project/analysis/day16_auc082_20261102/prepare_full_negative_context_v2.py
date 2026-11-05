"""Versioned API repair; adopt verified full-population inputs without rebuilding."""
from pathlib import Path
import ast
import hashlib
import json
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
source = HERE / 'full_negative_context_v1.py'
target = HERE / 'full_negative_context_v2.py'
runner = HERE / 'run_full_negative_context_v2.ps1'
record = HERE / 'full_negative_recovery_v2.json'
assert not target.exists() and not runner.exists() and not record.exists()
text = source.read_text(encoding='utf8')
text = text.replace("OUT = HERE / 'runs/full_negative_context_v1'", "OUT = HERE / 'runs/full_negative_context_v2'\nINPUT = HERE / 'runs/full_negative_context_v1'")
text = text.replace("'run_full_negative_context_v1.ps1'", "'run_full_negative_context_v2.ps1'")
text = text.replace("paths = [Path(__file__)]", "paths = [Path(__file__), HERE / 'prepare_full_negative_context_v2.py']")
start, end = text.index('def prepare():\n'), text.index('\n\ndef fit():\n')
adoption = '''def prepare():
    assert not OUT.exists(), 'Preserve all earlier runs and API failure evidence'
    previous = read(INPUT / 'protocol.json')
    for filename, expected in previous['sources_sha256'].items():
        assert digest(filename) == expected, filename
    assert previous['params'] == PARAMS and previous['arms'] == ARMS
    assert previous['checkpoints'] == CHECKPOINTS
    paths = dependencies() + [INPUT / 'protocol.json', INPUT / 'inputs_locked.json']
    locked = read(INPUT / 'inputs_locked.json')
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    contracts = {}
    for day in (15, 16):
        folder = INPUT / f'input_day{day}'
        contract = read(folder / 'contract.json')
        assert contract == locked[str(day)]['contract']
        assert digest(folder / 'contract.json') == locked[str(day)]['contract_sha256']
        assert digest(folder / 'features.npy') == contract['matrix_sha256']
        for name, expected in contract['files_sha256'].items():
            assert digest(folder / name) == expected, name
        rows, train, held = [np.load(folder / (name + '.npy')) for name in ('rows', 'train', 'held')]
        np.testing.assert_array_equal(rows[train], np.flatnonzero(raw.pt_d.to_numpy() < 20191100 + day))
        np.testing.assert_array_equal(rows[held], np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day))
        np.testing.assert_array_equal(np.load(folder / 'labels.npy'), raw.label.to_numpy()[rows[train]])
        assert contract['max_training_date'] < 20191100 + day
        assert contract['all_historical_negatives_retained'] and contract['all_prior_audited_common_rows_checked']
        paths += [folder / name for name in ('contract.json', 'features.npy', *contract['files_sha256'])]
        contracts[str(day)] = locked[str(day)]
    for filename, expected in previous['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    OUT.mkdir()
    protocol = {key: value for key, value in previous.items() if key != 'sources_sha256'}
    protocol.update(input_adoption=str(INPUT),
        repair='CatBoost base constructor takes one parameter dictionary; v1 feature matrices reused unchanged',
        sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)})
    write(OUT / 'protocol.json', protocol)
    write(OUT / 'inputs_locked.json', contracts)
    for day in (15, 16):
        np.save(OUT / f'day{day}_indices.npy', np.load(INPUT / f'day{day}_indices.npy'))
    folder = INPUT / 'input_day16'
    declared = read(folder / 'contract.json')
    labels = np.load(folder / 'labels.npy')
    train = np.load(folder / 'train.npy')
    selected = np.r_[np.flatnonzero(labels == 1)[:64], np.flatnonzero(labels == 0)[:1024]]
    matrix = np.load(folder / 'features.npy', mmap_mode='r')
    frame = pd.DataFrame(np.asarray(matrix[train[selected]]), columns=declared['schema']['columns'])
    cats = declared['schema']['categories']
    for name in cats:
        frame[name] = frame[name].to_numpy(dtype='int64')
    pool = Pool(frame, label=labels[selected], cat_features=cats)
    model = CatBoost(dict(iterations=20, depth=4, loss_function='Logloss', task_type='CPU',
                         random_seed=151661, thread_count=2, allow_writing_files=False, verbose=False))
    model.fit(pool)
    (OUT / 'preflight').mkdir()
    path = OUT / 'preflight/model.cbm'
    save_model(model, path)
    np.testing.assert_array_equal(model.predict(pool, prediction_type='RawFormulaVal'),
                                  load_model(path).predict(pool, prediction_type='RawFormulaVal', thread_count=2))
    dates = np.load(folder / 'dates.npy')[train[selected]]
    write(OUT / 'preflight.json', dict(status='passed', rows=len(frame), device='CPU',
        max_training_date=int(dates.max()), held_labels_read=False,
        serialization_maximum_error=0., full_gpu_capacity_proven=False,
        original_full_negative_inputs_reused_unchanged=True,
        protocol_sha256=digest(OUT / 'protocol.json'), input_lock_sha256=digest(OUT / 'inputs_locked.json')))
    check()
    emit('full_negative_v2_adopted_and_preflight_passed',
         population15=contracts['15']['contract']['training_rows'],
         population16=contracts['16']['contract']['training_rows'])
'''
text = text[:start] + adoption + text[end:]
text = text.replace("folder = OUT / f'input_day{day}'", "folder = INPUT / f'input_day{day}'")
text = text.replace("folder = OUT / (f'independent_day{day}' if independent else f'input_day{day}')",
                    "folder = (OUT / f'independent_day{day}' if independent else INPUT / f'input_day{day}')")
text = text.replace("read(OUT / f'input_day{day}/contract.json')", "read(INPUT / f'input_day{day}/contract.json')")
ast.parse(text, filename=str(target))
target.write_text(text, encoding='utf8')
runner.write_text((HERE / 'run_full_negative_context_v1.ps1').read_text(encoding='utf8').replace(
    'full_negative_context_v1.py', 'full_negative_context_v2.py'), encoding='utf8')
record.write_text(json.dumps(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
    failed_session=27513, failure='CatBoost.__init__() got an unexpected keyword argument iterations',
    preserved_run='runs/full_negative_context_v1', primary_full_populations_complete=True,
    repair='New v2 source adopts both hash-verified inputs and passes params as one dictionary',
    original_sources_modified=False, inferred17_labels_read=False,
    sources_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, target, runner)}),
    ensure_ascii=False, indent=2), encoding='utf8')
print('Created versioned complete-negative recovery; primary matrices preserved', flush=True)
