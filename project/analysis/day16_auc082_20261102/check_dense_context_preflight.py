"""Exercise the new category transform on real prior15 inputs without held scoring."""
import json
from pathlib import Path
import numpy as np
import dense_context_rows as run

run.OUT = run.HERE/'runs/dense_context_rows_preflight_v1'
assert not run.OUT.exists(), 'Preserve all prior preflight attempts'
x, training, held, y, weights, rows, contract = run.input_data(15, run.ARMS[1])
assert contract['max_training_date'] == 20191114
assert len(y) == len(training) == 569257 and len(held) == 424780
maps = np.load(run.OUT/'input/day15_context_numeric/category_maps.npz')
unknown = {}
for i in contract['categorical_columns']:
    vocabulary = maps[f'f{i}']
    values = x[training, i]
    assert np.all(values >= 0) and np.all(values < len(vocabulary))
    unknown[contract['feature_names'][i]] = int(np.sum(x[held, i] < 0))
sample = training[:40000]
dataset = run.lgb.Dataset(np.asarray(x[sample]), label=y[:len(sample)], weight=weights[:len(sample)],
    categorical_feature=contract['categorical_columns'], free_raw_data=True)
model = run.lgb.train({'objective': 'binary', 'num_leaves': 7, 'min_data_in_leaf': 50,
    'num_threads': 1, 'verbosity': -1, 'seed': 151637, 'max_bin': 31}, dataset, num_boost_round=15)
model_path = run.OUT/'serialization_smoke.txt'
model_path.write_text(model.model_to_string(), encoding='utf8')
before = model.predict(x[held[:1000]], num_threads=1)
reloaded = run.lgb.Booster(model_str=model_path.read_text(encoding='utf8'))
after = reloaded.predict(x[held[:1000]], num_threads=1)
error = float(np.max(np.abs(before-after)))
assert error < 1e-12 and np.isfinite(after).all() and np.ptp(after) > 0
report = {'status': 'passed', 'day': 20191115, 'shape': list(x.shape),
    'max_training_date': 20191114, 'training_rows': len(training),
    'held_rows': len(held), 'held_ad_labels_used_for_fit_or_evaluation': False,
    'training_dictionaries_dense_and_bounded': True, 'serialization_maximum_error': error,
    'unknown_held_category_counts': unknown, 'model_sha256': run.digest(model_path),
    'input_contract': contract}
run.write(run.OUT/'verification.json', report)
print(json.dumps({k: v for k, v in report.items() if k != 'input_contract'}, indent=2))
