"""Inspect trained row splits and category representation without held outcomes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from fit_stacker import ROOT, HERE, write, digest
from rich_userday_activity import array_hash
import global_row_context15 as sampling

folder = HERE/'runs/rich_augmented_rows_v1'
contract = json.loads((folder/'input_contract15.json').read_text(encoding='utf8'))
x = np.load(folder/'input_day15/features.npy', mmap_mode='r')
raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
raw.loc[raw.pt_d >= 20191115, 'label'] = -1
chosen, _ = sampling.selected_training(raw.label.to_numpy(), raw.pt_d.to_numpy())
held = np.flatnonzero(raw.pt_d.to_numpy() == 20191115)
rows = np.sort(np.r_[chosen, held])
training = np.flatnonzero(raw.pt_d.to_numpy()[rows] < 20191115)
assert array_hash(rows[training]) == contract['training_rows_sha256']
assert x.shape == tuple(contract['shape'])
categories = []
for column in contract['categorical_columns']:
    values = x[training, column].astype('int64')
    unique = np.unique(values[values >= 0])
    categories.append({'column': column, 'name': contract['feature_names'][column],
        'unique_training_categories': len(unique), 'minimum': int(values.min()), 'maximum': int(values.max()),
        'range_to_unique_ratio': float((unique.max()+1)/len(unique)) if len(unique) else None,
        'negative_missing_rows': int(np.sum(values < 0))})
path = folder/'day15_leaves31_weight5_trees1800.txt'
gains = np.zeros(x.shape[1])
counts = np.zeros(x.shape[1], dtype='int64')
cat_bytes = trees = 0
features = None
with path.open(encoding='utf8') as stream:
    for line in stream:
        if line.startswith('Tree='):
            trees += 1
        elif line.startswith('split_feature='):
            features = np.fromstring(line.split('=', 1)[1], sep=' ', dtype='int64')
            np.add.at(counts, features, 1)
        elif line.startswith('split_gain='):
            gain = np.fromstring(line.split('=', 1)[1], sep=' ')
            assert features is not None and len(gain) == len(features)
            np.add.at(gains, features, gain)
        elif line.startswith('cat_threshold='):
            cat_bytes += len(line.encode('utf8'))
order = np.argsort(-gains, kind='stable')[:35]
report = {'held_ad_labels_used_for_fit_or_evaluation': False, 'fit_max_date': 20191114,
    'model': str(path), 'model_sha256': digest(path), 'model_bytes': path.stat().st_size,
    'trees': trees, 'categorical_threshold_bytes': cat_bytes,
    'categorical_threshold_fraction': cat_bytes/path.stat().st_size,
    'categories': sorted(categories, key=lambda r: -r['range_to_unique_ratio']),
    'top_training_gain_features': [{'name': contract['feature_names'][i], 'gain': float(gains[i]),
        'splits': int(counts[i])} for i in order],
    'feature_family_gain': {name: float(gains[columns].sum()/gains.sum()) for name, columns in [
        ('original_augmented472', list(range(472))), ('rich_daily128', list(range(472, 600))),
        ('news_quality64', list(range(600, 664))), ('exact_news_snapshot80', list(range(664, 744)))]}}
write(HERE/'rich_row_capacity_diagnostic.json', report)
print(json.dumps({key: value for key, value in report.items() if key != 'categories'}, indent=2))
print(json.dumps({'most_sparse_category_codes': report['categories'][:6]}, indent=2))
