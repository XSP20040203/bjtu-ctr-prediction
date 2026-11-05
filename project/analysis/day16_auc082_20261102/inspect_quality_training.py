"""Inspect only fitted historical models, without opening held advertisement outcomes."""
import json
from pathlib import Path
from rich_group_residual import lgb
from fit_stacker import HERE, digest, write

directory = HERE/'runs/news_quality_daily_v1'
records = []
for path in sorted(directory.glob('day15_*_trees1600.txt')):
    model = lgb.Booster(model_str=path.read_text(encoding='utf8'))
    names = model.feature_name()
    gain = model.feature_importance('gain')
    quality = [i for i, name in enumerate(names) if name.startswith('quality__')]
    chosen = sorted(quality, key=lambda i: -gain[i])[:10]
    records.append({'model': path.name, 'sha256': digest(path),
                    'quality_gain_fraction': float(gain[quality].sum()/gain.sum()),
                    'top_quality_features': [{'name': names[i], 'gain': float(gain[i])} for i in chosen]})
report = {'held_ad_labels_read': False, 'training_max_ad_label_date': 20191114, 'models': records}
write(directory/'historical_training_quality_importance.json', report)
print(json.dumps(report, indent=2))
