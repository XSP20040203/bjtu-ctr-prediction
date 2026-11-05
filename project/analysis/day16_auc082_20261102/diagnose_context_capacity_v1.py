"""Inspect frozen model capacity and feedback cohorts without reading final17 labels."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoost
from sklearn.metrics import roc_auc_score
from conditional_daily import read
from fit_stacker import HERE, ROOT, digest, write

OUT = HERE / 'context_capacity_diagnostic_v1.json'


def main():
    assert not OUT.exists(), 'Preserve diagnostics'
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'user_id', 'task_id', 'label'], dtype='int32')
    reports = {}
    for day in (15, 16):
        held = raw.loc[raw.pt_d.eq(20191100 + day)]
        past = raw.loc[raw.pt_d.lt(20191100 + day)]
        masks = dict(all=np.ones(len(held), dtype=bool),
            cold_task=~held.task_id.isin(past.task_id.unique()).to_numpy(),
            user_no_past_ad_click=~held.user_id.isin(past.loc[past.label.eq(1), 'user_id'].unique()).to_numpy())
        base = np.load(HERE / f'runs/refresh_family_quantiles_v1/day{day}_prediction.npy')
        selected = ('positive25_d4' if day == 15 else 'positive25_d6')
        specifications = [('class', 'full_negative_class_balance_v1', selected, 3200),
            ('semantic', 'full_negative_semantic_embeddings_v1', 'semantic_d4', 2400),
            ('semantic_deep', 'full_negative_semantic_embeddings_v1', 'semantic_d6', 2400)]
        models = []
        for kind, run, arm, trees in specifications:
            folder = HERE / 'runs' / run
            name = f'day{day}_{arm}_trees{trees}'
            path = folder / (name + '.cbm')
            model = CatBoost()
            with tempfile.TemporaryDirectory(prefix='context_probe_') as temporary:
                saved = Path(temporary) / 'model.cbm'
                saved.write_bytes(path.read_bytes())
                model.load_model(str(saved))
            importance = model.get_feature_importance(type='PredictionValuesChange', thread_count=2)
            names = model.feature_names_
            margin = np.load(folder / (name + '_held_margin.npy'))
            y = held.label.to_numpy()
            models.append(dict(kind=kind, model=str(path), model_sha256=digest(path),
                standalone_auc=float(roc_auc_score(y, margin)),
                cohort_auc={name: float(roc_auc_score(y[mask], margin[mask])) for name, mask in masks.items()},
                top_features=[dict(name=names[index], importance=float(importance[index]))
                    for index in np.argsort(-importance)[:25]],
                raw_user_id_importance=float(sum(value for name, value in zip(names, importance) if name == 'user_id')),
                neural_feature_importance=float(sum(value for name, value in zip(names, importance) if name.startswith('neural_')))))
        reports[str(20191100 + day)] = dict(rows=len(held), models=models,
            baseline_cohort_auc={name: float(roc_auc_score(held.label.to_numpy()[mask], base[mask])) for name, mask in masks.items()},
            cohort_rows={name: int(mask.sum()) for name, mask in masks.items()})
    write(OUT, dict(dates=reports, only_saved_models_inspected=True, inferred17_labels_read=False,
        development_dates_repeatedly_reused=True, source_sha256=digest(Path(__file__))))
    print({day: dict(baseline=value['baseline_cohort_auc'], models=[{
        key: model[key] for key in ('kind', 'standalone_auc', 'raw_user_id_importance', 'neural_feature_importance', 'top_features')}
        for model in value['models']]) for day, value in reports.items()}, flush=True)


if __name__ == '__main__':
    main()
