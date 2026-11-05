"""Inspect completed15 models without reading any advertisement labels."""
from pathlib import Path
import numpy as np
from conditional_daily import read
from fit_stacker import HERE, digest, write
from catboost_userday_residual import load_model


def main():
    run = HERE / 'runs/full_negative_neural_embeddings_v2'
    output = HERE / 'neural_embedding_usage_day15_v1.json'
    assert not output.exists(), 'Preserve diagnostic history'
    preflight = read(run / 'preflight.json')
    assert preflight['status'] == 'passed'
    assert digest(run / 'protocol.json') == preflight['protocol_sha256']
    contracts = read(run / 'inputs_locked.json')
    names = contracts['15']['schema']['columns']
    assert len(names) == len(set(names)) == 533
    embedded = np.array([name.startswith('neural_') for name in names])
    assert embedded.sum() == 138
    records = {}
    sources = {str(Path(__file__)): digest(Path(__file__)),
        str(run / 'protocol.json'): digest(run / 'protocol.json'),
        str(run / 'inputs_locked.json'): digest(run / 'inputs_locked.json')}
    for arm in ('neural_embedding_d4', 'neural_embedding_d6'):
        path = run / f'day15_{arm}_trees2400.cbm'
        before = digest(path)
        model = load_model(path)
        assert model.feature_names_ == names and model.tree_count_ == 2400
        values = model.get_feature_importance(type='PredictionValuesChange', thread_count=2)
        assert len(values) == len(names) and np.isfinite(values).all()
        assert abs(float(values.sum()) - 100.) < 1e-9
        order = np.argsort(-values, kind='stable')
        records[arm] = dict(day=20191115, trees=model.tree_count_, feature_columns=len(names),
            max_training_date=contracts['15']['max_training_date'],
            embedding_feature_count=int(embedded.sum()),
            embedding_importance=float(values[embedded].sum()),
            context_importance=float(values[~embedded].sum()),
            used_embedding_features=int((values[embedded] > 0).sum()),
            top_features=[dict(name=names[i], importance=float(values[i])) for i in order[:20]])
        assert digest(path) == before
        sources[str(path)] = before
    report = dict(status='passed', scope='Only completed15 models; no held-date scoring',
        advertisement_labels_read=False, inferred17_labels_read=False, models=records,
        model_files_unchanged=True, sources_sha256=sources)
    write(output, report)
    print({name: {key: record[key] for key in ('embedding_importance', 'used_embedding_features')}
        for name, record in records.items()}, flush=True)


if __name__ == '__main__':
    main()
