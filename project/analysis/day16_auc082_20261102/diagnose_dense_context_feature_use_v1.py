"""Inspect completed native model structure without reading held labels."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = HERE / 'runs/full_negative_dense_context_lgb_v1'
OUT = HERE / 'dense_context_feature_use_diagnostic_v1.json'
REMOVE = ['user_id', 'task_id', 'adv_id', 'city', 'device_name', 'device_size']


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    assert not OUT.exists(), 'Preserve earlier diagnostic evidence'
    inputs = json.loads((SOURCE / 'inputs_locked.json').read_text(encoding='utf-8'))
    schema = inputs['15']['source_contract']['schema']
    assert len(schema['columns']) == 395 and set(REMOVE) <= set(schema['categories'])
    records = []
    for arm, leaves in [('dense_context_leaf31_w5', 31), ('dense_context_leaf63_w25', 63)]:
        path = SOURCE / f'day15_{arm}_trees2400.txt'
        before = digest(path)
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 131072))
            tail = stream.read().decode('utf-8')
        assert 'end of parameters' in tail
        section = tail.split('feature_importances:\n', 1)[1].split('parameters:', 1)[0]
        importance = {}
        for line in section.strip().splitlines():
            if '=' in line:
                name, value = line.rsplit('=', 1)
                importance[name] = int(value)
        assert set(importance) <= set(schema['columns'])
        with path.open('rb') as stream:
            trees = sum(line.startswith(b'Tree=') for line in stream)
        total = sum(importance.values())
        assert trees == 2400 and 0 < total <= trees * (leaves - 1)
        removed = sum(importance.get(name, 0) for name in REMOVE)
        assert digest(path) == before
        records.append(dict(model=str(path), model_sha256=before, trees=trees,
            model_bytes=path.stat().st_size, input_features=len(schema['columns']),
            used_features=len(importance), total_splits=total,
            six_identifier_splits=removed, six_identifier_split_fraction=removed / total,
            category_split_counts={name: importance.get(name, 0) for name in schema['categories']},
            all_feature_split_counts=importance,
            top_features=sorted(importance.items(), key=lambda row: (-row[1], row[0]))[:25]))
    report = dict(status='checked', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        diagnostic_only=True, models=records, proposed_removed_features=REMOVE,
        proposed_retained_features=389, held_labels_read=False, inferred17_labels_read=False,
        auc_evaluated=False, models_or_predictions_modified=False,
        interpretation='These are native model split counts, not gain importance or proof of overfitting. '
            'Test a separate frozen 389-column ablation to measure whether forcing historical/context '
            'features improves complete chronological cohorts; preserve the current fit and controls.',
        sources_sha256={str(p): digest(p) for p in (Path(__file__), SOURCE / 'protocol.json', SOURCE / 'inputs_locked.json')})
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='checked', models=[dict(model=Path(row['model']).name,
        used_features=row['used_features'], identifier_split_fraction=row['six_identifier_split_fraction'])
        for row in records], held_labels_read=False, auc_evaluated=False)), flush=True)


if __name__ == '__main__':
    main()
