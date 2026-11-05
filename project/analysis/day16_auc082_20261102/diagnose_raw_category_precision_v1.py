"""Audit raw category ID precision without reading advertisement labels."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from conditional_daily import read
from fit_stacker import HERE, ROOT, digest, write


def main():
    output = HERE / 'raw_category_precision_diagnostic_v1.json'
    assert not output.exists(), 'Preserve previous diagnostic records'
    source = ROOT / '2026ml/train.csv'
    audit = read(ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json')
    assert audit['audit_status'] == 'passed'
    expected = audit['raw_datasets_sha256']['train.csv']
    assert digest(source) == expected
    protocol_path = HERE / 'runs/full_negative_dense_context_lgb_v1/protocol.json'
    inputs_path = HERE / 'runs/full_negative_dense_context_lgb_v1/inputs_locked.json'
    contract = read(inputs_path)['16']['source_contract']
    categories = contract['schema']['categories']
    header = pd.read_csv(source, nrows=0).columns.tolist()
    columns = [name for name in categories if name in header]
    assert 'label' not in columns and 'pt_d' not in columns
    stats = {name: dict(rows=0, minimum=None, maximum=None,
        float32_roundtrip_mismatches=0, signed_int32_range_violations=0,
        maximum_absolute_float32_roundtrip_error=0) for name in columns}
    for block in pd.read_csv(source, usecols=columns, dtype='int64', chunksize=60000):
        for name in columns:
            original = block[name].to_numpy(dtype='int64')
            # int32 source IDs cannot approach int64 casting overflow.
            assert np.all(np.abs(original.astype('float64')) < 2**62)
            restored = original.astype('float32').astype('int64')
            record = stats[name]
            low, high = int(original.min()), int(original.max())
            record['rows'] += len(original)
            record['minimum'] = low if record['minimum'] is None else min(record['minimum'], low)
            record['maximum'] = high if record['maximum'] is None else max(record['maximum'], high)
            record['float32_roundtrip_mismatches'] += int(np.count_nonzero(restored != original))
            record['signed_int32_range_violations'] += int(np.count_nonzero(
                (original < np.iinfo('int32').min) | (original > np.iinfo('int32').max)))
            record['maximum_absolute_float32_roundtrip_error'] = max(
                record['maximum_absolute_float32_roundtrip_error'],
                int(np.max(np.abs(restored - original))))
    assert len({record['rows'] for record in stats.values()}) == 1
    assert digest(source) == expected
    losses = {name: record for name, record in stats.items()
              if record['float32_roundtrip_mismatches'] or record['signed_int32_range_violations']}
    report = dict(status='checked', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        diagnostic_only=True, raw_advertisement_rows=next(iter(stats.values()))['rows'],
        raw_categories_checked=columns, derived_or_feed_categories_not_in_train_csv=[
            name for name in categories if name not in columns], categories=stats,
        any_raw_category_id_information_loss=bool(losses), affected_categories=list(losses),
        advertisement_labels_read=False, inferred17_labels_read=False,
        fitted_models_or_predictions_changed=False, original_train_csv_unchanged=True,
        train_csv_sha256=expected,
        sources_sha256={str(path): digest(path) for path in
            (Path(__file__), protocol_path, inputs_path)},
        interpretation=(
            'Raw advertisement category IDs require a preserved-ID feature implementation.' if losses else
            'All checked raw advertisement category IDs roundtrip exactly through float32 and fit signed int32; no raw advertisement ID precision repair is indicated.'),
        scope='All rows of train.csv, raw advertised category columns only; this is not a model or AUC evaluation.')
    write(output, report)
    print(dict(status=report['status'], rows=report['raw_advertisement_rows'],
        categories=len(columns), affected_categories=list(losses),
        advertisement_labels_read=False, auc_evaluated=False), flush=True)


if __name__ == '__main__':
    main()
