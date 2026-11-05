"""Freeze the next experiment after passing real-input and category checks."""
from datetime import datetime, timezone
import json
import dense_context_rows as run

assert not run.QUEUE.exists() and not run.OUT.exists(), 'Preserve earlier queue versions'
preflight_path = run.HERE/'runs/dense_context_rows_preflight_v1/verification.json'
diagnostic_path = run.HERE/'rich_row_capacity_diagnostic.json'
for path in [preflight_path, diagnostic_path]:
    record = run.read(path)
    if 'held_ad_labels_read' in record:
        assert record.pop('held_ad_labels_read') is False
        record['held_ad_labels_used_for_fit_or_evaluation'] = False
        record['metadata_clarification'] = ('The CSV label column was loaded and held labels immediately masked; '
            'no held labels were used by these input/representation checks or scored.')
        run.write(path, record)
preflight = run.read(preflight_path)
assert preflight['status'] == 'passed' and preflight['max_training_date'] == 20191114
assert preflight['serialization_maximum_error'] == 0.
paths = [run.HERE/name for name in ['dense_context_rows.py', 'test_dense_context_rows.py',
    'run_dense_context_rows.ps1', 'check_dense_context_preflight.py']]
paths += [preflight_path, diagnostic_path]
record = {'status': 'queued_waiting_for_source_independent_verification',
    'frozen_utc': datetime.now(timezone.utc).isoformat(), 'wait_for_process_id': 97416,
    'source_run': str(run.SOURCE), 'output_run': str(run.OUT), 'arms': run.ARMS,
    'params': run.PARAMS, 'checkpoints': run.CHECKS, 'blend_weights': run.WEIGHTS,
    'targets': {'20191115': .815, '20191116': .82},
    'preflight_shape': preflight['shape'], 'preflight_maximum_serialization_error': 0.,
    'inferred17_labels_used': False,
    'sources_sha256': {str(path): run.digest(path) for path in paths}}
run.write(run.QUEUE, record)
print(json.dumps(record, indent=2))
