"""Check the frozen queue and its real source datasets before GPU dispatch."""
from pathlib import Path
import json
from fit_stacker import ROOT, HERE, digest

path = HERE/'full_quality_row_catboost_queued_v1.json'
record = json.loads(path.read_text(encoding='utf8'))
for source, expected in record['sources_sha256'].items():
    assert digest(Path(source)) == expected, source
for name, expected in record['raw_datasets_sha256'].items():
    assert digest(ROOT/'2026ml'/name) == expected, name
print(json.dumps({'event': 'full_quality_queue_verified', 'tests_passed': record['boundary_tests_passed'],
                  'preflight_shape': record['preflight_shape']}, ensure_ascii=False), flush=True)
