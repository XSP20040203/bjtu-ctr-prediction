"""Preserve failed v1 and correct its unbound identity-helper import in v2."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
TARGET=HERE/'export_firstday_task_v2.py'


def main():
    assert not TARGET.exists(), 'Preserve all exporter versions'
    source=(HERE/'export_firstday_task_v1.py').read_text(encoding='utf8')
    old='from firstday_task_core import identity_columns,cold_mask'
    assert old in source
    source=source.replace(old,'from cold_task_core import identity_columns,cold_mask')
    source=source.replace('firstday_task_run_v1','firstday_task_run_v2')
    source=source.replace('prepare_firstday_task_export_v1.py','prepare_firstday_task_export_v2.py')
    compile(source,str(TARGET),'exec')
    TARGET.write_text(source,encoding='utf8')
    (HERE/'firstday_task_export_recovery_v2.json').write_text(json.dumps(dict(
        failed_source=str(HERE/'export_firstday_task_v1.py'),failed_session=34079,
        error='ImportError: identity_columns and cold_mask belong to cold_task_core',
        failure_before_protocol_freeze=True,no_prediction_outputs_created=True,
        recovery_source=str(TARGET),verified_best_submission_unaffected=True),indent=2),encoding='utf8')


if __name__=='__main__':
    main()
