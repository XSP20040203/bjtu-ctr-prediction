"""Fresh CSV checks and preserved delivery pointers for the v6 export."""
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / 'analysis/day16_auc082_20261102'
sys.path.insert(0, str(EXP))
from conditional_daily import read
from fit_stacker import digest, write
from rich_userday_activity import array_hash
from publish_progress import publication_lock


def main():
    folder = ROOT / 'output/current_best_20191117_20261103_v6'
    run = HERE / 'current_best_17_run_v6'
    report = read(folder / 'verification.json')
    manifest = read(folder / 'prediction_manifest.json')
    assert report['status'] == 'passed' and report['inferred17_labels_read'] is False
    assert digest(run / 'protocol.json') == manifest['protocol_sha256']
    assert digest(run / 'input_contracts.json') == manifest['input_contracts_sha256']
    csv = folder / 'submission_best_20191117.csv'
    assert digest(csv) == report['submission_sha256'] == manifest['files_sha256'][csv.name]
    assert digest(folder / 'day17_prediction.npy') == manifest['files_sha256']['day17_prediction.npy']
    saved = pd.read_csv(csv, float_precision='round_trip')
    assert list(saved) == ['id', 'pctr'] and len(saved) == report['rows'] == 404182
    np.testing.assert_array_equal(saved.id, np.arange(len(saved)))
    np.testing.assert_array_equal(saved.pctr, np.load(folder / 'day17_prediction.npy'))
    assert np.isfinite(saved.pctr).all() and saved.pctr.between(0, 1, inclusive='neither').all()
    original = pd.read_csv(ROOT / '2026ml/final_test.csv', usecols=['pt_d', 'log_id'], dtype='int64')
    assert original.pt_d.eq(20191117).all() and len(original) == len(saved)
    assert array_hash(original.log_id.to_numpy()) == read(run / 'input_contracts.json')['20191117']['log_ids_sha256']
    assert digest(ROOT / '2026ml/final_test.csv') == report['raw_datasets_sha256']['final_test.csv']
    now = datetime.now(timezone.utc)
    delivery = dict(timestamp_utc=now.isoformat(), request_complete=True,
        submission=str(csv), verification=str(folder / 'verification.json'), rows=len(saved),
        development16_auc=report['development16_auc'], submission_sha256=digest(csv),
        inferred17_labels_read=False, platform_auc=None, original_target_goal_remains_active=True,
        independently_replayed=True, fresh_csv_probability_id_order_and_original_test_checks='passed',
        probability_min=float(saved.pctr.min()), probability_max=float(saved.pctr.max()))
    destination = HERE / 'delivery_current_best_17_v6.json'
    assert not destination.exists(), 'Preserve prior delivery records'
    write(destination, delivery)
    with publication_lock():
        archive = EXP / 'runtime_history' / now.strftime('%Y%m%dT%H%M%S%fZ_delivery17_v6')
        archive.mkdir(parents=True, exist_ok=False)
        for filename in ('STATUS.json', 'resumed_goal_runtime_v1.json'):
            path = EXP / filename
            if path.exists():
                shutil.copy2(path, archive / filename)
        runtime = read(EXP / 'resumed_goal_runtime_v1.json')
        previous = runtime.get('latest_day17_delivery')
        history = runtime.setdefault('day17_delivery_history', [])
        if previous and previous not in history:
            history.append(previous)
        runtime['latest_day17_delivery'] = delivery
        runtime['last_day17_delivery_update_utc'] = now.isoformat()
        write(EXP / 'resumed_goal_runtime_v1.json', runtime)
        state = read(EXP / 'STATUS.json')
        previous = state.get('latest_best_recipe17_delivery')
        history = state.setdefault('latest_best_recipe17_delivery_history', [])
        if previous and previous not in history:
            history.append(previous)
        state['latest_best_recipe17_delivery'] = delivery
        state['resumed_goal_runtime'] = runtime
        write(EXP / 'STATUS.json', state)
    print(delivery, flush=True)


if __name__ == '__main__':
    main()
