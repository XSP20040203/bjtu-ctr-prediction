"""Supplement frozen v2 verification without changing models or locked forecasts."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'

import gc
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.metrics import roc_auc_score
import catboost_within_ranker_v2 as frozen
from catboost_userday_residual import load_model
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash
from within_user_ranker import centered_unit

RECOVERY = HERE/'runs/catboost_ranker_precision_verification_v1'
AGGREGATE_ATOL = 2e-13
PROBABILITY_ATOL = 2e-14


def main():
    assert not RECOVERY.exists(), 'Preserve prior verification attempts'
    assert not (frozen.OUT/'verification.json').exists()
    protocol = frozen.check_sources()
    lock = read(frozen.OUT/'predictions_locked.json')
    models = read(frozen.OUT/'models_locked.json')
    assert lock['protocol_sha256'] == digest(frozen.OUT/'protocol.json')
    assert lock['models_lock_sha256'] == digest(frozen.OUT/'models_locked.json')
    paths = [Path(__file__), frozen.OUT/'protocol.json', frozen.OUT/'models_locked.json',
             frozen.OUT/'predictions_locked.json', frozen.OUT/'result.json']
    paths += list(frozen.OUT.glob('*.npy')) + list(frozen.OUT.glob('*.cbm'))
    paths += [frozen.OUT/f'{part}_contract{day}.json' for day in [15,16] for part in ['input','held']]
    bindings = {str(path): digest(path) for path in paths}
    RECOVERY.mkdir()
    write(RECOVERY/'protocol.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        frozen_run=str(frozen.OUT), sources_sha256=bindings,
        recovery_reason='Original verification required bitwise equality for independent pandas vs numpy group reductions; observed mean-margin difference 1.55431223e-14',
        raw_model_margins_require_bitwise_equality=True,
        aggregate_absolute_tolerance=AGGREGATE_ATOL, probability_absolute_tolerance=PROBABILITY_ATOL,
        auc_absolute_tolerance=1e-12, training_or_reselection=False, inferred17_labels_read=False))
    summaries = {}
    for day in [15,16]:
        train, cats, info, audit = frozen.inputs(day,'train',True)
        assert audit == read(frozen.OUT/f'input_contract{day}.json')
        del train, info
        gc.collect()
        frame, cats, info, audit = frozen.inputs(day,'held',True)
        assert audit == read(frozen.OUT/f'held_contract{day}.json')
        np.testing.assert_array_equal(info['held_rows'],np.load(frozen.OUT/f'day{day}_indices.npy'))
        pool = Pool(frame,cat_features=cats)
        del frame
        primary, independent, primary_last, independent_last = {}, {}, [], []
        for record in models['models']:
            if record['day'] != day:
                continue
            path = frozen.OUT/(record['name']+'.cbm')
            assert digest(path) == record['model_sha256']
            assert record['input_contract_sha256'] == digest(frozen.OUT/f'input_contract{day}.json')
            value = load_model(path).predict(pool,prediction_type='RawFormulaVal',thread_count=2)
            saved = np.load(frozen.OUT/(record['name']+'_held_margin.npy'))
            np.testing.assert_array_equal(value,saved)
            assert array_hash(value) == lock['dates'][str(day)]['forecasts_sha256'][record['name']]
            primary[record['name']] = value
            independent[record['name']] = value
            if record['trees'] == max(frozen.CHECKPOINTS):
                primary_last.append(centered_unit(value,info['groups'],False))
                independent_last.append(centered_unit(value,info['groups'],True))
        assert len(primary) == 6 and len(primary_last) == 2
        name = f'day{day}_mean_final_within_unit'
        primary[name] = np.mean(primary_last,axis=0)
        independent[name] = np.mean(independent_last,axis=0)
        np.testing.assert_array_equal(primary[name],np.load(frozen.OUT/(name+'_held_margin.npy')))
        aggregate_error = float(np.max(np.abs(primary[name]-independent[name])))
        assert aggregate_error < AGGREGATE_ATOL
        actual = pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
        y = actual.loc[actual.pt_d.eq(20191100+day),'label'].to_numpy()
        del actual
        candidate_error = 0.
        auc_error = 0.
        count = 0
        for (record,p), (other,q) in zip(frozen.candidates(day,primary,info['groups']),
                                      frozen.candidates(day,independent,info['groups'],True)):
            assert record == other
            assert array_hash(p) == lock['dates'][str(day)]['candidates_sha256'][frozen.candidate_key(record)]
            error = float(np.max(np.abs(p-q)))
            assert error < PROBABILITY_ATOL
            score_error = abs(float(roc_auc_score(y,p))-float(roc_auc_score(y,q)))
            assert score_error < 1e-12
            candidate_error = max(candidate_error,error)
            auc_error = max(auc_error,score_error)
            count += 1
        assert count == 113
        summaries[str(20191100+day)] = dict(raw_model_margins_exact=True,
            original_numpy_aggregate_exact=True, maximum_independent_aggregate_error=aggregate_error,
            maximum_independent_candidate_error=candidate_error,
            maximum_independent_candidate_auc_error=auc_error, candidates=count,
            training_and_held_frames_exact=True, full337_quality_rebuilt_from_raw=True)
        emit('catboost_ranker_precision_replay_passed',day=day,**summaries[str(20191100+day)])
        del pool, primary, independent, primary_last, independent_last, info
        frozen.QUALITY_REPLAY.pop(day,None)
        gc.collect()
    for path, expected in bindings.items():
        assert digest(path) == expected,path
    frozen.check_sources()
    write(RECOVERY/'verification.json',dict(status='passed',dates=summaries,
        original_frozen_sources_models_forecasts_and_selection_preserved=True,
        inferred17_labels_read=False,raw_datasets_sha256=protocol['raw_datasets_sha256']))
    frozen.score(True)
    report = read(frozen.OUT/'verification.json')
    report['supplemental_precision_verification'] = str(RECOVERY/'verification.json')
    report['supplemental_precision_verification_sha256'] = digest(RECOVERY/'verification.json')
    report['independent_aggregate_uses_documented_float64_tolerance'] = True
    write(frozen.OUT/'verification.json',report)
    emit('catboost_ranker_supplemental_verification_complete',aucs={day:r['auc'] for day,r in report['dates'].items()})


if __name__ == '__main__':
    main()
