"""Prior-only feature selection and deeper complete-count daily CatBoost models."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
from sklearn.metrics import roc_auc_score
from quality_daily_catboost import inputs as native_inputs, training_targets, PARAMS as OLD_PARAMS
from bayesian_daily_offset import offsets
from catboost_userday_residual import save_model, load_model
from conditional_daily import read, emit
from fit_stacker import ROOT, HERE, digest, write
from rich_userday_activity import array_hash, correct
from verify_stacker import rank_sum_auc

SOURCE = HERE/'runs/quality_daily_catboost_v1'
BASE = HERE/'runs/compose_quality_daily_transfer_v1'
OUT = HERE/'runs/compact_quality_daily_catboost_v1'
PREFLIGHT = HERE/'runs/compact_quality_daily_catboost_preflight_v1'
ARMS = [dict(name='poisson_d6_top384',depth=6,numeric_features=384),
        dict(name='poisson_d7_top192',depth=7,numeric_features=192)]
CHECKS = [600,1600,3200,4800]
POWERS = [.0625,.125,.25,.5]
PARAMS = dict(OLD_PARAMS, random_seed=151643, l2_leaf_reg=100., random_strength=2.)


def selected_columns(names, categories, importance, numeric_features):
    importance = np.asarray(importance)
    if importance.shape != (len(names),) or not np.isfinite(importance).all():
        raise ValueError('Finite training-model importance must match the feature schema')
    if len(set(names)) != len(names) or not set(categories).issubset(names):
        raise ValueError('Unique names and known categorical inputs are required')
    quality = [i for i,name in enumerate(names) if name.startswith('quality__')]
    ordinary = [i for i,name in enumerate(names) if name not in categories and i not in set(quality)]
    if numeric_features <= 0 or numeric_features > len(ordinary):
        raise ValueError('Numeric budget must fit the ordinary feature set')
    ranked = sorted(ordinary,key=lambda i:(-importance[i],i))[:numeric_features]
    indices = sorted(set(ranked+quality+[names.index(name) for name in categories]))
    return [names[i] for i in indices]


def inputs(day):
    data, audit = native_inputs(day)
    assert audit == read(SOURCE/f'input_contract{day}.json')
    name = f'day{day}_poisson_d5_offset_trees1600'
    record = next(item for item in read(SOURCE/'models_locked.json')['models'] if item['name'] == name)
    path = SOURCE/(name+'.cbm')
    assert digest(path) == record['model_sha256']
    importance = np.asarray(load_model(path).get_feature_importance(type='PredictionValuesChange',thread_count=1))
    names = data['frame'].columns.tolist()
    choices = {arm['name']:selected_columns(names,data['categories'],importance,arm['numeric_features']) for arm in ARMS}
    for columns in choices.values():
        assert sum(name.startswith('quality__') for name in columns) == 337
        assert 'log_bayesian_click_mass' in columns and set(data['categories']).issubset(columns)
    report = dict(native_contract=audit, selector_name=name, selector_model_sha256=record['model_sha256'],
        importance_sha256=array_hash(importance), selected_columns=choices,
        selector_max_ad_training_date=20191100+day-1, selection_uses_held_labels=False)
    return data, report


def preflight():
    assert not PREFLIGHT.exists(), 'Preserve previous preflights'
    assert read(SOURCE/'verification.json')['status'] == 'passed'
    assert read(BASE/'verification.json')['status'] == 'passed'
    PREFLIGHT.mkdir()
    data, contract = inputs(15)
    training, target = training_targets(data['keys'],data['counts'],20191115,'poisson')
    arm = ARMS[1]
    columns = contract['selected_columns'][arm['name']]
    frame = data['frame'].iloc[training][columns]
    pool = Pool(frame,label=target,cat_features=data['categories'],baseline=offsets(data,'poisson')[training])
    model = CatBoost(dict(PARAMS,depth=arm['depth'],iterations=20,loss_function='Poisson'))
    model.fit(pool)
    sample = Pool(frame.iloc[:2000],cat_features=data['categories'])
    before = model.predict(sample,prediction_type='RawFormulaVal',thread_count=2)
    save_model(model,PREFLIGHT/'smoke.cbm')
    after = load_model(PREFLIGHT/'smoke.cbm').predict(sample,prediction_type='RawFormulaVal',thread_count=2)
    error = float(np.max(np.abs(before-after)))
    assert error < 1e-12 and np.all(data['keys'][training,0] < 20191115)
    write(PREFLIGHT/'verification.json',dict(status='passed',source_code_sha256=digest(__file__),
        full_prior_training_groups=len(training),shape=list(frame.shape),depth=arm['depth'],
        full_cohort_memory_preflight=True,held_ad_labels_used=False,serialization_maximum_error=error,
        selected_columns=columns,selector_model_sha256=contract['selector_model_sha256']))
    emit('compact_daily_preflight_passed',shape=list(frame.shape),serialization_error=error)


def check_sources():
    protocol = read(OUT/'protocol.json')
    for path,expected in protocol['sources_sha256'].items():
        assert digest(path) == expected,path
    for name,expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected,name
    return protocol


def fit():
    assert not OUT.exists(), 'Preserve earlier runs and failures'
    smoke = read(PREFLIGHT/'verification.json')
    assert smoke['status'] == 'passed' and smoke['source_code_sha256'] == digest(__file__)
    parent = read(SOURCE/'protocol.json')
    assert read(SOURCE/'verification.json')['status'] == read(BASE/'verification.json')['status'] == 'passed'
    paths = [Path(__file__),HERE/'test_compact_quality_daily_catboost.py',HERE/'test_quality_daily_catboost.py',
        HERE/'run_compact_quality_daily_catboost.ps1',HERE/'quality_daily_catboost.py',
        HERE/'news_quality_daily.py',HERE/'news_quality_features.py',HERE/'bayesian_daily_offset.py',
        HERE/'rich_userday_activity.py',HERE/'catboost_userday_residual.py',HERE/'fit_stacker.py',
        HERE/'conditional_daily.py',HERE/'verify_stacker.py',PREFLIGHT/'verification.json',
        SOURCE/'protocol.json',SOURCE/'verification.json',SOURCE/'models_locked.json',
        BASE/'protocol.json',BASE/'verification.json']
    for path,expected in parent['sources_sha256'].items():
        assert digest(path) == expected,path
        paths.append(Path(path))
    for day in [15,16]:
        paths += [SOURCE/f'input_contract{day}.json',SOURCE/f'day{day}_poisson_d5_offset_trees1600.cbm',
            BASE/f'day{day}_prediction.npy',BASE/f'day{day}_indices.npy']
    for name,expected in parent['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected,name
    OUT.mkdir()
    protocol = dict(dates=[20191115,20191116],arms=ARMS,params=PARAMS,checkpoints=CHECKS,powers=POWERS,
        all337_quality_features_retained=True,all_native_categorical_fields_retained=True,
        selector='Training leaf weights of fixed prior-ad-fitted1600-tree Poisson d5 model; ordinary columns only',
        selector_uses_held_ad_labels=False,own_held_ad_labels_fit=False,inferred17_labels_read=False,
        training='Complete strictly earlier user/day counts with their Bayesian rate-mass Pool baseline',
        selection='Both-date models lock before held scoring; both-date candidate hashes lock before labels; reused development dates',
        raw_datasets_sha256=parent['raw_datasets_sha256'],sources_sha256={str(path):digest(path) for path in dict.fromkeys(paths)})
    write(OUT/'protocol.json',protocol)
    locks = []
    for day in [15,16]:
        data,contract = inputs(day)
        write(OUT/f'input_contract{day}.json',contract)
        np.save(OUT/f'day{day}_indices.npy',data['rows'])
        training,target = training_targets(data['keys'],data['counts'],20191100+day,'poisson')
        baseline = offsets(data,'poisson')[training]
        for arm in ARMS:
            columns = contract['selected_columns'][arm['name']]
            pool = Pool(data['frame'].iloc[training][columns],label=target,
                        baseline=baseline,cat_features=data['categories'])
            emit('compact_daily_training_ready',day=day,arm=arm['name'],shape=[len(training),len(columns)])
            model = CatBoost(dict(PARAMS,loss_function='Poisson',depth=arm['depth'],iterations=max(CHECKS)))
            model.fit(pool)
            del pool
            for trees in CHECKS:
                name = f'day{day}_{arm["name"]}_trees{trees}'
                path = OUT/(name+'.cbm')
                saved = model.copy()
                saved.shrink(trees)
                save_model(saved,path)
                locks.append(dict(arm,day=day,trees=trees,name=name,objective='poisson',
                    model_sha256=digest(path),target_sha256=array_hash(target),baseline_sha256=array_hash(baseline),
                    columns=columns))
                del saved
            emit('compact_daily_arm_complete',day=day,arm=arm['name'])
            del model
            gc.collect()
        del data
        gc.collect()
    write(OUT/'models_locked.json',dict(protocol_sha256=digest(OUT/'protocol.json'),models=locks))


def candidate(state,record):
    if record['name'] == 'base':
        return state['base'].copy()
    group_prediction = np.zeros(state['nkeys'])
    group_prediction[state['held']] = state['forecasts'][record['name']]
    return correct(state['base'],group_prediction,state['groups'],'poisson',record['power'])


def independent_correction(base,row_groups,response,power):
    p = np.clip(np.asarray(base,dtype='float64'),1e-8,1-1e-8)
    frame = pd.DataFrame(dict(group=row_groups,p=p))
    mass = frame.groupby('group',sort=False).p.transform('sum').to_numpy()
    ratio = np.clip(np.maximum(response[row_groups],1e-6)/np.maximum(mass,1e-8),.05,20.)**power
    odds = p/(1-p)*ratio
    return odds/(1+odds)


def evaluate(verifying):
    check_sources()
    assert not verifying or not (OUT/'verification.json').exists(), 'Preserve completed verification'
    locked_models = read(OUT/'models_locked.json')
    assert locked_models['protocol_sha256'] == digest(OUT/'protocol.json')
    states,records_by_day,hashes = {},{},{}
    for day in [15,16]:
        data,audit = inputs(day)
        assert audit == read(OUT/f'input_contract{day}.json')
        np.testing.assert_array_equal(data['rows'],np.load(OUT/f'day{day}_indices.npy'))
        np.testing.assert_array_equal(data['rows'],np.load(BASE/f'day{day}_indices.npy'))
        held = np.flatnonzero(data['keys'][:,0] == 20191100+day)
        state = dict(base=np.load(BASE/f'day{day}_prediction.npy'),held=held,nkeys=len(data['keys']),
            groups=data['row_groups'].copy(),rows=data['rows'].copy(),forecasts={})
        records = [dict(name='base',recipe='base',power=0.)]
        maximum_error = 0.
        for arm in ARMS:
            columns = audit['selected_columns'][arm['name']]
            pool = Pool(data['frame'].iloc[held][columns],cat_features=data['categories'])
            for record in locked_models['models']:
                if record['day'] != day or record['name'].split('_trees')[0] != f'day{day}_{arm["name"]}':
                    continue
                assert record['columns'] == columns
                path = OUT/(record['name']+'.cbm')
                assert digest(path) == record['model_sha256']
                margin = load_model(path).predict(pool,prediction_type='RawFormulaVal',thread_count=2)
                response = np.exp((margin+offsets(data,'poisson')[held]).clip(-15,8))
                path = OUT/(record['name']+'_held_group_prediction.npy')
                if verifying:
                    error = float(np.max(np.abs(response-np.load(path))))
                    assert error < 2e-7
                    maximum_error = max(maximum_error,error)
                else:
                    np.save(path,response)
                state['forecasts'][record['name']] = response
                records += [dict(name=record['name'],objective='poisson',power=power) for power in POWERS]
            del pool
        assert len(state['forecasts']) == len(ARMS)*len(CHECKS)
        states[day],records_by_day[day] = state,records
        hashes[str(day)] = {f'{r["name"]}_power{r["power"]:g}':array_hash(candidate(state,r)) for r in records}
        state['maximum_replay_error'] = maximum_error
        emit('compact_daily_candidates_ready',day=day,candidates=len(records),verifying=verifying)
        del data
        gc.collect()
    lock_path = OUT/'predictions_locked.json'
    lock = dict(protocol_sha256=digest(OUT/'protocol.json'),predictions_sha256=hashes,held_labels_opened=False)
    if verifying:
        assert lock == read(lock_path)
    else:
        assert not lock_path.exists()
        write(lock_path,lock)
    # Both full-date forecast hashes exist before any held advertisement labels are read here.
    raw = pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    results = {}
    for day in [15,16]:
        state = states[day]
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
        np.testing.assert_array_equal(rows,state['rows'])
        y = raw.label.to_numpy()[rows]
        assert len(y) == {15:424780,16:421983}[day] and int(y.sum()) == {15:3523,16:3724}[day]
        records = [dict(r,auc=rank_sum_auc(y,candidate(state,r))) for r in records_by_day[day]]
        best = max(records,key=lambda r:r['auc'])
        prediction = candidate(state,best)
        error = 0.
        if best['name'] != 'base':
            response = np.zeros(state['nkeys'])
            response[state['held']] = state['forecasts'][best['name']]
            independent = independent_correction(state['base'],state['groups'],response,best['power'])
            error = float(np.max(np.abs(prediction-independent)))
            assert error < 2e-14
        assert abs(roc_auc_score(y,prediction)-best['auc']) < 1e-12
        if verifying:
            np.testing.assert_allclose(prediction,np.load(OUT/f'day{day}_prediction.npy'),rtol=0,atol=2e-14)
        else:
            np.save(OUT/f'day{day}_prediction.npy',prediction)
        results[str(20191100+day)] = dict(auc=best['auc'],best=best,records=records,rows=len(y),clicks=int(y.sum()),
            independent_pandas_maximum_error=error,maximum_model_replay_error=state['maximum_replay_error'],
            all_both_date_candidate_hashes_replayed_exactly=verifying,
            features_rebuilt_from_raw=True,held_labels_used_to_select_fixed_recipe=True,
            evaluation_day_labels_fit=False,target_met=best['auc'] >= {15:.815,16:.82}[day])
        emit('compact_daily_scored',day=day,best=best,verifying=verifying)
    check_sources()
    write(OUT/('verification.json' if verifying else 'result.json'),dict(status='passed' if verifying else 'evaluated',
        dates=results,original_datasets_unchanged=True,both_targets_met=all(r['target_met'] for r in results.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=['preflight','fit','evaluate','verify'])
    args = parser.parse_args()
    if args.phase == 'preflight':
        preflight()
    elif args.phase == 'fit':
        fit()
    else:
        evaluate(args.phase == 'verify')
