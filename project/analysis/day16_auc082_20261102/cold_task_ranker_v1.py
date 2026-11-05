"""Strict-prior cross-user row ranking with all native and337 news-quality features."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from fit_stacker import HERE, ROOT, digest, write
from conditional_daily import read, emit
from rich_userday_activity import array_hash
from rich_row_catboost import selection, frame_hash
from full_quality_row_catboost import full_frame, quality_rows
from news_quality_features import READ, build_quality
from catboost_userday_residual import save_model, load_model
from verify_stacker import rank_sum_auc
from cross_user_pair_core import design
from within_user_ranking_core import within_prediction
import cross_user_pair_ranker_v1 as base_rows
from cold_task_core import identity_columns,cold_mask

OUT = HERE / 'runs/cold_task_ranker_v1'
PREFLIGHT = HERE / 'runs/cold_task_preflight_v1'
SOURCE = HERE / 'runs/rich_augmented_rows_v1'
QUALITY = HERE / 'runs/news_quality_daily_v1'
BASE = HERE / 'runs/cross_user_pair_ranker_v1'
ARMS = [dict(name='cold_task_d6', depth=6), dict(name='cold_task_d8', depth=8)]
CHECKPOINTS = [200, 600, 1600]
WEIGHTS = [.05, .1, .2, .35, .5, .75, 1.]
MODES = ['add_global', 'replace_global']
PARAMS = dict(loss_function='PairLogit', learning_rate=.025, l2_leaf_reg=100.,
    random_strength=1., bootstrap_type='Bernoulli', subsample=.8, border_count=32,
    one_hot_max_size=16, max_ctr_complexity=1, thread_count=2, random_seed=151648,
    task_type='GPU', devices='0', gpu_ram_part=.5, allow_writing_files=False, verbose=200)
REPLAY = {}
COLD = {}


def metadata(day):
    cutoff = 20191100 + day
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label', 'user_id'], dtype='int32')
    dates = raw.pt_d.to_numpy()
    labels = raw.label.to_numpy(copy=True)
    labels[dates >= cutoff] = -1
    chosen, weights, held_rows, train, held, y = selection(day, labels, dates)
    source = read(SOURCE/f'input_contract{day}.json')
    assert array_hash(chosen) == source['training_rows_sha256']
    assert array_hash(y) == source['training_labels_sha256']
    assert array_hash(weights) == source['weights_sha256']
    rows = np.sort(np.r_[chosen, held_rows])
    positive = train[labels[rows[train]] == 1]
    negative = train[labels[rows[train]] == 0]
    sampled = np.random.default_rng(151648).choice(negative, len(negative)//4, replace=False)
    subset = np.sort(np.r_[positive, sampled])
    order, groups, pairs = design(dates[rows[subset]], labels[rows[subset]], cutoff)
    positions = subset[order]
    target = labels[rows[positions]]
    np.testing.assert_array_equal(held_rows, np.load(BASE/f'day{day}_indices.npy'))
    assert np.all(dates[rows[positions]] < cutoff)
    held_users = raw.user_id.to_numpy()[held_rows]
    row_groups, _ = pd.factorize(held_users, sort=True)
    return dict(raw=raw, rows=rows, positions=positions, held=held, held_rows=held_rows,
        target=target, groups=groups, pairs=pairs, row_groups=row_groups,
        source=source, max_training_date=int(dates[rows[positions]].max()))


def inputs(day, partition, independent=False):
    frame,cats,info,parent_audit=base_rows.inputs(day,partition,independent)
    indices,masked_names=identity_columns(info['source']['feature_names'])
    columns=[frame.columns[i] for i in indices]
    for name in columns:
        frame[name]=0
    audit=dict(parent_audit,frame_sha256=frame_hash(frame),
        masked_identity_columns=columns,masked_identity_feature_names=masked_names,
        shared_advertiser_and_application_inputs_preserved=True)
    if independent and partition=='held':
        base_rows.REPLAY.pop(day,None)
    gc.collect()
    emit('cold_task_input_ready',day=day,partition=partition,independent=independent,masked=len(columns))
    return frame,cats,info,audit


def dependencies():
    paths = [Path(__file__), HERE/'cross_user_pair_core.py', HERE/'test_cross_user_pair_core.py',
        HERE/'run_cold_task_ranker_v1.ps1',HERE/'cold_task_core.py',HERE/'test_cold_task_core.py',HERE/'prepare_cold_task_ranker_v1.py',HERE/'cross_user_pair_ranker_v1.py', HERE/'fit_stacker.py', HERE/'conditional_daily.py',
        HERE/'rich_userday_activity.py', HERE/'rich_row_catboost.py', HERE/'full_quality_row_catboost.py',
        HERE/'news_quality_features.py', HERE/'catboost_userday_residual.py', HERE/'verify_stacker.py',
        HERE/'within_user_ranking_core.py', HERE/'global_row_context.py', HERE/'global_row_context15.py']
    for folder in [SOURCE, QUALITY, BASE]:
        assert read(folder/'verification.json')['status'] == 'passed'
        parent = read(folder/'protocol.json')
        for filename, expected in parent['sources_sha256'].items():
            assert digest(filename) == expected, filename
        paths += [folder/'protocol.json', folder/'verification.json']
    for day in [15, 16]:
        paths += [SOURCE/f'input_contract{day}.json', SOURCE/f'input_day{day}/features.npy',
            SOURCE/f'independent_day{day}/features.npy', QUALITY/f'quality{day}.npz',
            QUALITY/f'input_contract{day}.json', BASE/f'day{day}_prediction.npy', BASE/f'day{day}_indices.npy']
    return list(dict.fromkeys(paths))


def check_sources(directory=OUT):
    protocol = read(directory/'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    return protocol


def pool_from(frame, cats, info):
    return Pool(frame, label=info['target'], group_id=info['groups'], pairs=info['pairs'], cat_features=cats)


def preflight():
    assert not PREFLIGHT.exists(), 'Preserve all preflights'
    hashes = read(BASE/'protocol.json')['raw_datasets_sha256']
    paths = dependencies()
    PREFLIGHT.mkdir()
    write(PREFLIGHT/'protocol.json', dict(params=PARAMS, arms=ARMS,
        raw_datasets_sha256=hashes, sources_sha256={str(path): digest(path) for path in paths}))
    check_sources(PREFLIGHT)
    frame, cats, info, audit = inputs(16, 'train')
    pool = pool_from(frame, cats, info)
    sample = Pool(frame.iloc[:2000], cat_features=cats)
    del frame
    gc.collect()
    results = []
    for arm in ARMS:
        model = CatBoostRanker(**dict(PARAMS, depth=arm['depth'], iterations=20, verbose=False))
        model.fit(pool)
        before = model.predict(sample, thread_count=2)
        path = PREFLIGHT/(arm['name']+'.cbm')
        save_model(model, path)
        after = load_model(path).predict(sample, prediction_type='RawFormulaVal', thread_count=2)
        np.testing.assert_array_equal(before, after)
        results.append(dict(arm, serialization_maximum_error=0., model_sha256=digest(path)))
        del model
        gc.collect()
    check_sources(PREFLIGHT)
    write(PREFLIGHT/'verification.json', dict(status='passed', audit=audit, results=results,
        source_code_sha256=digest(__file__), full_prior_cohort_capacity_tested=True,
        held_auc_evaluated=False, inferred17_labels_read=False))
    emit('cold_task_preflight_passed', shape=audit['shape'])


def fit():
    assert not OUT.exists(), 'Preserve all previous full fits'
    preflight_report = read(PREFLIGHT/'verification.json')
    assert preflight_report['status'] == 'passed' and preflight_report['source_code_sha256'] == digest(__file__)
    frozen = check_sources(PREFLIGHT)
    paths = dependencies()+[PREFLIGHT/'protocol.json', PREFLIGHT/'verification.json']
    OUT.mkdir()
    write(OUT/'protocol.json', dict(params=PARAMS, arms=ARMS, checkpoints=CHECKPOINTS,
        weights=WEIGHTS, modes=MODES, dates=[20191115, 20191116],
        training='All earlier positive rows and one quarter of fixed sampled negatives; random1024-row date queries;48 negative pairs per positive',
        objective='Cross-user within-date row PairLogit; zero-click users retained',
        baseline_run=str(BASE), features='1017 full-quality inputs with task/creative identities and associated statistics zeroed; preserve advertiser/app metadata',
        selection='All both-date models and candidate hashes locked before scoring; dates repeatedly reused',
        evaluation_day_labels_fit=False, inferred17_labels_read=False,
        raw_datasets_sha256=frozen['raw_datasets_sha256'], sources_sha256={str(path):digest(path) for path in paths}))
    models = []
    for day in [15, 16]:
        frame, cats, info, audit = inputs(day, 'train')
        write(OUT/f'input_contract{day}.json', audit)
        np.save(OUT/f'day{day}_indices.npy', info['held_rows'])
        pool = pool_from(frame, cats, info)
        del frame
        gc.collect()
        for arm in ARMS:
            emit('cold_task_fit_started', day=day, arm=arm['name'])
            model = CatBoostRanker(**dict(PARAMS, depth=arm['depth'], iterations=max(CHECKPOINTS)))
            model.fit(pool)
            for trees in CHECKPOINTS:
                name = f'day{day}_{arm["name"]}_trees{trees}'
                saved = model.copy()
                saved.shrink(trees)
                path = OUT/(name+'.cbm')
                save_model(saved, path)
                models.append(dict(arm, day=day, trees=trees, name=name, model_sha256=digest(path),
                    input_contract_sha256=digest(OUT/f'input_contract{day}.json')))
                del saved
            del model
            gc.collect()
            emit('cold_task_fit_complete', day=day, arm=arm['name'])
        del pool, info
        gc.collect()
    write(OUT/'models_locked.json', dict(protocol_sha256=digest(OUT/'protocol.json'), models=models))
    check_sources()


def forecasts(day, independent=False):
    train, _, info, audit = inputs(day, 'train', independent)
    assert audit == read(OUT/f'input_contract{day}.json')
    del train, info
    gc.collect()
    frame, cats, info, audit = inputs(day, 'held', independent)
    if independent:
        assert audit == read(OUT/f'held_contract{day}.json')
    else:
        write(OUT/f'held_contract{day}.json', audit)
    pool = Pool(frame, cat_features=cats)
    del frame
    locked = read(OUT/'models_locked.json')
    assert locked['protocol_sha256'] == digest(OUT/'protocol.json')
    result = {}
    for record in locked['models']:
        if record['day'] != day:
            continue
        path = OUT/(record['name']+'.cbm')
        assert digest(path) == record['model_sha256']
        assert record['input_contract_sha256'] == digest(OUT/f'input_contract{day}.json')
        value = load_model(path).predict(pool, prediction_type='RawFormulaVal', thread_count=2)
        assert value.shape == info['held_rows'].shape and np.isfinite(value).all()
        result[record['name']] = value
        saved = OUT/(record['name']+'_held_margin.npy')
        if independent:
            np.testing.assert_array_equal(value, np.load(saved))
        else:
            assert not saved.exists()
            np.save(saved, value)
    REPLAY.pop(day, None)
    gc.collect()
    return result, info


def prediction(base, margin, groups, weight, mode, independent=False):
    if mode == 'add_within':
        return within_prediction(base, margin, groups, weight, 'add_within', independent)
    z = logit(np.clip(base, 1e-8, 1-1e-8))
    normalized = (margin-margin.mean())/max(margin.std(), 1e-6)*z.std()
    delta = normalized if mode == 'add_global' else normalized+z.mean()-z
    if independent:
        odds = np.exp(z)*np.exp(weight*delta)
        return odds/(1+odds)
    return expit(z+weight*delta)


def cold_query(day):
    if day not in COLD:
        raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','task_id'],dtype='int32')[['pt_d','task_id']]
        query=raw.loc[raw.pt_d.eq(20191100+day)].reset_index(drop=True)
        COLD[day]=cold_mask(raw,query,20191100+day)
    return COLD[day]


def candidates(day, values, groups, independent=False):
    base=np.load(BASE/f'day{day}_prediction.npy')
    mask=cold_query(day)
    assert len(mask)==len(base)
    yield dict(name='base',mode='base',weight=0.),base.copy()
    for name,margin in values.items():
        for mode in MODES:
            for weight in WEIGHTS:
                record=dict(name=name,mode=mode,weight=weight)
                p=prediction(base,margin,groups,weight,mode,independent)
                p[~mask]=base[~mask]
                yield record,p


def candidate_key(record):
    return f'{record["name"]}__{record["mode"]}__weight{record["weight"]:g}'


def evaluate():
    check_sources()
    assert not (OUT/'predictions_locked.json').exists()
    locks = {}
    for day in [15, 16]:
        values, info = forecasts(day)
        locks[str(day)] = {candidate_key(record):array_hash(p)
            for record,p in candidates(day, values, info['row_groups'])}
    write(OUT/'predictions_locked.json', dict(protocol_sha256=digest(OUT/'protocol.json'),
        candidates_sha256=locks, both_date_model_hashes_locked=True, held_labels_opened_for_scoring=False))
    emit('cold_task_candidates_locked')


def verify():
    check_sources()
    assert not (OUT/'verification.json').exists()
    locks = read(OUT/'predictions_locked.json')
    assert locks['protocol_sha256'] == digest(OUT/'protocol.json')
    values_by_day, infos = {}, {}
    # Reconstruct both-date training/held inputs and all12 model forecasts before scoring.
    for day in [15, 16]:
        values, info = forecasts(day, True)
        for record, p in candidates(day, values, info['row_groups']):
            assert array_hash(p) == locks['candidates_sha256'][str(day)][candidate_key(record)]
        values_by_day[day], infos[day] = values, info
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    reports = {}
    for day in [15, 16]:
        rows = infos[day]['held_rows']
        np.testing.assert_array_equal(rows, np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day))
        y = raw.label.to_numpy()[rows]
        assert len(y) == {15:424780,16:421983}[day] and y.sum() == {15:3523,16:3724}[day]
        records, maximum = [], 0.
        standard = list(candidates(day, values_by_day[day], infos[day]['row_groups']))
        independent = candidates(day, values_by_day[day], infos[day]['row_groups'], True)
        for (record,p),(other,q) in zip(standard,independent):
            assert record == other and np.isfinite(p).all() and np.all((p>0)&(p<1))
            error = float(np.max(np.abs(p-q)))
            assert error < 2e-14
            maximum = max(maximum,error)
            records.append(dict(record, auc=rank_sum_auc(y,p)))
        best = max(records, key=lambda record:record['auc'])
        p = next(p for record,p in standard if candidate_key(record)==candidate_key(best))
        assert abs(roc_auc_score(y,p)-best['auc']) < 1e-12
        np.save(OUT/f'day{day}_prediction.npy', p)
        reports[str(20191100+day)] = dict(auc=best['auc'], best=best, records=records,
            rows=len(y), clicks=int(y.sum()), evaluation_day_labels_fit=False,
            held_labels_used_to_select_fixed_recipe=True, all_both_date_candidates_locked_before_scoring=True,
            independent_native_source_and_raw_news_features_replayed=True, all12_checkpoint_margins_replayed_exactly=True,
            maximum_independent_prediction_error=maximum, candidate_hashes_replayed_exactly=True,
            target_met=best['auc'] >= {15:.815,16:.82}[day])
        emit('cold_task_scored', day=day, best=best)
    check_sources()
    write(OUT/'verification.json',dict(status='passed', dates=reports, original_datasets_unchanged=True,
        both_targets_met=all(record['target_met'] for record in reports.values())))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['preflight','fit','evaluate','verify'])
    {'preflight':preflight,'fit':fit,'evaluate':evaluate,'verify':verify}[parser.parse_args().phase]()

