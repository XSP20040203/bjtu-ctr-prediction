"""Chronological conditional choice model for supplied within-day ID positions.

No claim that log_id is a timestamp is needed. Features describe its observed
relative position among the supplied user/day exposures. Nuisance ad effects
are fitted jointly, but only the position terms are transferred to the base.
"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, logit
from fit_stacker import ROOT, HERE, digest, write
from verify_stacker import rank_sum_auc
from rich_userday_activity import array_hash

OUT = HERE/'runs/conditional_position_v1'
BASES = {15: HERE/'runs/rich_binary15_completed_v1',
         16: HERE/'runs/rolling_previous_app_v1'}
BASE_FILES = {15: 'prediction.npy', 16: 'day16_prediction.npy'}
FIELDS = ['pt_d', 'log_id', 'user_id', 'task_id', 'spread_app_id',
          'slot_id', 'adv_prim_id', 'creat_type_cd']
NUISANCE = ['task_id', 'spread_app_id', 'slot_id', 'adv_prim_id', 'creat_type_cd']
GAP_EDGES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256])
ARMS = [{'name': 'all_l30', 'position_l2': 30., 'last_days': None},
        {'name': 'all_l100', 'position_l2': 100., 'last_days': None},
        {'name': 'recent3_l30', 'position_l2': 30., 'last_days': 3}]
POWERS = [0., .125, .25, .5, .75, 1.]
MODES = ['logit', 'preserve_user_mass']


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def position_columns(frame):
    """Pure unlabeled per-day transformations; no outcome column accepted."""
    assert list(frame.columns) == FIELDS
    columns = []
    names = []
    for title, group_fields, bins, gaps in [
        ('user', ['pt_d', 'user_id'], 10, True),
        ('user_task', ['pt_d', 'user_id', 'task_id'], 8, True),
        ('user_app', ['pt_d', 'user_id', 'spread_app_id'], 8, False),
    ]:
        group = frame.groupby(group_fields, sort=False).log_id
        count = group.transform('size').to_numpy()
        rank = group.rank(method='average').to_numpy()
        minimum = group.transform('min').to_numpy()
        maximum = group.transform('max').to_numpy()
        columns.append(np.minimum(bins-1, ((rank-.5)*bins/count).astype('int32')))
        names.append((title+'_rank_bin', bins))
        if gaps:
            left = frame.log_id.to_numpy()-minimum
            right = maximum-frame.log_id.to_numpy()
            for suffix, values in [('left_gap', left), ('right_gap', right)]:
                columns.append(np.searchsorted(GAP_EDGES, values, side='left').astype('int32'))
                names.append((title+'_'+suffix, len(GAP_EDGES)+1))
        if title == 'user':
            relative = (frame.log_id.to_numpy()-minimum+.5)/(maximum-minimum+1)
            columns.append(np.minimum(9, (relative*10).astype('int32')))
            names.append(('user_coordinate_bin', 10))
    return columns, names


def build(day):
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=FIELDS+['label'], dtype='int32')
    raw = raw.loc[raw.pt_d <= 20191100+day].copy()
    original_rows = raw.index.to_numpy()
    dates = raw.pt_d.to_numpy()
    raw.loc[raw.pt_d >= 20191100+day, 'label'] = -1
    y = raw.label.to_numpy(dtype='float64')
    assert np.all(y[dates == 20191100+day] == -1)
    frame = raw[FIELDS]
    columns, definitions = position_columns(frame)
    names = []
    offsets = []
    offset = 0
    for title, width in definitions:
        offsets.append(offset)
        names.extend(f'{title}_{i}' for i in range(width))
        offset += width
    nposition = offset
    # Slot interactions use only a historical vocabulary.
    slot_vocab = np.sort(raw.loc[raw.pt_d < 20191100+day, 'slot_id'].unique())
    slot = pd.Index(slot_vocab).get_indexer(raw.slot_id)
    safe_slot = np.where(slot >= 0, slot, len(slot_vocab))
    columns.append(columns[0]*(len(slot_vocab)+1)+safe_slot)
    offsets.append(offset)
    names.extend(f'user_rank_x_slot_{r}_{s}' for r in range(10)
                 for s in list(slot_vocab)+['unknown'])
    offset = len(names)
    nposition = offset
    vocabularies = {'interaction_slot': slot_vocab.tolist()}
    for field in NUISANCE:
        vocab = np.sort(raw.loc[raw.pt_d < 20191100+day, field].unique())
        vocabularies[field] = vocab.tolist()
        code = pd.Index(vocab).get_indexer(raw[field])
        columns.append(np.where(code >= 0, code, len(vocab)))
        offsets.append(offset)
        names.extend(f'nuisance_{field}_{value}' for value in list(vocab)+['unknown'])
        offset = len(names)
    row = np.tile(np.arange(len(raw), dtype='int32'), len(columns))
    col = np.concatenate([values+start for values, start in zip(columns, offsets)])
    x = sparse.coo_matrix((np.ones(len(row), dtype='float64'), (row, col)),
                          shape=(len(raw), len(names))).tocsr()
    codes, keys = pd.factorize(pd.MultiIndex.from_frame(raw[['pt_d', 'user_id']]), sort=True)
    training = dates < 20191100+day
    assert np.isin(y[training], [0, 1]).all()
    contract = {'shape': list(x.shape), 'nposition': nposition,
                'matrix_data_sha256': array_hash(x.data),
                'matrix_indices_sha256': array_hash(x.indices),
                'matrix_indptr_sha256': array_hash(x.indptr),
                'rows_sha256': array_hash(original_rows), 'groups_sha256': array_hash(codes),
                'training_labels_sha256': array_hash(y[training]),
                'training_max_date': int(dates[training].max()),
                'held_label_values': [-1], 'feature_names': names,
                'vocabularies': vocabularies}
    return {'x': x, 'y': y, 'dates': dates, 'groups': codes, 'rows': original_rows,
            'nposition': nposition, 'contract': contract}


def objective(coef, x, y, groups, starts, counts, penalty):
    z = np.asarray(x@coef)
    maxima = np.maximum.reduceat(z, starts)
    numerator = np.exp(z-maxima[groups])
    normalizer = np.bincount(groups, weights=numerator, minlength=len(counts))
    expectation = counts[groups]*numerator/normalizer[groups]
    loss = float(counts@(maxima+np.log(normalizer))-y@z+.5*np.dot(penalty*coef, coef))
    gradient = np.asarray(x.T@(expectation-y))+penalty*coef
    return loss, gradient


def fit_arm(data, day, arm):
    train = data['dates'] < 20191100+day
    if arm['last_days']:
        train &= data['dates'] >= 20191100+day-arm['last_days']
    count = np.bincount(data['groups'][train], weights=data['y'][train],
                        minlength=int(data['groups'].max())+1)
    chosen = np.flatnonzero(train & (count[data['groups']] > 0))
    chosen = chosen[np.argsort(data['groups'][chosen], kind='stable')]
    groups = pd.factorize(data['groups'][chosen], sort=True)[0]
    x = data['x'][chosen]
    y = data['y'][chosen]
    starts = np.r_[0, np.flatnonzero(np.diff(groups))+1]
    counts = np.bincount(groups, weights=y)
    assert data['dates'][chosen].max() < 20191100+day
    penalty = np.full(x.shape[1], 30., dtype='float64')
    penalty[:data['nposition']] = arm['position_l2']
    iterations = [0]
    def callback(coef):
        iterations[0] += 1
        if iterations[0] % 25 == 0:
            loss, gradient = objective(coef, x, y, groups, starts, counts, penalty)
            emit('conditional_position_fit', day=day, arm=arm['name'],
                 iteration=iterations[0], loss=loss, gradient_max=float(np.max(np.abs(gradient))))
    result = minimize(objective, np.zeros(x.shape[1]),
                      args=(x, y, groups, starts, counts, penalty), jac=True,
                      method='L-BFGS-B', callback=callback,
                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-5, 'maxcor': 15})
    assert np.isfinite(result.x).all() and np.isfinite(result.fun)
    name = f'day{day}_{arm["name"]}'
    model_path = OUT/(name+'.npy')
    np.save(model_path, result.x)
    record = dict(arm, day=day, model_name=name, model_sha256=digest(model_path),
                  fit_rows=len(chosen), fit_clicks=int(y.sum()), fit_groups=len(counts),
                  fit_max_date=int(data['dates'][chosen].max()),
                  fit_rows_sha256=array_hash(data['rows'][chosen]),
                  fit_labels_sha256=array_hash(y), optimizer_success=bool(result.success),
                  optimizer_message=str(result.message), iterations=int(result.nit),
                  loss=float(result.fun), gradient_max=float(np.max(np.abs(result.jac))))
    emit('conditional_position_model_complete', **record)
    return record


def base(day):
    return np.load(BASES[day]/BASE_FILES[day])


def corrected(p, delta, groups, power, mode):
    if power == 0:
        return p.copy()
    delta = np.clip(delta, -2., 2.)*power
    if mode == 'logit':
        return expit(logit(p.clip(1e-8, 1-1e-8))+delta)
    corrected_mass = p*np.exp(delta)
    old_sum = np.bincount(groups, weights=p)
    new_sum = np.bincount(groups, weights=corrected_mass)
    value = corrected_mass*old_sum[groups]/new_sum[groups]
    assert np.isfinite(value).all()
    return value.clip(1e-8, 1-1e-8)


def fit():
    assert not OUT.exists(), 'Preserving conditional position experiment'
    expected = json.loads((HERE/'runs/weighted_v1_day16/protocol.json').read_text(encoding='utf8'))['raw_datasets_sha256']
    for name, h in expected.items():
        assert digest(ROOT/'2026ml'/name) == h, name
    paths = [Path(__file__), HERE/'fit_stacker.py', HERE/'verify_stacker.py',
             HERE/'rich_userday_activity.py']
    for day, directory in BASES.items():
        assert json.loads((directory/'verification.json').read_text(encoding='utf8'))['status'] == 'passed'
        paths += [directory/'verification.json', directory/BASE_FILES[day]]
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json', {'dates': [20191115, 20191116], 'arms': ARMS,
          'powers': POWERS, 'modes': MODES, 'raw_datasets_sha256': expected,
          'sources_sha256': {str(path): digest(path) for path in paths},
          'objective': 'Conditional multinomial log likelihood given historical user-day click counts; ridge penalty',
          'transfer': 'Only fitted position terms; nuisance ad effects excluded from correction',
          'feature_availability': 'Relative log_id positions among supplied unlabeled same-day rows; no timestamp assumption',
          'selection': 'All six models lock before either held cohort is scored; reused development dates',
          'own_evaluation_day_labels_fit': False, 'inferred17_labels_read': False})
    locks = []
    for day in [15, 16]:
        data = build(day)
        write(OUT/f'input_contract{day}.json', data['contract'])
        np.save(OUT/f'day{day}_indices.npy', data['rows'][data['dates'] == 20191100+day])
        emit('conditional_position_ready', day=day, shape=list(data['x'].shape),
             position_columns=data['nposition'])
        for arm in ARMS:
            locks.append(fit_arm(data, day, arm))
        del data
        gc.collect()
    write(OUT/'models_locked.json', {'models': locks, 'protocol_sha256': digest(OUT/'protocol.json')})
    evaluate(False)


def evaluate(verifying):
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    locks = json.loads((OUT/'models_locked.json').read_text(encoding='utf8'))
    assert locks['protocol_sha256'] == digest(OUT/'protocol.json')
    for path, h in protocol['sources_sha256'].items():
        assert digest(path) == h, path
    for name, h in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == h, name
    # Outcomes are opened for scoring only after the complete models lock exists.
    outcomes = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    results = {}
    for day in [15, 16]:
        data = build(day)
        assert data['contract'] == json.loads((OUT/f'input_contract{day}.json').read_text(encoding='utf8'))
        held = data['dates'] == 20191100+day
        rows = np.flatnonzero(outcomes.pt_d.to_numpy() == 20191100+day)
        np.testing.assert_array_equal(rows, data['rows'][held])
        np.testing.assert_array_equal(rows, np.load(OUT/f'day{day}_indices.npy'))
        y = outcomes.label.to_numpy()[rows]
        assert len(y) == {15: 424780, 16: 421983}[day]
        assert int(y.sum()) == {15: 3523, 16: 3724}[day]
        pbase = base(day)
        position = data['x'][held, :data['nposition']]
        groups = pd.factorize(data['groups'][held], sort=True)[0]
        best = {'model_name': 'base', 'power': 0., 'mode': 'base', 'auc': rank_sum_auc(y, pbase)}
        best_p = pbase.copy()
        records = []
        for model in locks['models']:
            if model['day'] != day:
                continue
            path = OUT/(model['model_name']+'.npy')
            assert digest(path) == model['model_sha256']
            coef = np.load(path)
            delta = np.asarray(position@coef[:data['nposition']])
            for mode in MODES:
                for power in POWERS:
                    p = corrected(pbase, delta, groups, power, mode)
                    record = {'model_name': model['model_name'], 'power': power,
                              'mode': mode, 'auc': rank_sum_auc(y, p)}
                    records.append(record)
                    if record['auc'] > best['auc']:
                        best = record
                        best_p = p.copy()
        error = 0.
        if verifying:
            error = float(np.max(np.abs(best_p-np.load(OUT/f'day{day}_prediction.npy'))))
            assert error < 1e-12, error
        else:
            np.save(OUT/f'day{day}_prediction.npy', best_p)
        results[str(20191100+day)] = {'auc': best['auc'], 'best': best, 'records': records,
            'rows': len(y), 'clicks': int(y.sum()), 'maximum_replay_error': error,
            'evaluation_day_labels_fit': False, 'own_held_day_ad_labels_fit': False,
            'target_met': best['auc'] >= {15: .815, 16: .82}[day]}
        emit('conditional_position_scored', day=day, **best)
        del data
        gc.collect()
    report = {'status': 'passed' if verifying else 'evaluated', 'dates': results,
              'original_datasets_unchanged': True,
              'both_targets_met': all(r['target_met'] for r in results.values())}
    write(OUT/('verification.json' if verifying else 'result.json'), report)
    emit('conditional_position_finished', status=report['status'],
         dates={d: r['auc'] for d, r in results.items()})


def self_test():
    x = sparse.csr_matrix(np.array([[1., 0, .2], [0, 1., .4], [1., 0, .9],
                                    [0, 1., -.3], [1., 1., .5]]))
    y = np.array([1., 0, 0, 1., 1.])
    groups = np.array([0, 0, 1, 1, 1])
    starts = np.array([0, 2])
    counts = np.array([1., 2.])
    coef = np.array([.2, -.4, .1])
    penalty = np.array([1., 2., 3.])
    _, gradient = objective(coef, x, y, groups, starts, counts, penalty)
    numerical = np.zeros(3)
    for i in range(3):
        perturb = np.zeros(3)
        perturb[i] = 1e-6
        numerical[i] = (objective(coef+perturb, x, y, groups, starts, counts, penalty)[0]
                        -objective(coef-perturb, x, y, groups, starts, counts, penalty)[0])/2e-6
    np.testing.assert_allclose(gradient, numerical, rtol=1e-7, atol=1e-8)
    base_values = np.array([.01, .02, .03, .04, .02])
    delta = np.array([.3, -.3, -.1, .2, .4])
    adjusted = corrected(base_values, delta, groups, 1., 'preserve_user_mass')
    np.testing.assert_allclose(np.bincount(groups, weights=base_values),
                               np.bincount(groups, weights=adjusted), rtol=1e-12)
    emit('conditional_position_self_test_passed', gradient_max_error=float(np.max(np.abs(gradient-numerical))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else evaluate(True) if args.verify else fit()
