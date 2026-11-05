"""Strict-prior posterior calibration of rare unordered task-history evidence."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from fit_stacker import HERE,ROOT,digest,write
from conditional_daily import read,emit
from rich_userday_activity import array_hash
from verify_stacker import rank_sum_auc
from transition_posterior_core_v1 import FIELDS,task_bins,fit_rates,prior_shift,predict

OUT = HERE/'runs/transition_posterior_v1'
BASES = {name:HERE/'runs'/name for name in ['pointwise_compact_v1','compose_probability_quantiles_v1']}
STRENGTHS = [5.,20.,50.]
WEIGHTS = [.25,.5,.75,1.]


def inputs(independent=False):
    raw = pd.read_csv(ROOT/'2026ml/train.csv',usecols=FIELDS+['label'],
        dtype={'ad_click_list_v001':str})
    rows = np.flatnonzero(raw.pt_d.to_numpy() <= 20191116)
    local = raw.iloc[rows].reset_index(drop=True)
    bins = task_bins(local[FIELDS],independent)
    return rows,local.pt_d.to_numpy(),local.label.to_numpy(dtype='int32'),bins


def rates_for(dates,labels,bins,independent=False):
    rates = {}
    for day in [15,16]:
        cutoff = 20191100+day
        hidden = labels.copy()
        hidden[dates>=cutoff] = -1
        rates[str(day)] = {}
        for recent in [False,True]:
            for strength in STRENGTHS:
                key = f'{"recent" if recent else "all"}_shrink{strength:g}'
                rates[str(day)][key] = fit_rates(bins,dates,hidden,cutoff,strength,recent,independent)
    return rates


def check_fit():
    protocol = read(OUT/'protocol.json')
    for filename,expected in protocol['sources_sha256'].items():
        assert digest(filename)==expected,filename
    for filename,expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/filename)==expected,filename
    lock = read(OUT/'rates_locked.json')
    assert lock['protocol_sha256']==digest(OUT/'protocol.json')
    assert lock['inputs_file_sha256']==digest(OUT/'inputs.npz')
    return protocol,lock


def fit():
    assert not OUT.exists(),'Preserve previous fits'
    paths = [Path(__file__),HERE/'transition_posterior_core_v1.py',
        HERE/'test_transition_posterior_core_v1.py',HERE/'run_transition_posterior_v1.ps1',
        HERE/'fit_stacker.py',HERE/'conditional_daily.py',HERE/'rich_userday_activity.py',HERE/'verify_stacker.py']
    hashes = read(HERE/'runs/compose_unordered_masks_v1/protocol.json')['raw_datasets_sha256']
    OUT.mkdir()
    write(OUT/'protocol.json',dict(strengths=STRENGTHS,weights=WEIGHTS,thresholds=[1,2,3,4],
        baselines=list(BASES),training='Per-bin Bernoulli rates using strictly earlier real advertisement labels',
        features='Unordered same-user/date task-history additions; no assumed within-day row chronology',
        bins=['no_addition','addition','single_added_task','single_task_unique_exposure','unique_without_removal'],
        posterior='Beta shrinkage toward historical overall click rate; all-prior and latest-prior-date windows',
        selection='Both-date candidate hashes locked before scoring; repeatedly reused development dates',
        evaluation_day_labels_fit=False,inferred17_labels_read=False,
        sources_sha256={str(p):digest(p) for p in paths},raw_datasets_sha256=hashes))
    rows,dates,labels,bins = inputs()
    np.savez_compressed(OUT/'inputs.npz',rows=rows,dates=dates,bins=bins)
    rates = rates_for(dates,labels,bins)
    write(OUT/'rates_locked.json',dict(protocol_sha256=digest(OUT/'protocol.json'),
        inputs_file_sha256=digest(OUT/'inputs.npz'),rates=rates,
        labels_at_or_after_own_prediction_day_always_masked=True))
    check_fit()
    emit('transition_posteriors_fitted',max_dates={d:r['all_shrink20']['max_training_date'] for d,r in rates.items()},
         historical14_bins=np.bincount(bins[dates==20191114],minlength=5).tolist())


def baseline_inputs(day,bins,dates,rows):
    selected = np.flatnonzero(dates==20191100+day)
    actual_rows = rows[selected]
    values = {}
    for name,directory in BASES.items():
        assert read(directory/'verification.json')['status']=='passed'
        np.testing.assert_array_equal(actual_rows,np.load(directory/f'day{day}_indices.npy'))
        values[name] = np.load(directory/f'day{day}_prediction.npy')
    return actual_rows,bins[selected],values


def candidates(day,bins,bases,rates,independent=False):
    for name,base in bases.items():
        yield dict(base=name,model='identity',threshold=0,weight=0.),base.copy()
        shifts = {}
        for model,record in rates[str(day)].items():
            rate = record['global_rate']
            if rate not in shifts:
                shifts[rate] = prior_shift(base,rate)
                assert abs(float(np.mean(1/(1+np.exp(-(np.log(base/(1-base))+shifts[rate])))))-rate)<1e-12
            for threshold in [1,2,3,4]:
                for weight in WEIGHTS:
                    yield dict(base=name,model=model,threshold=threshold,weight=weight),predict(
                        base,bins,record,threshold,weight,independent,shifts[rate])


def key(record):
    return f'{record["base"]}__{record["model"]}__t{record["threshold"]}__w{record["weight"]:g}'


def check_baselines():
    frozen = read(OUT/'baselines_locked.json')
    for filename,expected in frozen['sources_sha256'].items():
        assert digest(filename)==expected,filename
    return frozen


def prepare():
    _,lock = check_fit()
    assert not (OUT/'baselines_locked.json').exists(),'Preserve prediction locks'
    paths = []
    for directory in BASES.values():
        assert read(directory/'verification.json')['status']=='passed'
        for filename,expected in read(directory/'protocol.json')['sources_sha256'].items():
            assert digest(filename)==expected,filename
        paths += [directory/'protocol.json',directory/'verification.json']
        for day in [15,16]:
            paths += [directory/f'day{day}_indices.npy',directory/f'day{day}_prediction.npy']
    write(OUT/'baselines_locked.json',dict(sources_sha256={str(p):digest(p) for p in paths},
        rates_sha256=digest(OUT/'rates_locked.json')))
    saved = np.load(OUT/'inputs.npz')
    predictions = {}
    for day in [15,16]:
        rows,bins,bases = baseline_inputs(day,saved['bins'],saved['dates'],saved['rows'])
        np.save(OUT/f'day{day}_indices.npy',rows)
        predictions[str(day)] = {key(r):array_hash(p) for r,p in candidates(day,bins,bases,lock['rates'])}
    write(OUT/'predictions_locked.json',dict(baselines_sha256=digest(OUT/'baselines_locked.json'),
        rates_sha256=digest(OUT/'rates_locked.json'),dates=predictions,held_labels_used_to_select_recipes=False))
    emit('transition_posterior_predictions_locked',candidates={d:len(v) for d,v in predictions.items()})


def verify():
    _,lock = check_fit()
    check_baselines()
    assert not (OUT/'verification.json').exists(),'Preserve previous verification'
    frozen = read(OUT/'predictions_locked.json')
    assert frozen['baselines_sha256']==digest(OUT/'baselines_locked.json')
    assert frozen['rates_sha256']==digest(OUT/'rates_locked.json')
    rows,dates,labels,bins = inputs(True)
    saved = np.load(OUT/'inputs.npz')
    for name,value in [('rows',rows),('dates',dates),('bins',bins)]:
        np.testing.assert_array_equal(saved[name],value)
    assert rates_for(dates,labels,bins,True)==lock['rates']
    reports = {}
    for day in [15,16]:
        indices,local,bases = baseline_inputs(day,bins,dates,rows)
        np.testing.assert_array_equal(indices,np.load(OUT/f'day{day}_indices.npy'))
        y = labels[dates==20191100+day]
        assert len(y)=={15:424780,16:421983}[day] and y.sum()=={15:3523,16:3724}[day]
        records = []
        maximum,best_value = 0.,-1.
        for (r,p),(other,q) in zip(candidates(day,local,bases,lock['rates']),
                                  candidates(day,local,bases,lock['rates'],True)):
            assert r==other and array_hash(p)==frozen['dates'][str(day)][key(r)]
            error = float(np.max(np.abs(p-q)))
            assert error<2e-13
            maximum = max(maximum,error)
            value = rank_sum_auc(y,p)
            record = dict(r,auc=value)
            records.append(record)
            if value>best_value:
                best_value,best,best_prediction = value,record,p.copy()
        assert abs(roc_auc_score(y,best_prediction)-best_value)<1e-12
        np.save(OUT/f'day{day}_prediction.npy',best_prediction)
        reports[str(20191100+day)] = dict(auc=best_value,best=best,records=records,rows=len(y),clicks=int(y.sum()),
            bin_counts=np.bincount(local,minlength=5).tolist(),evaluation_day_labels_fit=False,
            held_labels_used_to_select_fixed_recipe=True,all_both_date_candidates_locked_before_scoring=True,
            independent_bins_rates_and_odds_replayed=True,maximum_independent_prediction_error=maximum,
            target_met=best_value>={15:.815,16:.82}[day])
        emit('transition_posterior_scored',day=day,best=best)
    check_fit();check_baselines()
    write(OUT/'verification.json',dict(status='passed',dates=reports,original_datasets_unchanged=True,
        both_targets_met=all(r['target_met'] for r in reports.values())))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=['fit','prepare','verify'])
    {'fit':fit,'prepare':prepare,'verify':verify}[parser.parse_args().phase]()
