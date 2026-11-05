"""Synthetic large-query API recovery; compare with the Pool's stored baseline."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='2'
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRanker,Pool
from scipy.special import logit
from daily_auc_pair_core import sample_pairs
from catboost_within_ranker_v2 import PARAMS
from catboost_userday_residual import save_model,load_model
from fit_stacker import HERE,digest,write
from conditional_daily import emit
from rich_userday_activity import array_hash

OUT=HERE/'runs/daily_auc_pair_api_preflight_v2'


def main():
    assert not OUT.exists(),'Preserve completed and failed large-query preflights'
    params=dict(PARAMS,loss_function='PairLogit',depth=6,iterations=5,verbose=False)
    OUT.mkdir()
    paths=[Path(__file__),HERE/'daily_auc_pair_core.py',HERE/'test_daily_auc_pair_core.py',
        HERE/'catboost_within_ranker_v2.py',HERE/'catboost_userday_residual.py',
        HERE/'check_daily_auc_pair_api.py',HERE/'runs/daily_auc_pair_api_preflight_v1/protocol.json']
    write(OUT/'protocol.json',dict(params=params,synthetic_data_only=True,
        previous_failure='The original check compared the prediction to the supplied float64 baseline rather than the baseline actually stored by CatBoost Pool',
        sources_sha256={str(path):digest(path) for path in paths}))
    nusers=8504
    rng=np.random.default_rng(151646)
    keys=np.column_stack([np.repeat([20191113,20191114],nusers),np.tile(np.arange(nusers),2)])
    sizes=rng.integers(10,80,size=len(keys))
    counts=np.minimum(sizes,rng.poisson(.4,size=len(keys)))
    training,pairs,groups=sample_pairs(keys,sizes,counts,20191115)
    frame=pd.DataFrame(dict(user=keys[training,1],news=rng.normal(size=len(training)),exposures=np.log1p(sizes[training])))
    baseline=logit(np.clip(.008*np.exp(.2*frame.news.to_numpy()),1e-5,.2))
    pool=Pool(frame,label=counts[training],group_id=groups,pairs=pairs,baseline=baseline,cat_features=['user'])
    stored=np.asarray(pool.get_baseline()).reshape(-1)
    model=CatBoostRanker(**params)
    model.fit(pool)
    sample=Pool(frame,cat_features=['user'])
    raw=model.predict(sample,thread_count=2)
    with_baseline=model.predict(pool,thread_count=2)
    baseline_error=float(np.max(np.abs(with_baseline-raw-stored)))
    diagnostics=dict(stored_baseline_dtype=str(stored.dtype),
        stored_vs_supplied_baseline_error=float(np.max(np.abs(stored-baseline))),
        prediction_vs_supplied_baseline_error=float(np.max(np.abs(with_baseline-raw-baseline))),
        prediction_vs_stored_baseline_error=baseline_error)
    write(OUT/'diagnostics.json',diagnostics)
    emit('daily_auc_pair_baseline_diagnostics',**diagnostics)
    np.testing.assert_array_equal(stored,baseline.astype('float32'))
    assert baseline_error<1e-12
    save_model(model,OUT/'large_query_smoke.cbm')
    replay=load_model(OUT/'large_query_smoke.cbm').predict(sample,prediction_type='RawFormulaVal',thread_count=2)
    error=float(np.max(np.abs(raw-replay)))
    assert error==0. and np.isfinite(raw).all()
    report=dict(status='passed',synthetic_data_only=True,largest_query=nusers,rows=len(training),pairs=len(pairs),
        explicit_pairs_with_count_targets_supported=True,pairs_sha256=array_hash(pairs),
        pool_baseline_added_by_predict=True,pool_stores_baseline_as_float32=True,
        baseline_maximum_error=baseline_error,serialization_maximum_error=error,
        inferred17_labels_read=False,held_auc_evaluated=False)
    write(OUT/'verification.json',report)
    emit('daily_auc_pair_large_query_api_passed',**report)


if __name__=='__main__':
    main()
