"""Independent gather/removal fixture and actual native serialization checks."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'

from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
import full_negative_context_no_ids_lgb_v1 as experiment
from dense_categorical_context_core_v1 import frame as complete_frame
from conditional_daily import read
from fit_stacker import HERE, digest, write
from rich_userday_activity import lgb

OUT = HERE / 'runs/context_no_ids_native_fixture_v1'


def tables(day, independent=False):
    path = experiment.PARENT / f'category_dictionaries_day{day}{"_independent" if independent else ""}.npz'
    locked = read(experiment.PARENT / 'inputs_locked.json')[str(day)]
    assert digest(path) == locked['independent_dictionary_sha256' if independent else 'dictionary_sha256']
    with np.load(path) as cache:
        return {name:cache[name] for name in cache.files}


def main():
    assert not OUT.exists(), 'Preserve prior fixture runs and failures'
    OUT.mkdir()
    rows = []
    for day in (15,16):
        for partition in ('train','held'):
            folder = experiment.dense.folder(day)
            total = len(np.load(folder / ('train.npy' if partition=='train' else 'held.npy')))
            selected = np.unique(np.linspace(0,total-1,2000,dtype='int64'))
            left,cats = experiment.frame(folder,partition,tables(day),selected)
            right,other = experiment.frame(experiment.dense.folder(day,True),partition,tables(day,True),selected,True)
            reference,reference_cats = complete_frame(folder,partition,tables(day),selected)
            expected = reference.drop(columns=experiment.REMOVE)
            assert cats == other == [n for n in reference_cats if n not in experiment.REMOVE]
            assert list(left) == list(right) == list(expected) and len(left.columns)==389
            np.testing.assert_array_equal(left.to_numpy(),right.to_numpy())
            np.testing.assert_array_equal(left.to_numpy(),expected.to_numpy())
            assert not set(experiment.REMOVE) & set(left)
            rows.append(dict(day=day,partition=partition,rows=len(left),columns=389,
                explicit_full_frame_column_removal_exact=True,independent_gather_and_category_mapping_exact=True))
    folder = experiment.dense.folder(15)
    labels = np.load(folder / 'labels.npy')
    positive,negative = np.flatnonzero(labels==1),np.flatnonzero(labels==0)
    selected = np.sort(np.r_[positive[np.linspace(0,len(positive)-1,512,dtype='int64')],
        negative[np.linspace(0,len(negative)-1,3584,dtype='int64')]])
    dates = np.load(folder/'dates.npy')[np.load(folder/'train.npy')[selected]]
    assert np.all(dates<20191115)
    data,cats = experiment.frame(folder,'train',tables(15),selected)
    held,_ = experiment.frame(folder,'held',tables(15),np.arange(1000))
    independent,_ = experiment.frame(experiment.dense.folder(15,True),'held',tables(15,True),np.arange(1000),True)
    models = []
    for arm in experiment.ARMS:
        params = dict(experiment.dense.parameters(arm),min_data_in_leaf=16,min_data_per_group=4,
            lambda_l2=4.,cat_l2=4.,cat_smooth=4.)
        dataset = lgb.Dataset(data,label=labels[selected],categorical_feature=cats,params=params)
        def callback(environment):
            trees = environment.iteration+1
            if trees in (5,12,20):
                path=OUT/f'{arm["name"]}_trees{trees}.txt'
                environment.model.save_model(experiment.dense.native_path(path),num_iteration=trees)
        model = lgb.train(params,dataset,num_boost_round=20,callbacks=[callback],keep_training_booster=True)
        for trees in (5,12,20):
            path = OUT/f'{arm["name"]}_trees{trees}.txt'
            restored = lgb.Booster(model_file=experiment.dense.native_path(path))
            assert restored.current_iteration()==trees and restored.feature_name()==list(data)
            for a,b in [(data,data),(held,independent)]:
                np.testing.assert_array_equal(model.predict(a,raw_score=True,num_iteration=trees,num_threads=2),
                    restored.predict(b,raw_score=True,num_threads=2))
            models.append(dict(arm=arm['name'],trees=trees,model_sha256=digest(path),serialization_maximum_error=0.))
    assert not torch.cuda.is_initialized()
    sources=[Path(__file__),HERE/'full_negative_context_no_ids_lgb_v1.py',
        HERE/'dense_categorical_context_core_v1.py',HERE/'full_negative_dense_context_lgb_v1.py']
    report=dict(status='passed',timestamp_utc=datetime.now(timezone.utc).isoformat(),projection_fixtures=rows,
        actual_native_models=models,train_rows=len(data),held_feature_rows=len(held),columns=389,
        training_max_date=int(dates.max()),removed_features=experiment.REMOVE,
        dictionary_fits_remain_strictly_before_each_date=True,held_advertisement_labels_read=False,
        inferred17_labels_read=False,gpu_memory_allocated=False,auc_evaluated=False,
        production_capacity_proven=False,full_population_projection_proven=False,
        sources_sha256={str(path):digest(path) for path in sources})
    write(OUT/'verification.json',report)
    print(dict(status='passed',projection_rows=sum(row['rows'] for row in rows),columns=389,
        native_saved_and_reloaded_models=len(models),held_labels_read=False,auc_evaluated=False),flush=True)


if __name__=='__main__':
    main()
