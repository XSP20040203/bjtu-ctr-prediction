"""Independent CPU boosted-tree comparator with chronological input boundaries."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,log_loss,average_precision_score

HERE=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main(args):
    output=HERE/'lgb_results'/args.tag
    if output.exists():raise FileExistsError('Preserving existing LightGBM experiment')
    cache=HERE.parent/'user_personalization/cache_day15_stats' if args.day==20191115 else HERE.parent/'auc085/cache_rich_stats_time16'
    manifest=read(cache/'manifest.json');train=np.load(cache/'train_indices.npy');valid=np.load(cache/'valid_indices.npy')
    dates=np.load(cache/'date.npy',mmap_mode='r')
    if dates[train].max()>=args.day or not np.all(dates[valid]==args.day):raise AssertionError('Invalid date split')
    labels=np.load(cache/'label.npy',mmap_mode='r');category=np.load(cache/'categories.npy',mmap_mode='r');sequence=np.load(cache/'sequences.npy',mmap_mode='r')
    numeric=np.load(HERE/f'cache_day{str(args.day)[-2:]}'/'numeric.npy',mmap_mode='r')
    selected=np.r_[train,valid];columns={};cat_fields=[]
    for index,field in enumerate(manifest['categorical_fields']):
        if args.no_identity and field in ['user_id','task_id','adv_id']:continue
        columns[field]=category[selected,index].astype(np.int32);cat_fields.append(field)
    for index,field in enumerate(manifest['numeric_fields']):columns[field]=numeric[selected,index]
    for index,field in enumerate(manifest['sequence_fields']):
        if args.no_sequence_ids:continue
        for position in range(5):
            name=field+'_token'+str(position);columns[name]=sequence[selected,index,position].astype(np.int32);cat_fields.append(name)
    for name in ['exposure_features','snapshot_features']:
        p=HERE/(name+'.npy');values=np.load(p,mmap_mode='r');meta=read(p.with_suffix('.json'))
        if meta['advertisement_labels_loaded'] is not False:raise AssertionError('Input target boundary')
        for index,field in enumerate(meta['features']):columns[field]=values[selected,index]
    if args.extra_features:
        values=np.load(args.extra_features,mmap_mode='r');meta=read(args.extra_features.with_suffix('.json'))
        historical_only=meta.get('advertisement_validation_labels_used') is False and meta.get('validation_day')==args.day
        if meta.get('advertisement_labels_loaded') is not False and not historical_only:raise ValueError('Additional inputs require no current or validation advertisement labels')
        for index,field in enumerate(meta['features']):columns[field]=np.asarray(values[selected,index],dtype=np.float32)
    frame=pd.DataFrame(columns);del columns;gc.collect();ntrain=len(train)
    print(f'LGB_FEATURES train={ntrain} valid={len(valid)} fields={len(frame.columns)} categories={len(cat_fields)}',flush=True)
    params={'objective':'binary','metric':['auc','binary_logloss'],'learning_rate':args.learning_rate,'num_leaves':args.leaves,
            'max_depth':-1,'min_data_in_leaf':args.min_leaf,'max_bin':63,'feature_fraction':.9,'bagging_fraction':.9,'bagging_freq':1,
            'lambda_l2':10.,'cat_l2':20.,'cat_smooth':20.,'max_cat_threshold':32,'verbosity':-1,'num_threads':6,
            'seed':42,'deterministic':True,'force_col_wise':True,'feature_pre_filter':False}
    fit=lgb.Dataset(frame.iloc[:ntrain],label=labels[train],categorical_feature=cat_fields,free_raw_data=True,params={'max_bin':63})
    check=lgb.Dataset(frame.iloc[ntrain:],label=labels[valid],categorical_feature=cat_fields,reference=fit,free_raw_data=True)
    started=time.perf_counter();history={};output.mkdir(parents=True)
    (output/'protocol.json').write_text(json.dumps({'params':params,'rounds':args.rounds,'early_stopping_rounds':100,
            'validation_day':args.day,'train_rows':len(train),'valid_rows':len(valid),'features':frame.columns.tolist(),
            'categorical_fields':cat_fields,'extra_features':str(args.extra_features) if args.extra_features else None,'no_identity':args.no_identity,'no_sequence_ids':args.no_sequence_ids,
            'source_rule':'Supplied same-day source and input-only day exposures; strictly previous ad labels',
            'final_labels_available':False},ensure_ascii=False,indent=2),encoding='utf-8')
    model=lgb.train(params,fit,num_boost_round=args.rounds,valid_sets=[check],valid_names=['valid'],
                    callbacks=[lgb.early_stopping(100,first_metric_only=True),lgb.log_evaluation(50),lgb.record_evaluation(history)])
    prediction=model.predict(frame.iloc[ntrain:],num_iteration=model.best_iteration)
    (output/'model.txt').write_text(model.model_to_string(),encoding='utf-8');np.save(output/'validation_prediction.npy',prediction)
    restored=lgb.Booster(model_str=(output/'model.txt').read_text(encoding='utf-8'));replay=restored.predict(frame.iloc[ntrain:])
    error=float(np.max(np.abs(replay-prediction)))
    if error>1e-12:raise AssertionError('LGB checkpoint mismatch')
    record={'status':'complete_and_verified','tag':args.tag,'day':args.day,'auc':float(roc_auc_score(labels[valid],prediction)),
            'logloss':float(log_loss(labels[valid],prediction,labels=[0,1])),'average_precision':float(average_precision_score(labels[valid],prediction)),
            'mean_prediction':float(prediction.mean()),'best_iteration':model.best_iteration,'fields':len(frame.columns),
            'params':params,'cache':str(cache.resolve()),'train_rows':len(train),'valid_rows':len(valid),'maximum_reload_error':error,
            'training_seconds':time.perf_counter()-started,'extra_features':str(args.extra_features) if args.extra_features else None,'no_identity':args.no_identity,'no_sequence_ids':args.no_sequence_ids,
            'model_sha256':hashlib.sha256((output/'model.txt').read_bytes()).hexdigest(),'final_labels_used':False,
            'validation_scope':'Whole future day; repeatedly used development validation, not independent final test'}
    (output/'metrics.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'history.json').write_text(json.dumps(history),encoding='utf-8')
    pd.DataFrame({'feature':model.feature_name(),'gain':model.feature_importance('gain')}).sort_values('gain',ascending=False).to_csv(output/'importance.csv',index=False)
    print('LGB_COMPLETE '+json.dumps(record),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--day',type=int,choices=[20191115,20191116],required=True);parser.add_argument('--tag',required=True)
    parser.add_argument('--leaves',type=int,default=31);parser.add_argument('--min-leaf',type=int,default=200);parser.add_argument('--learning-rate',type=float,default=.03)
    parser.add_argument('--no-identity',action='store_true');parser.add_argument('--no-sequence-ids',action='store_true')
    parser.add_argument('--rounds',type=int,default=2400);parser.add_argument('--extra-features',type=Path);main(parser.parse_args())
