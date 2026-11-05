"""Fit full labeled-date preprocessing, keeping final advertisements unlabeled."""
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=ROOT/'2026ml'
sys.path.insert(0,str(HERE.parent/'auc085'))
from predict_final import SavedFeatureTransform
from benchmark_core import category_key
from target_statistics import SPECS
from interest_data import build_past_statistics
from prepare_exposure_features import build_exposure_features
from prepare_snapshot_features import build_features as build_snapshot_features

OUT=HERE/'full_refit_cache'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    if OUT.exists():raise FileExistsError('Preserving full-refit preprocessing')
    started=time.perf_counter();transform=SavedFeatureTransform();old=copy.deepcopy(transform.manifests['signed'])
    vocabs=copy.deepcopy(transform.vocabs);groups=old['group_for_field']
    print('EXPAND_FULL_TRAINING_VOCABULARY',flush=True)
    usecols=transform.raw_categories+transform.raw_sequences+['pt_d']
    keys=set()
    for chunk in pd.read_csv(DATA/'train.csv',usecols=usecols,dtype={f:str for f in transform.raw_sequences},chunksize=60000):
        for field in transform.raw_categories:
            vocab=vocabs[groups[field]]
            for raw in chunk[field].unique():
                key=category_key(raw)
                if key not in vocab:vocab[key]=len(vocab)+2
        for field in transform.raw_sequences:
            vocab=vocabs[groups[field]]
            for value in chunk[field].fillna('').unique():
                for token in value.split('^'):
                    key=category_key(token)
                    if token and key not in vocab:vocab[key]=len(vocab)+2
        keys.update((chunk.pt_d.to_numpy(dtype=np.int64)*10000000+chunk.user_id.to_numpy(dtype=np.int64)).tolist())
    fit=transform.profiles.reindex(sorted(keys))
    for field in old['categorical_fields'][24:]:
        vocab=vocabs[groups[field]]
        for raw in fit[field].fillna('__MISSING__').unique():
            key=category_key(raw)
            if key not in vocab:vocab[key]=len(vocab)+2
    for field in old['sequence_fields'][7:]:
        vocab=vocabs[groups[field]]
        for raw in fit[field].fillna('').unique():
            for token in str(raw).split('^'):
                key=category_key(token)
                if token and key not in vocab:vocab[key]=len(vocab)+2
    del fit,keys
    transform.vocabs=vocabs
    transform.raw_vocab={field:{int(raw):code for raw,code in vocabs[groups[field]].items()} for field in transform.raw_categories}
    sizes={group:len(vocab)+2 for group,vocab in vocabs.items()}
    for branch in transform.manifests:transform.manifests[branch]['vocabulary_sizes']=sizes
    transform.manifest['vocabulary_sizes']=sizes
    ntrain=2526320;nfinal=404182;n=ntrain+nfinal
    OUT.mkdir(parents=True)
    arrays={'categories':open_memmap(OUT/'categories.npy',mode='w+',dtype='int32',shape=(n,27)),
            'sequences':open_memmap(OUT/'sequences.npy',mode='w+',dtype='int32',shape=(n,12,5)),
            'numeric':open_memmap(OUT/'numeric.npy',mode='w+',dtype='float32',shape=(n,166)),
            'label':open_memmap(OUT/'label.npy',mode='w+',dtype='int8',shape=(n,)),
            'date':open_memmap(OUT/'date.npy',mode='w+',dtype='int32',shape=(n,)),
            'log_id':open_memmap(OUT/'log_id.npy',mode='w+',dtype='int32',shape=(n,)),
            'user_id':open_memmap(OUT/'user_id.npy',mode='w+',dtype='int32',shape=(n,))}
    offset=0
    for name in ['train.csv','final_test.csv']:
        cols=transform.read_columns+(['label'] if name=='train.csv' else [])
        for chunk in pd.read_csv(DATA/name,usecols=cols,dtype={field:str for field in transform.raw_sequences},chunksize=30000):
            cats,seqs,numeric=transform.transform(chunk);end=offset+len(chunk)
            arrays['categories'][offset:end]=cats;arrays['sequences'][offset:end]=seqs
            # Undo the previous training-only scaler before fitting the full-training scaler.
            raw=numeric['signed'][:,:39].copy()
            for field,scale in old['numeric_scaler'].items():
                col=old['numeric_fields'].index(field)
                if col<39:raw[:,col]=raw[:,col]*scale['std']+scale['mean']
            arrays['numeric'][offset:end,:39]=raw
            arrays['label'][offset:end]=chunk.label.to_numpy() if name=='train.csv' else -1
            for field in ['date','log_id','user_id']:arrays[field][offset:end]=chunk['pt_d' if field=='date' else field].to_numpy()
            offset=end;print(f'ENCODE_FULL {offset}/{n}',flush=True)
    if offset!=n or not np.all(arrays['label'][ntrain:]==-1):raise AssertionError('Full data alignment/unknown labels failure')
    scalers={}
    for field in old['numeric_scaler']:
        col=old['numeric_fields'].index(field)
        if col>=39:continue
        values=arrays['numeric'][:ntrain,col];mean=float(values.mean());std=max(float(values.std()),1e-6)
        arrays['numeric'][:,col]=(arrays['numeric'][:,col]-mean)/std;scalers[field]={'mean':mean,'std':std}
    del transform;gc.collect()
    fields=sorted({f for spec,_ in SPECS for f in spec}|{'user_id','pt_d','label'})
    labeled=pd.read_csv(DATA/'train.csv',usecols=fields,dtype='int32')
    final=pd.read_csv(DATA/'final_test.csv',usecols=[f for f in fields if f!='label'],dtype='int32');final['label']=-1
    allads=pd.concat([labeled,final],ignore_index=True)
    strict=build_past_statistics(allads,20191117,SPECS)
    for col,field in enumerate(old['numeric_fields'][39:]):
        values=strict[:ntrain,col];mean=float(values.mean());std=max(float(values.std()),1e-6)
        arrays['numeric'][:,39+col]=(strict[:,col]-mean)/std;scalers[field]={'mean':mean,'std':std}
    del strict,labeled,final,allads;gc.collect()
    exposure_fields=['pt_d','user_id','task_id','adv_id','adv_prim_id','spread_app_id','app_second_class','slot_id']
    data=pd.concat([pd.read_csv(DATA/name,usecols=exposure_fields,dtype='int32') for name in ['train.csv','final_test.csv']],ignore_index=True)
    exposure,names_exposure=build_exposure_features(data)
    oldexposure=np.load(HERE/'exposure_features.npy',mmap_mode='r')
    np.testing.assert_array_equal(exposure[:ntrain],oldexposure)
    del data;gc.collect()
    snapshot_fields=['user_id','pt_d','task_id','adv_prim_id','spread_app_id']+[f'ad_{a}_list_v00{i}' for a in ['click','close'] for i in [1,2,3]]
    data=pd.concat([pd.read_csv(DATA/name,usecols=snapshot_fields,dtype={f:str for f in snapshot_fields if 'list' in f}) for name in ['train.csv','final_test.csv']],ignore_index=True)
    snapshot,names_snapshot=build_snapshot_features(data)
    np.testing.assert_array_equal(snapshot[:ntrain],np.load(HERE/'snapshot_features.npy',mmap_mode='r'))
    del data;gc.collect()
    extra_names=names_exposure+names_snapshot
    for source,start in [(exposure,75),(snapshot,118)]:
        for col in range(source.shape[1]):
            values=source[:,col];mean=float(values[:ntrain].mean());std=max(float(values[:ntrain].std()),1e-6)
            arrays['numeric'][:,start+col]=(values-mean)/std;scalers[(old['numeric_fields']+extra_names)[start+col]]={'mean':mean,'std':std}
    np.save(OUT/'exposure_raw.npy',exposure);np.save(OUT/'snapshot_raw.npy',snapshot)
    for value in arrays.values():value.flush()
    if not np.isfinite(arrays['numeric']).all():raise AssertionError('Nonfinite full refit input')
    manifest=copy.deepcopy(old)
    manifest.update(version='full_chronological_refit_v1',rows=n,train_rows=ntrain,final_rows=nfinal,validation_day=None,
        train_clicks=int(arrays['label'][:ntrain].sum()),numeric_fields=old['numeric_fields']+extra_names,
        numeric_scaler=scalers,vocabulary_sizes=sizes,training_fraction=1.,
        source_rule='Supplied same-day source snapshot plus strictly historical source; no final advertisement labels',
        advertisement_statistics='Per-row strictly earlier dates; initial prior .0084 mass1000',
        exposure_rule='Full supplied unlabeled current-day inputs for each day, including final day',
        final_labels_available=False,first39_schema=old['numeric_fields'][:39],
        input_sha256={name:digest(DATA/name) for name in ['train.csv','train_feeds.csv','final_test.csv','test_feeds.csv']},
        source_sha256={str(p.resolve()):digest(p) for p in [Path(__file__),HERE/'interest_data.py',HERE/'prepare_exposure_features.py',HERE/'prepare_snapshot_features.py']},
        preparation_seconds=time.perf_counter()-started)
    (OUT/'vocabs.json').write_text(json.dumps(vocabs,ensure_ascii=False),encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FULL_CACHE_COMPLETE '+json.dumps({'train_rows':ntrain,'final_rows':nfinal,'numeric_dim':166,'seconds':manifest['preparation_seconds']}),flush=True)


if __name__=='__main__':main()
