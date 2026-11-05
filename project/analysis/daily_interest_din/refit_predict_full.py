"""Fixed-budget full-data refit of the cross-day exposure model and final prediction."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from catboost import CatBoostClassifier,Pool

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CACHE=HERE/'full_refit_cache'
OUT=HERE/'full_refit_models'
EXPORT=ROOT/'output/chronological_exposure_v1'
sys.path.insert(0,str(HERE.parent/'model_comparison'))
from bounded_id_models import BoundedIDDCN


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def tree_frame(rows,arrays,manifest):
    columns={};categorical=[]
    for col,name in enumerate(manifest['categorical_fields']):
        columns[name]=pd.Categorical(arrays['categories'][rows,col]);categorical.append(name)
    for col,name in enumerate(manifest['numeric_fields'][:75]):columns[name]=arrays['numeric'][rows,col]
    for col,name in enumerate(manifest['sequence_fields']):
        for index in range(5):
            field=name+'_token'+str(index);columns[field]=pd.Categorical(arrays['sequences'][rows,col,index]);categorical.append(field)
    exposure=np.load(CACHE/'exposure_raw.npy',mmap_mode='r');snapshot=np.load(CACHE/'snapshot_raw.npy',mmap_mode='r')
    for col,name in enumerate(read(HERE/'exposure_features.json')['features']):columns[name]=exposure[rows,col]
    for col,name in enumerate(read(HERE/'snapshot_features.json')['features']):columns[name]=snapshot[rows,col]
    return pd.DataFrame(columns),categorical


def main():
    started=time.perf_counter()
    if EXPORT.exists():raise FileExistsError('Preserving existing final refit exports')
    OUT.mkdir(exist_ok=True)
    manifest=read(CACHE/'manifest.json')
    arrays={name:np.load(CACHE/(name+'.npy'),mmap_mode='r') for name in ['categories','sequences','numeric','label','date','log_id','user_id']}
    n=manifest['train_rows'];nfinal=manifest['final_rows'];train=np.arange(n,dtype=np.int32);final=np.arange(n,n+nfinal,dtype=np.int32)
    if not np.all(arrays['label'][final]==-1) or arrays['date'][train].max()!=20191116 or not np.all(arrays['date'][final]==20191117):raise AssertionError('Final label or date boundary failure')
    protocol={'status':'fixed_before_final_inference','train_rows':n,'train_clicks':int(arrays['label'][train].sum()),
        'dcn_epochs':1,'tree_iterations':2000,'tree_depth':6,'seed':42,'ensemble':{'dcn':.5,'tree':.5},
        'budget_source':'Both chronological development dates selected DCN epoch1 and snapshot-exposure trees about2000; no final labels',
        'advertisement_statistics':'strictly earlier dates for every training/test row','same_day_inputs':'Provided full current-day source, exposures and ad-history snapshots',
        'full_training_refit':True,'final_labels_available':False,'final_auc':None}
    if (OUT/'protocol.json').exists() and read(OUT/'protocol.json')!=protocol:raise ValueError('Full refit protocol changed')
    write(OUT/'protocol.json',protocol)
    predictions={}
    tree_path=OUT/'catboost.cbm'
    if not tree_path.exists():
        frame,cat=tree_frame(train,arrays,manifest);pool=Pool(frame,arrays['label'][train],cat_features=cat);del frame;gc.collect()
        params=dict(iterations=2000,depth=6,learning_rate=.04,l2_leaf_reg=20,loss_function='Logloss',random_seed=42,
                    task_type='GPU',border_count=64,one_hot_max_size=12,max_ctr_complexity=1,gpu_ram_part=.60,
                    gpu_cat_features_storage='CpuPinnedMemory',pinned_memory_size='512mb',thread_count=6,allow_writing_files=False)
        model=CatBoostClassifier(**params);model.fit(pool,verbose=100);model.save_model(str(tree_path))
        write(OUT/'tree_parameters.json',params);del model,pool;gc.collect()
    tree=CatBoostClassifier();tree.load_model(str(tree_path));parts=[]
    for start in range(0,len(final),25000):
        rows=final[start:start+25000];frame,cat=tree_frame(rows,arrays,manifest)
        if frame.columns.tolist()!=tree.feature_names_:raise AssertionError('Tree feature order mismatch')
        parts.append(tree.predict_proba(Pool(frame,cat_features=cat),thread_count=6)[:,1]);print(f'FINAL_TREE {min(start+25000,len(final))}/{len(final)}',flush=True)
    predictions['tree']=np.concatenate(parts);np.save(OUT/'tree_final_prediction.npy',predictions['tree'])
    del tree,frame,parts;gc.collect()
    torch.set_num_threads(6);torch.manual_seed(42);np.random.seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    options={'numeric_dim':166,'embedding_dim':8,'hidden_dims':[128,64],'dropout':.1,'unknown_dropout':.01,'cross_layers':2,'group_norm_caps':{'user_id':.2,'ad_task':.2,'adv_id':.2}}
    model=BoundedIDDCN(manifest['categorical_fields'],manifest['sequence_fields'],manifest['group_for_field'],manifest['vocabulary_sizes'],initial_ctr=float(arrays['label'][train].mean()),**options).to(device)
    def batch(rows):
        return torch.as_tensor(arrays['categories'][rows].astype(np.int64),device=device),torch.as_tensor(arrays['sequences'][rows].astype(np.int64),device=device),torch.as_tensor(np.array(arrays['numeric'][rows]),device=device)
    path=OUT/'dcn.pt'
    if not path.exists():
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5);order=np.random.default_rng(43).permutation(train);model.train();total=0.;seen=0
        for start in range(0,len(order),2048):
            rows=order[start:start+2048];y=torch.as_tensor(arrays['label'][rows].astype(np.float32),device=device)
            optimizer.zero_grad(set_to_none=True);loss=nn.functional.binary_cross_entropy_with_logits(model(*batch(rows)),y)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite full-training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total+=float(loss.detach())*len(rows);seen+=len(rows)
            if start% (2048*250)==0:print(f'FULL_DCN {seen}/{len(train)} logloss={total/seen:.8f}',flush=True)
        torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'manifest':manifest,'options':options,'epochs':1,'seed':42,'training_online_logloss':total/seen},path)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval();parts=[]
    with torch.no_grad():
        for start in range(0,len(final),4096):parts.append(torch.sigmoid(model(*batch(final[start:start+4096]))).cpu().numpy())
    predictions['dcn']=np.concatenate(parts).astype(np.float64);np.save(OUT/'dcn_final_prediction.npy',predictions['dcn'])
    probability=.5*predictions['tree']+.5*predictions['dcn']
    if not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)):raise AssertionError('Invalid final probabilities')
    EXPORT.mkdir(parents=True)
    ids=pd.read_csv(ROOT/'2026ml/final_test.csv',usecols=['log_id','user_id','pt_d'])
    for field,col in [('log_id','log_id'),('user_id','user_id'),('pt_d','date')]:np.testing.assert_array_equal(ids[field],arrays[col][final])
    pd.DataFrame({'id':np.arange(nfinal),'pctr':probability}).to_csv(EXPORT/'submission.csv',index=False,float_format='%.6f',encoding='utf-8')
    pd.DataFrame({'log_id':ids.log_id,'pctr':probability}).to_csv(EXPORT/'result_log_id.csv',index=False,float_format='%.6f',encoding='utf-8')
    detail=pd.DataFrame({'id':np.arange(nfinal)}).join(ids);detail['pctr']=probability;detail['dcn_pctr']=predictions['dcn'];detail['tree_pctr']=predictions['tree']
    detail.to_csv(EXPORT/'prediction_details.csv',index=False,float_format='%.15g',encoding='utf-8')
    actual=pd.read_csv(EXPORT/'submission.csv');np.testing.assert_array_equal(actual.id,np.arange(nfinal))
    if np.max(np.abs(actual.pctr-probability))>5.000001e-7:raise AssertionError('Export rounding mismatch')
    output={**protocol,'status':'predicted','models':{'tree':{'path':str(tree_path.resolve()),'sha256':digest(tree_path)},'dcn':{'path':str(path.resolve()),'sha256':digest(path)}},
            'full_cache_manifest_sha256':digest(CACHE/'manifest.json'),'input_sha256':manifest['input_sha256'],
            'probability':{'minimum':float(probability.min()),'maximum':float(probability.max()),'mean':float(probability.mean()),'quantiles':{str(q):float(np.quantile(probability,q)) for q in [.01,.1,.5,.9,.99]}},
            'development_evidence':{'day15_dense_tree_equal_auc':.7858631196467175,'day16_dense_tree_equal_auc':.7917860844136609,'scope':'Repeated development dates, not final score'},
            'files':{name:{'path':str((EXPORT/name).resolve()),'sha256':digest(EXPORT/name),'bytes':(EXPORT/name).stat().st_size} for name in ['submission.csv','result_log_id.csv','prediction_details.csv']},
            'elapsed_seconds':time.perf_counter()-started}
    write(EXPORT/'manifest.json',output);print('FULL_FINAL_COMPLETE '+json.dumps({'submission':str(EXPORT/'submission.csv'),'rows':nfinal,'probability':output['probability'],'seconds':output['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
