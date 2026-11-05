"""Independent full-refit input probes, checkpoint replay and actual CSV verification."""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier,Pool

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'model_comparison'))
from bounded_id_models import BoundedIDDCN
from refit_predict_full import tree_frame,digest,read,CACHE,OUT,EXPORT
from interest_data import build_past_statistics
sys.path.insert(0,str(HERE.parent/'auc085'))
from target_statistics import SPECS


def main():
    manifest=read(CACHE/'manifest.json');record=read(EXPORT/'manifest.json')
    arrays={name:np.load(CACHE/(name+'.npy'),mmap_mode='r') for name in ['categories','sequences','numeric','label','date','log_id','user_id']}
    n=manifest['train_rows'];final=np.arange(n,manifest['rows'],dtype=np.int32)
    if not np.all(arrays['label'][final]==-1) or int(arrays['label'][:n].sum())!=21354:raise AssertionError('Unknown-label/training-label boundary failed')
    raw=pd.read_csv(ROOT/'2026ml/final_test.csv',usecols=['log_id','user_id','pt_d'])
    result=pd.read_csv(EXPORT/'submission.csv');detail=pd.read_csv(EXPORT/'prediction_details.csv')
    if result.columns.tolist()!=['id','pctr'] or len(result)!=404182:raise AssertionError('Platform submission format error')
    np.testing.assert_array_equal(result.id,np.arange(len(raw)))
    np.testing.assert_array_equal(detail[['log_id','user_id','pt_d']],raw[['log_id','user_id','pt_d']])
    for field,col in [('log_id','log_id'),('user_id','user_id'),('pt_d','date')]:np.testing.assert_array_equal(raw[field],arrays[col][final])
    fields=sorted({f for spec,_ in SPECS for f in spec}|{'pt_d','label'})
    train=pd.read_csv(ROOT/'2026ml/train.csv',usecols=fields,dtype='int32')
    selected=[0,len(final)//2,len(final)-1]
    query=pd.read_csv(ROOT/'2026ml/final_test.csv',usecols=[f for f in fields if f!='label']).iloc[selected].copy();query['label']=-1
    combined=pd.concat([train,query],ignore_index=True)
    # Recompute raw statistics directly per final probe from all and only labeled prior dates.
    prior=(train.label.sum()+1000*.0084)/(len(train)+1000);maximum_stats=0.;checks=0
    for row,q in zip(selected,query.to_dict('records')):
        for position,(group,alpha) in enumerate(SPECS):
            mask=np.ones(len(train),dtype=bool)
            for field in group:mask &= train[field].to_numpy()==q[field]
            count=int(mask.sum());clicks=int(train.loc[mask,'label'].sum());p=(clicks+alpha*prior)/(count+alpha)
            raw_values=[np.log1p(count),np.log(p/(1-p))-np.log(prior/(1-prior))]
            for offset,value in enumerate(raw_values):
                column=39+2*position+offset;field=manifest['numeric_fields'][column];scale=manifest['numeric_scaler'][field]
                expected=(np.float32(value)-scale['mean'])/scale['std'];actual=float(arrays['numeric'][n+row,column]);error=abs(expected-actual)
                maximum_stats=max(maximum_stats,error);checks+=1
                if error>2e-5:raise AssertionError(f'Statistics error {field} {error}')
    del train,combined,query
    tree=CatBoostClassifier();tree.load_model(str(OUT/'catboost.cbm'));parts=[]
    for start in range(0,len(final),25000):
        rows=final[start:start+25000];frame,cat=tree_frame(rows,arrays,manifest)
        parts.append(tree.predict_proba(Pool(frame,cat_features=cat),thread_count=6)[:,1])
    treep=np.concatenate(parts);saved_tree=np.load(OUT/'tree_final_prediction.npy');tree_error=float(np.max(np.abs(treep-saved_tree)))
    if tree_error>1e-10:raise AssertionError('Tree checkpoint final replay differs')
    del tree,frame,parts
    torch.set_num_threads(6);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ck=torch.load(OUT/'dcn.pt',map_location='cpu',weights_only=False)
    model=BoundedIDDCN(manifest['categorical_fields'],manifest['sequence_fields'],manifest['group_for_field'],manifest['vocabulary_sizes'],**ck['options']).to(device)
    model.load_state_dict(ck['state_dict'],strict=True);model.eval();parts=[]
    with torch.no_grad():
        for start in range(0,len(final),4096):
            rows=final[start:start+4096];inputs=(torch.as_tensor(arrays['categories'][rows].astype(np.int64),device=device),torch.as_tensor(arrays['sequences'][rows].astype(np.int64),device=device),torch.as_tensor(np.array(arrays['numeric'][rows]),device=device))
            parts.append(torch.sigmoid(model(*inputs)).cpu().numpy())
    dcnp=np.concatenate(parts).astype(np.float64);saved_dcn=np.load(OUT/'dcn_final_prediction.npy');dcn_error=float(np.max(np.abs(dcnp-saved_dcn)))
    if dcn_error>2e-7:raise AssertionError('DCN final replay differs')
    probability=.5*treep+.5*dcnp
    if not np.isfinite(probability).all() or not result.pctr.between(0,1).all():raise AssertionError('Invalid probability')
    rounding=float(np.max(np.abs(result.pctr-probability)));detail_error=float(np.max(np.abs(detail.pctr-probability)))
    if rounding>5.000001e-7 or detail_error>1e-14:raise AssertionError('Export values differ')
    with (EXPORT/'submission.csv').open('rb') as f:
        if f.read(3)==b'\xef\xbb\xbf':raise AssertionError('Unexpected BOM')
    with (EXPORT/'submission.csv').open(encoding='utf-8',newline='') as f:
        reader=csv.reader(f);next(reader)
        for row in reader:
            if not re.fullmatch(r'[01]\.\d{6}',row[1]):raise AssertionError('Expected six decimal places')
    for name,expected in manifest['input_sha256'].items():
        if digest(ROOT/'2026ml'/name)!=expected:raise AssertionError('Input changed '+name)
    for name,item in record['files'].items():
        if digest(EXPORT/name)!=item['sha256']:raise AssertionError('Export hash differs')
    for item in record['models'].values():
        if digest(Path(item['path']))!=item['sha256']:raise AssertionError('Model hash differs')
    output={'status':'passed','rows':len(final),'training_rows':n,'training_clicks':21354,'original_ids_and_row_order':True,
          'all_final_advertisement_labels_unknown':True,'strict_past_stat_probes':checks,'maximum_stat_probe_error':maximum_stats,
          'full_tree_replay_error':tree_error,'full_dcn_replay_error':dcn_error,'maximum_rounding_error':rounding,'maximum_detail_error':detail_error,
          'input_model_and_export_hashes_match':True,'six_decimals_utf8_no_bom':True,'final_auc':None,'final_labels_available':False}
    (EXPORT/'verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output),flush=True)


if __name__=='__main__':main()
