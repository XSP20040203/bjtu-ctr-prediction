"""Preserve and verify fully completed arms from the GPU-memory-limited run."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rolling_rich_daily_catboost import inputs,contract,group_prediction
from catboost_userday_residual import load_model
from rich_userday_activity import correct
from fit_stacker import ROOT,HERE,digest,write
from verify_stacker import rank_sum_auc

SOURCE=HERE/'runs/rolling_rich_daily_catboost_v1'
OUT=HERE/'runs/rich_binary15_completed_v1'
BASE=HERE/'runs/catboost_global_rows15_v1'
POWERS=[0.,.125,.25,.5,.75]


def prepare():
    assert not OUT.exists(),'Preserving completed-arm evaluation'
    original=json.loads((SOURCE/'protocol.json').read_text(encoding='utf8'))
    assert json.loads((BASE/'verification.json').read_text(encoding='utf8'))['status']=='passed'
    models=[]
    for offset in [False,True]:
        for trees in [300,700,1200]:
            name=f'day15_binary_d6_{"offset" if offset else "plain"}_trees{trees}'
            path=SOURCE/(name+'.cbm')
            assert path.exists() and path.stat().st_size>0
            models.append({'name':name,'objective':'binary','offset':offset,'trees':trees,
                           'path':str(path),'model_sha256':digest(path)})
    paths=[Path(__file__),HERE/'rolling_rich_daily_catboost.py',SOURCE/'protocol.json',
           SOURCE/'input_contract15.json',BASE/'prediction.npy',BASE/'verification.json']
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json',{'day':20191115,'models':models,'powers':POWERS,
          'evaluation15_labels_fit':False,'development15_model_selection':True,
          'source_run_outcome':'Two complete binary arms; subsequent depth8 Poisson stopped with GPU memory error',
          'sources_sha256':{**original['sources_sha256'],**{str(p):digest(p) for p in paths}},
          'raw_datasets_sha256':original['raw_datasets_sha256']})
    evaluate(False)


def evaluate(verifying):
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    for path,h in protocol['sources_sha256'].items():assert digest(path)==h,path
    for name,h in protocol['raw_datasets_sha256'].items():assert digest(ROOT/'2026ml'/name)==h,name
    data=inputs(15)
    assert contract(data,15)==json.loads((SOURCE/'input_contract15.json').read_text(encoding='utf8'))
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    rows=np.flatnonzero(raw.pt_d.to_numpy()==20191115)
    np.testing.assert_array_equal(rows,data['rows'])
    np.testing.assert_array_equal(rows,np.load(BASE/'valid_indices.npy'))
    held=np.flatnonzero(data['valid'])
    y=raw.label.to_numpy()[rows]
    assert len(y)==424780 and int(y.sum())==3523
    base=np.load(BASE/'prediction.npy');best={'name':'base','power':0.,'auc':rank_sum_auc(y,base)}
    best_p=base.copy();records=[]
    for record in protocol['models']:
        path=Path(record['path']);assert digest(path)==record['model_sha256']
        gp=np.zeros(len(data['keys']))
        gp[held]=group_prediction(load_model(path),data,record,held)
        for power in POWERS:
            p=correct(base,gp,data['row_groups'],'binary',power)
            result={'name':record['name'],'power':power,'auc':rank_sum_auc(y,p)}
            records.append(result)
            if result['auc']>best['auc']:best=result;best_p=p.copy()
    error=0.
    if verifying:
        np.testing.assert_array_equal(rows,np.load(OUT/'valid_indices.npy'))
        error=float(np.max(np.abs(best_p-np.load(OUT/'prediction.npy'))));assert error<1e-12,error
    else:
        np.save(OUT/'prediction.npy',best_p);np.save(OUT/'valid_indices.npy',rows)
    report={'status':'passed' if verifying else 'evaluated','auc':best['auc'],'best':best,'records':records,
            'rows':len(rows),'clicks':int(y.sum()),'maximum_replay_error':error,
            'evaluation15_labels_fit':False,'development15_model_selection':True,
            'original_datasets_unchanged':True,'target_met':best['auc']>=.815}
    write(OUT/('verification.json' if verifying else 'result.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    evaluate(True) if parser.parse_args().verify else prepare()
