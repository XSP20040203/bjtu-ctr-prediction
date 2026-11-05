"""Score aligned stored temporal predictions, without granting promotion."""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from fit_stacker import ROOT,HERE,REFERENCE,digest,write

def auc(y,p):
    n=int(y.sum())
    return float((rankdata(p,method='average')[y==1].sum()-n*(n+1)/2)/(n*(len(y)-n)))

def main():
    source=HERE/'paired_prediction_inventory.json'
    inventory=json.loads(source.read_text(encoding='utf8'))
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    ys={day:raw.label.to_numpy()[raw.pt_d.to_numpy()==20191100+day] for day in [15,16]}
    records=[]
    for r in inventory['records']:
        if not r['alignment_verified']:continue
        result={'family':r['family'],'path15':r['path15'],'path16':r['path16'],'promotion_allowed':False}
        for i,day in enumerate([15,16]):
            path=ROOT/r[f'path{day}']
            assert digest(path)==r['sha256'][i],path
            np.testing.assert_array_equal(np.load(r['indices'][i]),np.load(REFERENCE/f'day{day}_indices.npy'))
            p=np.load(path)
            result[f'auc{day}']=auc(ys[day],p)
        records.append(result)
    records.sort(key=lambda r:r['auc15'],reverse=True)
    report={'status':'diagnostic_only','training_boundary_must_be_audited_before_promotion':True,
            'inventory_sha256':digest(source),'held_dates_reused_for_development':True,'records':records}
    write(HERE/'historical_pair_scores.json',report)
    print(json.dumps(records[:12],ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
