"""Inspect frozen feature importance without accessing advertisement evaluation labels."""
import numpy as np
from fit_stacker import HERE,digest,write
from conditional_daily import read,emit
from catboost_userday_residual import load_model

OUT=HERE/'runs/snapshot_pair_ranker_v1'
REPORT=OUT/'feature_usage_diagnostic.json'

if __name__=='__main__':
    assert not REPORT.exists(),'Preserve previous diagnostics'
    lock=read(OUT/'models_locked.json')
    assert lock['protocol_sha256']==digest(OUT/'protocol.json')
    records=[]
    for record in lock['models']:
        if record['trees']!=1600:
            continue
        path=OUT/(record['name']+'.cbm')
        assert digest(path)==record['model_sha256']
        model=load_model(path)
        weights=model.get_feature_importance(type='PredictionValuesChange')
        names=model.feature_names_
        assert len(weights)==len(names)==1257
        snapshot=np.array([name.startswith('snapshot_quality__') for name in names])
        assert snapshot.sum()==240
        top=np.argsort(-weights,kind='stable')[:25]
        records.append(dict(name=record['name'],new_feature_importance_total=float(weights[snapshot].sum()),
            new_features_with_positive_importance=int(np.sum(weights[snapshot]>0)),
            top_features=[dict(name=names[i],importance=float(weights[i])) for i in top]))
    write(REPORT,dict(status='passed',advertisement_labels_read=False,
        model_selection_performed=False,source_sha256=digest(__file__),models=records))
    emit('snapshot_feature_usage',records=[{k:v for k,v in r.items() if k!='top_features'} for r in records])
