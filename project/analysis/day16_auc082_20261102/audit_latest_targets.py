"""Independently check both full published cohorts with sklearn and raw hashes."""
from datetime import datetime,timezone
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from fit_stacker import ROOT,HERE,digest,write
from publish_progress import publication_lock


def main():
    output=ROOT/'output/two_day_auc_targets_20261102'
    progress=json.loads((output/'progress.json').read_text(encoding='utf8'))
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label','log_id','user_id'],dtype='int32')
    results={}
    for day in [15,16]:
        record=progress['dates'][str(20191100+day)]
        rows=np.flatnonzero(raw.pt_d.to_numpy()==20191100+day)
        path=(output/'day15_predictions.csv' if day==15 else ROOT/'output/day16_auc082_progress_20261102/day16_predictions.csv')
        frame=pd.read_csv(path,float_precision='round_trip')
        np.testing.assert_array_equal(frame.row_index.to_numpy(),rows)
        for field in ['label','log_id','user_id']:
            np.testing.assert_array_equal(frame[field].to_numpy(),raw[field].to_numpy()[rows])
        p=np.load(record['prediction'])
        np.testing.assert_array_equal(frame.pctr.to_numpy(),p)
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        assert digest(record['prediction'])==record['prediction_sha256']
        value=float(roc_auc_score(raw.label.to_numpy()[rows],p))
        assert abs(value-record['auc'])<1e-12
        verification=json.loads(open(record['verification'],encoding='utf8').read())
        published_auc=verification.get('auc',verification.get('csv_auc'))
        assert abs(value-published_auc)<1e-12
        results[str(20191100+day)]={'auc':value,'target_auc':record['target_auc'],
                'target_met':value>=record['target_auc'],'rows':len(rows),'clicks':int(frame.label.sum()),
                'csv_sha256':digest(path),'prediction_sha256':digest(record['prediction']),
                'raw_row_alignment_verified':True,'sklearn_auc_matches_published':True}
    expected=json.loads((HERE/'runs/weighted_v1_day16/protocol.json').read_text(encoding='utf8'))['raw_datasets_sha256']
    actual={name:digest(ROOT/'2026ml'/name) for name in expected}
    assert actual==expected
    result={'audit_status':'passed','goal_achieved':all(r['target_met'] for r in results.values()),
            'timestamp_utc':datetime.now(timezone.utc).isoformat(),'dates':results,
            'original_datasets_unchanged':True,'raw_datasets_sha256':actual}
    write(output/'independent_target_audit.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with publication_lock():main()
