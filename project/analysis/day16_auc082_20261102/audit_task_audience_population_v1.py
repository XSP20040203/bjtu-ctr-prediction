"""Compare aggregate training targets with full original per-date totals."""
import numpy as np
import pandas as pd
from fit_stacker import HERE,ROOT,digest,write


def main():
    path=HERE/'task_audience_full_population_audit_v1.json'
    assert not path.exists(),'Preserve prior audits'
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    records={}
    for day in [15,16]:
        source=HERE/f'runs/task_audience_rate_v1/input_day{day}/metadata.npz'
        if not source.exists() and day==16:
            source=HERE/'runs/task_audience_preflight_v1/input_day16/metadata.npz'
        metadata=np.load(source)
        keys,sizes,counts=(metadata[k] for k in ['keys','sizes','counts'])
        result={}
        for date in sorted(raw.pt_d.unique()):
            if date>20191100+day:continue
            groups=keys[:,0]==date
            rows=raw.pt_d.eq(date)
            expected=int(rows.sum())
            assert sizes[groups].sum()==expected
            if date<20191100+day:
                actual=int(raw.loc[rows,'label'].sum())
                assert counts[groups].sum()==actual
                assert np.all((counts[groups]>=0)&(counts[groups]<=sizes[groups]))
            else:
                actual=None
                assert np.all(counts[groups]==-1)
            result[str(date)]=dict(exposures=expected,training_clicks=actual)
        records[str(20191100+day)]=dict(metadata=str(source),metadata_sha256=digest(source),dates=result)
    report=dict(status='passed',training_ad_exposures_and_clicks_match_full_raw_population=True,
        current_date_click_counts_masked=True,no_held_auc_scored=True,inferred17_labels_read=False,
        train_csv_sha256=digest(ROOT/'2026ml/train.csv'),dates=records)
    write(path,report)
    print(report,flush=True)


if __name__=='__main__':main()
