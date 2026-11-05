"""Real prior-date sparse encoding, gradient and coefficient replay preflight."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='2'
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from fit_stacker import ROOT,HERE,digest,write
from sparse_cross_core_v1 import FIELDS,CHANNELS,WIDTH,encode_chunks,encode,margin


def main():
    out=HERE/'sparse_cross_core_preflight_v1.json'
    assert not out.exists(),'Preserve earlier preflight'
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=FIELDS+['pt_d','label'],
        dtype={name:str for name in CHANNELS})
    prior=raw[raw.pt_d==20191114].reset_index(drop=True)
    del raw
    start=time.perf_counter()
    x=encode_chunks(prior)
    encoding_seconds=time.perf_counter()-start
    probe=encode(prior.iloc[:1000],independent=True)
    assert (x[:1000]!=probe).nnz==0
    y=prior.label.to_numpy(dtype='int32')
    model=SGDClassifier(loss='log_loss',alpha=1e-5,learning_rate='constant',eta0=.01,
                        average=len(y)//2,random_state=151655)
    model.partial_fit(x,y,classes=np.array([0,1]),sample_weight=np.where(y==1,3.,1.))
    coefficients=model.coef_.reshape(-1).astype('float64')
    intercept=float(model.intercept_[0])
    a=margin(x[:2000],coefficients,intercept)
    b=margin(x[:2000],coefficients,intercept,True)
    error=float(np.max(np.abs(a-b)))
    assert error<1e-11 and np.isfinite(a).all()
    report=dict(status='passed',source_sha256=digest(HERE/'sparse_cross_core_v1.py'),
        raw_datasets_sha256={name:digest(ROOT/'2026ml'/name) for name in ['train.csv','train_feeds.csv','final_test.csv','test_feeds.csv']},
        date=20191114,shape=list(x.shape),nonzeros=int(x.nnz),encoding_seconds=encoding_seconds,
        total_seconds=time.perf_counter()-start,independent_encoded_probe_rows=1000,
        independent_prediction_probe_rows=2000,maximum_prediction_error=error,
        fit_only_prior14_labels=True,evaluation_days_used=False,inferred17_labels_read=False)
    write(out,report)
    print(report,flush=True)


if __name__=='__main__':main()
