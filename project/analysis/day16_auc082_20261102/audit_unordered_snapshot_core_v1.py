"""Rebuild label-free unordered snapshot features on complete historical day14."""
import time
import numpy as np
import pandas as pd
from fit_stacker import ROOT, HERE, digest, write
from rich_userday_activity import array_hash
from unordered_snapshot_core_v1 import FIELDS, NAMES, build


def main():
    out = HERE/'unordered_snapshot_core_preflight_v1.json'
    assert not out.exists(), 'Preserve previous feature audits'
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=FIELDS,
        dtype={name:str for name in FIELDS if 'list' in name})
    prior = raw[raw.pt_d == 20191114].reset_index(drop=True)
    del raw
    started = time.perf_counter()
    actual, names = build(prior)
    primary_seconds = time.perf_counter()-started
    started = time.perf_counter()
    independent, other = build(prior, independent=True)
    independent_seconds = time.perf_counter()-started
    np.testing.assert_array_equal(actual, independent)
    assert names == other == NAMES and actual.shape == (382961,72)
    report = dict(status='passed', source_sha256=digest(HERE/'unordered_snapshot_core_v1.py'),
        shape=list(actual.shape), features_sha256=array_hash(actual),
        primary_seconds=primary_seconds, independent_seconds=independent_seconds,
        advertisement_labels_read=False, evaluation_days_used=False, inferred17_labels_read=False,
        date=20191114, nonzero_rows=int(np.any(actual != 0,axis=1).sum()),
        per_column_nonzero=np.count_nonzero(actual,axis=0).tolist())
    write(out,report)
    print({key:value for key,value in report.items() if key!='per_column_nonzero'},flush=True)


if __name__ == '__main__':
    main()
