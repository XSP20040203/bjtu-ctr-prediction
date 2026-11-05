"""Check graph reconstruction on real inputs without scoring any target labels."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import time
import numpy as np
import pandas as pd
from audience_graph_core_v1 import build_day, NAMES
from fit_stacker import ROOT, HERE, digest, write
from rich_userday_activity import array_hash


def main():
    destination = HERE / 'audience_graph_core_preflight_v1.json'
    assert not destination.exists(), 'Preserve previous graph input audits'
    source = HERE / 'audience_graph_core_v1.py'
    source_hash = digest(source)
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'user_id', 'task_id', 'label'], dtype='int32')
    day = 20191116
    labels = np.where(raw.pt_d.to_numpy() < day, raw.label.to_numpy(), -1)
    candidates = np.flatnonzero(raw.pt_d.to_numpy() == day)
    rows = np.sort(np.random.default_rng(152601).choice(candidates, 1000, replace=False))
    started = time.monotonic()
    primary = build_day(raw[['pt_d', 'user_id', 'task_id']], labels, day, rows)
    elapsed_primary = time.monotonic() - started
    started = time.monotonic()
    independent = build_day(raw[['pt_d', 'user_id', 'task_id']], labels, day, rows, True)
    elapsed_independent = time.monotonic() - started
    np.testing.assert_array_equal(primary, independent)
    assert digest(source) == source_hash
    report = dict(status='passed', date=day, shape=list(primary.shape), names=NAMES,
                  primary_seconds=elapsed_primary, independent_seconds=elapsed_independent,
                  maximum_replay_error=0., features_sha256=array_hash(primary),
                  query_rows_sha256=array_hash(rows), source_sha256=source_hash,
                  advertisement_labels_strictly_before_date=True,
                  unlabeled_exposures_at_most_date=True, target_labels_scored=False,
                  inferred17_labels_read=False)
    write(destination, report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
