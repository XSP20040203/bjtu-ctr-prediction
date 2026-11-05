"""Validate the new pair design on both complete training populations before GPU use."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import gc
from pathlib import Path
import numpy as np
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from full_negative_pair_core_v1 import design
from rich_userday_activity import array_hash

OUTPUT = HERE / 'full_negative_pair_population_audit_v1.json'
SOURCE = HERE / 'runs/full_negative_context_v1'
PARAMS = dict(seed=151681, negative_block=512, positive_draws=64)


def main():
    assert not OUTPUT.exists(), 'Preserve every completed or failed audit'
    report = {}
    for day in (15, 16):
        folder = SOURCE / f'input_day{day}'
        declared = read(folder / 'contract.json')
        for name, expected in declared['files_sha256'].items():
            assert digest(folder / name) == expected, name
        rows, train, held, labels, all_dates = [np.load(folder / (name + '.npy'))
                                               for name in ('rows', 'train', 'held', 'labels', 'dates')]
        dates = all_dates[train]
        assert len(labels) == len(train) and np.all(dates < 20191100 + day)
        assert np.all(all_dates[held] == 20191100 + day)
        order, groups, pairs = design(dates, labels, 20191100 + day, **PARAMS)
        # Check actual pair incidence independently of the generator's unique-index checks.
        positive_degree = np.bincount(pairs[:, 0], minlength=len(order))
        negative_degree = np.bincount(pairs[:, 1], minlength=len(order))
        assert np.all(positive_degree[labels[order] == 1] > 0)
        assert np.all(negative_degree[labels[order] == 0] > 0)
        assert not positive_degree[labels[order] == 0].any()
        assert not negative_degree[labels[order] == 1].any()
        maximum = int(np.bincount(groups).max())
        assert maximum <= 1024
        np.testing.assert_array_equal(np.sort(rows[train][order]), rows[train])
        assert not np.isin(rows[train][order], rows[held]).any()
        report[str(20191100 + day)] = dict(training_rows=len(order),
            positives=int(labels.sum()), negatives=int((labels == 0).sum()), pairs=len(pairs),
            maximum_query_size=maximum, every_negative_in_pairs=True, every_positive_in_pairs=True,
            date_isolation=True, held_rows_in_training=0, max_training_date=int(dates.max()),
            order_sha256=array_hash(order), groups_sha256=array_hash(groups), pairs_sha256=array_hash(pairs),
            context_contract_sha256=digest(folder / 'contract.json'))
        emit('full_negative_pair_complete_population_audited', day=day, **report[str(20191100 + day)])
        del order, groups, pairs, positive_degree, negative_degree
        gc.collect()
    expected = read(SOURCE / 'protocol.json')['raw_datasets_sha256']
    assert {name: digest(ROOT / '2026ml' / name) for name in expected} == expected
    paths = [Path(__file__), HERE / 'full_negative_pair_core_v1.py', HERE / 'full_rich_pair_ranker_v1.py',
             HERE / 'runs/full_rich_pair_preflight_v1/verification.json']
    write(OUTPUT, dict(status='passed', feature_only=True, dates=report, pair_design=PARAMS,
        inferred17_labels_read=False, held_labels_read=False, original_datasets_unchanged=True,
        raw_datasets_sha256=expected, sources_sha256={str(path): digest(path) for path in paths}))


if __name__ == '__main__':
    main()
