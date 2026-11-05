"""Row-level category interactions and MLP, fitted strictly before each held date."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash
from rich_row_catboost import frame_hash
from verify_stacker import rank_sum_auc
import compact_full_sample_ranker_v1 as source
from cross_user_pair_ranker_v1 import prediction as row_prediction
from row_deepfm_core_v1 import fit_schema, transform, prior_rows, DeepFM

OUT = HERE / 'runs/row_deepfm_v1'
BASE = HERE / 'runs/compose_transition_priors_v1'
ARMS = [dict(name='embedding8', embedding_dim=8, seed=151658),
        dict(name='embedding16', embedding_dim=16, seed=151659)]
CHECKPOINTS = [2, 5, 10]
WEIGHTS = [.025, .05, .1, .2, .35, .5, .75, 1.]
MODES = ['replace_global', 'add_within']


def check():
    protocol = read(OUT / 'protocol.json')
    for filename, expected in protocol['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    return protocol


def prepare():
    assert not OUT.exists(), 'Preserve previous runs'
    assert read(BASE / 'verification.json')['status'] == 'passed'
    paths = source.dependencies() + [Path(__file__), HERE / 'row_deepfm_core_v1.py',
        HERE / 'test_row_deepfm_core_v1.py', HERE / 'run_row_deepfm_v1.ps1',
        BASE / 'verification.json', BASE / 'protocol.json']
    for day in [15, 16]:
        paths += [BASE / f'day{day}_prediction.npy', BASE / f'day{day}_indices.npy']
    OUT.mkdir()
    write(OUT / 'protocol.json', dict(
        arms=ARMS, checkpoints=CHECKPOINTS, epochs=max(CHECKPOINTS), batch_size=2048,
        optimizer='AdamW', learning_rate=.001, weight_decay=.0001, dropout=.2,
        objective='Binary cross entropy, positive weight3, fixed one-third source negatives',
        architecture='Field-specific category embeddings, second-order FM, 192/96 MLP, all384 compact numeric inputs',
        normalization_and_vocabulary='Fitted on strictly prior training rows only; unknown category index0',
        selection='All both-date checkpoints locked before scoring; reused development dates',
        weights=WEIGHTS, modes=MODES, evaluation_day_labels_fit=False, inferred17_labels_read=False,
        raw_datasets_sha256=read(BASE / 'protocol.json')['raw_datasets_sha256'],
        sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)}))
    contracts = {}
    for day in [15, 16]:
        contracts[str(day)] = {}
        for partition in ['train', 'held']:
            frame, cats, info, audit = source.inputs(day, partition)
            if partition == 'train':
                schema = fit_schema(frame, cats)
                write(OUT / f'schema{day}.json', schema)
                chosen_dates = info['raw'].pt_d.to_numpy()[info['rows'][info['positions']]]
                rows = prior_rows(chosen_dates, info['target'], 20191100 + day)
                np.testing.assert_array_equal(rows, np.arange(len(frame)))
                np.save(OUT / f'target{day}.npy', info['target'])
                np.save(OUT / f'day{day}_indices.npy', info['held_rows'])
            else:
                schema = read(OUT / f'schema{day}.json')
            numeric, encoded = transform(frame, schema)
            files = {}
            for name, values in [('numeric', numeric), ('categories', encoded)]:
                path = OUT / f'{name}{day}_{partition}.npy'
                np.save(path, values)
                files[path.name] = digest(path)
            contracts[str(day)][partition] = dict(source_audit=audit, files_sha256=files,
                numeric_sha256=array_hash(numeric), categories_sha256=array_hash(encoded),
                schema_sha256=digest(OUT / f'schema{day}.json'))
            write(OUT / 'inputs_locked.json', contracts)
            del frame, numeric, encoded, info
            gc.collect()
            emit('row_deepfm_inputs_prepared', day=day, partition=partition, shape=audit['shape'])
    check()


def load_data(day, partition):
    contract = read(OUT / 'inputs_locked.json')[str(day)][partition]
    for filename, expected in contract['files_sha256'].items():
        assert digest(OUT / filename) == expected, filename
    schema = read(OUT / f'schema{day}.json')
    assert digest(OUT / f'schema{day}.json') == contract['schema_sha256']
    return (np.load(OUT / f'numeric{day}_{partition}.npy'),
            np.load(OUT / f'categories{day}_{partition}.npy'), schema)


def network(schema, arm):
    return DeepFM(len(schema['numeric']), [len(schema['vocabularies'][name]) for name in schema['categories']],
                  arm['embedding_dim'], .2)


def fit():
    check()
    assert not (OUT / 'models_locked.json').exists(), 'Preserve fitted models'
    assert not list(OUT.glob('*.pt')), 'Partial fits require explicit recovery'
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    records = []
    for day in [15, 16]:
        numeric, categories, schema = load_data(day, 'train')
        target = np.load(OUT / f'target{day}.npy')
        assert np.isin(target, [0, 1]).all() and len(target) == len(numeric)
        for arm in ARMS:
            torch.manual_seed(arm['seed'])
            np.random.seed(arm['seed'])
            model = network(schema, arm).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            generator = np.random.default_rng(arm['seed'])
            for epoch in range(1, max(CHECKPOINTS) + 1):
                model.train()
                order = generator.permutation(len(target))
                total = 0.
                for first in range(0, len(order), 2048):
                    batch = order[first:first + 2048]
                    x = torch.from_numpy(numeric[batch]).cuda()
                    c = torch.from_numpy(categories[batch]).cuda()
                    y = torch.from_numpy(target[batch].astype('float32')).cuda()
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        model(x, c), y, pos_weight=torch.tensor(3., device='cuda'))
                    assert torch.isfinite(loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                    total += float(loss.detach()) * len(batch)
                emit('row_deepfm_epoch', day=day, arm=arm['name'], epoch=epoch, loss=total / len(target))
                if epoch in CHECKPOINTS:
                    name = f'day{day}_{arm["name"]}_epoch{epoch}'
                    path = OUT / (name + '.pt')
                    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, path)
                    records.append(dict(arm, day=day, epoch=epoch, name=name, model_sha256=digest(path),
                                        maximum_training_date=20191100 + day - 1,
                                        inputs_locked_sha256=digest(OUT / 'inputs_locked.json')))
            del optimizer, model
            torch.cuda.empty_cache()
            gc.collect()
        del numeric, categories, target
        gc.collect()
    write(OUT / 'models_locked.json', dict(models=records, protocol_sha256=digest(OUT / 'protocol.json')))
    check()


def forecasts(day):
    torch.set_num_threads(2)
    numeric, categories, schema = load_data(day, 'held')
    values = {}
    for record in read(OUT / 'models_locked.json')['models']:
        if record['day'] != day:
            continue
        path = OUT / (record['name'] + '.pt')
        assert digest(path) == record['model_sha256']
        model = network(schema, record)
        model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
        model.eval()
        margins = []
        with torch.no_grad():
            for first in range(0, len(numeric), 4096):
                margins.append(model(torch.from_numpy(numeric[first:first + 4096]),
                                     torch.from_numpy(categories[first:first + 4096])).numpy())
        values[record['name']] = np.concatenate(margins).astype('float64')
    return values


def candidates(day, margins, independent=False):
    base = np.load(BASE / f'day{day}_prediction.npy')
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'user_id'], dtype='int32')
    groups, _ = pd.factorize(raw.loc[raw.pt_d.eq(20191100 + day), 'user_id'], sort=True)
    yield dict(name='base', mode='base', weight=0.), base.copy()
    for name, margin in margins.items():
        for mode in MODES:
            for weight in WEIGHTS:
                yield dict(name=name, mode=mode, weight=weight), row_prediction(base, margin, groups, weight, mode, independent)


def key(record):
    return f'{record["name"]}__{record["mode"]}__{record["weight"]:g}'


def evaluate():
    check()
    assert not (OUT / 'predictions_locked.json').exists()
    locked = {}
    for day in [15, 16]:
        margins = forecasts(day)
        for name, margin in margins.items():
            np.save(OUT / (name + '_held_margin.npy'), margin)
        locked[str(day)] = {key(r): array_hash(p) for r, p in candidates(day, margins)}
    write(OUT / 'predictions_locked.json', dict(
        models_locked_sha256=digest(OUT / 'models_locked.json'), inputs_locked_sha256=digest(OUT / 'inputs_locked.json'),
        dates=locked, all_both_date_models_locked=True, held_labels_opened_for_scoring=False))
    emit('row_deepfm_candidates_locked')


def verify():
    check()
    assert not (OUT / 'verification.json').exists()
    locked = read(OUT / 'predictions_locked.json')
    assert locked['models_locked_sha256'] == digest(OUT / 'models_locked.json')
    assert locked['inputs_locked_sha256'] == digest(OUT / 'inputs_locked.json')
    checked = {}
    for day in [15, 16]:
        for partition in ['train', 'held']:
            frame, cats, info, audit = source.inputs(day, partition, True)
            contract = read(OUT / 'inputs_locked.json')[str(day)][partition]
            assert audit == contract['source_audit']
            schema = read(OUT / f'schema{day}.json')
            if partition == 'train':
                assert fit_schema(frame, cats) == schema
                np.testing.assert_array_equal(info['target'], np.load(OUT / f'target{day}.npy'))
            numeric, encoded = transform(frame, schema, True)
            assert array_hash(numeric) == contract['numeric_sha256']
            assert array_hash(encoded) == contract['categories_sha256']
            del frame, info, numeric, encoded
            gc.collect()
        margins = forecasts(day)
        for name, margin in margins.items():
            np.testing.assert_array_equal(margin, np.load(OUT / (name + '_held_margin.npy')))
        checked[day] = margins
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    reports = {}
    for day in [15, 16]:
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day)
        np.testing.assert_array_equal(rows, np.load(OUT / f'day{day}_indices.npy'))
        y = raw.label.to_numpy()[rows]
        records, best_value, maximum = [], -1., 0.
        for (record, prediction), (other, replay) in zip(candidates(day, checked[day]), candidates(day, checked[day], True)):
            assert record == other and array_hash(prediction) == locked['dates'][str(day)][key(record)]
            assert np.isfinite(prediction).all() and np.all((prediction > 0) & (prediction < 1))
            error = float(np.max(np.abs(prediction - replay)))
            assert error < 2e-13
            maximum = max(maximum, error)
            value = rank_sum_auc(y, prediction)
            record = dict(record, auc=value)
            records.append(record)
            if value > best_value:
                best_value, best, best_prediction = value, record, prediction.copy()
        assert abs(roc_auc_score(y, best_prediction) - best_value) < 1e-12
        np.save(OUT / f'day{day}_prediction.npy', best_prediction)
        reports[str(20191100 + day)] = dict(auc=best_value, best=best, records=records, rows=len(y), clicks=int(y.sum()),
            evaluation_day_labels_fit=False, inferred17_labels_read=False, held_labels_used_to_select_fixed_recipe=True,
            all_both_date_candidates_locked_before_scoring=True, independent_all_features_and_models_replayed=True,
            maximum_independent_prediction_error=maximum, target_met=best_value >= {15: .815, 16: .82}[day])
        emit('row_deepfm_scored', day=day, best=best)
    check()
    write(OUT / 'verification.json', dict(status='passed', dates=reports, original_datasets_unchanged=True,
                                         both_targets_met=all(r['target_met'] for r in reports.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['prepare', 'fit', 'evaluate', 'verify'])
    {'prepare': prepare, 'fit': fit, 'evaluate': evaluate, 'verify': verify}[parser.parse_args().phase]()
