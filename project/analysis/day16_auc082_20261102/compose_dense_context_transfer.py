"""Transfer independently verified dense/context improvements to the current winner."""
import argparse
from pathlib import Path
import numpy as np
import compose_quality_daily_transfer as engine
from fit_stacker import HERE,ROOT,digest,write
from conditional_daily import read,emit
from rich_userday_activity import array_hash

engine.OUT=HERE/'runs/compose_dense_context_transfer_v1'
engine.SOURCES=dict(current=HERE/'runs/compose_quality_daily_transfer_v1',
    daily=HERE/'runs/dense_context_rows_v1',anchor=HERE/'runs/rich_augmented_rows_v1')


def prepare():
    assert not engine.OUT.exists(),'Preserve all earlier compositions and failures'
    hashes=read(engine.SOURCES['current']/'protocol.json')['raw_datasets_sha256']
    paths=[Path(__file__),Path(engine.__file__),HERE/'test_compose_quality_daily_transfer.py',
        HERE/'run_compose_dense_context_transfer.ps1',HERE/'fit_stacker.py',HERE/'conditional_daily.py',
        HERE/'rich_userday_activity.py',HERE/'verify_stacker.py',HERE/'group_temperature.py']
    for folder in engine.SOURCES.values():
        report=read(folder/'verification.json')
        assert report['status']=='passed'
        assert read(folder/'protocol.json')['raw_datasets_sha256']==hashes
        for date in ['20191115','20191116']:
            assert not report['dates'][date]['evaluation_day_labels_fit']
        paths += [folder/'protocol.json',folder/'verification.json']
        for day in [15,16]:
            paths += [folder/f'day{day}_indices.npy',folder/f'day{day}_prediction.npy']
    for name,expected in hashes.items():
        assert digest(ROOT/'2026ml'/name)==expected,name
    engine.OUT.mkdir()
    write(engine.OUT/'protocol.json',dict(dates=[20191115,20191116],recipes=list(engine.recipes()),
        sources={name:str(folder) for name,folder in engine.SOURCES.items()},
        new_model_training=False,inferred17_labels_read=False,
        selection='Both-date fixed candidate hashes lock before held scoring; reused development dates',
        raw_datasets_sha256=hashes,sources_sha256={str(p):digest(p) for p in paths}))
    data,locks=engine.inputs(),{}
    for day in [15,16]:
        np.save(engine.OUT/f'day{day}_indices.npy',data[day]['rows'])
        locks[str(day)]={record['name']:array_hash(engine.predict(data[day],record)) for record in engine.recipes()}
        emit('dense_context_transfer_candidates_locked',day=day,candidates=len(locks[str(day)]))
    write(engine.OUT/'predictions_locked.json',dict(protocol_sha256=digest(engine.OUT/'protocol.json'),
        predictions_sha256=locks,held_labels_opened=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    engine.verify() if parser.parse_args().verify else prepare()
