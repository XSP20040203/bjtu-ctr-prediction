"""Create a row ranker with prior-selected task audience context features."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
TARGET=HERE/'task_audience_row_ranker_v1.py'
INPUTS='''def inputs(day, partition, independent=False):
    frame,cats,info,parent_audit=base_rows.inputs(day,partition,independent)
    cache_key=(day,independent)
    if cache_key not in AUDIENCE_CACHE:
        if independent:
            auxiliary,meta,contract=audience.inputs(day,True)
            assert contract==read(AUDIENCE/f'input_day{day}/contract.json')
        else:
            auxiliary,meta,contract=audience.load_inputs(AUDIENCE/f'input_day{day}')
        model_path=AUDIENCE/f'day{day}_audience_d6_trees600.cbm'
        model=load_model(model_path)
        importance=model.get_feature_importance()
        names=select_columns(list(auxiliary),contract['categories'],importance,192)
        AUDIENCE_CACHE[cache_key]=(auxiliary[names].to_numpy(dtype='float32'),meta['keys'],names,digest(model_path))
    values,keys,names,selector_hash=AUDIENCE_CACHE[cache_key]
    positions=info['positions'] if partition=='train' else info['held']
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','task_id'],dtype='int32')
    query=raw.iloc[info['rows'][positions]][['pt_d','task_id']].to_numpy()
    extra=aligned(keys,values,query,20191100+day)
    frame=pd.concat([frame.reset_index(drop=True),pd.DataFrame(extra,columns=['task_audience__'+name for name in names])],axis=1)
    audit=dict(parent_audit,shape=list(frame.shape),frame_sha256=frame_hash(frame),
        audience_columns=names,audience_selector_model_sha256=selector_hash,
        audience_features_sha256=array_hash(extra),audience_keys_sha256=array_hash(keys),
        selector_fitted_strictly_before_prediction_date=True)
    if partition=='held':
        AUDIENCE_CACHE.pop(cache_key,None)
        if independent:base_rows.REPLAY.pop(day,None)
    gc.collect()
    emit('task_audience_row_input_ready',day=day,partition=partition,independent=independent,shape=list(frame.shape))
    return frame,cats,info,audit


'''


def main():
    assert not TARGET.exists(),'Preserve previous versions'
    source=(HERE/'cross_user_pair_ranker_v1.py').read_text(encoding='utf8')
    source=source.replace('from within_user_ranking_core import within_prediction',
        'from within_user_ranking_core import within_prediction\nimport cross_user_pair_ranker_v1 as base_rows\nimport task_audience_rate_v1 as audience\nfrom task_audience_row_core import select_columns,aligned')
    start=source.index('def inputs(');end=source.index('def dependencies():',start)
    source=source[:start]+INPUTS+source[end:]
    changes=[
        ("OUT = HERE / 'runs/cross_user_pair_ranker_v1'","OUT = HERE / 'runs/task_audience_row_ranker_v1'"),
        ("PREFLIGHT = HERE / 'runs/cross_user_pair_preflight_v1'","PREFLIGHT = HERE / 'runs/task_audience_row_preflight_v1'"),
        ("BASE = HERE / 'runs/compose_daily_objectives_v1'","AUDIENCE=HERE/'runs/task_audience_rate_v1'\nBASES={day:AUDIENCE for day in [15,16]}"),
        ('BASE/f','BASES[day]/f'),
        ("read(BASE/'protocol.json')","read(BASES[16]/'protocol.json')"),
        ('for folder in [SOURCE, QUALITY, BASE]:','for folder in [SOURCE, QUALITY, *BASES.values(), AUDIENCE]:'),
        ("HERE/'run_cross_user_pair_ranker_v1.ps1'","HERE/'run_task_audience_row_ranker_v1.ps1',HERE/'prepare_task_audience_row_ranker_v1.py',HERE/'task_audience_row_core.py',HERE/'test_task_audience_row_core.py',HERE/'task_audience_rate_v1.py',HERE/'task_audience_core.py',HERE/'cross_user_pair_ranker_v1.py'"),
        ("'global_pair_d6'","'audience_row_d6'"),("'global_pair_d8'","'audience_row_d8'"),
        ('REPLAY = {}','REPLAY = {}\nAUDIENCE_CACHE = {}'),
        ("baseline_run=str(BASE)","baseline_runs={day:str(path) for day,path in BASES.items()}"),
        ("features='680 native row columns plus all337 news-quality columns'","features='1017 row features plus192 task audience features selected only by a fixed prior-trained600-tree depth6 task model'"),
        ("emit('cross_user_pair_","emit('task_audience_row_")
    ]
    for old,new in changes:
        assert old in source,old
        source=source.replace(old,new)
    anchor='    return list(dict.fromkeys(paths))'
    extra="""    for day in [15,16]:
        paths += [AUDIENCE/f'day{day}_audience_d6_trees600.cbm',
                  AUDIENCE/f'input_day{day}/frame.pkl',AUDIENCE/f'input_day{day}/metadata.npz',
                  AUDIENCE/f'input_day{day}/contract.json',AUDIENCE/f'input_day{day}/files_locked.json']
"""
    assert anchor in source
    source=source.replace(anchor,extra+anchor)
    compile(source,str(TARGET),'exec')
    TARGET.write_text(source,encoding='utf8')
    print(TARGET)


if __name__=='__main__':main()
