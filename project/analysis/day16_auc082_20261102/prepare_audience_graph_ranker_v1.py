"""Prepare chronological graph-response ranking without changing frozen predecessors."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / 'audience_graph_ranker_v1.py'


def main():
    assert not TARGET.exists(), 'Preserve previous graph experiments'
    source = (HERE / 'task_audience_row_ranker_v1.py').read_text(encoding='utf8')
    start = source.index('def inputs(day, partition, independent=False):')
    end = source.index('\n\ndef dependencies():', start)
    replacement = '''def inputs(day, partition, independent=False):
    frame,cats,info,parent_audit=audience_rows.inputs(day,partition,independent)
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','user_id','task_id','label'],dtype='int32')
    positions=info['positions'] if partition=='train' else info['held']
    original_rows=info['rows'][positions]
    dates=raw.pt_d.to_numpy()
    labels=raw.label.to_numpy()
    query_dates=dates[original_rows]
    extra=np.empty((len(original_rows),len(GRAPH_NAMES)),dtype='float32')
    for date in np.unique(query_dates):
        local=np.flatnonzero(query_dates==date)
        masked=np.where(dates<date,labels,-1)
        extra[local]=graph_day(raw[['pt_d','user_id','task_id']],masked,int(date),original_rows[local],independent)
        emit('audience_graph_date_ready',prediction_day=day,partition=partition,
             feature_date=int(date),rows=len(local),independent=independent)
    frame=pd.concat([frame.reset_index(drop=True),pd.DataFrame(extra,columns=GRAPH_NAMES)],axis=1)
    audit=dict(parent_audit,shape=list(frame.shape),frame_sha256=frame_hash(frame),
               graph_columns=GRAPH_NAMES,graph_features_sha256=array_hash(extra),
               graph_ad_labels_strictly_before_each_feature_date=True,
               graph_unlabeled_exposure_dates_at_most_each_feature_date=True)
    gc.collect()
    emit('audience_graph_input_ready',day=day,partition=partition,independent=independent,shape=list(frame.shape))
    return frame,cats,info,audit'''
    source = source[:start] + replacement + source[end:]
    source = source.replace("import task_audience_rate_v1 as audience",
        "import task_audience_rate_v1 as audience\nimport task_audience_row_ranker_v1 as audience_rows\n"
        "from audience_graph_core_v1 import build_day as graph_day, NAMES as GRAPH_NAMES")
    changes = [
        ('def dependencies():\n    paths =',
         "def dependencies():\n    graph_check=read(HERE/'audience_graph_core_preflight_v1.json')\n"
         "    assert graph_check['status']=='passed' and graph_check['source_sha256']==digest(HERE/'audience_graph_core_v1.py')\n"
         "    paths ="),
        ("'runs/task_audience_row_ranker_v1'", "'runs/audience_graph_ranker_v1'"),
        ("'runs/task_audience_row_preflight_v1'", "'runs/audience_graph_preflight_v1'"),
        ('BASES={day:AUDIENCE for day in [15,16]}',
         "BASES={day:HERE/'runs/task_audience_row_ranker_v1' for day in [15,16]}"),
        ("name='audience_row_d6'", "name='graph_row_d6'"),
        ("name='audience_row_d8'", "name='graph_row_d8'"),
        ("HERE/'run_task_audience_row_ranker_v1.ps1',HERE/'prepare_task_audience_row_ranker_v1.py'",
         "HERE/'run_audience_graph_ranker_v1.ps1',HERE/'prepare_audience_graph_ranker_v1.py',"
         "HERE/'task_audience_row_ranker_v1.py',HERE/'audience_graph_core_v1.py',HERE/'test_audience_graph_core_v1.py',"
         "HERE/'audit_audience_graph_core_v1.py',HERE/'audience_graph_core_preflight_v1.json'"),
    ]
    for old, new in changes:
        assert source.count(old) == 1, (old, source.count(old))
        source = source.replace(old, new)
    source = source.replace("emit('task_audience_row_", "emit('audience_graph_")
    source = source.replace('1017 row features plus192 task audience features selected only by a fixed prior-trained600-tree depth6 task model',
        '1209 validated row/audience features plus24 chronological collaborative graph features; graph ranks8/16/32/64 fixed before held scoring')
    source = source.replace('Strict-prior cross-user row ranking with all native and337 news-quality features.',
        'Chronological collaborative graph responses added to1209 audience/row inputs.')
    compile(source, str(TARGET), 'exec')
    TARGET.write_text(source, encoding='utf8')
    print(TARGET)


if __name__ == '__main__':
    main()
