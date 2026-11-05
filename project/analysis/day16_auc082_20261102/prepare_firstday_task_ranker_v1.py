"""Create a separate ranker trained on uncensored first-day task examples."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE/'firstday_task_ranker_v1.py'
INPUTS = '''def inputs(day, partition, independent=False):
    frame,cats,info,parent_audit=base_rows.inputs(day,partition,independent)
    history=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','task_id'],dtype='int32')[['pt_d','task_id']]
    training_rows=info['rows'][info['positions']]
    order,groups,pairs=firstday_design(history,training_rows,info['target'],20191100+day)
    info=dict(info)
    info['positions']=info['positions'][order]
    info['target']=info['target'][order]
    info['groups']=groups
    info['pairs']=pairs
    info['max_training_date']=int(history.pt_d.to_numpy()[info['rows'][info['positions']]].max())
    if partition=='train':
        frame=frame.iloc[order].reset_index(drop=True)
    indices,masked_names=identity_columns(info['source']['feature_names'])
    columns=[frame.columns[i] for i in indices]
    for name in columns:
        frame[name]=0
    audit=dict(parent_audit,shape=list(frame.shape),frame_sha256=frame_hash(frame),
        training_rows_sha256=array_hash(info['rows'][info['positions']]),
        training_targets_sha256=array_hash(info['target']),pairs_sha256=array_hash(pairs),
        groups_sha256=array_hash(groups),max_training_date=info['max_training_date'],
        firstday_training_rows=len(order),firstday_training_clicks=int(info['target'].sum()),
        firstday_min_training_date=20191112,
        masked_identity_columns=columns,masked_identity_feature_names=masked_names,
        shared_advertiser_and_application_inputs_preserved=True)
    if independent and partition=='held':
        base_rows.REPLAY.pop(day,None)
    gc.collect()
    emit('firstday_task_input_ready',day=day,partition=partition,independent=independent,
         training_rows=len(order),training_clicks=int(info['target'].sum()))
    return frame,cats,info,audit


'''


def main():
    assert not TARGET.exists(), 'Preserve all model versions'
    source=(HERE/'cold_task_ranker_v1.py').read_text(encoding='utf8')
    source=source.replace('from cold_task_core import identity_columns,cold_mask',
        'from cold_task_core import identity_columns,cold_mask\nfrom firstday_task_core import firstday_design')
    start=source.index('def inputs(')
    end=source.index('def dependencies():',start)
    source=source[:start]+INPUTS+source[end:]
    for old,new in [
        ('runs/cold_task_ranker_v1','runs/firstday_task_ranker_v1'),
        ('runs/cold_task_preflight_v1','runs/firstday_task_preflight_v1'),
        ('cold_task_d6','firstday_task_d6'),('cold_task_d8','firstday_task_d8'),
        ('run_cold_task_ranker_v1.ps1','run_firstday_task_ranker_v1.ps1'),
        ("HERE/'prepare_cold_task_ranker_v1.py'", "HERE/'prepare_firstday_task_ranker_v1.py',HERE/'firstday_task_core.py',HERE/'test_firstday_task_core.py'"),
        ("'All earlier positive rows and one quarter of fixed sampled negatives; random1024-row date queries;48 negative pairs per positive'",
         "'Uncensored first-day task rows from Nov12 through cutoff-1; all available sampled positives and quarter sampled negatives; random1024-row date queries;48 negative pairs per positive'"),
        ("emit('cold_task_", "emit('firstday_task_")
    ]:
        assert old in source, old
        source=source.replace(old,new)
    compile(source,str(TARGET),'exec')
    TARGET.write_text(source,encoding='utf8')
    print(TARGET)


if __name__=='__main__':
    main()
