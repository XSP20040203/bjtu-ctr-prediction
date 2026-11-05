"""Create a separately versioned gated transfer model for historically unseen tasks."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
TARGET=HERE/'cold_task_ranker_v1.py'
INPUTS='''def inputs(day, partition, independent=False):
    frame,cats,info,parent_audit=base_rows.inputs(day,partition,independent)
    indices,masked_names=identity_columns(info['source']['feature_names'])
    columns=[frame.columns[i] for i in indices]
    for name in columns:
        frame[name]=0
    audit=dict(parent_audit,frame_sha256=frame_hash(frame),
        masked_identity_columns=columns,masked_identity_feature_names=masked_names,
        shared_advertiser_and_application_inputs_preserved=True)
    if independent and partition=='held':
        base_rows.REPLAY.pop(day,None)
    gc.collect()
    emit('cold_task_input_ready',day=day,partition=partition,independent=independent,masked=len(columns))
    return frame,cats,info,audit


'''

CANDIDATES='''def cold_query(day):
    if day not in COLD:
        raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','task_id'],dtype='int32')[['pt_d','task_id']]
        query=raw.loc[raw.pt_d.eq(20191100+day)].reset_index(drop=True)
        COLD[day]=cold_mask(raw,query,20191100+day)
    return COLD[day]


def candidates(day, values, groups, independent=False):
    base=np.load(BASE/f'day{day}_prediction.npy')
    mask=cold_query(day)
    assert len(mask)==len(base)
    yield dict(name='base',mode='base',weight=0.),base.copy()
    for name,margin in values.items():
        for mode in MODES:
            for weight in WEIGHTS:
                record=dict(name=name,mode=mode,weight=weight)
                p=prediction(base,margin,groups,weight,mode,independent)
                p[~mask]=base[~mask]
                yield record,p


'''


def main():
    assert not TARGET.exists(),'Preserve all model versions'
    source=(HERE/'cross_user_pair_ranker_v1.py').read_text(encoding='utf8')
    source=source.replace('from within_user_ranking_core import within_prediction',
        'from within_user_ranking_core import within_prediction\nimport cross_user_pair_ranker_v1 as base_rows\nfrom cold_task_core import identity_columns,cold_mask')
    start=source.index('def inputs(')
    end=source.index('def dependencies():',start)
    source=source[:start]+INPUTS+source[end:]
    start=source.index('def candidates(')
    end=source.index('def candidate_key(',start)
    source=source[:start]+CANDIDATES+source[end:]
    changes=[
        ("OUT = HERE / 'runs/cross_user_pair_ranker_v1'","OUT = HERE / 'runs/cold_task_ranker_v1'"),
        ("PREFLIGHT = HERE / 'runs/cross_user_pair_preflight_v1'","PREFLIGHT = HERE / 'runs/cold_task_preflight_v1'"),
        ("BASE = HERE / 'runs/compose_daily_objectives_v1'","BASE = HERE / 'runs/cross_user_pair_ranker_v1'"),
        ("'global_pair_d6'","'cold_task_d6'"),
        ("'global_pair_d8'","'cold_task_d8'"),
        ("WEIGHTS = [.025, .05, .1, .2, .35, .5, .75]","WEIGHTS = [.05, .1, .2, .35, .5, .75, 1.]"),
        ("MODES = ['add_global', 'replace_global', 'add_within']","MODES = ['add_global', 'replace_global']"),
        ("REPLAY = {}","REPLAY = {}\nCOLD = {}"),
        ("HERE/'run_cross_user_pair_ranker_v1.ps1'",
         "HERE/'run_cold_task_ranker_v1.ps1',HERE/'cold_task_core.py',HERE/'test_cold_task_core.py',HERE/'prepare_cold_task_ranker_v1.py',HERE/'cross_user_pair_ranker_v1.py'"),
        ("features='680 native row columns plus all337 news-quality columns'",
         "features='1017 full-quality inputs with task/creative identities and associated statistics zeroed; preserve advertiser/app metadata'"),
    ]
    for old,new in changes:
        assert old in source,old
        source=source.replace(old,new)
    source=source.replace("emit('cross_user_pair_","emit('cold_task_")
    compile(source,str(TARGET),'exec')
    TARGET.write_text(source,encoding='utf8')
    print(TARGET)


if __name__=='__main__':
    main()
