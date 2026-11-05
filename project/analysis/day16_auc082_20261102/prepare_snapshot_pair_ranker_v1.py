"""Create a versioned row ranker extending verified inputs with240 exact-context feedback features."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
TARGET=HERE/'snapshot_pair_ranker_v1.py'
INPUTS='''def inputs(day, partition, independent=False):
    frame,cats,info,parent_audit=base_rows.inputs(day,partition,independent)
    positions=info['positions'] if partition=='train' else info['held']
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=AD_FIELDS,
        dtype={name:(str if name=='u_newsCatInterestsST' else 'int64') for name in AD_FIELDS},keep_default_na=False)
    ads=raw.iloc[info['rows'][positions]][AD_FIELDS].reset_index(drop=True)
    del raw
    news=pd.read_csv(ROOT/'2026ml/train_feeds.csv',usecols=NEWS_FIELDS,
        dtype={'u_newsCatInterestsST':str,'i_docId':str},keep_default_na=False)[NEWS_FIELDS]
    values,names=build_snapshot(ads,news,20191100+day)
    del ads,news
    extra=pd.DataFrame(values,columns=names)
    frame=pd.concat([frame.reset_index(drop=True),extra],axis=1)
    assert frame.shape==(len(positions),1257)
    audit=dict(parent_audit,shape=list(frame.shape),frame_sha256=frame_hash(frame),
        snapshot_quality_sha256=array_hash(values),snapshot_feature_names=names,
        only_same_or_earlier_news_dates=True,own_user_excluded_from_content_expectation=True)
    if independent and partition=='held':
        base_rows.REPLAY.pop(day,None)
    del values,extra
    gc.collect()
    emit('snapshot_pair_input_ready',day=day,partition=partition,independent=independent,shape=audit['shape'])
    return frame,cats,info,audit


'''


def main():
    assert not TARGET.exists(),'Preserve all versioned scripts'
    source=(HERE/'cross_user_pair_ranker_v1.py').read_text(encoding='utf8')
    source=source.replace('from within_user_ranking_core import within_prediction',
        'from within_user_ranking_core import within_prediction\nimport cross_user_pair_ranker_v1 as base_rows\nfrom snapshot_quality_core import AD_FIELDS,NEWS_FIELDS,build as build_snapshot')
    start=source.index('def inputs(')
    end=source.index('def dependencies():',start)
    source=source[:start]+INPUTS+source[end:]
    changes=[
        ("OUT = HERE / 'runs/cross_user_pair_ranker_v1'","OUT = HERE / 'runs/snapshot_pair_ranker_v1'"),
        ("PREFLIGHT = HERE / 'runs/cross_user_pair_preflight_v1'","PREFLIGHT = HERE / 'runs/snapshot_pair_preflight_v1'"),
        ("BASE = HERE / 'runs/compose_daily_objectives_v1'","BASE = HERE / 'runs/cross_user_pair_ranker_v1'"),
        ("'global_pair_d6'","'snapshot_pair_d6'"),
        ("'global_pair_d8'","'snapshot_pair_d8'"),
        ("HERE/'run_cross_user_pair_ranker_v1.ps1'",
         "HERE/'run_snapshot_pair_ranker_v1.ps1',HERE/'snapshot_quality_core.py',HERE/'test_snapshot_quality_core.py',HERE/'prepare_snapshot_pair_ranker_v1.py',HERE/'cross_user_pair_ranker_v1.py'"),
        ("features='680 native row columns plus all337 news-quality columns'",
         "features='680 native row columns,337 daily quality and240 exact-context adjusted feedback columns'"),
        ('cross_user_pair_','snapshot_pair_'),
    ]
    # Apply event rename separately; preserve the imported parent and its bound dependencies.
    for old,new in changes[:-1]:
        assert old in source,old
        source=source.replace(old,new)
    source=source.replace("emit('cross_user_pair_","emit('snapshot_pair_")
    compile(source,str(TARGET),'exec')
    TARGET.write_text(source,encoding='utf8')
    print(TARGET)


if __name__=='__main__':
    main()
