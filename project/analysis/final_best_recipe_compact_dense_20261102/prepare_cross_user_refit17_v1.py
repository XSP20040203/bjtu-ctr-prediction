"""Create an immutable final-date row refit, gated on the strict real16 improvement."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
TARGET=HERE/'export_cross_user_refit17_v1.py'

TRAINING='''
def training_design(make_pool=True):
    # Extend the exact validated prior sample with genuine16 positives and sampled16 negatives.
    info=ranker.metadata(16)
    raw=info['raw']
    dates=raw.pt_d.to_numpy()[info['rows']]
    labels=raw.label.to_numpy()[info['rows']]
    assert dates.max()==20191116 and raw.pt_d.max()==20191116
    prior=info['positions']
    assert np.all(dates[prior]<20191116) and np.isin(labels,[0,1]).all()
    fresh=np.flatnonzero(dates==20191116)
    positive=fresh[labels[fresh]==1]
    negative=fresh[labels[fresh]==0]
    initial=np.random.default_rng(4163).choice(negative,len(negative)//3,replace=False)
    sampled=np.random.default_rng(151648).choice(initial,len(initial)//4,replace=False)
    subset=np.sort(np.r_[prior,positive,sampled])
    assert len(np.unique(subset))==len(subset)
    assert np.isin(positive,subset).all()
    order,groups,pairs=ranker.design(dates[subset],labels[subset],20191117)
    positions=subset[order]
    query=raw.iloc[info['rows'][positions]][['pt_d','user_id']].to_numpy()
    quality=np.load(ranker.QUALITY/'quality16.npz')
    aligned=quality_rows(quality['keys'],quality['values'],query)
    native=read(SOURCE/'input_contract16.json')
    path=SOURCE/'input_day16/features.npy'
    assert digest(path)==native['matrix_sha256']
    matrix=np.load(path,mmap_mode='r')
    frame,cats,_=full_frame(matrix,positions,native['feature_names'],native['categorical_columns'],aligned)
    target=labels[positions]
    audit=dict(prediction_date=20191117,maximum_training_date=int(dates[positions].max()),
        rows=len(positions),clicks=int(target.sum()),pairs=len(pairs),categories=cats,
        training_raw_rows_sha256=array_hash(info['rows'][positions]),target_sha256=array_hash(target),
        groups_sha256=array_hash(groups),pairs_sha256=array_hash(pairs),frame_sha256=frame_hash(frame),
        validated_prior_training_sample_preserved=True,all_genuine16_positive_rows_added=True,
        original_prior_rows=len(prior),new16_rows=len(positive)+len(sampled),
        only_genuine_ad_labels_strictly_before_prediction_date=True,inferred17_labels_read=False)
    pool=Pool(frame,label=target,group_id=groups,pairs=pairs,cat_features=cats) if make_pool else None
    del frame,matrix,info,raw,quality,aligned
    gc.collect()
    return pool,audit


def fit():
    protocol=check_sources()
    assert not (RUN/'fit_locked.json').exists() and not (RUN/'refit17.cbm').exists()
    selected=protocol['selected']
    record=next(item for item in read(RANKER/'models_locked.json')['models'] if item['name']==selected['name'])
    params=dict(read(RANKER/'protocol.json')['params'],depth=record['depth'],iterations=record['trees'])
    pool,audit=training_design()
    assert audit['maximum_training_date']==20191116
    write(RUN/'training17_input_contract.json',audit)
    emit('cross_user_refit17_started',rows=audit['rows'],trees=record['trees'],pairs=audit['pairs'])
    model=CatBoostRanker(**params)
    model.fit(pool)
    expected=model.predict(pool,thread_count=2)
    save_model(model,RUN/'refit17.cbm')
    actual=load_model(RUN/'refit17.cbm').predict(pool,prediction_type='RawFormulaVal',thread_count=2)
    np.testing.assert_array_equal(expected,actual)
    write(RUN/'fit_locked.json',dict(status='passed',model_sha256=digest(RUN/'refit17.cbm'),
        training_contract_sha256=digest(RUN/'training17_input_contract.json'),
        maximum_training_date=20191116,selected_original16_recipe=selected,
        training_serialization_replay_exact=True,inferred17_labels_read=False))
    check_sources()
    emit('cross_user_refit17_model_locked')

'''


def main():
    assert not TARGET.exists(), 'Preserve all final exporters'
    source=(HERE/'export_cross_user_pair_v1.py').read_text(encoding='utf8')
    changes=[
        ('from catboost import Pool','from catboost import CatBoostRanker,Pool'),
        ('from catboost_userday_residual import load_model','from catboost_userday_residual import load_model,save_model'),
        ('from cross_user_pair_ranker_v1 import prediction as row_prediction',
         'import cross_user_pair_ranker_v1 as ranker\nfrom cross_user_pair_ranker_v1 import prediction as row_prediction'),
        ("RUN=HERE/'cross_user_pair_run_v1'","RUN=HERE/'cross_user_refit17_run_v1'"),
        ('output/best_recipe_cross_user_pair_20191117_20261102','output/best_recipe_cross_user_refit17_20191117_20261102'),
        ("paths=[Path(__file__),HERE/'prepare_cross_user_pair_export_v1.py'",
         "paths=[Path(__file__),HERE/'prepare_cross_user_refit17_v1.py',HERE/'export_cross_user_pair_v1.py',HERE/'prepare_cross_user_pair_export_v1.py'"),
        ('new_model_training=False','new_model_training=True'),
        ('correction_training_max_date=20191115','correction_training_max_date=20191116'),
        ("    model_path=RANKER/(selected['name']+'.cbm')\n    assert digest(model_path)==model_record['model_sha256']",
         "    if day==20191117:\n        fitted=read(RUN/'fit_locked.json')\n        assert fitted['status']=='passed' and fitted['maximum_training_date']==20191116\n        model_path=RUN/'refit17.cbm'\n        assert digest(model_path)==fitted['model_sha256']\n    else:\n        model_path=RANKER/(selected['name']+'.cbm')\n        assert digest(model_path)==model_record['model_sha256']"),
        ("    manifest=read(OUTPUT/'prediction_manifest.json')",
         "    _,training_audit=training_design(False)\n    assert training_audit==read(RUN/'training17_input_contract.json')\n    assert digest(RUN/'training17_input_contract.json')==read(RUN/'fit_locked.json')['training_contract_sha256']\n    manifest=read(OUTPUT/'prediction_manifest.json')"),
        ("day17_group_correction_independently_replayed=True,maximum_prediction_replay_error=error,",
         "day17_group_correction_independently_replayed=True,maximum_prediction_replay_error=error,\n        development16_uses_original_strict_prior_model=True,final17_refitted_through_genuine16=True,\n        refit_training_design_independently_replayed=True,"),
        ('delivery_cross_user_pair_v1.json','delivery_cross_user_refit17_v1.json'),
        ("choices=['freeze','predict','verify','all']","choices=['freeze','fit','predict','verify','all']"),
        ('        if freeze():\n            predict()','        if freeze():\n            fit()\n            predict()'),
        ("{'freeze':freeze,'predict':predict,'verify':verify}","{'freeze':freeze,'fit':fit,'predict':predict,'verify':verify}"),
    ]
    for old,new in changes:
        assert old in source,old
        source=source.replace(old,new)
    source=source.replace('\ndef forecast(day,independent=False):','\n'+TRAINING+'\ndef forecast(day,independent=False):')
    compile(source,str(TARGET),'exec')
    TARGET.write_text(source,encoding='utf8')
    print(TARGET)


if __name__=='__main__':
    main()
