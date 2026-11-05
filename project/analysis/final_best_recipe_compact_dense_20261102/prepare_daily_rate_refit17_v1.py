"""Create a versioned final-date refit exporter from the audited validation exporter."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
TARGET=HERE/'export_daily_rate_refit17_v1.py'

TRAINING='''
def training_design(make_pool=True):
    protocol=read(RUN/'protocol.json')
    folder=Path(protocol['chosen_source'])
    selected=protocol['selected']
    native=read(folder/'input_day16/contract.json')
    for filename,expected in read(folder/'input_day16/files_locked.json').items():
        assert digest(folder/'input_day16'/filename)==expected,filename
    frame=pd.read_pickle(folder/'input_day16/frame.pkl')
    saved=np.load(folder/'input_day16/metadata.npz')
    meta={name:saved[name] for name in saved.files}
    raw=pd.read_csv(base.ROOT/'2026ml/train.csv',usecols=['pt_d','user_id','label'],dtype='int32')
    assert raw.pt_d.max()==20191116
    counts=restore_prior_counts(meta['keys'],meta['sizes'],meta['counts'],raw,20191117)
    record=next(item for item in read(folder/'models_locked.json')['models'] if item['name']==selected['name'])
    columns=record['columns']
    assert np.all(meta['keys'][:,0]<20191117)
    assert np.all(meta['counts'][meta['keys'][:,0]==20191116]==-1)
    assert np.all(counts>=0)
    kwargs={}
    if folder.name=='daily_auc_pair_ranker_v1':
        training,pairs,groups=sample_pairs(meta['keys'],meta['sizes'],counts,20191117,48,151646)
        target=counts[training]/meta['sizes'][training]
        kwargs=dict(group_id=groups,pairs=pairs)
        objective_audit=dict(kind='explicit_daily_pairs',pair_count=len(pairs),pairs_sha256=array_hash(pairs),
            date_groups_sha256=array_hash(groups))
    elif folder.name=='daily_binomial_rate_v1':
        training,target,weights=binomial_targets(meta['keys'],counts,meta['sizes'],20191117,record['half_life'])
        kwargs=dict(weight=weights)
        objective_audit=dict(kind='exposure_binomial',weights_sha256=array_hash(weights),half_life=record['half_life'])
    else:
        raise ValueError('Only the two verified daily objectives are declared')
    audit=dict(prediction_date=20191117,maximum_training_date=int(meta['keys'][training,0].max()),
        training_rows=len(training),training_clicks=int(counts[training].sum()),
        training_exposures=int(meta['sizes'][training].sum()),
        training_keys_sha256=array_hash(meta['keys'][training]),target_sha256=array_hash(target),
        restored_click_counts_sha256=array_hash(counts),training_frame_sha256=frame_hash(frame.iloc[training][columns]),
        baseline_sha256=array_hash(meta['baseline'][training]),columns=columns,objective=objective_audit,
        only_genuine_advertisement_labels_strictly_before17=True,inferred17_labels_read=False,
        selected_hyperparameters_frozen_on_real16_before_refit=True)
    assert audit['maximum_training_date']==20191116
    pool=(Pool(frame.iloc[training][columns],label=target,baseline=meta['baseline'][training],
        cat_features=native['categories'],**kwargs) if make_pool else None)
    return pool,audit


def fit():
    protocol=check_sources()
    assert not (RUN/'refit17.cbm').exists() and not (RUN/'fit_locked.json').exists()
    folder=Path(protocol['chosen_source'])
    selected=protocol['selected']
    record=next(item for item in read(folder/'models_locked.json')['models'] if item['name']==selected['name'])
    params=dict(read(folder/'protocol.json')['params'],depth=record['depth'],iterations=record['trees'])
    pool,audit=training_design()
    write(RUN/'training17_input_contract.json',audit)
    emit('daily_rate_refit17_started',source=folder.name,trees=record['trees'],
        training_rows=audit['training_rows'],max_training_date=audit['maximum_training_date'])
    model=CatBoostRanker(**params) if folder.name=='daily_auc_pair_ranker_v1' else CatBoost(params)
    model.fit(pool)
    expected=model.predict(pool,thread_count=2)
    save_model(model,RUN/'refit17.cbm')
    replay=load_model(RUN/'refit17.cbm').predict(pool,prediction_type='RawFormulaVal',thread_count=2)
    np.testing.assert_array_equal(expected,replay)
    write(RUN/'fit_locked.json',dict(status='passed',model_sha256=digest(RUN/'refit17.cbm'),
        training_input_contract_sha256=digest(RUN/'training17_input_contract.json'),
        maximum_training_date=20191116,selected_original16_recipe=selected,
        training_serialization_replay_exact=True,inferred17_labels_read=False))
    check_sources()
    emit('daily_rate_refit17_locked',model_sha256=digest(RUN/'refit17.cbm'))

'''


def main():
    assert not TARGET.exists(),'Preserve all final refit exporter versions'
    source=(HERE/'export_daily_rate_best_v1.py').read_text(encoding='utf8')
    replacements=[
        ('Select a verified daily-rate recipe on real16 and export its day17 predictions.',
         'Refit a verified real16 daily recipe through genuine day16 labels for day17 prediction.'),
        ('from catboost import Pool','from catboost import CatBoost,CatBoostRanker,Pool'),
        ('from catboost_userday_residual import load_model','from catboost_userday_residual import load_model,save_model'),
        ('from daily_auc_pair_math import daily_prediction,rate_baseline',
         'from daily_auc_pair_math import daily_prediction,rate_baseline\nfrom daily_auc_pair_core import sample_pairs\nfrom exposure_binomial_core import training_targets as binomial_targets\nfrom day17_refit_core import restore_prior_counts'),
        ("RUN=HERE/'daily_rate_best_run_v1'","RUN=HERE/'daily_rate_refit17_run_v1'"),
        ("output/best_recipe_daily_rate_20191117_20261102","output/best_recipe_daily_rate_refit17_20191117_20261102"),
        ("paths=[Path(__file__),PARENT_RUN/'protocol.json'",
         "paths=[Path(__file__),HERE/'prepare_daily_rate_refit17_v1.py',HERE/'export_daily_rate_best_v1.py',\n        base.EXP/'day17_refit_core.py',base.EXP/'test_day17_refit_core.py',\n        base.EXP/'daily_auc_pair_core.py',base.EXP/'exposure_binomial_core.py',PARENT_RUN/'protocol.json'"),
        ("frozen_before_daily_rate17_predictions_and_inferred_label_evaluation",
         "frozen_before_final_refit17_training_predictions_and_inferred_label_evaluation"),
        ('new_model_training=False','new_model_training=True'),
        ('correction_training_max_date=20191115','correction_training_max_date=20191116'),
        ("    path=folder/(selected['name']+'.cbm')\n    assert digest(path)==model_record['model_sha256']",
         "    if day==20191117:\n        fitted=read(RUN/'fit_locked.json')\n        assert fitted['status']=='passed' and fitted['maximum_training_date']==20191116\n        path=RUN/'refit17.cbm'\n        assert digest(path)==fitted['model_sha256']\n    else:\n        path=folder/(selected['name']+'.cbm')\n        assert digest(path)==model_record['model_sha256']"),
        ("own_prediction_date_ad_labels_masked=True)",
         "own_prediction_date_ad_labels_masked=True,refit_model_lock_sha256=digest(RUN/'fit_locked.json'))"),
        ("    manifest=read(OUTPUT/'prediction_manifest.json')",
         "    _,training_audit=training_design(False)\n    assert training_audit==read(RUN/'training17_input_contract.json')\n    assert digest(RUN/'training17_input_contract.json')==read(RUN/'fit_locked.json')['training_input_contract_sha256']\n    manifest=read(OUTPUT/'prediction_manifest.json')"),
        ("day16_winning_forecasts_replayed=True,day17_daily_features_rebuilt_twice=True,",
         "day16_winning_forecasts_replayed=True,day17_daily_features_rebuilt_twice=True,\n        development16_score_uses_original_strict_prior_model=True,\n        final_day17_model_refitted_using_genuine_labels_through16=True,\n        refit_training_design_independently_replayed=True,"),
        ("f'按真实16号开发集选择模型，AUC={auc16:.12f}；15/16号曾反复用于开发。\\n'",
         "f'按真实16号开发集选择模型，AUC={auc16:.12f}；15/16号曾反复用于开发。\\n'\n        '最终17号日级模型按选定参数，使用截至16号的真实广告标签重训。\\n'"),
        ("delivery_daily_rate_best_v1.json","delivery_daily_rate_refit17_v1.json"),
        ("choices=['freeze','predict','verify','all']","choices=['freeze','fit','predict','verify','all']"),
        ("        if freeze():\n            predict()","        if freeze():\n            fit()\n            predict()"),
        ("{'freeze':freeze,'predict':predict,'verify':verify}","{'freeze':freeze,'fit':fit,'predict':predict,'verify':verify}"),
        ('daily_rate17_','daily_rate_refit17_')]
    for old,new in replacements:
        assert old in source,old
        source=source.replace(old,new)
    marker='\ndef forecast(day,independent=False):'
    assert marker in source
    source=source.replace(marker,'\n'+TRAINING+marker)
    compile(source,str(TARGET),'exec')
    TARGET.write_text(source,encoding='utf8')
    print(str(TARGET))


if __name__=='__main__':
    main()
