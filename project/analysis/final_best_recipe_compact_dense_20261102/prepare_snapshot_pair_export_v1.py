"""Create a frozen-on-use snapshot-quality export gated by verified real16 improvement."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
TARGET=HERE/'export_snapshot_pair_v1.py'


def main():
    assert not TARGET.exists(),'Preserve all export versions'
    source=(HERE/'export_cross_user_pair_v1.py').read_text(encoding='utf8')
    changes=[
        ('from cross_user_pair_ranker_v1 import prediction as row_prediction',
         'from snapshot_pair_ranker_v1 import prediction as row_prediction\nfrom snapshot_quality_core import AD_FIELDS,NEWS_FIELDS,build as build_snapshot'),
        ("RUN=HERE/'cross_user_pair_run_v1'","RUN=HERE/'snapshot_pair_run_v1'"),
        ('output/best_recipe_cross_user_pair_20191117_20261102','output/best_recipe_snapshot_pair_20191117_20261102'),
        ('output/best_recipe_daily_rate_refit17_v2_20191117_20261102','output/best_recipe_cross_user_refit17_20191117_20261102'),
        ("PARENT_RUN=HERE/'daily_rate_refit17_run_v2'","PARENT_RUN=HERE/'cross_user_refit17_run_v1'"),
        ("RANKER=base.EXP/'runs/cross_user_pair_ranker_v1'","RANKER=base.EXP/'runs/snapshot_pair_ranker_v1'"),
        ("paths=[Path(__file__),HERE/'prepare_cross_user_pair_export_v1.py'",
         "paths=[Path(__file__),HERE/'prepare_snapshot_pair_export_v1.py',HERE/'export_cross_user_pair_v1.py',HERE/'prepare_cross_user_pair_export_v1.py'"),
        ("    assert frame.shape==(len(raw),1017)",
         "    ad_path=base.ROOT/'2026ml'/('train.csv' if day==20191116 else 'final_test.csv')\n    ad_context=pd.read_csv(ad_path,usecols=AD_FIELDS,dtype={name:(str if name=='u_newsCatInterestsST' else 'int64') for name in AD_FIELDS},keep_default_na=False)\n    ad_context=ad_context.loc[ad_context.pt_d.eq(day),AD_FIELDS].reset_index(drop=True)\n    news_paths=[base.ROOT/'2026ml/train_feeds.csv']\n    if day==20191117:\n        news_paths.append(base.ROOT/'2026ml/test_feeds.csv')\n    news=pd.concat([pd.read_csv(path,usecols=NEWS_FIELDS,dtype={'u_newsCatInterestsST':str,'i_docId':str},keep_default_na=False)[NEWS_FIELDS] for path in news_paths],ignore_index=True)\n    values,names=build_snapshot(ad_context,news,day)\n    assert len(values)==len(raw)\n    frame=pd.concat([frame.reset_index(drop=True),pd.DataFrame(values,columns=names)],axis=1)\n    del ad_context,news,values\n    assert frame.shape==(len(raw),1257)"),
        ('day17_full1017_feature_frame_and_model_margins_replayed_exactly','day17_full1257_feature_frame_and_model_margins_replayed_exactly'),
        ('delivery_cross_user_pair_v1.json','delivery_snapshot_pair_v1.json'),
        ('cross_user17_','snapshot17_'),
    ]
    for old,new in changes:
        assert old in source,old
        source=source.replace(old,new)
    compile(source,str(TARGET),'exec')
    TARGET.write_text(source,encoding='utf8')
    print(TARGET)


if __name__=='__main__':
    main()
