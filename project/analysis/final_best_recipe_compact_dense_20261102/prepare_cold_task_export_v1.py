"""Prepare a gated final-date cold-task export; select only with verified real16 AUC."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
TARGET=HERE/'export_cold_task_v1.py'


def main():
    assert not TARGET.exists(),'Preserve previous exporters'
    source=(HERE/'export_cross_user_pair_v1.py').read_text(encoding='utf8')
    changes=[
        ('from cross_user_pair_ranker_v1 import prediction as row_prediction',
         'from cold_task_ranker_v1 import prediction as row_prediction\nfrom cold_task_core import identity_columns,cold_mask'),
        ("RUN=HERE/'cross_user_pair_run_v1'","RUN=HERE/'cold_task_run_v1'"),
        ('output/best_recipe_cross_user_pair_20191117_20261102','output/best_recipe_cold_task_20191117_20261103'),
        ('output/best_recipe_daily_rate_refit17_v2_20191117_20261102','output/best_recipe_cross_user_refit17_20191117_20261102'),
        ("PARENT_RUN=HERE/'daily_rate_refit17_run_v2'","PARENT_RUN=HERE/'cross_user_refit17_run_v1'"),
        ("RANKER=base.EXP/'runs/cross_user_pair_ranker_v1'","RANKER=base.EXP/'runs/cold_task_ranker_v1'"),
        ("paths=[Path(__file__),HERE/'prepare_cross_user_pair_export_v1.py'",
         "paths=[Path(__file__),HERE/'prepare_cold_task_export_v1.py',HERE/'export_cross_user_pair_v1.py',HERE/'prepare_cross_user_pair_export_v1.py'"),
        ("    assert frame.shape==(len(raw),1017)",
         "    masked,_=identity_columns(native['feature_names'])\n    for column in masked:\n        frame[frame.columns[column]]=0\n    assert frame.shape==(len(raw),1017)"),
        ("    if day==20191116:\n        np.testing.assert_allclose(prediction,np.load(RANKER/'day16_prediction.npy'),rtol=0,atol=2e-14)",
         "    history=pd.read_csv(base.ROOT/'2026ml/train.csv',usecols=['pt_d','task_id'],dtype='int32')[['pt_d','task_id']]\n    mask=cold_mask(history,raw[['pt_d','task_id']],day)\n    prediction[~mask]=current[~mask]\n    if day==20191116:\n        np.testing.assert_allclose(prediction,np.load(RANKER/'day16_prediction.npy'),rtol=0,atol=2e-14)"),
        ("    assert abs(audit['dates']['20191116']['auc']-selected['auc'])<1e-12",
         "    if audit['dates']['20191116']['auc']>selected['auc']+1e-12:\n        emit('cold_task17_export_superseded',candidate_auc=selected['auc'],best_auc=audit['dates']['20191116']['auc'])\n        return False\n    assert abs(audit['dates']['20191116']['auc']-selected['auc'])<1e-12"),
        ('delivery_cross_user_pair_v1.json','delivery_cold_task_v1.json'),
        ('cross_user17_','cold_task17_'),
    ]
    for old,new in changes:
        assert old in source,old
        source=source.replace(old,new)
    compile(source,str(TARGET),'exec')
    TARGET.write_text(source,encoding='utf8')
    print(TARGET)


if __name__=='__main__':
    main()
