"""Prepare an immutable 1209-feature export from the verified row exporter."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / 'export_task_audience_row_v1.py'


def main():
    assert not TARGET.exists(), 'Preserve previous export sources'
    source = (HERE / 'export_cross_user_pair_v1.py').read_text(encoding='utf8')
    helper = '''def audience_for_raw(raw, day, independent):
    auxiliary, meta, audit = audience_export.inputs(day, independent)
    names = read(RANKER/'input_contract16.json')['audience_columns']
    values = align_audience(meta['keys'], auxiliary[names].to_numpy(dtype='float32'),
                            raw[['pt_d','task_id']].to_numpy(), day)
    contract = dict(audience_frame_sha256=audit['frame_sha256'],
                    selected_audience_columns=names, extra_sha256=array_hash(values))
    del auxiliary, meta
    gc.collect()
    return values, contract


def forecast(day,independent=False):'''
    changes = [
        ('from cross_user_pair_ranker_v1 import prediction as row_prediction',
         'from task_audience_row_ranker_v1 import prediction as row_prediction\n'
         'from task_audience_row_core import aligned as align_audience\n'
         'import export_task_audience_v1 as audience_export'),
        ("RUN=HERE/'cross_user_pair_run_v1'", "RUN=HERE/'task_audience_row_run_v1'"),
        ('output/best_recipe_cross_user_pair_20191117_20261102',
         'output/best_recipe_task_audience_row_20191117_20261103'),
        ('output/best_recipe_daily_rate_refit17_v2_20191117_20261102',
         'output/best_recipe_task_audience_20191117_20261103'),
        ("PARENT_RUN=HERE/'daily_rate_refit17_run_v2'", "PARENT_RUN=HERE/'task_audience_run_v1'"),
        ("RANKER=base.EXP/'runs/cross_user_pair_ranker_v1'",
         "RANKER=base.EXP/'runs/task_audience_row_ranker_v1'"),
        ("paths=[Path(__file__),HERE/'prepare_cross_user_pair_export_v1.py'",
         "paths=[Path(__file__),HERE/'prepare_task_audience_row_export_v1.py',"
         "HERE/'export_cross_user_pair_v1.py',HERE/'export_task_audience_v1.py',"
         "HERE/'prepare_cross_user_pair_export_v1.py'"),
        ('def forecast(day,independent=False):', helper),
        ('    raw=base.query_raw(day)\n',
         '    raw=base.query_raw(day)\n    extra, audience_contract=audience_for_raw(raw,day,independent)\n'),
        ('    assert frame.shape==(len(raw),1017)',
         "    names=audience_contract['selected_audience_columns']\n"
         "    frame=pd.concat([frame.reset_index(drop=True),pd.DataFrame(extra,columns=['task_audience__'+name for name in names])],axis=1)\n"
         "    del extra\n    gc.collect()\n    assert frame.shape==(len(raw),1209)"),
        ('        raw_model_margin_sha256=array_hash(margin),advertisement_prediction_date_labels_masked=True)',
         '        raw_model_margin_sha256=array_hash(margin),audience_contract=audience_contract,advertisement_prediction_date_labels_masked=True)'),
        ('day17_full1017_feature_frame_and_model_margins_replayed_exactly=True',
         'day17_full1209_feature_frame_and_model_margins_replayed_exactly=True'),
        ('delivery_cross_user_pair_v1.json', 'delivery_task_audience_row_v1.json'),
    ]
    for old, new in changes:
        assert source.count(old) == 1, (old, source.count(old))
        source = source.replace(old, new)
    source = source.replace('cross_user17_', 'task_audience_row17_')
    compile(source, str(TARGET), 'exec')
    TARGET.write_text(source, encoding='utf8')
    print(TARGET)


if __name__ == '__main__':
    main()
