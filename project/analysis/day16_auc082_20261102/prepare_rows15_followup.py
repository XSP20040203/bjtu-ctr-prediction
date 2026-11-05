"""Create isolated day15 versions; preserve already hash-bound day16 sources."""
import json
from pathlib import Path
from fit_stacker import HERE,digest,write


def replace_checked(text,before,after):
    if before not in text:raise AssertionError('Expected source fragment missing: '+before)
    return text.replace(before,after)


def main():
    rolling=HERE/'runs/rolling_daily15_v1'
    verification=json.loads((rolling/'verification.json').read_text(encoding='utf8'))
    assert verification['status']=='passed' and not verification['evaluation15_labels_fit']
    assert not verification['held_label_feature_selection']
    schema=json.loads((rolling/'input_contract.json').read_text(encoding='utf8'))['schema']
    write(rolling/'news_schema.json',schema)
    source=HERE/'global_row_context.py'
    destination=HERE/'global_row_context15.py'
    text=source.read_text(encoding='utf8')
    changes=[('20191116','20191115'),
             ('from feed_behavior_userday import prepare as daily_inputs','from rolling_daily15 import prepare as daily_inputs'),
             ('global_row_context_v1','global_row_context15_v1'),
             ('runs/feed_behavior_userday_v1/binary_15.txt','runs/rolling_daily15_v1/binary_l15_trees700.txt'),
             ('runs/feed_behavior_userday_v1/news_schema.json','runs/rolling_daily15_v1/news_schema.json'),
             ('runs/previous_day_residual_v1','runs/rolling_daily15_v1'),
             ('daily, _ = daily_inputs(schema)','daily, _ = daily_inputs()'),
             ('day16_indices.npy','day15_indices.npy'),('421983','424780'),('3724','3523'),
             ('evaluation16_labels_fit','evaluation15_labels_fit'),
             ('records[best]["auc"] >= .82','records[best]["auc"] >= .815'),
             ('HERE/"feed_behavior_userday.py"','HERE/"rolling_daily15.py"')]
    for before,after in changes:text=replace_checked(text,before,after)
    if destination.exists():raise FileExistsError(destination)
    destination.write_text(text,encoding='utf8')
    cat_source=HERE/'catboost_global_rows.py'
    cat_destination=HERE/'catboost_global_rows15.py'
    text=cat_source.read_text(encoding='utf8')
    cat_changes=[('20191116','20191115'),('20191115\'','20191114\''),
                 ('from global_row_context import selected_training,CAT','from global_row_context15 import selected_training,CAT'),
                 ('catboost_global_rows_v1','catboost_global_rows15_v1'),
                 ('global_row_context_v1','global_row_context15_v1'),
                 ('runs/compose_verified_v1','runs/rolling_daily15_v1'),
                 ('global_row_context.py','global_row_context15.py'),
                 ('421983','424780'),('3724','3523'),
                 ('evaluation16_labels_fit','evaluation15_labels_fit'),
                 ('development16_model_selection','development15_model_selection'),
                 ('best[\'auc\']>=.82','best[\'auc\']>=.815')]
    # Dates in code comparisons use unquoted integers. Only the recorded maximum
    # training date needs the second date replacement below.
    cat_changes.pop(1)
    for before,after in cat_changes:text=replace_checked(text,before,after)
    text=replace_checked(text,"'max_training_date':20191115","'max_training_date':20191114")
    if cat_destination.exists():raise FileExistsError(cat_destination)
    cat_destination.write_text(text,encoding='utf8')
    write(HERE/'runs/rows15_followup_sources.json',{
        'goal15':.815,'goal16':.82,'held15_label_feature_selection':False,
        'original_sources_sha256':{str(p):digest(p) for p in [source,cat_source]},
        'new_sources_sha256':{str(p):digest(p) for p in [destination,cat_destination]},
        'selector_fit_max_date':20191114,'day15_base_auc':verification['auc']})
    print('Isolated day15 row pipelines prepared',flush=True)


if __name__=='__main__':main()
