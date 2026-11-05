"""Prepare a historical profile-response extension with immutable predecessors."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / 'profile_response_ranker_v1.py'


def main():
    assert not TARGET.exists(), 'Preserve previous profile-response experiments'
    source = (HERE / 'audience_graph_ranker_v1.py').read_text(encoding='utf8')
    source = source.replace('audience_graph', 'profile_response')
    changes = [
        ('from profile_response_core_v1 import build_day as graph_day, NAMES as GRAPH_NAMES',
         'from profile_response_core_v1 import build_day as profile_day, NAMES as PROFILE_NAMES, FIELDS as PROFILE_FIELDS'),
        ("BASES={day:HERE/'runs/task_audience_row_ranker_v1' for day in [15,16]}",
         "BASES={day:HERE/'runs/audience_graph_ranker_v1' for day in [15,16]}"),
        ("name='graph_row_d6'", "name='profile_row_d6'"),
        ("name='graph_row_d8'", "name='profile_row_d8'"),
        ("raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','user_id','task_id','label'],dtype='int32')",
         "raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=PROFILE_FIELDS+['label'])"),
        ("graph_day(raw[['pt_d','user_id','task_id']],masked,int(date),original_rows[local],independent)",
         "profile_day(raw[PROFILE_FIELDS],masked,int(date),original_rows[local],independent)"),
        ('GRAPH_NAMES', 'PROFILE_NAMES'),
        ('graph_columns=', 'profile_response_columns='),
        ('graph_features_sha256=', 'profile_response_features_sha256='),
        ('graph_ad_labels_strictly_before_each_feature_date=', 'profile_labels_strictly_before_each_feature_date='),
        ('graph_unlabeled_exposure_dates_at_most_each_feature_date=True',
         'profile_reference_ad_dates_strictly_before_each_feature_date=True'),
        ('Chronological collaborative graph responses added to1209 audience/row inputs.',
         'Chronological historical profile-response statistics added to1209 audience/row inputs.'),
        ('1209 validated row/audience features plus24 chronological collaborative graph features; graph ranks8/16/32/64 fixed before held scoring',
         '1209 validated row/audience features plus168 strictly historical profile response features; fixed app/owner/slot interactions and smoothing'),
    ]
    for old, new in changes:
        assert old in source, old
        source = source.replace(old, new)
    compile(source, str(TARGET), 'exec')
    TARGET.write_text(source, encoding='utf8')
    print(TARGET)


if __name__ == '__main__':
    main()
