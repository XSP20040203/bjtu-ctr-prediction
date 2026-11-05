"""Pointwise weighted-logloss counterpart of the verified compact snapshot ranker."""
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    target=HERE/'pointwise_compact_v1.py'
    assert not target.exists(),'Preserve previous sources'
    source=(HERE/'unordered_snapshot_ranker_v1.py').read_text(encoding='utf8')
    replacements=[
        ('Full-sample row ranking with label-free unordered click and close state changes.',
         'Weighted pointwise logloss on complete compact row and unordered snapshot features.'),
        ('from catboost import CatBoostRanker, Pool','from catboost import CatBoostClassifier, Pool'),
        ('CatBoostRanker(**','CatBoostClassifier(**'),
        ('before = model.predict(sample, thread_count=2)',
         "before = model.predict(sample, prediction_type='RawFormulaVal', thread_count=2)"),
        ("OUT = HERE / 'runs/unordered_snapshot_ranker_v1'", "OUT = HERE / 'runs/pointwise_compact_v1'"),
        ("PREFLIGHT = HERE / 'runs/unordered_snapshot_preflight_v1'", "PREFLIGHT = HERE / 'runs/pointwise_compact_preflight_v1'"),
        ("BASES={day:HERE/'runs/compact_full_sample_ranker_v1' for day in [15,16]}",
         "BASES={day:HERE/'runs/sparse_cross_ranker_v1' for day in [15,16]}"),
        ("dict(name='unordered_d6', depth=6)", "dict(name='pointwise_d5', depth=5)"),
        ("dict(name='unordered_d8', depth=8)", "dict(name='pointwise_d7', depth=7)"),
        ('CHECKPOINTS = [600, 1600, 2400]','CHECKPOINTS = [600, 1800, 3600]'),
        ("loss_function='PairLogit', learning_rate=.035", "loss_function='Logloss', class_weights=[1.,3.], learning_rate=.03"),
        ('random_seed=151654','random_seed=151656'),
        ("return Pool(frame, label=info['target'], group_id=info['groups'], pairs=info['pairs'], cat_features=cats)",
         "return Pool(frame, label=info['target'], cat_features=cats)"),
        ("HERE/'prepare_unordered_snapshot_ranker_v1.py'", "HERE/'prepare_pointwise_compact_v1.py'"),
        ("HERE/'run_unordered_snapshot_ranker_v1.ps1'", "HERE/'run_pointwise_compact_v1.ps1',HERE/'test_pointwise_compact_v1.py'"),
        ('All earlier positives and all fixed one-third source negatives; no second downsample;1024-row date queries;64 pairs per positive',
         'All earlier positives and all fixed one-third source negatives; pointwise binary labels with positive class weight3'),
        ('Cross-user within-date row PairLogit; zero-click users retained',
         'Pointwise weighted binary Logloss; zero-click users retained; date-isolated source rows'),
        ("emit('unordered_snapshot_", "emit('pointwise_compact_"),
    ]
    for old,new in replacements:
        assert old in source,old
        source=source.replace(old,new)
    compile(source,str(target),'exec')
    target.write_text(source,encoding='utf8')
    print(target)


if __name__=='__main__':main()
