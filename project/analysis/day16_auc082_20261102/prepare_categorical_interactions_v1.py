"""Prepare a separately frozen categorical-interaction probability experiment."""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def generated_source():
    source = (HERE/'pointwise_compact_v1.py').read_text(encoding='utf8')
    replacements = [
        ('Weighted pointwise logloss on complete compact row and unordered snapshot features.',
         'Pointwise category-pair interactions on strictly prior compact row features.'),
        ("OUT = HERE / 'runs/pointwise_compact_v1'", "OUT = HERE / 'runs/categorical_interactions_v1'"),
        ("PREFLIGHT = HERE / 'runs/pointwise_compact_preflight_v1'", "PREFLIGHT = HERE / 'runs/categorical_interactions_preflight_v1'"),
        ("BASES={day:HERE/'runs/sparse_cross_ranker_v1' for day in [15,16]}",
         "BASES={day:HERE/'runs/pointwise_compact_v1' for day in [15,16]}"),
        ("dict(name='pointwise_d5', depth=5)", "dict(name='interactions_d6', depth=6)"),
        ("dict(name='pointwise_d7', depth=7)", "dict(name='interactions_d8', depth=8)"),
        ('CHECKPOINTS = [600, 1800, 3600]', 'CHECKPOINTS = [400, 1200, 2400]'),
        ('class_weights=[1.,3.]', 'class_weights=[1.,1.]'),
        ('l2_leaf_reg=40.', 'l2_leaf_reg=20.'),
        ('max_ctr_complexity=1', 'max_ctr_complexity=2'),
        ('random_seed=151656', 'random_seed=151657'),
        ("HERE/'prepare_pointwise_compact_v1.py'", "HERE/'prepare_categorical_interactions_v1.py', HERE/'pointwise_compact_v1.py'"),
        ("HERE/'run_pointwise_compact_v1.ps1',HERE/'test_pointwise_compact_v1.py'",
         "HERE/'run_categorical_interactions_v1.ps1', HERE/'start_categorical_interactions_queue_v1.ps1'"),
        ('pointwise binary labels with positive class weight3', 'pointwise binary labels with equal class weights and categorical pairs'),
        ('Pointwise weighted binary Logloss; zero-click users retained; date-isolated source rows',
         'Pointwise binary Logloss with category-pair CTRs; zero-click users retained; date-isolated source rows'),
        ("emit('pointwise_compact_", "emit('categorical_interactions_"),
    ]
    for old, new in replacements:
        assert old in source, old
        source = source.replace(old, new)
    return source


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    target = HERE/'categorical_interactions_v1.py'
    source = generated_source()
    compile(source, str(target), 'exec')
    if args.check:
        assert target.read_text(encoding='utf8') == source
        print('Categorical interaction source independently regenerated exactly', flush=True)
    else:
        assert not target.exists(), 'Preserve frozen sources'
        target.write_text(source, encoding='utf8')
        print(target, flush=True)
