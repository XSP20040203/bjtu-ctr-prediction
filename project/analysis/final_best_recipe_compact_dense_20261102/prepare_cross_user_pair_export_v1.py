"""Create a separate export only if the new strict-prior row ranker improves real16."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE/'export_cross_user_pair_v1.py'


def main():
    assert not TARGET.exists(), 'Preserve all final export sources'
    source = (HERE/'export_catboost_ranked_v1.py').read_text(encoding='utf8')
    changes = [
        ('from within_user_ranking_core import within_prediction',
         'from cross_user_pair_ranker_v1 import prediction as row_prediction'),
        ("RUN=HERE/'catboost_ranked_run_v1'", "RUN=HERE/'cross_user_pair_run_v1'"),
        ('output/best_recipe_catboost_ranked_20191117_20261102',
         'output/best_recipe_cross_user_pair_20191117_20261102'),
        ('output/best_recipe_ranked_compact_dense_20191117_20261102',
         'output/best_recipe_daily_rate_refit17_v2_20191117_20261102'),
        ("PARENT_RUN=HERE/'ranked_run_v1'", "PARENT_RUN=HERE/'daily_rate_refit17_run_v2'"),
        ("runs/catboost_within_ranker_v2", "runs/cross_user_pair_ranker_v1"),
        ("    assert selected['name']=='day16_pairlogit_d6_trees1600'\n    assert selected['mode']=='add_within' and selected['weight']==.05\n    assert selected['auc']>current['development16_auc']",
         "    if selected['auc']<=current['development16_auc']:\n        emit('cross_user17_export_unnecessary',current_auc=current['development16_auc'],candidate_auc=selected['auc'])\n        return False\n    assert selected['name']!='base' and selected['mode'] in ['add_global','replace_global','add_within']"),
        ("paths=[Path(__file__),PARENT_RUN/'protocol.json'",
         "paths=[Path(__file__),HERE/'prepare_cross_user_pair_export_v1.py',HERE/'export_catboost_ranked_v1.py',PARENT_RUN/'protocol.json'"),
        (",\n        Path(report['supplemental_precision_verification'])]", "]"),
        ("    emit('catboost17_recipe_frozen',selected=selected)",
         "    emit('cross_user17_recipe_frozen',selected=selected)\n    return True"),
        ('prediction=within_prediction(current,margin,groups,selected[\'weight\'],selected[\'mode\'],independent)',
         'prediction=row_prediction(current,margin,groups,selected[\'weight\'],selected[\'mode\'],independent)'),
        ('delivery_catboost_ranked_v1.json', 'delivery_cross_user_pair_v1.json'),
        ('catboost17_', 'cross_user17_'),
        ('        freeze()\n        predict()\n        verify()',
         '        if freeze():\n            predict()\n            verify()'),
    ]
    for old,new in changes:
        assert old in source, old
        source=source.replace(old,new)
    compile(source,str(TARGET),'exec')
    TARGET.write_text(source,encoding='utf8')
    print(TARGET)


if __name__ == '__main__':
    main()
