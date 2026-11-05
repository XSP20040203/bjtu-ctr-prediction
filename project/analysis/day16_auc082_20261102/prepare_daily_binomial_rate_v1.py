"""Create a separate, reviewable binomial experiment from the frozen pair protocol."""
import ast
from pathlib import Path

HERE=Path(__file__).resolve().parent
TARGET=HERE/'daily_binomial_rate_v1.py'
WRAPPER=HERE/'run_daily_binomial_rate_v1.ps1'


def main():
    assert not TARGET.exists() and not WRAPPER.exists(),'Preserve all existing experiment versions'
    original=(HERE/'daily_auc_pair_ranker_v1.py').read_text(encoding='utf8')
    tree=ast.parse(original)
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='training_pool')
    old_pool=ast.get_source_segment(original,function)
    new_pool='''def training_pool(frame,meta,contract,arm):
    training,target,weights=binomial_targets(meta['keys'],meta['counts'],meta['sizes'],
        contract['day'],arm['half_life'])
    assert np.all(meta['keys'][training,0]<contract['day'])
    columns=contract['selected_columns'][arm['name']]
    pool=Pool(frame.iloc[training][columns],label=target,weight=weights,
        baseline=meta['baseline'][training],cat_features=contract['categories'])
    np.testing.assert_array_equal(pool.get_baseline().reshape(-1),meta['baseline'][training])
    return pool,columns,training


def make_model(**params):
    return CatBoost(params)'''
    text=original.replace(old_pool,new_pool)
    replacements=[
        ('Full-quality explicit daily pair models with frozen chronological evaluation.',
         'Full-quality exposure-weighted daily binomial models with chronological evaluation.'),
        ('from catboost import CatBoostRanker,Pool','from catboost import CatBoost,Pool'),
        ('from daily_auc_pair_core import sample_pairs',
         'from daily_auc_pair_core import sample_pairs\nfrom exposure_binomial_core import training_targets as binomial_targets'),
        ("OUT=HERE/'runs/daily_auc_pair_ranker_v1'","OUT=HERE/'runs/daily_binomial_rate_v1'"),
        ("PREFLIGHT=HERE/'runs/daily_auc_pair_ranker_preflight_v1'","PREFLIGHT=HERE/'runs/daily_binomial_rate_preflight_v1'"),
        ("ARMS=[dict(name='pair_d6_top384',depth=6,selector='poisson_d6_top384'),\n      dict(name='pair_d7_top192',depth=7,selector='poisson_d7_top192')]",
         "ARMS=[dict(name='binomial_d6_top384',depth=6,selector='poisson_d6_top384',half_life=None),\n      dict(name='binomial_recent2_d7_top192',depth=7,selector='poisson_d7_top192',half_life=2.)]"),
        ("PARAMS=dict(loss_function='PairLogit'","PARAMS=dict(loss_function='CrossEntropy'"),
        ('random_seed=151646','random_seed=151647'),
        ("HERE/'run_daily_auc_pair_ranker_v1.ps1',",
         "HERE/'run_daily_binomial_rate_v1.ps1',HERE/'prepare_daily_binomial_rate_v1.py',\n        HERE/'daily_auc_pair_ranker_v1.py',HERE/'exposure_binomial_core.py',HERE/'test_exposure_binomial_core.py',"),
        ('CatBoostRanker(**dict','make_model(**dict'),
        ('prediction=model.predict(sample,thread_count=2)',
         "prediction=model.predict(sample,prediction_type='RawFormulaVal',thread_count=2)"),
        ("objective='Explicit click vs nonclick exposure pairs within earlier advertisement dates'",
         "objective='CrossEntropy on click/exposure soft labels weighted by earlier exposure counts and optional recency'"),
        ("held_counts_masked=True,all337_quality_columns=True,selection_uses_held_labels=False)",
         "held_counts_masked=True,all337_quality_columns=True,selection_uses_held_labels=False,\n        ranking_pairs_not_used_in_binomial_fitting=True)"),
        ('daily_auc_pairs_','daily_binomial_')]
    for old,new in replacements:
        assert old in text,old
        text=text.replace(old,new)
    compile(text,str(TARGET),'exec')
    wrapper=(HERE/'run_daily_auc_pair_ranker_v1.ps1').read_text(encoding='utf8')
    wrapper=wrapper.replace("@('test_daily_auc_pair_core.py','test_daily_auc_pair_math.py')",
        "@('test_daily_auc_pair_core.py','test_daily_auc_pair_math.py','test_exposure_binomial_core.py')")
    wrapper=wrapper.replace('daily_auc_pair_ranker_v1.py','daily_binomial_rate_v1.py')
    TARGET.write_text(text,encoding='utf8')
    WRAPPER.write_text(wrapper,encoding='utf8')
    print(str(TARGET))
    print(str(WRAPPER))


if __name__=='__main__':
    main()
