"""Preserve the frozen v1 sources and generate the GPU memory-option API repair."""
from pathlib import Path
import ast
import hashlib
import json

HERE = Path(__file__).resolve().parent


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    native = json.loads((HERE / 'full_rich_gpu_native_preflight_v2.json').read_text(encoding='utf-8'))
    assert native['status'] == 'passed'
    pointwise = (HERE / 'full_rich_negative_v1.py').read_text(encoding='utf-8')
    pointwise = pointwise.replace('full_rich_negative_v1', 'full_rich_negative_v2')
    pointwise = pointwise.replace('full_rich_negative_preflight_v1', 'full_rich_negative_preflight_v2')
    pointwise = once(pointwise, 'pinned_memory_size=1073741824', "pinned_memory_size='1GB'")
    pointwise = once(pointwise,
        "    pool = Pool(primary, label=labels[selected], cat_features=cats)\n    model.fit(pool)",
        "    pool = Pool(primary, label=labels[selected], cat_features=cats)\n"
        "    gpu_probe = CatBoost(dict(PARAMS, iterations=20, depth=4))\n"
        "    gpu_probe._prepare_train_params(pool)\n"
        "    del gpu_probe\n    model.fit(pool)")
    pointwise = once(pointwise, 'full_gpu_capacity_proven=False, held_labels_read=False, serialization_error=0.,',
        "full_gpu_capacity_proven=False, held_labels_read=False, serialization_error=0.,\n"
        "        gpu_native_training_parameter_preparation='passed',")
    pointwise = once(pointwise,
        "    paths += [HERE / name for name in ('conditional_daily.py'",
        "    paths += [HERE / 'full_rich_gpu_native_preflight_v2.json',\n"
        "              HERE / 'full_rich_gpu_recovery_v2.json', Path(__file__).with_name('prepare_full_rich_gpu_recovery_v2.py')]\n"
        "    paths += [HERE / name for name in ('conditional_daily.py'")
    pairwise = (HERE / 'full_rich_pair_ranker_v1.py').read_text(encoding='utf-8')
    pairwise = pairwise.replace('full_rich_negative_v1', 'full_rich_negative_v2')
    pairwise = pairwise.replace('full_rich_pair_ranker_v1', 'full_rich_pair_ranker_v2')
    pairwise = pairwise.replace('full_rich_pair_preflight_v1', 'full_rich_pair_preflight_v2')
    pairwise = once(pairwise,
        "    pool = Pool(data, label=labels[chosen[order]], group_id=groups, pairs=pairs, cat_features=declared['categories'])\n    model.fit(pool)",
        "    pool = Pool(data, label=labels[chosen[order]], group_id=groups, pairs=pairs, cat_features=declared['categories'])\n"
        "    gpu_probe = CatBoostRanker(**dict(PARAMS, iterations=20, depth=4))\n"
        "    gpu_probe._prepare_train_params(pool)\n"
        "    del gpu_probe\n    model.fit(pool)")
    pairwise = once(pairwise, "full_gpu_capacity_proven=False,\n", "full_gpu_capacity_proven=False,\n        gpu_native_training_parameter_preparation='passed',\n")
    replacements = {'full_rich_negative_v2.py': pointwise, 'full_rich_pair_ranker_v2.py': pairwise}
    for name, text in replacements.items():
        ast.parse(text)
        assert not (HERE / name).exists(), 'Preserve previous versions'
    for name, text in replacements.items():
        (HERE / name).write_text(text, encoding='utf-8')
    report = dict(status='sources_generated', repair="pinned_memory_size: integer1073741824 -> string1GB",
        real_gpu_option_preparation_added_to_both_actual576_column_CPU_preflights=True,
        model_training_or_selection_changed=False, original_sources_preserved=True,
        sources_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in
            ['full_rich_negative_v1.py', 'full_rich_pair_ranker_v1.py', *replacements]})
    output = HERE / 'full_rich_gpu_sources_v2.json'
    assert not output.exists()
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
