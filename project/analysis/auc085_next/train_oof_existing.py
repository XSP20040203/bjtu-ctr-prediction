"""Ablate old train-only grouped-OOF statistics with the newer exposure inputs."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DAILY = HERE.parent / 'daily_interest_din'
spec = importlib.util.spec_from_file_location('dense_signals', DAILY / 'train_dense_signals.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main(args):
    original_load = module.np.load
    def load(path, *positional, **kwargs):
        resolved = Path(path).resolve()
        for day, cache in [(15, HERE.parent / 'user_personalization/cache_day15_stats'),
                           (16, HERE.parent / 'auc085/cache_rich_stats_time16')]:
            if resolved == (DAILY / f'cache_day{day}/numeric.npy').resolve():
                path = cache / 'numeric.npy'
        return original_load(path, *positional, **kwargs)
    module.np.load = load
    module.main(args)
    output = DAILY / 'dense_results' / args.tag
    path = output / 'metrics.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    record.update(advertisement_training_statistics='Existing duplicate-group 5-fold OOF across training dates; outer whole-day validation uses full training statistics',
                  per_row_strict_past=False,
                  outer_validation_labels_used=False,
                  feature_ablation='Swap first75 numeric source to original cache; retain current exposure+snapshot inputs')
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('OOF_ABLATION_COMPLETE', record['auc'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=int, choices=[20191115, 20191116], required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--generic-features', type=Path)
    parser.add_argument('--pos-weight', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--recent-days', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=.001)
    main(parser.parse_args())
