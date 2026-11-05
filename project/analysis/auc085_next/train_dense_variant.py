"""Controlled architecture ablations on the existing daily exposure feature pipeline."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DAILY = HERE.parent / 'daily_interest_din'
spec = importlib.util.spec_from_file_location('dense_variant_base', DAILY / 'train_dense_signals.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main(args):
    original_model = module.BoundedIDDCN
    def model(*positional, **options):
        options['embedding_dim'] = args.embedding_dim
        options['group_norm_caps'] = {} if args.id_cap == 0 else {k: args.id_cap for k in ['user_id', 'ad_task', 'adv_id']}
        options['dropout'] = args.dropout
        if args.group_dropout:
            options['group_dropout_rates'] = {k: args.group_dropout for k in ['user_id', 'ad_task', 'adv_id']}
        return original_model(*positional, **options)
    module.BoundedIDDCN = model
    module.main(args)
    output = DAILY / 'dense_results' / args.tag
    path = output / 'metrics.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    options = record['options']
    options.update(embedding_dim=args.embedding_dim, dropout=args.dropout,
                   group_norm_caps={} if args.id_cap == 0 else {k: args.id_cap for k in ['user_id', 'ad_task', 'adv_id']})
    if args.group_dropout:
        options['group_dropout_rates'] = {k: args.group_dropout for k in ['user_id', 'ad_task', 'adv_id']}
    checkpoint = module.torch.load(output / 'model.pt', map_location='cpu', weights_only=False)
    checkpoint['options'] = options
    module.torch.save(checkpoint, output / 'model.pt')
    record['model_sha256'] = module.hashlib.sha256((output / 'model.pt').read_bytes()).hexdigest()
    record['variant'] = {'embedding_dim': args.embedding_dim, 'id_cap': args.id_cap,
                         'dropout': args.dropout, 'group_dropout': args.group_dropout}
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('VARIANT_COMPLETE', record['auc'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=int, choices=[20191115, 20191116], required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--generic-features', type=Path)
    parser.add_argument('--pos-weight', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--recent-days', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--embedding-dim', type=int, default=8)
    parser.add_argument('--id-cap', type=float, default=.2, help='zero disables caps')
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--group-dropout', type=float, default=0.)
    main(parser.parse_args())
