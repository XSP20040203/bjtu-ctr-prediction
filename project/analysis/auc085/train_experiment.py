"""Run explicit random/temporal experiments and keep their metrics distinct."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "model_comparison"
sys.path.insert(0, str(BENCH))
from train_models import load_arrays, metrics, train_neural, write_json
from mainstream_models import DCNV2Model
from bounded_id_models import BoundedIDDCN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["dcn", "bounded"], required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = HERE / "results" / args.tag / args.model / f"seed{args.seed}"
    if (output / "model.pt").exists():
        raise FileExistsError(f"Preserving existing weights: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    arrays = load_arrays(args.cache_dir)
    train, valid = arrays["train_indices"], arrays["valid_indices"]
    assert not np.intersect1d(train, valid).size
    strategy = manifest.get("validation_strategy", "time" + str(manifest["validation_day"])[-2:])
    if manifest["validation_day"] is not None:
        assert arrays["date"][train].max() < manifest["validation_day"]
        assert np.all(arrays["date"][valid] == manifest["validation_day"])
    options = {"embedding_dim": 8, "hidden_dims": [128, 64], "dropout": .1, "unknown_dropout": .01,
               "cross_layers": 2, "numeric_dim": len(manifest["numeric_fields"])}
    builder = DCNV2Model
    if args.model == "bounded":
        builder = BoundedIDDCN
        options.update(group_norm_caps={"user_id": .2, "ad_task": .2, "adv_id": .2})
    print(f"EXPERIMENT {args.tag} model={args.model} strategy={strategy}", flush=True)
    prediction, metadata = train_neural(arrays, manifest, output, args.epochs, 2048,
                                       model_builder=builder, model_options=options, seed=args.seed)
    record = {**metrics(arrays["label"][valid], prediction), **metadata, "model": args.model,
              "seed": args.seed, "validation_strategy": strategy, "validation_day": manifest["validation_day"],
              "cache_dir": str(args.cache_dir.resolve()), "train_rows": len(train), "valid_rows": len(valid),
              "validation_clicks": int(arrays["label"][valid].sum()),
              "feature_note": f"{len(manifest['categorical_fields'])} categorical fields, {len(manifest['sequence_fields'])} masked-mean lists, {len(manifest['numeric_fields'])} numeric inputs; {manifest.get('feature_variant', 'original historical source features')}",
              "feature_availability_assumption": manifest.get("feature_availability_assumption", "Strictly earlier source dates"),
              "scope": "Offline validation, never interpreted as measured final-test AUC."}
    np.save(output / "validation_prediction.npy", prediction)
    write_json(output / "metrics.json", record)
    print("EXPERIMENT_RESULT " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
