"""Run one frozen architecture candidate on a specified time split."""
import argparse
import json
import platform
from pathlib import Path

import numpy as np

from train_models import load_arrays, metrics, train_neural, write_json
from v2_models import variant_definition

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "B1", "B2", "B3", "B4"], required=True)
    parser.add_argument("--cache-dir", type=Path, default=HERE / "cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    day = manifest["validation_day"]
    output = HERE / "results_v2" / str(day) / args.variant / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays(args.cache_dir)
    builder, options = variant_definition(args.variant)
    print(f"START V2 variant={args.variant}, validation_day={day}, seed={args.seed}", flush=True)
    prediction, metadata = train_neural(arrays, manifest, output, args.epochs, args.batch_size,
                                        model_builder=builder, model_options=options, seed=args.seed)
    valid, train = arrays["valid_indices"], arrays["train_indices"]
    known_tasks = np.unique(arrays["baseline"][train, 8])
    new_tasks = ~np.isin(arrays["baseline"][valid, 8], known_tasks)
    result = {"model": args.variant, "variant": args.variant, **metrics(arrays["label"][valid], prediction), **metadata,
              "train_rows": manifest["train_rows"], "valid_rows": manifest["valid_rows"],
              "validation_day": day, "validation_clicks": manifest["valid_clicks"], "seed": args.seed,
              "cache_dir": str(args.cache_dir.resolve()), "python": platform.python_version(),
              "feature_note": ("24 categorical embeddings, three target-aware click-history attention pools, four masked-mean pools, and the same 20 numeric inputs. " + ("Source has a separate 7-feature branch and a learned vector gate." if args.variant == "B4" else "Flat MLP fusion.")) if args.variant in ["B3", "B4"] else "24 categorical embeddings, seven masked-mean pools and the same 20 numeric inputs.",
              "new_task_rows": int(new_tasks.sum()), "new_task_clicks": int(arrays["label"][valid][new_tasks].sum()),
              "new_task_metrics": metrics(arrays["label"][valid][new_tasks], prediction[new_tasks]),
              "experiment_scope": "Architecture-only; same frozen preprocessing, numeric features, training settings, row order and seed for this date."}
    np.save(output / "validation_prediction.npy", prediction)
    write_json(output / "metrics.json", result)
    print("FINAL_V2_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
