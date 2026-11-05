"""Run one architecture without changing existing experiments or preprocessing."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from mainstream_models import mainstream_definition
from train_models import load_arrays, metrics, train_neural, write_json

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_mainstream"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["deepfm", "dcnv2", "mlp"], required=True)
    parser.add_argument("--cache-dir", type=Path, default=HERE / "cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    manifest_path = args.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    day = manifest["validation_day"]
    output = OUT / str(day) / args.model / f"seed{args.seed}"
    if (output / "model.pt").exists():
        raise FileExistsError(f"Preserving the existing experiment: {output}")
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"model": args.model, "validation_day": day, "seed": args.seed,
                  "cache_dir": str(args.cache_dir.resolve()),
                  "sha256": {str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in [manifest_path, HERE / "mainstream_models.py", HERE / "train_models.py",
                                          args.cache_dir / "train_indices.npy", args.cache_dir / "valid_indices.npy"]}}
    write_json(output / "provenance.json", provenance)
    arrays = load_arrays(args.cache_dir)
    train, valid = arrays["train_indices"], arrays["valid_indices"]
    assert arrays["date"][train].max() < day
    assert np.all(arrays["date"][valid] == day)
    builder, options = mainstream_definition(args.model)
    print(f"START_MAINSTREAM model={args.model}, validation_day={day}, seed={args.seed}", flush=True)
    prediction, metadata = train_neural(arrays, manifest, output, args.epochs, args.batch_size,
                                       model_builder=builder, model_options=options, seed=args.seed)
    new_tasks = ~np.isin(arrays["baseline"][valid, 8], np.unique(arrays["baseline"][train, 8]))
    result = {"model": args.model, **metrics(arrays["label"][valid], prediction), **metadata,
              "train_rows": manifest["train_rows"], "valid_rows": manifest["valid_rows"],
              "validation_day": day, "validation_clicks": manifest["valid_clicks"], "seed": args.seed,
              "cache_dir": str(args.cache_dir.resolve()), "python": platform.python_version(),
              "new_task_rows": int(new_tasks.sum()), "new_task_clicks": int(arrays["label"][valid][new_tasks].sum()),
              "new_task_metrics": metrics(arrays["label"][valid][new_tasks], prediction[new_tasks]),
              "experiment_scope": "Same raw inputs, frozen preprocessing, date boundaries, batch order and optimizer. Architecture and its input representation vary.",
              "feature_note": "24 categorical fields, seven masked-mean history fields, 20 numeric inputs. " + {
                  "deepfm": "Linear + second-order FM + DNN; FM/DNN share all field embeddings; numeric fields use x_i*v_i.",
                  "dcnv2": "Parallel full-matrix two-layer CrossNet + DNN; concatenate outputs before logistic head.",
                  "mlp": "B2-size embedding + MLP control, same dimensions and dropout."}[args.model]}
    np.save(output / "validation_prediction.npy", prediction)
    write_json(output / "metrics.json", result)
    print("FINAL_MAINSTREAM_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
