"""Training-only sparsity audit; click counts here never initialize model gates."""
import json
from pathlib import Path

import numpy as np

from hierarchical_id_model import fit_exposure_support

HERE = Path(__file__).resolve().parent


def main():
    cache = HERE / "cache_rich_stats_random20_aligned"
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    category = np.load(cache / "categories.npy", mmap_mode="r")
    train = np.load(cache / "train_indices.npy")
    labels = np.load(cache / "label.npy", mmap_mode="r")[train]
    support = fit_exposure_support(category, train, manifest)
    records = []
    for field, count in support.items():
        column = manifest["categorical_fields"].index(field)
        codes = category[train, column]
        clicks = np.bincount(codes, weights=labels, minlength=len(count))
        seen = count > 0
        records.append({"field": field, "seen_current_ids": int(seen.sum()),
                        "exposure_quantiles": dict(zip(["q10", "q50", "q90"], np.quantile(count[seen], [.1, .5, .9]).tolist())),
                        "ids_with_no_click_fraction": float((clicks[seen] == 0).mean()),
                        "ids_with_at_most_two_clicks_fraction": float((clicks[seen] <= 2).mean()),
                        "rows_from_ids_with_at_most_two_clicks_fraction": float((clicks[codes] <= 2).mean()),
                        "initial_support_weight_quantiles": np.quantile(count[codes] / (count[codes] + 200.), [.1, .5, .9]).tolist()})
    result = {"scope": "Training indices only; validation labels not read. Click counts are diagnostic and never passed to fit_exposure_support or model initialization",
              "train_rows": len(train), "train_positive_rate": float(labels.mean()), "fields": records}
    (HERE / "phase4_support_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
