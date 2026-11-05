"""Read only context inventory; no advertisement labels are read."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for relative in [
    "analysis/userday_aux_20261101/runs/protected_day_aux_v1_day15_seed42",
    "analysis/day16_auc082_20261102/runs/weighted_v1_day16",
]:
    path = ROOT / relative
    metadata = json.loads((path / "context_manifest.json").read_text(encoding="utf8"))
    data = np.load(path / "context.npz")
    print(json.dumps({"directory": relative, "arrays": {k: {"shape": data[k].shape, "dtype": str(data[k].dtype)} for k in data.files},
                      "metadata_keys": list(metadata), "feature_count":len(metadata["features"]),
                      "feature_tail": metadata["features"][-30:], "label_flag": metadata.get("ad_labels_in_context")},indent=2))
