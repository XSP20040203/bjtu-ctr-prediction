"""Training-only audit of whether supplied ad-history slots carry distinct information."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    cache = HERE / "cache_rich_stats_random20_aligned"
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    arrays = {name: np.load(cache / (name + ".npy"), mmap_mode="r")
              for name in ["categories", "sequences", "label", "train_indices"]}
    rows = arrays["train_indices"]
    y = arrays["label"][rows]
    categories, sequences = arrays["categories"][rows], arrays["sequences"][rows]
    result = {"scope": "Training indices only; validation labels never read", "train_rows": len(rows),
              "train_positive_rate": float(y.mean()), "position_semantics": "Provided list positions; chronology is undocumented", "fields": []}
    for field in manifest["sequence_fields"][:6]:
        index = manifest["sequence_fields"].index(field)
        group = manifest["group_for_field"][field]
        current = next(name for name in manifest["categorical_fields"] if manifest["group_for_field"][name] == group)
        query = categories[:, manifest["categorical_fields"].index(current)]
        history = sequences[:, index]
        valid = history >= 2
        matched = (history == query[:, None]) & valid & (query[:, None] >= 2)
        any_match = matched.any(axis=1)
        record = {"field": field, "current_field": current, "history_nonempty_fraction": float(valid.any(axis=1).mean()),
                  "match_fraction": float(any_match.mean()), "ctr_if_match": float(y[any_match].mean()) if any_match.any() else None,
                  "ctr_if_no_match": float(y[~any_match].mean()), "positions": []}
        for slot in range(history.shape[1]):
            mask = matched[:, slot]
            record["positions"].append({"slot": slot, "known_nonpadding_fraction": float(valid[:, slot].mean()),
                                         "match_rows": int(mask.sum()), "clicks_when_matching": int(y[mask].sum()),
                                         "ctr_if_match": float(y[mask].mean()) if mask.any() else None})
        result["fields"].append(record)
    (HERE / "phase3_ad_history_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
