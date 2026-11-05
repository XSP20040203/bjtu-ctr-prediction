"""Freeze a candidate using only the primary screening cell."""
import json
from pathlib import Path

from train_models import write_json

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_optimization"


def main():
    candidates = []
    for family, root, variants, runner in [
        ("id_dropout", "results_id_regularization", ["U20", "U50", "UA20", "UA50"], "run_id_regularization.py"),
        ("window3", "results_recent_window", ["baseline"], "run_recent_window.py"),
        ("norm_caps", "results_id_norm_caps", ["CU10", "CU20", "CUA20"], "run_id_norm_caps.py")]:
        for variant in variants:
            path = HERE / root / "20191116" / variant / "seed42" / "metrics.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            candidates.append({"family": family, "variant": variant, "runner": runner,
                               "results_root": root, "primary_auc": record["auc"],
                               "results_dir": str(path.parent), "primary_metrics": record})
    best = max(candidates, key=lambda row: row["primary_auc"])
    baseline = json.loads((HERE / "results_mainstream/20191116/dcnv2/seed42/metrics.json").read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    target = OUT / "frozen_selection.json"
    if target.exists():
        raise FileExistsError("The primary-cell choice has already been frozen")
    write_json(target, {"selection_source": "Only day16 seed42 completed screening scores", "selected": best,
                        "baseline_primary_auc": baseline["auc"],
                        "contender": best["primary_auc"] >= baseline["auc"] - .001,
                        "all_screening_candidates": [{k: row[k] for k in ["family", "variant", "primary_auc", "results_dir"]} for row in candidates],
                        "confirmation_scores_used": False})
    print(json.dumps({key: best[key] for key in ["family", "variant", "primary_auc"]}))


if __name__ == "__main__":
    main()
