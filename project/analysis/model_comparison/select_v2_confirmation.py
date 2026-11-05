"""Freeze candidates using only the primary date, before secondary-date testing."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results_v2"
records = [json.loads((ROOT / "20191116" / name / "seed42" / "metrics.json").read_text(encoding="utf-8")) for name in ["B1", "B2", "B3", "B4"]]
auc_winner = max(records, key=lambda row: row["auc"])
loss_winner = min(records, key=lambda row: row["logloss"])
selection = {"selection_date": 20191116, "confirmation_date": 20191115, "seed": 42,
             "criteria": ["highest primary-date validation AUC", "lowest primary-date validation Logloss"],
             "selected_variants": sorted({auc_winner["variant"], loss_winner["variant"]}),
             "auc_winner": auc_winner["variant"], "logloss_winner": loss_winner["variant"],
             "primary_results": [{key: row[key] for key in ["variant", "auc", "logloss", "parameter_count"]} for row in records],
             "selection_uses_secondary_date_scores": False}
(ROOT / "frozen_selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(selection, ensure_ascii=False, indent=2))
