"""Read-only inventory for the selected full refit."""
import json
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    for relative in [
        "analysis/daily_interest_din/full_refit_cache/manifest.json",
        "analysis/nextday_goal_20261029/full_best/oof/manifest.json",
        "analysis/architecture_goal_20261029/full_refit/gated166/input_manifest.json",
        "analysis/userday_aux_20261101/runs/protected_day_aux_v1_day15_seed42/both/metrics.json",
        "analysis/userday_aux_20261101/runs/protected_day_aux_v1_day16_seed42/both/metrics.json",
    ]:
        value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        keep = ["status", "rows", "train_rows", "final_rows", "train_clicks",
                "vocabulary_sizes", "target_stat_definitions", "best_checkpoint",
                "fixed_endpoint", "numeric_sha256"]
        print(relative, json.dumps({key: value[key] for key in keep if key in value}))
    print("MEMORY", psutil.virtual_memory()._asdict())
    print("PYTHON", [(p.pid, p.info["name"], p.info["memory_info"].rss / 1024**3)
                     for p in psutil.process_iter(["name", "memory_info"])
                     if p.info["name"] and "python" in p.info["name"].lower()])


if __name__ == "__main__":
    main()
