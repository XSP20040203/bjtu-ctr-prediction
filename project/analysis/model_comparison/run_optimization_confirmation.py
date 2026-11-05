"""Run three matching baseline/candidate comparisons after freezing the choice."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--candidate-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((HERE / "optimization_confirmation_protocol.json").read_text(encoding="utf-8"))
    selection = None
    if not args.baseline_only:
        selection = json.loads((HERE / "results_optimization/frozen_selection.json").read_text(encoding="utf-8"))
        if not selection["contender"]:
            print("No screening contender passed the prespecified threshold.")
            return
    for cell in protocol["confirmation_cells"]:
        day, seed = cell["day"], cell["seed"]
        cache = HERE / ("cache" if day == 20191116 else "cache_day15")
        if not args.candidate_only:
            output = HERE / "results_id_regularization" / str(day) / "baseline" / f"seed{seed}" / "metrics.json"
            if not output.exists():
                subprocess.run([sys.executable, "-X", "utf8", str(HERE / "run_id_regularization.py"),
                                "--variant", "baseline", "--cache-dir", str(cache), "--seed", str(seed)], check=True)
        if selection is not None:
            choice = selection["selected"]
            if choice["family"] == "window3":
                cache = HERE / ("cache_window3_day16" if day == 20191116 else "cache_window3_day15")
            output = HERE / choice["results_root"] / str(day) / choice["variant"] / f"seed{seed}" / "metrics.json"
            if not output.exists():
                subprocess.run([sys.executable, "-X", "utf8", str(HERE / choice["runner"]),
                                "--variant", choice["variant"], "--cache-dir", str(cache), "--seed", str(seed)], check=True)


if __name__ == "__main__":
    main()
