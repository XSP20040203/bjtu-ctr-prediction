"""Export a verified two-day development recipe, never a final-test submission."""
import argparse
from pathlib import Path
import sys
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = ROOT / "analysis/learned_audience_20261101"
sys.path.insert(0, str(ROOT / "analysis/strong_interest_control_20261031"))
from features import read, write, digest, emit, ARCH
from experiment import scores
sys.path.insert(0, str(REFERENCE))
from freeze_reference import V7_HASH
import numpy as np
import pandas as pd


def run(args):
    verification = read(HERE / "verification.json")
    if verification["status"] != "passed" or verification["verified_model_count"] < 16:
        raise RuntimeError("Complete independent two-day verification is required before freezing")
    if digest(ROOT / "output/best_nextday_arch_v7/submission.csv") != V7_HASH:
        raise AssertionError("Original v7 submission changed")
    if not re.fullmatch(r"[a-z0-9_]+", args.name):
        raise ValueError("Reference name must use lowercase letters, numbers and underscores")
    out = HERE / ("reference_" + args.name)
    if out.exists():
        raise FileExistsError("Preserving existing frozen reference")
    table = pd.read_csv(HERE / "comparison.csv")
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["pt_d", "label"], dtype="int32")
    suffixes = {"replacement": "v7_replace30", "blend10": "candidate_blend10",
                "equal": "candidate_equal_replacement"}
    control_suffixes = {"replacement": "mean_control_v7_replace30", "blend10": "mean_control_candidate_blend10",
                        "equal": "mean_control_equal_replacement"}
    results, inputs, pending = {}, {}, []
    for day in [20191115, 20191116]:
        suffix = day % 100
        root = HERE / "runs" / f"{args.family}_day{suffix}_seed{args.seed}"
        feature = read(root / "feature_manifest.json")
        for name, expected in feature["input_sha256"].items():
            if digest(ROOT / "2026ml" / name) != expected:
                raise AssertionError("Dataset changed after verification")
        directory = root / args.mode
        paths = {
            "source_prediction": directory / f"{args.endpoint}_prediction.npy",
            "original_v7": ARCH / f"fixed_gated20_allocated10_core30_day{suffix}.npy",
            "original_core": ARCH / f"results/gated166_day{suffix}/validation_prediction.npy",
            "earlier_reference": REFERENCE / f"reference/day{suffix}_prediction.npy",
        }
        prediction = np.load(paths["source_prediction"]).astype("float64")
        metric = read(directory / "metrics.json")
        if digest(paths["source_prediction"]) != metric["files_sha256"][args.endpoint + "_prediction.npy"]:
            raise AssertionError("Source prediction changed after verification")
        if args.mean_control:
            paths["matched_base"] = root / "base" / f"{args.endpoint}_prediction.npy"
            base = np.load(paths["matched_base"]).astype("float64")
            prediction = np.clip(base * prediction.mean() / max(float(base.mean()), 1e-12), 0., 1.)
        reference = np.load(paths["earlier_reference"])
        v7, old_core = np.load(paths["original_v7"]), np.load(paths["original_core"])
        replacement = v7 + .3 * (prediction - old_core)
        blended = replacement if args.recipe == "replacement" else (
            .9 * reference + .1 * prediction if args.recipe == "blend10" else .5 * reference + .5 * replacement)
        indices = np.flatnonzero(raw.pt_d.to_numpy() == day)
        np.testing.assert_array_equal(np.load(directory / "valid_indices.npy"), indices)
        actual = scores(raw.label.to_numpy()[indices], blended)
        descriptor = f"{args.family}_seed{args.seed}/{args.mode}/{args.endpoint}/" + (
            control_suffixes[args.recipe] if args.mean_control else suffixes[args.recipe])
        saved = table.loc[(table.day == day) & (table.model == descriptor)]
        if len(saved) != 1 or abs(float(saved.iloc[0].auc) - actual["auc"]) > 1e-12:
            raise AssertionError("Recipe does not match independent verification")
        earlier = scores(raw.label.to_numpy()[indices], reference)
        if actual["auc"] <= earlier["auc"]:
            raise ValueError("Candidate does not improve both existing development days")
        if actual["logloss"] - earlier["logloss"] > .0001:
            raise ValueError("Candidate logloss worsening exceeds the declared safeguard")
        results[str(day)] = actual
        inputs[str(day)] = {str(path): digest(path) for path in paths.values()}
        pending.append((suffix, blended, indices))
    out.mkdir()
    for suffix, prediction, indices in pending:
        np.save(out / f"day{suffix}_prediction.npy", prediction)
        np.save(out / f"day{suffix}_indices.npy", indices)
    manifest = {
        "status": "verified_development_reference", "development": results,
        "family": args.family, "mode": args.mode, "endpoint": args.endpoint, "seed": args.seed,
        "recipe": args.recipe, "mean_control": args.mean_control,
        "selection": "Compared fixed development methods on repeatedly reused dates; seed confirmation pending",
        "input_predictions_sha256": inputs,
        "verification_sha256": digest(HERE / "verification.json"),
        "comparison_sha256": digest(HERE / "comparison.csv"), "source_sha256": digest(Path(__file__)),
        "files_sha256": {path.name: digest(path) for path in out.glob("*.npy")},
        "original_v7_submission_sha256": digest(ROOT / "output/best_nextday_arch_v7/submission.csv"),
        "final_submission_exported": False, "final_test_auc": None,
        "development_threshold_met": all(result["auc"] >= .85 for result in results.values()),
    }
    write(out / "manifest.json", manifest)
    emit("day_candidate_frozen", path=str(out / "manifest.json"), development=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--family", choices=["day_aux_v1", "protected_day_aux_v1"], default="protected_day_aux_v1")
    parser.add_argument("--mode", choices=["base", "context", "aux", "both"], default="both")
    parser.add_argument("--endpoint", choices=["best", "final"], default="best")
    parser.add_argument("--recipe", choices=["replacement", "blend10", "equal"], default="replacement")
    parser.add_argument("--mean-control", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
