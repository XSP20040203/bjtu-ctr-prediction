"""Copy mutable evaluation reports into a reference before publishing its pointer."""
import argparse
from pathlib import Path
import shutil
import sys
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis/strong_interest_control_20261031"))
from features import read, write, digest, emit


def run(args):
    if not re.fullmatch(r"[a-z0-9_]+", args.name):
        raise ValueError("Reference name must be an internal artifact name")
    out = HERE / ("reference_" + args.name)
    manifest = read(out / "manifest.json")
    for name, expected in manifest["files_sha256"].items():
        if digest(out / name) != expected:
            raise AssertionError("Existing frozen reference changed")
    family, seed = manifest["family"], manifest["seed"]
    sources = {"audit_verification.json": HERE / "verification.json",
               "audit_comparison.csv": HERE / "comparison.csv",
               "audit_paired_changes.csv": HERE / "paired_changes.csv",
               "audit_freezer_source.txt": HERE / "freeze_candidate.py",
               "audit_verifier_source.txt": HERE / "verify_day_aux.py"}
    for name in ["confirmation_protocol.json", "protected_confirmation_protocol.json",
                 "mean_control_protocol.json", "equal_ensemble_protocol.json"]:
        sources["audit_" + name] = HERE / name
    for suffix in [15, 16]:
        root = HERE / "runs" / f"{family}_day{suffix}_seed{seed}"
        sources[f"audit_day{suffix}_features.json"] = root / "feature_manifest.json"
        sources[f"audit_day{suffix}_protocol.json"] = root / "protocol.json"
    for name, source in sources.items():
        destination = out / name
        if destination.exists():
            if digest(destination) != digest(source):
                raise AssertionError("Preserving prior audit snapshot: " + name)
        else:
            shutil.copyfile(source, destination)
    if digest(out / "audit_verification.json") != manifest["verification_sha256"]:
        raise AssertionError("Original verification snapshot does not match")
    if digest(out / "audit_comparison.csv") != manifest["comparison_sha256"]:
        raise AssertionError("Original comparison snapshot does not match")
    manifest["audit_snapshots_sha256"] = {name: digest(out / name) for name in sources}
    manifest["files_sha256"].update(manifest["audit_snapshots_sha256"])
    manifest["sealed_before_reference_publication"] = True
    write(out / "manifest.json", manifest)
    emit("candidate_audit_sealed", files=len(sources), path=str(out / "manifest.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    run(parser.parse_args())
