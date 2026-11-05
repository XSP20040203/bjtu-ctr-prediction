"""Training-only exposed-but-never-clicked relations for a comparative expert."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from collaborative_memory import build_collaborative_memory, MEMORY_ARRAYS

HERE = Path(__file__).resolve().parent


def build_negative_memory(categories, labels, train_indices, manifest, max_neighbors=16):
    train = np.asarray(train_indices)
    fields, groups, sizes = manifest["categorical_fields"], manifest["group_for_field"], manifest["vocabulary_sizes"]
    n_task = sizes[groups["task_id"]]
    users = categories[train, fields.index("user_id")].astype(np.int64)
    tasks = categories[train, fields.index("task_id")].astype(np.int64)
    target = labels[train]
    if not np.isin(target, [0, 1]).all():
        raise ValueError("Advertisement training labels must be binary")
    keys = users * n_task + tasks
    eligible = (users >= 2) & (tasks >= 2)
    positive_keys = np.unique(keys[(target == 1) & eligible])
    negative_keys = np.unique(keys[(target == 0) & eligible])
    observed_negative = (target == 0) & eligible & ~np.isin(keys, positive_keys)
    pseudo_label = np.zeros(len(labels), dtype=np.uint8)
    pseudo_label[train] = observed_negative
    result = build_collaborative_memory(categories, pseudo_label, train, manifest, max_neighbors)
    expected = np.setdiff1d(negative_keys, positive_keys)
    np.testing.assert_array_equal(result["positive_keys"], expected)
    result.update(relation="exposed_but_never_clicked", mixed_pairs_excluded=int(np.intersect1d(negative_keys, positive_keys).size))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-memory", type=Path, default=HERE / "collaborative_memory_random20_k16")
    parser.add_argument("--output-dir", type=Path, default=HERE / "negative_collaborative_memory_random20_k16")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving negative memory {args.output_dir}")
    positive_meta = json.loads((args.positive_memory / "manifest.json").read_text(encoding="utf-8"))
    cache = Path(positive_meta["cache_dir"])
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    train = np.load(cache / "train_indices.npy")
    categories = np.load(cache / "categories.npy", mmap_mode="r")
    labels = np.load(cache / "label.npy", mmap_mode="r")
    assert hashlib.sha256(np.asarray(labels[train]).tobytes()).hexdigest() == positive_meta["train_label_sha256"]
    data = build_negative_memory(categories, labels, train, manifest, positive_meta["max_neighbors"])
    np.testing.assert_array_equal(data["task_metadata"], np.load(args.positive_memory / "task_metadata.npy"))
    positive_keys = np.load(args.positive_memory / "positive_keys.npy")
    assert not np.intersect1d(positive_keys, data["positive_keys"]).size
    args.output_dir.mkdir(parents=True)
    for name in MEMORY_ARRAYS:
        np.save(args.output_dir / (name + ".npy"), data[name])
    metadata = {key: data[key] for key in ["metadata_fields", "n_users", "n_tasks", "max_neighbors", "relation", "mixed_pairs_excluded"]}
    metadata.update(status="complete", negative_edges=len(data["positive_keys"]), cache_dir=str(cache.resolve()),
        positive_memory_dir=str(args.positive_memory.resolve()),
        positive_memory_manifest_sha256=hashlib.sha256((args.positive_memory / "manifest.json").read_bytes()).hexdigest(),
        train_indices_sha256=hashlib.sha256((cache / "train_indices.npy").read_bytes()).hexdigest(),
        train_label_sha256=positive_meta["train_label_sha256"],
        array_sha256={name: hashlib.sha256((args.output_dir / (name + ".npy")).read_bytes()).hexdigest() for name in MEMORY_ARRAYS},
        advertisement_validation_labels_used=False)
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.output_dir / "audit.json").write_text(json.dumps({"status": "passed", "training_negative_keys_rebuilt": True,
        "positive_negative_relations_disjoint": True, "metadata_equals_training_x_positive_memory": True,
        "advertisement_validation_labels_used": False, "negative_edges": metadata["negative_edges"]}, indent=2), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ["negative_edges", "mixed_pairs_excluded", "relation", "advertisement_validation_labels_used"]}), flush=True)


if __name__ == "__main__":
    main()
