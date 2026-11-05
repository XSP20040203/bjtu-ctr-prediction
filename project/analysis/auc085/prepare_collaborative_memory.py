"""Prepare only training-click relationships and audited frozen source vectors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from collaborative_memory import build_collaborative_memory, MEMORY_ARRAYS

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=HERE / "cache_rich_stats_random20_aligned")
    parser.add_argument("--source-teacher", type=Path, default=HERE / "source_graph_teacher_day16")
    parser.add_argument("--output-dir", type=Path, default=HERE / "collaborative_memory_random20_k16")
    parser.add_argument("--max-neighbors", type=int, default=16)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving existing collaboration memory {args.output_dir}")
    manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    train = np.load(args.cache_dir / "train_indices.npy")
    categories = np.load(args.cache_dir / "categories.npy", mmap_mode="r")
    labels = np.load(args.cache_dir / "label.npy", mmap_mode="r")
    data = build_collaborative_memory(categories, labels, train, manifest, args.max_neighbors)
    source_meta = json.loads((args.source_teacher / "manifest.json").read_text(encoding="utf-8"))
    assert source_meta["status"] == "complete" and not source_meta["advertisement_labels_used"]
    dates = np.load(args.cache_dir / "date.npy", mmap_mode="r")
    assert source_meta["source_cutoff_day"] <= dates[train].max()
    vectors = np.load(args.source_teacher / "user_vectors.npy")
    raw_ids = np.load(args.source_teacher / "user_raw_ids.npy")
    source_count = np.load(args.source_teacher / "user_positive_count.npy")
    lookup = {str(int(raw)): index for index, raw in enumerate(raw_ids)}
    vocab = json.loads((args.cache_dir / "vocabs.json").read_text(encoding="utf-8"))["user_id"]
    aligned = np.zeros((data["n_users"], vectors.shape[1]), dtype=np.float32)
    counts = np.zeros(data["n_users"], dtype=np.float32)
    for raw, code in vocab.items():
        if raw in lookup:
            aligned[code] = vectors[lookup[raw]]
            counts[code] = source_count[lookup[raw]]
    args.output_dir.mkdir(parents=True)
    for name in MEMORY_ARRAYS:
        np.save(args.output_dir / (name + ".npy"), data[name])
    np.save(args.output_dir / "source_user_vectors.npy", aligned)
    np.save(args.output_dir / "source_positive_count.npy", counts)
    names = MEMORY_ARRAYS + ["source_user_vectors", "source_positive_count"]
    metadata = {name: data[name] for name in ["metadata_fields", "n_users", "n_tasks", "max_neighbors"]}
    metadata.update(status="complete", positive_edges=len(data["positive_keys"]), source_dimension=vectors.shape[1],
        cache_dir=str(args.cache_dir.resolve()), source_teacher_dir=str(args.source_teacher.resolve()),
        train_rows=len(train), train_click_rows=int(labels[train].sum()),
        train_indices_sha256=hashlib.sha256((args.cache_dir / "train_indices.npy").read_bytes()).hexdigest(),
        train_label_sha256=hashlib.sha256(np.asarray(labels[train]).tobytes()).hexdigest(),
        cache_manifest_sha256=hashlib.sha256((args.cache_dir / "manifest.json").read_bytes()).hexdigest(),
        source_teacher_manifest_sha256=hashlib.sha256((args.source_teacher / "manifest.json").read_bytes()).hexdigest(),
        array_sha256={name: hashlib.sha256((args.output_dir / (name + ".npy")).read_bytes()).hexdigest() for name in names},
        advertisement_validation_labels_used=False, query_edge_exclusion="Both directions, before neighbor truncation, including degree correction",
        source_scope="Provided offline source feedback snapshot through Nov16; no causal online availability claim")
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ["status", "positive_edges", "train_rows", "source_dimension", "metadata_fields"]}), flush=True)


if __name__ == "__main__":
    main()
