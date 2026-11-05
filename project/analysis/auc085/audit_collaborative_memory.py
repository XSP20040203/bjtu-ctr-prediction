"""Rebuild training-only memory and independently check every positive query."""
import hashlib
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from collaborative_memory import build_collaborative_memory, QueryExcludedMemory, MEMORY_ARRAYS

HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-dir", type=Path, default=HERE / "collaborative_memory_random20_k16")
    args = parser.parse_args()
    directory = args.memory_dir.resolve()
    info = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    cache = Path(info["cache_dir"])
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    categories = np.load(cache / "categories.npy", mmap_mode="r")
    label = np.load(cache / "label.npy", mmap_mode="r")
    train = np.load(cache / "train_indices.npy")
    assert hashlib.sha256((cache / "train_indices.npy").read_bytes()).hexdigest() == info["train_indices_sha256"]
    assert hashlib.sha256(np.asarray(label[train]).tobytes()).hexdigest() == info["train_label_sha256"]
    assert not info["advertisement_validation_labels_used"]
    rebuilt = build_collaborative_memory(categories, label, train, manifest, info["max_neighbors"])
    for name in MEMORY_ARRAYS:
        value = np.load(directory / (name + ".npy"))
        np.testing.assert_array_equal(value, rebuilt[name])
        assert hashlib.sha256((directory / (name + ".npy")).read_bytes()).hexdigest() == info["array_sha256"][name]
    user = categories[train, manifest["categorical_fields"].index("user_id")].astype(np.int64)
    task = categories[train, manifest["categorical_fields"].index("task_id")].astype(np.int64)
    nt = info["n_tasks"]
    independent_keys = np.unique(user[label[train] == 1] * nt + task[label[train] == 1])
    np.testing.assert_array_equal(independent_keys, rebuilt["positive_keys"])
    memory = QueryExcludedMemory(info["n_users"], nt, info["max_neighbors"], info["positive_edges"], len(info["metadata_fields"]))
    memory.install(rebuilt)
    users, tasks = independent_keys // nt, independent_keys % nt
    user_book, task_book = {}, {}
    for u, t in zip(users.tolist(), tasks.tolist()):
        user_book.setdefault(u, []).append(t)
        task_book.setdefault(t, []).append(u)
    user_book = {key: np.array(sorted(values)) for key, values in user_book.items()}
    task_book = {key: np.array(sorted(values)) for key, values in task_book.items()}
    negative = np.unique((user * nt + task)[label[train] == 0])[:1024]
    queries = np.unique(np.r_[independent_keys, negative, 0, nt + 1])
    qu, qt = queries // nt, queries % nt
    actual = memory.query(torch.from_numpy(qu), torch.from_numpy(qt))
    k = info["max_neighbors"]
    empty = np.zeros(0, dtype=np.int64)
    for index, (u, t) in enumerate(zip(qu.tolist(), qt.tolist())):
        task_neighbors = user_book.get(u, empty)
        task_neighbors = task_neighbors[task_neighbors != t]
        user_neighbors = task_book.get(t, empty)
        user_neighbors = user_neighbors[user_neighbors != u]
        expected_tasks, expected_users = np.zeros(k, dtype=np.int64), np.zeros(k, dtype=np.int64)
        expected_tasks[:min(k, len(task_neighbors))] = task_neighbors[:k]
        expected_users[:min(k, len(user_neighbors))] = user_neighbors[:k]
        np.testing.assert_array_equal(actual["user_tasks"][index].numpy(), expected_tasks)
        np.testing.assert_array_equal(actual["task_users"][index].numpy(), expected_users)
        assert actual["user_degree"][index].item() == len(task_neighbors)
        assert actual["task_degree"][index].item() == len(user_neighbors)
    source = Path(info["source_teacher_dir"])
    assert hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest() == info["source_teacher_manifest_sha256"]
    raw_ids = np.load(source / "user_raw_ids.npy")
    source_vectors = np.load(source / "user_vectors.npy")
    source_counts = np.load(source / "user_positive_count.npy")
    vocab = json.loads((cache / "vocabs.json").read_text(encoding="utf-8"))["user_id"]
    lookup = {str(int(raw)): index for index, raw in enumerate(raw_ids)}
    expected_vectors = np.zeros((info["n_users"], info["source_dimension"]), dtype=np.float32)
    expected_counts = np.zeros(info["n_users"], dtype=np.float32)
    for raw, code in vocab.items():
        if raw in lookup:
            expected_vectors[code] = source_vectors[lookup[raw]]
            expected_counts[code] = source_counts[lookup[raw]]
    np.testing.assert_array_equal(expected_vectors, np.load(directory / "source_user_vectors.npy"))
    np.testing.assert_array_equal(expected_counts, np.load(directory / "source_positive_count.npy"))
    result = {"status": "passed", "memory_dir": str(directory.resolve()), "training_edges_rebuilt": len(independent_keys),
              "independent_queries_checked": len(queries), "all_positive_queries_checked": True,
              "query_removed_before_neighbor_cap": True, "degree_correction_verified": True,
              "metadata_training_x_rebuilt": True, "frozen_source_user_mapping_verified": True,
              "advertisement_validation_labels_used": False}
    (directory / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if info["max_neighbors"] == 16:
        (HERE / "phase5_memory_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
