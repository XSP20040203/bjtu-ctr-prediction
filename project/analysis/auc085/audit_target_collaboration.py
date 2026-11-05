"""Feasibility evidence for a training-click graph, without fitting on held-out labels."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent


def main():
    cache = HERE / "cache_rich_stats_random20_aligned"
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    train = np.load(cache / "train_indices.npy")
    categories = np.load(cache / "categories.npy", mmap_mode="r")
    y = np.load(cache / "label.npy", mmap_mode="r")[train]
    users = categories[train, manifest["categorical_fields"].index("user_id")].astype(np.int64)
    tasks = categories[train, manifest["categorical_fields"].index("task_id")].astype(np.int64)
    n_user, n_task = manifest["vocabulary_sizes"]["user_id"], manifest["vocabulary_sizes"]["ad_task"]
    keys = users * n_task + tasks
    positive = np.unique(keys[y == 1])
    negative = np.unique(keys[y == 0])
    pu, pt = positive // n_task, positive % n_task
    assert int(y.sum()) == 17083 and (pu >= 2).all() and (pt >= 2).all()
    user_degree = np.bincount(pu, minlength=n_user)
    task_degree = np.bincount(pt, minlength=n_task)
    adjacency = coo_matrix((np.ones(len(pu) * 2), (np.r_[pu, pt + n_user], np.r_[pt + n_user, pu])),
                           shape=(n_user + n_task, n_user + n_task)).tocsr()
    components, groups = connected_components(adjacency, directed=False)
    sizes = np.bincount(groups)
    largest = int(np.argmax(sizes))
    nonisolated = np.r_[user_degree > 0, task_degree > 0]
    metadata = {}
    for name in ["adv_prim_id", "spread_app_id", "app_second_class", "creat_type_cd"]:
        values = categories[train, manifest["categorical_fields"].index(name)]
        unique_pairs = np.unique(np.column_stack([tasks, values]), axis=0)
        variation = np.bincount(unique_pairs[:, 0], minlength=n_task)
        metadata[name] = {"observed_values": int(np.unique(values).size),
                          "task_ids_with_multiple_values": int((variation > 1).sum()),
                          "max_values_for_one_task": int(variation.max())}
    result = {"scope": "Training advertisement labels only; no validation labels read; no aggregation from later ad-history snapshots",
              "train_rows": len(train), "train_click_rows": int(y.sum()), "unique_positive_user_task_edges": len(positive),
              "users_with_positive_edges": int((user_degree > 0).sum()), "tasks_with_positive_edges": int((task_degree > 0).sum()),
              "user_positive_degree_quantiles": np.quantile(user_degree[user_degree > 0], [.1, .5, .9]).tolist(),
              "task_positive_degree_quantiles": np.quantile(task_degree[task_degree > 0], [.1, .5, .9]).tolist(),
              "positive_edges_whose_pair_also_has_negative_rows": int(np.intersect1d(positive, negative).size),
              "connected_components_including_isolated_vocab_nodes": int(components),
              "nonisolated_components": int(np.unique(groups[nonisolated]).size),
              "largest_component_nodes": int(sizes[largest]),
              "positive_edge_fraction_in_largest_component": float((groups[pu] == largest).mean()),
              "metadata_consistency": metadata,
              "next_architecture_constraints": ["Source/ad user alignment must use the audited common user mapping",
                "Train graph edges and negative pools must use only advertisement training labels",
                "Exclude the queried user-task positive edge when computing training-query messages or use grouped cross-fitting",
                "Known user-task pairs can contain both outcomes, so graph edges encode preference evidence rather than deterministic answers",
                "Ad metadata consistency can support shared item priors, with explicit fallback for missing/cold nodes"]}
    (HERE / "phase5_target_collaboration_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
