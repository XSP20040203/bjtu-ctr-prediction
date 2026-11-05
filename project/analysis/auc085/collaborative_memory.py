"""Training-click neighborhoods with query-edge exclusion before truncation."""
import numpy as np
import torch
from torch import nn

METADATA_FIELDS = ["adv_prim_id", "spread_app_id", "app_second_class", "creat_type_cd", "hispace_app_tags", "inter_type_cd"]
MEMORY_ARRAYS = ["positive_keys", "user_tasks", "task_users", "user_degree", "task_degree", "task_metadata"]


def _prefix_neighbors(primary, neighbor, size, width):
    result = np.zeros((size, width), dtype=np.int64)
    if not len(primary):
        return result
    order = np.lexsort((neighbor, primary))
    first, second = primary[order], neighbor[order]
    starts = np.r_[0, np.flatnonzero(np.diff(first)) + 1]
    stops = np.r_[starts[1:], len(first)]
    for start, stop in zip(starts, stops):
        selected = second[start:min(stop, start + width)]
        result[first[start], :len(selected)] = selected
    return result


def build_collaborative_memory(categories, labels, train_indices, manifest, max_neighbors=16):
    if max_neighbors < 1:
        raise ValueError("At least one neighbor is required")
    rows = np.asarray(train_indices)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("Training indices must be an integer vector")
    fields = manifest["categorical_fields"]
    sizes, groups = manifest["vocabulary_sizes"], manifest["group_for_field"]
    n_user, n_task = sizes[groups["user_id"]], sizes[groups["task_id"]]
    users = categories[rows, fields.index("user_id")].astype(np.int64)
    tasks = categories[rows, fields.index("task_id")].astype(np.int64)
    target = labels[rows]
    if not np.isin(target, [0, 1]).all():
        raise ValueError("Advertisement training labels must be binary")
    positive = (target == 1) & (users >= 2) & (tasks >= 2)
    keys = np.unique(users[positive] * n_task + tasks[positive])
    pu, pt = keys // n_task, keys % n_task
    metadata_fields = [name for name in METADATA_FIELDS if name in fields]
    metadata = np.zeros((n_task, len(metadata_fields)), dtype=np.int64)
    for column, name in enumerate(metadata_fields):
        # Task prototypes use X from all training exposures, not only clicks.
        values = categories[rows, fields.index(name)].astype(np.int64)
        size = sizes[groups[name]]
        packed, counts = np.unique(tasks[tasks >= 2] * size + values[tasks >= 2], return_counts=True)
        task_codes, value_codes = packed // size, packed % size
        order = np.lexsort((value_codes, -counts, task_codes))
        ordered = task_codes[order]
        first = np.r_[True, np.diff(ordered) != 0] if len(ordered) else np.zeros(0, dtype=bool)
        selected = order[first]
        metadata[task_codes[selected], column] = value_codes[selected]
    return {"positive_keys": keys.astype(np.int64),
            "user_tasks": _prefix_neighbors(pu, pt, n_user, max_neighbors + 1),
            "task_users": _prefix_neighbors(pt, pu, n_task, max_neighbors + 1),
            "user_degree": np.bincount(pu, minlength=n_user).astype(np.int64),
            "task_degree": np.bincount(pt, minlength=n_task).astype(np.int64),
            "task_metadata": metadata, "metadata_fields": metadata_fields,
            "n_users": n_user, "n_tasks": n_task, "max_neighbors": max_neighbors}


class QueryExcludedMemory(nn.Module):
    def __init__(self, n_users, n_tasks, max_neighbors, positive_edge_count, metadata_dim):
        super().__init__()
        self.n_tasks, self.max_neighbors = n_tasks, max_neighbors
        self.register_buffer("positive_keys", torch.zeros(positive_edge_count, dtype=torch.long))
        self.register_buffer("user_tasks", torch.zeros(n_users, max_neighbors + 1, dtype=torch.long))
        self.register_buffer("task_users", torch.zeros(n_tasks, max_neighbors + 1, dtype=torch.long))
        self.register_buffer("user_degree", torch.zeros(n_users, dtype=torch.long))
        self.register_buffer("task_degree", torch.zeros(n_tasks, dtype=torch.long))
        self.register_buffer("task_metadata", torch.zeros(n_tasks, metadata_dim, dtype=torch.long))

    def install(self, data):
        with torch.no_grad():
            for name in MEMORY_ARRAYS:
                value = torch.as_tensor(data[name], dtype=torch.long, device=getattr(self, name).device)
                if value.shape != getattr(self, name).shape:
                    raise ValueError(f"Memory shape mismatch for {name}")
                getattr(self, name).copy_(value)

    def _remove_and_limit(self, neighbors, excluded):
        mask = neighbors.ge(2) & neighbors.ne(excluded[:, None])
        width = neighbors.shape[1]
        scores = torch.arange(width, device=neighbors.device)[None] + (~mask).long() * width
        selected = scores.argsort(dim=1)[:, :self.max_neighbors]
        values = neighbors.gather(1, selected)
        return values * mask.gather(1, selected)

    def query(self, users, tasks):
        # Membership is used only to remove its contribution. It is never
        # returned as a feature, including when outside the K+1 reservoir.
        query_key = users * self.n_tasks + tasks
        present = torch.zeros_like(users, dtype=torch.bool)
        if self.positive_keys.numel():
            position = torch.searchsorted(self.positive_keys, query_key)
            bounded = position.clamp_max(len(self.positive_keys) - 1)
            present = (position < len(self.positive_keys)) & self.positive_keys[bounded].eq(query_key) & users.ge(2) & tasks.ge(2)
        return {"user_tasks": self._remove_and_limit(self.user_tasks[users], tasks),
                "task_users": self._remove_and_limit(self.task_users[tasks], users),
                "user_degree": (self.user_degree[users] - present.long()).clamp_min(0).float(),
                "task_degree": (self.task_degree[tasks] - present.long()).clamp_min(0).float()}
