"""Average observed training outcomes only for exactly equal model inputs."""
import numpy as np


def duplicate_targets(data):
    rows = np.asarray(data["train"])
    labels = np.asarray(data["label"])[rows]
    if len(np.unique(rows)) != len(rows) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Unique training rows with real binary labels required")
    if np.intersect1d(rows, data["valid"]).size:
        raise ValueError("Held rows cannot enter averaging")
    # Keep views of existing inputs. Copying all training blocks would allocate
    # another ~2 GB while the original neural inputs are still resident.
    blocks = [np.asarray(data[name]).reshape(len(data["label"]), -1)
              for name in ["categories", "sequences", "numeric", "day_rows"]]
    h1 = np.full(len(rows), np.uint64(1469598103934665603))
    h2 = np.full(len(rows), np.uint64(7809847782465536322))
    for block in blocks:
        if block.dtype.kind == "f":
            if block.dtype != np.float32:
                raise ValueError("Float32 model inputs required")
        for column in range(block.shape[1]):
            values = np.ascontiguousarray(block[rows, column])
            if values.dtype.kind == "f":
                if not np.isfinite(values).all():
                    raise ValueError("Finite model inputs required")
                values = values.view(np.uint32)
            v = values.astype("uint64")
            h1 ^= v + np.uint64(0x9E3779B97F4A7C15)
            h1 *= np.uint64(1099511628211)
            h2 = ((h2 << np.uint64(13)) | (h2 >> np.uint64(51))) ^ v
            h2 *= np.uint64(0xC2B2AE3D27D4EB4F)
    keys = np.empty(len(rows), dtype=[("a", "u8"), ("b", "u8")])
    keys["a"], keys["b"] = h1, h2
    _, representatives, groups = np.unique(keys, return_index=True, return_inverse=True)
    # Hashes are merely a grouping accelerator. Accept a group only after an
    # exhaustive, exact comparison of all its actual model input values.
    for block in blocks:
        for start in range(0, len(rows), 16384):
            end = min(start+16384, len(rows))
            if not np.array_equal(block[rows[start:end]], block[rows[representatives[groups[start:end]]]]):
                raise AssertionError("Hash group contains unequal model inputs")
    counts = np.bincount(groups)
    clicks = np.bincount(groups, weights=labels)
    means = clicks/counts
    target = np.full(len(data["label"]), -1., dtype="float64")
    target[rows] = means[groups]
    if abs(float(target[rows].sum()) - float(labels.sum())) > 1e-6:
        raise AssertionError("Total observed click mass changed")
    mixed = (clicks > 0) & (clicks < counts)
    return target, {
        "training_rows": len(rows), "unique_exact_input_groups": len(counts),
        "duplicate_extra_rows": int(len(rows)-len(counts)),
        "mixed_outcome_groups": int(mixed.sum()),
        "rows_with_fractional_target": int(np.sum(counts[mixed])),
        "exact_group_features_checked": True, "held_day_used": False,
        "observed_click_mass": float(labels.sum()),
        "averaged_target_mass": float(target[rows].sum()),
        "definition": "Mean of observed pre-held-day labels for bitwise equal full model inputs; original rows retained",
    }
