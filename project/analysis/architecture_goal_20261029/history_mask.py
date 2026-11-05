"""Label-free, optional masking of dominant close-history defaults.

Use the *sequence cache* manifest and its vocabs with create_mask_spec. For each
batch, apply_sequence_mask returns a new sequence array and three float32 missing
flags, ordered v001/v002/v003. PAD=0 and UNK=1 keep their existing meanings; only
the resolved default ID in its own close field is replaced by PAD. Click fields,
candidate categorical IDs and supplied slot positions are untouched.

For already standardized numeric inputs, select feature_column_selection(names)
and append those missing flags, named MISSING_FLAG_NAMES. This drops old close
snapshot, length and candidate-match columns instead of pretending standardized
zero means raw zero. The newer 95 history features already remove defaults and
are retained. No scaler fitting, data loading, labels, training or writes occur.

Reinstating dropped columns requires rebuilding their original input features:
lengths and matches from default-filtered raw row histories, and snapshot metrics
from complete default-filtered user/day unions plus preceding snapshot state.
Architecture scalers alone cannot undo upstream transforms or clipped values.
"""

import numpy as np


DEFAULT_CLOSE = {
    "ad_close_list_v001": 24107,
    "ad_close_list_v002": 1218,
    "ad_close_list_v003": 173,
}
CLOSE_FIELDS = tuple(DEFAULT_CLOSE)
MISSING_FLAG_NAMES = tuple(f"history_mask__close_{index}__missing" for index in (1, 2, 3))
CLOSE_LENGTH_FIELDS = tuple(f"{field}__length" for field in CLOSE_FIELDS)


def _names(values, description):
    if isinstance(values, (str, bytes)):
        raise ValueError(description + " must be a sequence of names")
    names = list(values)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(description + " must contain nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError(description + " must contain unique names")
    return names


def create_mask_spec(manifest, vocabs):
    """Resolve defaults through each close field's actual vocabulary group.

    The JSON-safe result records field positions and encoded IDs. Missing default
    tokens are an error: falling back to UNK would also remove unrelated histories.
    Only manifest schema/vocabulary metadata is accessed, never target columns.
    """
    sequence_fields = _names(manifest["sequence_fields"], "sequence_fields")
    groups = manifest["group_for_field"]
    sizes = manifest.get("vocabulary_sizes", {})
    rules = []
    for field, raw_default in DEFAULT_CLOSE.items():
        if field not in sequence_fields:
            raise ValueError("Missing close sequence field: " + field)
        if field not in groups:
            raise ValueError("Missing vocabulary group for " + field)
        group = groups[field]
        if group not in vocabs:
            raise ValueError("Missing vocabulary group: " + str(group))
        vocabulary = vocabs[group]
        keys = [key for key in (str(raw_default), raw_default) if key in vocabulary]
        if not keys:
            raise ValueError(f"Default token {raw_default} absent from vocabulary {group}; cannot mask UNK safely")
        encoded = vocabulary[keys[0]]
        if any(vocabulary[key] != encoded for key in keys):
            raise ValueError("Conflicting string/integer default token IDs in " + str(group))
        if isinstance(encoded, (bool, np.bool_)) or not isinstance(encoded, (int, np.integer)) or encoded < 2:
            raise ValueError("Default token must have a distinct integer ID >= 2 in " + str(group))
        if group in sizes and encoded >= sizes[group]:
            raise ValueError("Default token ID exceeds vocabulary size in " + str(group))
        rules.append({"field": field, "sequence_index": sequence_fields.index(field),
                      "group": group, "raw_default": raw_default, "encoded_default": int(encoded)})
    return {"version": 1, "pad_id": 0, "unknown_id": 1,
            "sequence_fields": sequence_fields, "fields": rules,
            "missing_flag_names": list(MISSING_FLAG_NAMES)}


def apply_sequence_mask(sequences, spec):
    """Return (masked_copy, missing_flags) without modifying a batch or memmap.

    Input shape is [rows, len(spec['sequence_fields']), slots], with integer IDs.
    Flags are [rows, 3] float32: 1 exactly when a field has no surviving non-PAD
    slots. Unknown ID 1 remains a supplied token; it cannot be decoded as a default.
    Empty batches and zero-slot histories are supported. Slots are never shifted.
    """
    values = np.asarray(sequences)
    if values.ndim != 3 or values.shape[1] != len(spec["sequence_fields"]):
        raise ValueError("sequences must have shape [rows, manifest sequence fields, slots]")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("sequences must contain integer encoded IDs")
    if np.any(values < 0):
        raise ValueError("Encoded sequence IDs must be nonnegative; PAD is 0")
    result = values.copy()
    missing = np.empty((len(values), len(CLOSE_FIELDS)), dtype=np.float32)
    for column, rule in enumerate(spec["fields"]):
        history = result[:, rule["sequence_index"], :]
        history[history == rule["encoded_default"]] = 0
        missing[:, column] = ~np.any(history != 0, axis=1)
    return result, missing


def _drop_reason(name):
    if name.startswith("snapshot_close_"):
        return "snapshot_close"
    if name in CLOSE_LENGTH_FIELDS:
        return "close_length"
    if name.startswith("hist_close_"):
        return "hist_close"
    return None


def feature_column_selection(names):
    """Return ordered np.intp indices of numerics compatible with the mask.

    Select from the caller's current numeric width before appending missing flags.
    All snapshot_close_* and hist_close_* columns and the three original close
    lengths are omitted. Existing missing indicators and history95 are preserved.
    Values and scalers are never edited or refit; kept standardized values remain
    exactly as cached. For the current schemas: 500 -> 470 kept, 166 -> 136 kept.
    """
    names = _names(names, "numeric field names")
    if set(names).intersection(MISSING_FLAG_NAMES):
        raise ValueError("Numeric fields already contain appended history-mask flags")
    return np.asarray([index for index, name in enumerate(names) if _drop_reason(name) is None], dtype=np.intp)


def mask_metadata(spec, numeric_fields):
    """Return JSON-safe protocol metadata for selection plus three flag columns."""
    names = _names(numeric_fields, "numeric field names")
    kept = feature_column_selection(names).tolist()
    removed = [index for index, name in enumerate(names) if _drop_reason(name) is not None]
    counts = {reason: sum(_drop_reason(names[index]) == reason for index in removed)
              for reason in ("snapshot_close", "close_length", "hist_close")}
    output_names = [names[index] for index in kept] + list(MISSING_FLAG_NAMES)
    return {
        "enabled": True,
        "spec": {"version": spec["version"], "pad_id": spec["pad_id"], "unknown_id": spec["unknown_id"],
                 "sequence_fields": list(spec["sequence_fields"]),
                 "fields": [dict(rule) for rule in spec["fields"]],
                 "missing_flag_names": list(MISSING_FLAG_NAMES)},
        "input_numeric_count": len(names), "removed_numeric_count": len(removed),
        "retained_numeric_count": len(kept), "appended_missing_count": len(MISSING_FLAG_NAMES),
        "output_numeric_count": len(output_names), "kept_columns": kept, "removed_columns": removed,
        "removed_fields": [names[index] for index in removed], "removed_field_counts": counts,
        "numeric_fields": output_names,
        "retained_history_feature_count": sum(names[index].startswith(("history_input__", "history_metadata__"))
                                               for index in kept),
        "missing_rule": "1 iff no non-PAD encoded close-history slots survive default removal; UNK=1 is retained",
        "numeric_rule": "Select cached columns without changing values or scalers; append three unstandardized 0/1 flags",
        "raw_sources_requiring_rebuild": {
            "close_length_and_matches": "Original per-row raw close histories with defaults removed, and candidate IDs for matches",
            "snapshot_close": "Complete per-user/day raw close snapshot unions and past-union state, rebuilt after default removal",
        },
        "labels_used": False,
    }
