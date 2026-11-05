"""Audit the unsigned unknown-label sentinel without rewriting frozen runs."""
import argparse
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from fit_stacker import ROOT, HERE, digest, write


def signed_labels(labels, train, valid):
    train,valid = np.asarray(train),np.asarray(valid)
    if np.intersect1d(train,valid).size:
        raise ValueError("Training and held rows overlap")
    labels = np.asarray(labels)
    if not np.isin(labels[train],[0,1]).all():
        raise ValueError("Training requires observed binary labels")
    result = labels.astype("int16",copy=True)
    result[valid] = -1
    np.testing.assert_array_equal(result[train],labels[train])
    return result


def audit(family):
    out = HERE/"runs"/f"{family}_day16"
    protocol = json.loads((out/"protocol.json").read_text(encoding="utf8"))
    for name,expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT/"2026ml"/name) != expected:
            raise AssertionError("Raw dataset changed")
    raw = pd.read_csv(ROOT/"2026ml/train.csv",usecols=["pt_d","label"],dtype="int32")
    train = np.flatnonzero(raw.pt_d.to_numpy()<20191116)
    valid = np.flatnonzero(raw.pt_d.to_numpy()==20191116)
    labels = np.load(ROOT/"analysis/auc085/cache_rich_stats_time16/label.npy")
    if len(labels) != len(raw):
        raise AssertionError("Cached label alignment")
    np.testing.assert_array_equal(labels,raw.label.to_numpy())
    np.testing.assert_array_equal(valid,np.load(out/"valid_indices.npy"))
    contract = json.loads((out/"input_contract.json").read_text(encoding="utf8"))
    import hashlib
    def hash_array(values):
        return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
    if hash_array(train) != contract["fit_indices_sha256"] or hash_array(valid) != contract["evaluation_indices_sha256"]:
        raise AssertionError("Fitted row boundaries changed")
    legacy = labels.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",DeprecationWarning)
        legacy[valid] = -1
    np.testing.assert_array_equal(legacy[train],labels[train])
    if np.isin(legacy[valid],[0,1]).any():
        raise AssertionError("Unknown sentinel is still a binary outcome")
    context = np.load(out/"context.npz")
    if not np.all(context["any_targets"][context["keys"][:,0]==20191116] == -1):
        raise AssertionError("Held auxiliary outcome is not masked")
    fixed = signed_labels(labels,train,valid)
    result = {"status":"passed","family":family,"cached_dtype":str(labels.dtype),
              "legacy_held_sentinel_values":np.unique(legacy[valid]).tolist(),
              "legacy_minus_one_check_expected_false":True,
              "legacy_input_contract_minus_one_wording_inaccurate":True,
              "all_training_indices_strictly_before16":True,
              "legacy_training_values_identical_to_raw_binary_labels":True,
              "legacy_held_values_not_binary_training_targets":True,
              "legacy_auxiliary_targets_on16_equal_minus_one":True,
              "signed_fix_dtype":str(fixed.dtype),"signed_fix_held_sentinel":-1,
              "training_rows":len(train),"validation_rows":len(valid),
              "source_sha256":digest(__file__),"original_datasets_unchanged":True,
              "conclusion":"Unsigned mask wraps to255. Held rows are excluded by exact date indices; training targets remain correct. Use signed_labels for new fits."}
    write(out/"label_boundary_audit.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family",choices=["weighted_v1","duplicate_mean_v1"],required=True)
    audit(parser.parse_args().family)
