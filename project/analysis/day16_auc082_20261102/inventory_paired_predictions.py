"""Inventory real-day paired predictors without reading advertisement outcomes."""
from pathlib import Path
import json
import re
import numpy as np
from fit_stacker import ROOT, HERE, REFERENCE, digest, write

FAMILIES = ["strong_interest_control_20261031", "userday_aux_20261101",
            "candidate_day_context_20261101", "day_seed_confirmation_20261101",
            "learned_audience_20261101", "content_audience_20261101",
            "regularized_auc_20261101", "pair_state_20261101",
            "conditional_pair_state_20261101", "typed_interest_association_20261101",
            "supervised_interest_20261101", "temporal_response_20261101",
            "target_consistency_20261101", "userday_count_20261101",
            "architecture_goal_20261029", "nextday_goal_20261029",
            "auc85_goal_20261031", "strict_nextday_interest_20261031",
            "transductive_interest_20261101", "frozen_history_transfer_20261101"]


def rows_path(path, day):
    for parent in [path.parent,path.parent.parent,path.parent.parent.parent]:
        for name in ["valid_indices.npy",f"day{day}_indices.npy",f"indices_day{day}.npy"]:
            candidate = parent/name
            if candidate.is_file():
                values = np.load(candidate,mmap_mode="r")
                if values.shape == (424780 if day==15 else 421983,):
                    return candidate
    return None


def main():
    records,seen = [],set()
    reference_rows = {d:np.load(REFERENCE/f"day{d}_indices.npy") for d in [15,16]}
    for family in FAMILIES:
        for path in sorted((ROOT/"analysis"/family).rglob("*prediction*.npy")):
            relative = str(path.relative_to(ROOT)).replace("\\","/")
            if "day15" not in relative and "20191115" not in relative:
                continue
            other = ROOT/relative.replace("day15","day16").replace("20191115","20191116")
            if not other.is_file():
                continue
            try:
                p15,p16 = np.load(path,mmap_mode="r"),np.load(other,mmap_mode="r")
            except Exception:
                continue
            if p15.shape != (424780,) or p16.shape != (421983,):
                continue
            if not all(np.isfinite(p).all() and np.all((p>=0)&(p<=1)) for p in [p15,p16]):
                continue
            hashes = (digest(path),digest(other))
            if hashes in seen:
                continue
            seen.add(hashes)
            aligned,indices = [],[]
            for candidate,day in [(path,15),(other,16)]:
                index_path = rows_path(candidate,day)
                aligned.append(index_path is not None and np.array_equal(np.load(index_path),reference_rows[day]))
                indices.append(str(index_path) if index_path else None)
            records.append({"family":family,"path15":relative,
                            "path16":str(other.relative_to(ROOT)).replace("\\","/"),
                            "sha256":hashes,"alignment_verified":all(aligned),"indices":indices})
    write(HERE/"paired_prediction_inventory.json",{"ad_labels_read":False,"records":records})
    print(json.dumps({"unique_pairs":len(records),"aligned_pairs":sum(r["alignment_verified"] for r in records),
                      "families":{family:sum(r["family"]==family for r in records) for family in FAMILIES}},indent=2),flush=True)


if __name__ == "__main__":
    main()
