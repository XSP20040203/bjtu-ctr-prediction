"""Read trained group models and show feature-gain diagnostics without outcomes."""
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT / "analysis/auc085_next/deps"))
import lightgbm as lgb

metadata=json.loads((HERE / "runs/weighted_v1_day16/context_manifest.json").read_text(encoding="utf8"))
names=metadata["features"]+[f"history_{window}_{kind}" for window in ["all","3d","7d"]
    for kind in ["log_exposures","log_clicks","log_days","smoothed_ctr","active_day_fraction"]]+[
    "days_since_observed","days_since_click","base_log_size","base_log_mass","base_mean_logit","base_max_logit"]
for filename in ["poisson_7.txt","binary_3.txt"]:
    model=lgb.Booster(model_str=(HERE / "runs/rich_group_residual_v1" / filename).read_text(encoding="utf8"))
    gain=model.feature_importance(importance_type="gain")
    order=np.argsort(-gain)[:25]
    print(json.dumps({"model":filename,"gain":[{"feature":names[i],"gain":float(gain[i])} for i in order]},indent=2))
