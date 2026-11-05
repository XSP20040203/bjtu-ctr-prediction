"""Descriptive day-level source evidence coverage, never used as ad target features."""
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    result=[]
    for day in [15,16]:
        folder=HERE/f"cache_day{day}"
        m=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        keys=np.load(folder/"keys.npy",mmap_mode="r")
        mass=np.load(folder/"topic_mass.npy",mmap_mode="r")
        context=np.load(folder/"interest_context.npy",mmap_mode="r")
        today=keys%100000000==m["validation_day"]
        for name,mask in [("all_eligible_user_days",keys!=0),("validation_user_days",today)]:
            row={"validation_day":m["validation_day"],"scope":name,"requests":int(mask.sum()),
                 "same_day_source_missing_rate":float(context[mask,1,-2].mean()),
                 "same_day_has_click_rate":float((mass[mask,1,1]>0).mean()),
                 "same_day_exposure_quantiles":np.quantile(mass[mask,1,0],[0,.1,.5,.9,1]).tolist(),
                 "same_day_click_quantiles":np.quantile(mass[mask,1,1],[0,.1,.5,.9,1]).tolist(),
                 "previous3_has_click_rate":float((mass[mask,0,1]>0).mean()),
                 "explicit_dislike_available_rate":float((mass[mask,1,3]>0).mean())}
            result.append(row)
    output={"scope":"Source feedback only; no ad labels used","records":result}
    (HERE/"source_interest_coverage.json").write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(output),flush=True)


if __name__=="__main__":main()
