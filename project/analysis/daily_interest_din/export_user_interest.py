"""Export interpretable known-category interest strengths without ad-label statistics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def main(args):
    directory=HERE/f"cache_day{args.day}"
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    keys=np.load(directory/"keys.npy",mmap_mode="r")
    masses=np.load(directory/"topic_mass.npy",mmap_mode="r")
    weights=np.load(directory/"topic_weights.npy",mmap_mode="r")
    inverse={index:raw for raw,index in manifest["topic_vocabulary"].items()}
    selected=np.flatnonzero(keys%100000000==manifest["validation_day"])
    records=[]
    for row in selected:
        entry={"user_id":int(keys[row]//100000000),"day":manifest["validation_day"],
               "today_exposures":float(masses[row,1,0]),"today_clicks":float(masses[row,1,1]),
               "past3_clicks":float(masses[row,0,1])}
        for axis,prefix in [(0,"past3"),(1,"today")]:
            for channel,name in [(0,"exposure"),(1,"click"),(2,"reading"),(3,"dislike")]:
                values=np.array(weights[row,axis,channel]);values[:2]=0
                indices=np.argsort(values)[::-1][:5]
                entry[f"{prefix}_{name}_top5"]="^".join(inverse.get(int(i),"UNK") for i in indices if values[i]>0)
                entry[f"{prefix}_{name}_top5_weights"]="^".join(f"{float(values[i]):.6f}" for i in indices if values[i]>0)
        records.append(entry)
    output=HERE/f"用户当日兴趣示例_day{args.day}.csv"
    if output.exists():raise FileExistsError("Preserving existing interest export")
    pd.DataFrame(records).to_csv(output,index=False,encoding="utf-8-sig")
    print(str(output),len(records),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--day",type=int,choices=[15,16],default=16);main(parser.parse_args())
