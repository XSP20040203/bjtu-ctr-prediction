"""Complete source-day aggregate descriptors, independent of advertisement labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NUMERIC=["u_phonePrice","u_browserLifeCycle","u_browserMode","u_feedLifeCycle","u_refreshTimes",
         "i_regionEntity","i_dislikeTimes","i_upTimes","i_dtype","e_ch","e_m","e_po","e_pl","e_rn","e_section"]
SEQUENCES=["u_newsCatInterests","u_newsCatDislike","u_newsCatInterestsST","u_click_ca2_news","i_entities"]


def aggregate_window(source, users):
    source=source.copy()
    source["clicked"]=source.label.eq(1).astype(np.float32)
    source["liked"]=source.cillabel.eq(1).astype(np.float32)
    source["finished"]=(source.pro.ge(80)&source.label.eq(1)).astype(np.float32)
    source["clicked_progress"]=source.pro.where(source.label.eq(1))
    groups=source.groupby("u_userId",sort=False)
    output=groups.agg(source_rows=("label","size"),clicks=("clicked","sum"),likes=("liked","sum"),
          click_rate=("clicked","mean"),progress_mean=("pro","mean"),progress_std=("pro","std"),
          progress_max=("pro","max"),progress_median=("pro","median"),finish_rate=("finished","mean"),
          click_progress_mean=("clicked_progress","mean"),click_progress_max=("clicked_progress","max"),
          click_progress_std=("clicked_progress","std"),documents_unique=("i_docId","nunique"),
          sources_unique=("i_s_sourceId","nunique"),topics_unique=("i_cat","nunique"))
    for field in NUMERIC:
        stats=groups[field].agg(["mean","std","nunique"])
        stats.columns=[field+"__"+suffix for suffix in stats]
        output=output.join(stats)
    for field in SEQUENCES:
        text=source[field].fillna("").astype(str)
        source[field+"__length"]=np.where(text.eq(""),0,text.str.count(r"\^")+1)
        stats=source.groupby("u_userId")[field+"__length"].agg(["mean","std"])
        stats.columns=[field+"__length_"+suffix for suffix in stats]
        output=output.join(stats)
        output[field+"__snapshot_nunique"]=groups[field].nunique()
    for click_only,prefix in [(False,"exposure"),(True,"clicked")]:
        subset=source.loc[source.clicked.eq(1)] if click_only else source
        counts=subset.groupby(["u_userId","i_cat"]).size()
        probabilities=counts/counts.groupby(level=0).transform('sum')
        entropy=-(probabilities*np.log(np.maximum(probabilities,1e-12))).groupby(level=0).sum()
        output[prefix+"_topic_entropy"]=entropy
        output[prefix+"_topic_top_share"]=probabilities.groupby(level=0).max()
    output["document_repeat_fraction"]=1-output.documents_unique/output.source_rows
    output["clicked_finish_fraction"]=source.groupby('u_userId').finished.sum()/np.maximum(output.clicks,1)
    output["source_missing"]=0.
    result=output.reindex(np.asarray(users,dtype=np.int64)).copy()
    result["source_missing"]=result.source_rows.isna().astype(np.float32)
    return result.fillna(0).astype(np.float32)


def build_rich_source_features(feeds,requests):
    if not feeds.label.isin([-1,1]).all():raise ValueError('Expected source labels -1/1')
    frames=[]
    for day,group in requests.groupby('pt_d',sort=True):
        date=pd.Timestamp(str(int(day)))
        cutoff=int((date-pd.Timedelta(days=3)).strftime('%Y%m%d'))
        today=feeds.loc[feeds.e_et.eq(day)]
        past=feeds.loc[(feeds.e_et<day)&(feeds.e_et>=cutoff)]
        values=[]
        for prefix,source in [('today',today),('past3',past)]:
            out=aggregate_window(source,group.user_id.to_numpy())
            out.columns=[f'rich_source__{prefix}__{name}' for name in out]
            out.index=group.index
            values.append(out)
        frames.append(pd.concat(values,axis=1))
    return pd.concat(frames).reindex(requests.index)


def main(args):
    if args.output.exists():raise FileExistsError('Preserving existing rich source features')
    folder=HERE/f'cache_day{str(args.day)[-2:]}'
    keys=np.load(folder/'keys.npy')
    mapping=np.load(folder/'request_index.npy',mmap_mode='r')
    valid=np.flatnonzero(keys>0)
    requests=pd.DataFrame({'user_id':keys[valid]//100000000,'pt_d':keys[valid]%100000000},index=valid)
    feeds=pd.read_csv(ROOT/'2026ml/train_feeds.csv',dtype={field:str for field in SEQUENCES+['i_docId','i_s_sourceId']})
    feeds=feeds.loc[feeds.e_et<=args.day].copy()
    features=build_rich_source_features(feeds,requests)
    names=features.columns.tolist()
    compact=np.zeros((len(keys),len(names)),dtype=np.float32);compact[valid]=features.to_numpy()
    if not np.isfinite(compact).all():raise AssertionError('Nonfinite source aggregate')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    output=open_memmap(args.output,mode='w+',dtype='float16',shape=(len(mapping),len(names)))
    for start in range(0,len(mapping),50000):output[start:start+50000]=compact[mapping[start:start+50000]]
    output.flush()
    meta={'shape':list(output.shape),'dtype':'float16','features':names,'advertisement_labels_loaded':False,
       'validation_day':args.day,'source_rule':'Complete same-day and prior3 natural dates source feedback; no later source input',
       'encoding_note':'Anonymous numeric codes provide empirical aggregate descriptors, not physical measurement semantics',
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k!='features'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--day',type=int,choices=[20191115,20191116],required=True)
    parser.add_argument('--output',type=Path,required=True);main(parser.parse_args())
