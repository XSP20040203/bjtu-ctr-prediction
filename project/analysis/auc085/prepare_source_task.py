"""Encode news-click supervision sharing the exact advertisement vocabularies."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "model_comparison"
sys.path.insert(0, str(BENCH))
from benchmark_core import category_key, make_vocab, encode_sequence
from train_models import write_json

SHARED = {"u_userId": "user_id", "u_phonePrice": "feed_phone_price",
          "u_browserLifeCycle": "feed_browser_lifecycle", "u_browserMode": "feed_browser_mode",
          "u_feedLifeCycle": "u_feedLifeCycle", "u_refreshTimes": "u_refreshTimes", "i_cat": "news_interest"}
OWN = ["i_docId", "i_s_sourceId", "i_regionEntity", "i_dtype", "e_ch", "e_m", "e_po", "e_pl", "e_rn", "e_section"]
SEQUENCES = ["u_newsCatInterests", "u_newsCatDislike", "u_newsCatInterestsST", "u_click_ca2_news"]
NUMERIC = ["i_dislikeTimes", "i_upTimes"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ad-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-doc-id", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving existing source task: {args.output_dir}")
    ad_manifest = json.loads((args.ad_cache / "manifest.json").read_text(encoding="utf-8"))
    ad_vocabs = json.loads((args.ad_cache / "vocabs.json").read_text(encoding="utf-8"))
    dates = np.load(args.ad_cache / "date.npy", mmap_mode="r")
    train = np.load(args.ad_cache / "train_indices.npy")
    valid = np.load(args.ad_cache / "valid_indices.npy")
    last_day = int(max(dates[train].max(), dates[valid].max()))
    own = [name for name in OWN if not (args.exclude_doc_id and name == "i_docId")]
    categorical = list(SHARED) + own
    frame = pd.read_csv(HERE.parents[1] / "2026ml/train_feeds.csv",
                        usecols=categorical + SEQUENCES + NUMERIC + ["label", "e_et"],
                        dtype={name: str for name in SEQUENCES})
    frame = frame.loc[frame.e_et.le(last_day)].reset_index(drop=True)
    group_for = {**SHARED, **{name: "source_" + name for name in own}, **{name: "news_interest" for name in SEQUENCES}}
    vocabs = {group: ad_vocabs[group] for group in set(SHARED.values())}
    for name in own:
        vocabs[group_for[name]] = make_vocab(frame[name])
    categories = np.empty((len(frame), len(categorical)), dtype=np.int32)
    for index, name in enumerate(categorical):
        categories[:, index] = frame[name].map(category_key).map(vocabs[group_for[name]]).fillna(1).to_numpy(dtype=np.int32)
    sequences = np.empty((len(frame), len(SEQUENCES), 5), dtype=np.int32)
    for index, name in enumerate(SEQUENCES):
        vocabulary = vocabs["news_interest"]
        values = frame[name].fillna("")
        lookup = {value: encode_sequence(value, vocabulary) for value in values.unique()}
        sequences[:, index] = np.stack([lookup[value] for value in values])
    numeric = np.log1p(frame[NUMERIC].to_numpy(dtype=np.float32).clip(min=0))
    mean, std = numeric.mean(axis=0), np.maximum(numeric.std(axis=0), 1e-6)
    numeric = (numeric - mean) / std
    labels = frame.label.eq(1).to_numpy(dtype=np.uint8)
    args.output_dir.mkdir(parents=True)
    for name, values in [("categories", categories), ("sequences", sequences), ("numeric", numeric),
                         ("label", labels), ("date", frame.e_et.to_numpy(dtype=np.int32))]:
        np.save(args.output_dir / (name + ".npy"), values)
    schema = {"categorical_fields": categorical, "sequence_fields": SEQUENCES, "numeric_fields": NUMERIC,
              "group_for_field": group_for, "vocabulary_sizes": {group: len(vocabulary) + 2 for group, vocabulary in vocabs.items()},
              "positive_rate": float(labels.mean()), "rows": len(frame), "latest_day": last_day,
              "source_input_exclusions": ["label", "pro", "cillabel"] + (["i_docId"] if args.exclude_doc_id else []),
              "ad_cache_dir": str(args.ad_cache.resolve()),
              "ad_manifest_sha256": hashlib.sha256((args.ad_cache / "manifest.json").read_bytes()).hexdigest(),
              "numeric_scaler": {"mean": mean.tolist(), "std": std.tolist()},
              "availability": "Provided source-domain responses through the latest validation day; no advertisement validation labels"}
    write_json(args.output_dir / "manifest.json", schema)
    write_json(args.output_dir / "vocabs.json", vocabs)
    print(json.dumps({"rows": len(frame), "positive": int(labels.sum()), "latest_day": last_day,
                      "categorical_fields": len(categorical), "sequence_fields": len(SEQUENCES), "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
