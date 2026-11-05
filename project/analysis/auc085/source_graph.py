"""Typed source-only preference graph and light relation-aware propagation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent


def normalized_relation(edges, n_nodes):
    pairs = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    pairs = np.unique(np.concatenate([pairs, pairs[:, ::-1]], axis=0), axis=0)
    degree = np.bincount(pairs[:, 0], minlength=n_nodes).astype(np.float32)
    values = 1. / np.sqrt(np.maximum(degree[pairs[:, 0]] * degree[pairs[:, 1]], 1.))
    adjacency = torch.sparse_coo_tensor(torch.from_numpy(pairs.T.copy()), torch.from_numpy(values),
                                      size=(n_nodes, n_nodes)).coalesce()
    return adjacency, torch.from_numpy((degree > 0).astype(np.float32))


def build_source_graph(frame, cutoff_day, min_entity_docs=2, max_entities=20000):
    frame = frame.loc[frame.e_et.le(cutoff_day)].copy()
    users = sorted(int(value) for value in frame.u_userId.unique())
    documents = sorted(frame.i_docId.astype(str).unique())
    categories = sorted(int(value) for value in frame.i_cat.unique())
    user_lookup, doc_lookup, cat_lookup = ({value: index for index, value in enumerate(values)}
                                          for values in [users, documents, categories])
    entity_links = set()
    for doc, value in frame[["i_docId", "i_entities"]].drop_duplicates().itertuples(index=False, name=None):
        if pd.isna(value):
            continue
        for entity in str(value).split("^"):
            if entity and entity not in ["nan", "None"]:
                entity_links.add((doc_lookup[str(doc)], entity))
    counts = Counter(entity for _, entity in entity_links)
    entities = [value for value, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
                if count >= min_entity_docs][:max_entities]
    entity_lookup = {value: index for index, value in enumerate(entities)}
    n_users, n_docs, n_cats = len(users), len(documents), len(categories)
    offsets = {"users": 0, "documents": n_users, "categories": n_users + n_docs,
               "entities": n_users + n_docs + n_cats}
    u = frame.u_userId.map(user_lookup).to_numpy(dtype=np.int64)
    d = frame.i_docId.astype(str).map(doc_lookup).to_numpy(dtype=np.int64)
    positive = np.unique(np.column_stack([u[frame.label.eq(1)], d[frame.label.eq(1)]]), axis=0)
    negative = np.unique(np.column_stack([u[frame.label.eq(-1)], d[frame.label.eq(-1)]]), axis=0)
    positive_key = positive[:, 0] * n_docs + positive[:, 1]
    negative_key = negative[:, 0] * n_docs + negative[:, 1]
    negative = negative[~np.isin(negative_key, positive_key)]
    order = np.lexsort((negative[:, 1], negative[:, 0]))
    negative = negative[order]
    negative_count = np.bincount(negative[:, 0], minlength=n_users)
    negative_indptr = np.r_[0, np.cumsum(negative_count)].astype(np.int64)
    eligible_positive = positive[negative_count[positive[:, 0]] > 0]
    doc_cat = frame[["i_docId", "i_cat"]].drop_duplicates()
    doc_cat_edges = np.column_stack([doc_cat.i_docId.astype(str).map(doc_lookup).to_numpy() + offsets["documents"],
                                    doc_cat.i_cat.map(cat_lookup).to_numpy() + offsets["categories"]]).astype(np.int64)
    doc_entity_edges = np.array([(doc + offsets["documents"], entity_lookup[entity] + offsets["entities"])
                                 for doc, entity in entity_links if entity in entity_lookup], dtype=np.int64).reshape(-1, 2)
    click_edges = positive.copy()
    click_edges[:, 1] += offsets["documents"]
    return {"vocabularies": {"users": users, "documents": documents, "categories": categories, "entities": entities},
            "offsets": offsets, "n_nodes": offsets["entities"] + len(entities),
            "relations": {"user_document": click_edges, "document_category": doc_cat_edges,
                          "document_entity": doc_entity_edges},
            "positive_pairs": positive, "eligible_positive_pairs": eligible_positive,
            "negative_indptr": negative_indptr, "negative_documents": negative[:, 1],
            "source_positive_count": np.bincount(positive[:, 0], minlength=n_users).astype(np.int32),
            "cutoff_day": cutoff_day, "source_rows": len(frame)}


class SourceGraphEncoder(nn.Module):
    """LightGCN-style propagation extended with typed relation mixing."""
    def __init__(self, n_nodes, relations, dimension=32, n_layers=2):
        super().__init__()
        self.embeddings = nn.Embedding(n_nodes, dimension)  # Node zero is a real source user.
        nn.init.normal_(self.embeddings.weight, std=.1)
        self.relation_names = list(relations)
        activity = []
        for index, edges in enumerate(relations.values()):
            adjacency, active = normalized_relation(edges, n_nodes)
            self.register_buffer(f"adjacency_{index}", adjacency)
            activity.append(active)
        self.register_buffer("relation_activity", torch.stack(activity))
        self.relation_logits = nn.Parameter(torch.zeros(n_layers, len(relations)))

    def forward(self):
        current = self.embeddings.weight
        layers = [current]
        for weights in self.relation_logits.softmax(dim=-1):
            active_weight = weights @ self.relation_activity
            messages = sum(weight * torch.sparse.mm(getattr(self, f"adjacency_{index}"), current)
                           for index, weight in enumerate(weights))
            current = messages / active_weight.clamp_min(1e-8)[:, None]
            layers.append(current)
        return torch.stack(layers).mean(dim=0)


def load_graph(directory):
    metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    relations = {name: np.load(directory / ("edges_" + name + ".npy")) for name in metadata["relation_names"]}
    return metadata, relations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff-day", type=int, default=20191116)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving source graph {args.output_dir}")
    source_path = HERE.parents[1] / "2026ml/train_feeds.csv"
    frame = pd.read_csv(source_path, usecols=["u_userId", "i_docId", "i_cat", "i_entities", "label", "e_et"])
    graph = build_source_graph(frame, args.cutoff_day)
    args.output_dir.mkdir(parents=True)
    for name, edges in graph["relations"].items():
        np.save(args.output_dir / ("edges_" + name + ".npy"), edges)
    for name in ["positive_pairs", "eligible_positive_pairs", "negative_indptr", "negative_documents", "source_positive_count"]:
        np.save(args.output_dir / (name + ".npy"), graph[name])
    manifest = {key: graph[key] for key in ["vocabularies", "offsets", "n_nodes", "cutoff_day", "source_rows"]}
    manifest.update(relation_names=list(graph["relations"]),
                    relation_edges={name: len(value) for name, value in graph["relations"].items()},
                    positive_pairs=len(graph["positive_pairs"]), eligible_positive_pairs=len(graph["eligible_positive_pairs"]),
                    source_path=str(source_path.resolve()), source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    ad_labels_used=False, singleton_entities_omitted=True, observed_clicks_excluded_from_negative_pool=True,
                    availability="Source responses and static descriptors supplied through the validation cutoff, under the offline-source-snapshot scenario")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"n_nodes": graph["n_nodes"], "node_types": {key: len(value) for key, value in graph["vocabularies"].items()},
                      "relation_edges": manifest["relation_edges"], "positive_pairs": manifest["positive_pairs"],
                      "eligible_positive_pairs": manifest["eligible_positive_pairs"]}), flush=True)


if __name__ == "__main__":
    main()
