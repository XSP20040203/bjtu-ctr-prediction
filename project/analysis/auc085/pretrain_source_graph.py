"""Train source preference representations without any advertisement labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from source_graph import SourceGraphEncoder, load_graph

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving graph teacher {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(6)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest, relations = load_graph(args.graph_dir)
    model = SourceGraphEncoder(manifest["n_nodes"], relations, dimension=32, n_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    pairs = np.load(args.graph_dir / "eligible_positive_pairs.npy")
    pointers = np.load(args.graph_dir / "negative_indptr.npy")
    negatives = np.load(args.graph_dir / "negative_documents.npy")
    counts = np.load(args.graph_dir / "source_positive_count.npy")
    doc_offset = manifest["offsets"]["documents"]
    history = []
    started = time.perf_counter()
    print(f"GRAPH_PRETRAIN nodes={manifest['n_nodes']} pairs={len(pairs)} device={device}", flush=True)
    for epoch in range(1, args.epochs + 1):
        rng = np.random.default_rng(42 + epoch)
        order = rng.permutation(len(pairs))
        total, examples = 0., 0
        for start in range(0, len(order), 2048):
            selected = pairs[order[start:start + 2048]]
            user = selected[:, 0]
            negative_position = rng.integers(pointers[user], pointers[user + 1])
            negative_doc = negatives[negative_position]
            users = torch.as_tensor(user, device=device)
            positive = torch.as_tensor(selected[:, 1] + doc_offset, device=device)
            negative = torch.as_tensor(negative_doc + doc_offset, device=device)
            optimizer.zero_grad(set_to_none=True)
            representations = model()
            positive_score = (representations[users] * representations[positive]).sum(dim=-1)
            negative_score = (representations[users] * representations[negative]).sum(dim=-1)
            ranking = F.softplus(negative_score - positive_score).mean()
            raw = model.embeddings.weight
            regularizer = (raw[users].square().sum(dim=-1) + raw[positive].square().sum(dim=-1) + raw[negative].square().sum(dim=-1)).mean()
            loss = ranking + .0001 * regularizer
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite source graph ranking objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(ranking.detach()) * len(selected)
            examples += len(selected)
        record = {"epoch": epoch, "source_train_bpr_loss": total / examples,
                  "elapsed_seconds": time.perf_counter() - started,
                  "relation_weights": model.relation_logits.softmax(dim=-1).detach().cpu().tolist()}
        history.append(record)
        pd.DataFrame([{key: row[key] for key in ["epoch", "source_train_bpr_loss", "elapsed_seconds"]} for row in history]).to_csv(args.output_dir / "training_history.csv", index=False)
        print(json.dumps(record), flush=True)
    model.eval()
    with torch.no_grad():
        vectors = model()[:len(manifest["vocabularies"]["users"])].cpu().numpy()
    observed = counts > 0
    center = vectors[observed].mean(axis=0)
    scale = np.maximum(vectors[observed].std(axis=0), 1e-6)
    vectors = (vectors - center) / scale
    vectors[~observed] = 0
    np.save(args.output_dir / "user_vectors.npy", vectors.astype(np.float32))
    np.save(args.output_dir / "user_raw_ids.npy", np.asarray(manifest["vocabularies"]["users"], dtype=np.int64))
    np.save(args.output_dir / "user_positive_count.npy", counts)
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "graph_dir": str(args.graph_dir.resolve()), "dimension": 32, "layers": 2,
                "center": center, "scale": scale, "epochs": args.epochs}, args.output_dir / "graph_teacher.pt")
    metadata = {"status": "complete", "source_graph_dir": str(args.graph_dir.resolve()), "source_cutoff_day": manifest["cutoff_day"],
                "graph_manifest_sha256": hashlib.sha256((args.graph_dir / "manifest.json").read_bytes()).hexdigest(),
                "dimension": 32, "layers": 2, "epochs": args.epochs, "source_users": len(vectors),
                "source_train_bpr_loss": history[-1]["source_train_bpr_loss"], "training_seconds": time.perf_counter() - started,
                "advertisement_labels_used": False, "advertisement_auc_claimed": False}
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("GRAPH_TEACHER_COMPLETE " + json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
