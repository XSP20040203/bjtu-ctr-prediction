"""Rebuild source-only graph inputs and verify exported and mapped teacher vectors."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from source_graph import SourceGraphEncoder, build_source_graph, load_graph

HERE = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def sorted_edges(values):
    return values[np.lexsort((values[:, 1], values[:, 0]))]


def main():
    torch.set_num_threads(4)
    teacher = HERE / "source_graph_teacher_day16"
    metadata = json.loads((teacher / "manifest.json").read_text(encoding="utf-8"))
    graph_dir = Path(metadata["source_graph_dir"])
    manifest, relations = load_graph(graph_dir)
    assert metadata["graph_manifest_sha256"] == digest(graph_dir / "manifest.json")
    assert not manifest["ad_labels_used"] and not metadata["advertisement_labels_used"]
    source = Path(manifest["source_path"])
    assert digest(source) == manifest["source_sha256"]
    frame = pd.read_csv(source, usecols=["u_userId", "i_docId", "i_cat", "i_entities", "label", "e_et"])
    rebuilt = build_source_graph(frame, metadata["source_cutoff_day"])
    assert rebuilt["vocabularies"] == manifest["vocabularies"]
    assert rebuilt["source_rows"] == manifest["source_rows"]
    for name, values in relations.items():
        np.testing.assert_array_equal(sorted_edges(values), sorted_edges(rebuilt["relations"][name]))
    for name in ["positive_pairs", "eligible_positive_pairs", "negative_indptr", "negative_documents", "source_positive_count"]:
        np.testing.assert_array_equal(np.load(graph_dir / (name + ".npy")), rebuilt[name])
    checkpoint = torch.load(teacher / "graph_teacher.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SourceGraphEncoder(manifest["n_nodes"], relations, checkpoint["dimension"], checkpoint["layers"])
    # Independently reconstructed relation buffers must match those saved by pretraining.
    for name, values in encoder.state_dict().items():
        if name.startswith("adjacency_"):
            saved = checkpoint["state_dict"][name].coalesce()
            np.testing.assert_array_equal(saved.indices(), values.coalesce().indices())
            np.testing.assert_array_equal(saved.values(), values.coalesce().values())
        elif name == "relation_activity":
            torch.testing.assert_close(values, checkpoint["state_dict"][name], rtol=0, atol=0)
    encoder.load_state_dict(checkpoint["state_dict"], strict=True)
    encoder.to(device).eval()
    count = np.load(teacher / "user_positive_count.npy")
    with torch.no_grad():
        reproduced = encoder()[:metadata["source_users"]].cpu().numpy()
    reproduced = (reproduced - checkpoint["center"]) / checkpoint["scale"]
    reproduced[count == 0] = 0
    exported = np.load(teacher / "user_vectors.npy")
    np.testing.assert_allclose(reproduced, exported, rtol=3e-5, atol=3e-5)
    raw_ids = np.load(teacher / "user_raw_ids.npy")
    np.testing.assert_array_equal(raw_ids, manifest["vocabularies"]["users"])
    mapped_records = []
    for run in sorted((HERE / "results_modern").glob("phase3_graph_*/graph/seed*/metrics.json")):
        record = json.loads(run.read_text(encoding="utf-8"))
        initialization = json.loads((run.parent / "graph_initialization.json").read_text(encoding="utf-8"))
        assert initialization["teacher_manifest_sha256"] == digest(teacher / "manifest.json")
        cache = Path(record["cache_dir"])
        vocabs = json.loads((cache / "vocabs.json").read_text(encoding="utf-8"))["user_id"]
        saved = torch.load(run.parent / "model.pt", map_location="cpu", weights_only=False)
        vectors = saved["state_dict"]["source_user_vectors"].numpy()
        counts = saved["state_dict"]["source_positive_count"].numpy()
        expected_vectors, expected_counts = np.zeros_like(vectors), np.zeros_like(counts)
        lookup = {str(int(raw)): index for index, raw in enumerate(raw_ids)}
        for raw, encoded in vocabs.items():
            if raw in lookup:
                expected_vectors[encoded] = exported[lookup[raw]]
                expected_counts[encoded] = count[lookup[raw]]
        np.testing.assert_array_equal(counts, expected_counts)
        difference = float(np.max(np.abs(vectors - expected_vectors)))
        if record["protected_source"]:
            np.testing.assert_array_equal(vectors, expected_vectors)
        else:
            assert difference > 0, "Fine-tuned graph vectors never changed"
        arrays = {name: np.load(cache / (name + ".npy"), mmap_mode="r") for name in ["date", "train_indices", "valid_indices"]}
        assert metadata["source_cutoff_day"] <= max(arrays["date"][arrays["train_indices"]].max(), arrays["date"][arrays["valid_indices"]].max())
        mapped_records.append({"experiment": run.parts[-4], "protected_source": record["protected_source"],
                               "max_source_vector_update": difference, "mapping_verified": True})
    result = {"status": "passed", "source_graph_rebuilt_from_original_feeds": True,
              "source_sha256": manifest["source_sha256"], "cutoff_day": metadata["source_cutoff_day"],
              "graph_relations_and_negative_pools_recomputed": True, "teacher_checkpoint_export_replayed": True,
              "teacher_export_max_error": float(np.max(np.abs(reproduced - exported))),
              "ad_model_mapping_checks": mapped_records, "advertisement_validation_labels_used": False,
              "scope": "Provided offline news feedback snapshot through Nov16; no online causal availability claim"}
    (HERE / "phase3_graph_source_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
