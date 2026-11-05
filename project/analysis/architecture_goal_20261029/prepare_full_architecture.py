"""CPU-only full architecture inputs in original train-then-final row order.

Reuse existing feature builders without importing GPU trainers. The architecture
applies a second training-only scaler to base numeric[:75], and a first scaler
to the raw exposure/snapshot/multiscale/history/source-context blocks.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "full_refit/features"
DAILY = ROOT / "analysis/daily_interest_din"
CACHE = DAILY / "full_refit_cache"
NEXT = ROOT / "analysis/auc085_next"
PREVIOUS = ROOT / "analysis/nextday_goal_20261029"
OOF = PREVIOUS / "full_best/oof"
MULTISCALE = ROOT / "output/chronological_multiscale_v3"
NTRAIN, NFINAL = 2526320, 404182
NROWS = NTRAIN + NFINAL
FINAL_DAY = 20191117
IDENTITY_RULE = "SHA256 of contiguous little-endian signed int64 (<i8), no header, original train.csv then final_test.csv rows"
SCALER_RULE = "Per-column float64 training-prefix mean and population std (ddof=0), std=max(std,1e-5); float32 input minus float64 mean divided by float64 std; clip [-8,8] then cast float32"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    staged = path.with_name(path.name + ".tmp")
    staged.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    staged.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def emit(phase, **values):
    print(json.dumps({"phase": phase, **values}, ensure_ascii=False), flush=True)


class Matrix:
    """Short-lived mappings bound resident memory for both C and F sources."""
    def __init__(self, path, width):
        self.path = Path(path)
        mapped = np.load(self.path, mmap_mode="r", allow_pickle=False)
        self.shape = tuple(mapped.shape)
        self.dtype = mapped.dtype
        self.offset = mapped.offset
        self.order = "F" if mapped.flags.f_contiguous else "C"
        mapped._mmap.close()
        if self.shape != (NROWS, width) or self.dtype != np.dtype("float32"):
            raise AssertionError("Input matrix shape/dtype mismatch: " + str(path))

    def rows(self, start, stop, first=0, last=None):
        mapped = np.memmap(self.path, mode="r", dtype=self.dtype, shape=self.shape,
                           order=self.order, offset=self.offset)
        try:
            return np.array(mapped[start:stop, first:last], dtype="float32", order="C", copy=True)
        finally:
            mapped._mmap.close()

    def column(self, column, stop=NTRAIN):
        mapped = np.memmap(self.path, mode="r", dtype=self.dtype, shape=self.shape,
                           order=self.order, offset=self.offset)
        try:
            return np.array(mapped[:stop, column], dtype="float64", copy=True)
        finally:
            mapped._mmap.close()


def identity_check(chunksize):
    cache = {field: np.load(CACHE / (name + ".npy"), mmap_mode="r")
             for field, name in (("log_id", "log_id"), ("user_id", "user_id"), ("pt_d", "date"))}
    hashes = {field: hashlib.sha256() for field in cache}
    multiscale_hashes = {field: hashlib.sha256() for field in cache}
    offset, sections, days = 0, [], {}
    for filename, expected in (("train.csv", NTRAIN), ("final_test.csv", NFINAL)):
        source = ROOT / "2026ml" / filename
        stat = source.stat()
        begin = offset
        seen_days = set()
        for frame in pd.read_csv(source, usecols=list(cache), dtype="int64", chunksize=chunksize):
            end = offset + len(frame)
            if end > NROWS:
                raise AssertionError("Too many original input rows")
            for field, mapped in cache.items():
                if mapped.shape != (NROWS,):
                    raise AssertionError("Identity cache length mismatch")
                values = frame[field].to_numpy(dtype="<i8")
                np.testing.assert_array_equal(values, mapped[offset:end])
                hashes[field].update(np.ascontiguousarray(values).tobytes())
                multiscale_hashes[field].update(np.asarray(values, dtype="<i4").tobytes())
            unique = np.unique(frame.pt_d)
            seen_days.update(int(value) for value in unique)
            if filename == "train.csv" and np.any(unique >= FINAL_DAY):
                raise AssertionError("Training dates reach final day")
            if filename == "final_test.csv" and not np.all(unique == FINAL_DAY):
                raise AssertionError("Final inputs must all have date 20191117")
            offset = end
        if offset - begin != expected or source.stat().st_size != stat.st_size or source.stat().st_mtime_ns != stat.st_mtime_ns:
            raise AssertionError("Input identity count or source changed")
        sections.append({"path": str(source), "start": begin, "stop": offset,
                         "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        days[filename] = sorted(seen_days)
    if offset != NROWS:
        raise AssertionError("Original combined row count differs")
    final_ids = np.load(OOF / "final_log_id.npy", mmap_mode="r")
    np.testing.assert_array_equal(final_ids, cache["log_id"][NTRAIN:])
    np.testing.assert_array_equal(np.load(OOF / "final_indices.npy"), np.arange(NTRAIN, NROWS))
    for mapped in list(cache.values()) + [final_ids]:
        mapped._mmap.close()
    values = {field: value.hexdigest() for field, value in hashes.items()}
    alignment = read(MULTISCALE / "input_alignment.json")
    multiscale_values = {field: value.hexdigest() for field, value in multiscale_hashes.items()}
    if multiscale_values != alignment["order_sha256"] or alignment["all_log_id_user_id_date_equal_existing_full_cache"] is not True:
        raise AssertionError("Multiscale int32 identity provenance differs from original rows")
    emit("identity_verified", rows=offset, dates=days, row_identity_sha256=values)
    return values, sections, days


def ensure_raw(name, script, width, additional):
    output = OUT / (name + "_raw.npy")
    metadata = output.with_suffix(".json")
    if output.exists() != metadata.exists():
        raise FileExistsError("Incomplete existing raw output: " + str(output))
    if not output.exists():
        command = [sys.executable, "-u", str(script), "--include-final", "--output", str(output)] + additional
        emit("raw_builder_start", name=name, command=command)
        with (OUT / (name + "_prepare.log")).open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        emit("raw_builder_complete", name=name)
    meta = read(metadata)
    if meta.get("include_final") is not True or meta.get("advertisement_labels_loaded") is not False:
        raise AssertionError("Raw feature final/label boundary differs: " + str(output))
    if meta["shape"] != [NROWS, width]:
        raise AssertionError("Raw feature shape differs")
    return Matrix(output, width), meta


def prefix_probes(matrix, old_path, names, seed):
    old = np.load(old_path, mmap_mode="r")
    old_meta = read(old_path.with_suffix(".json"))
    if old.shape != (NTRAIN, len(names)) or old_meta["features"] != names:
        raise AssertionError("Development raw feature schema differs")
    rng = np.random.default_rng(seed)
    rows = np.unique(np.r_[0, 1, NTRAIN // 2, NTRAIN - 1, rng.integers(0, NTRAIN, 1024)])
    mapped = np.load(matrix.path, mmap_mode="r")
    np.testing.assert_array_equal(mapped[rows], old[rows])
    mapped._mmap.close()
    old._mmap.close()
    emit("raw_prefix_probes_verified", source=str(matrix.path), probes=len(rows), columns=len(names))
    return {"reference": str(old_path), "training_probes": len(rows), "seed": seed,
            "all_columns_exact": True}


def fit_scalers(matrix, columns, names):
    means, stds = np.empty(len(names), dtype="float64"), np.empty(len(names), dtype="float64")
    for local, column in enumerate(columns):
        sample = matrix.column(column)
        if not np.isfinite(sample).all():
            raise AssertionError("Nonfinite scaler fit source")
        means[local] = sample.mean()
        stds[local] = max(float(sample.std()), 1e-5)
    scales = {name: {"mean": float(mean), "std": float(std)}
              for name, mean, std in zip(names, means, stds)}
    emit("scalers_fit", source=str(matrix.path), columns=len(names), train_rows=NTRAIN)
    return means, stds, scales


def create_output(path):
    with path.open("xb") as stream:
        np.lib.format.write_array_header_2_0(stream, {"descr": "<f4", "fortran_order": False,
                                                     "shape": (NROWS, 500)})
        offset = stream.tell()
        stream.truncate(offset + NROWS * 500 * 4)
    return offset


def put_rows(path, offset, start, values):
    mapped = np.memmap(path, mode="r+", dtype="<f4", shape=values.shape,
                       offset=offset + start * 500 * 4, order="C")
    try:
        mapped[:] = values
        mapped.flush()
    finally:
        mapped._mmap.close()


def prepare(chunksize):
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Preserving completed architecture cache; use --verify-only")
    finals = {mode: OUT / ("numeric_" + mode + "500.npy") for mode in ("oof", "past")}
    staging = {mode: OUT / ("numeric_" + mode + "500.partial.npy") for mode in finals}
    for path in list(finals.values()) + list(staging.values()):
        if path.exists():
            raise FileExistsError("Preserving previous/incomplete numeric output: " + str(path))
    free = shutil.disk_usage(OUT).free
    required = NROWS * (1000 + 175) * 4 + 4 * 1024**3
    if free < required:
        raise OSError("Insufficient free disk space for both numeric maps, raw blocks, scratch")
    emit("preparation_start", cpu_threads=2, cuda_visible_devices="", free_bytes=free)
    names = read(HERE / "cache/day16_oof/manifest.json")["numeric_fields"]
    past_meta, oof_meta = read(CACHE / "manifest.json"), read(OOF / "manifest.json")
    if len(names) != 500 or len(set(names)) != 500:
        raise AssertionError("Architecture requires 500 unique numeric fields")
    for meta in (past_meta, oof_meta):
        if meta["numeric_fields"] != names[:166] or meta["rows"] != NROWS or meta["train_rows"] != NTRAIN or meta["final_rows"] != NFINAL:
            raise AssertionError("Full-cache numeric schema/row boundaries differ")
    if past_meta.get("final_labels_available") is not False or oof_meta.get("target_stat_final_labels_used") is not False:
        raise AssertionError("Full-cache final target boundary differs")
    identities, sections, days = identity_check(chunksize)
    history, history_meta = ensure_raw("history", PREVIOUS / "prepare_history.py", 95,
                                       ["--chunksize", "25000"])
    source, source_meta = ensure_raw("source", NEXT / "prepare_source_context.py", 80,
                                     ["--buckets", "64", "--chunksize", "25000", "--memory-budget-mib", "1536"])
    if history_meta["row_identity_sha256"] != identities:
        raise AssertionError("History raw row identity mismatch")
    expected_sections = [(0, NTRAIN), (NTRAIN, NROWS)]
    if [(part["start"], part["stop"]) for part in source_meta["advertisement_row_sections"]] != expected_sections:
        raise AssertionError("Source context input row sections differ")
    blocks = [
        (Matrix(CACHE / "exposure_raw.npy", 43), read(DAILY / "exposure_features.json")["features"], 75),
        (Matrix(CACHE / "snapshot_raw.npy", 48), read(DAILY / "snapshot_features.json")["features"], 118),
        (Matrix(MULTISCALE / "multiscale_raw.npy", 159), read(MULTISCALE / "multiscale_raw.json")["features"], 166),
        (history, history_meta["features"], 325),
        (source, source_meta["features"], 420),
    ]
    multiscale_meta = read(MULTISCALE / "multiscale_raw.json")
    if multiscale_meta.get("final_labels_read") is not False or multiscale_meta.get("advertisement_validation_labels_used") is not False:
        raise AssertionError("Multiscale final target boundary differs")
    probe_results = {}
    for matrix, fields, offset in blocks:
        if fields != names[offset:offset + len(fields)] or matrix.shape[1] != len(fields):
            raise AssertionError("Raw block order differs from architecture")
    for matrix, fields, reference, seed in (
        (history, history_meta["features"], PREVIOUS / "history_metadata.npy", 20261029),
        (source, source_meta["features"], NEXT / "source_context.npy", 20261030),
        (blocks[0][0], blocks[0][1], DAILY / "exposure_features.npy", 20261031),
        (blocks[1][0], blocks[1][1], DAILY / "snapshot_features.npy", 20261101),
        (blocks[2][0], blocks[2][1], NEXT / "multiscale_day16.npy", 20261102)):
        probe_results[matrix.path.stem] = prefix_probes(matrix, reference, fields, seed)
    bases = {"oof": Matrix(OOF / "numeric.npy", 166), "past": Matrix(CACHE / "numeric.npy", 166)}
    base_stats = {mode: fit_scalers(matrix, range(75), names[:75]) for mode, matrix in bases.items()}
    common_stats = [fit_scalers(matrix, range(len(fields)), fields) for matrix, fields, _ in blocks]
    scalers = {}
    for mode in bases:
        scalers[mode] = dict(base_stats[mode][2])
        for _, _, scales in common_stats:
            scalers[mode].update(scales)
        if list(scalers[mode]) != names:
            raise AssertionError("Scaler schema differs from numeric order")
    headers = {mode: create_output(path) for mode, path in staging.items()}
    progress = {"status": "writing", "rows": NROWS, "train_rows": NTRAIN, "final_rows": NFINAL,
                "numeric_fields": names, "numeric_paths": {k: str(v) for k, v in finals.items()},
                "source_scalers_oof": scalers["oof"], "source_scalers_past": scalers["past"],
                "final_labels_read": False}
    write(OUT / "progress.json", progress)
    last_log = 0.
    for start in range(0, NROWS, chunksize):
        stop = min(start + chunksize, NROWS)
        common = np.empty((stop - start, 500), dtype="float32")
        for (matrix, fields, offset), (means, stds, _) in zip(blocks, common_stats):
            raw = matrix.rows(start, stop)
            common[:, offset:offset + len(fields)] = np.clip((raw - means) / stds, -8, 8)
        first39 = None
        for mode, matrix in bases.items():
            raw = matrix.rows(start, stop, 0, 75)
            if first39 is None:
                first39 = raw[:, :39].copy()
            else:
                np.testing.assert_array_equal(first39, raw[:, :39])
            means, stds, _ = base_stats[mode]
            common[:, :75] = np.clip((raw - means) / stds, -8, 8)
            if not np.isfinite(common).all() or np.any(np.abs(common) > 8):
                raise AssertionError("Nonfinite/out-of-clip numeric output")
            put_rows(staging[mode], headers[mode], start, common)
        now = time.perf_counter()
        if now - last_log >= 10 or stop == NROWS:
            emit("numeric_write", rows=stop, total=NROWS, seconds=round(now - started, 1))
            progress["written_rows"] = stop
            write(OUT / "progress.json", progress)
            last_log = now
    del common, raw, first39
    gc.collect()
    rng = np.random.default_rng(20261029)
    probes = np.unique(np.r_[0, 1, NTRAIN - 1, NTRAIN, NROWS - 1, rng.integers(0, NROWS, 512)])
    verification = {}
    for mode, path in staging.items():
        output = Matrix(path, 500)
        sample = np.vstack([output.rows(int(row), int(row) + 1) for row in probes])
        base = np.vstack([bases[mode].rows(int(row), int(row) + 1, 0, 75) for row in probes])
        means, stds, _ = base_stats[mode]
        np.testing.assert_array_equal(sample[:, :75], np.clip((base - means) / stds, -8, 8).astype("float32"))
        for (matrix, fields, offset), (means, stds, _) in zip(blocks, common_stats):
            raw = np.vstack([matrix.rows(int(row), int(row) + 1) for row in probes])
            np.testing.assert_array_equal(sample[:, offset:offset + len(fields)],
                                          np.clip((raw - means) / stds, -8, 8).astype("float32"))
        minimum, maximum = float("inf"), -float("inf")
        for start in range(0, NROWS, chunksize):
            block = output.rows(start, min(start + chunksize, NROWS))
            if not np.isfinite(block).all() or np.any(np.abs(block) > 8):
                raise AssertionError("Saved numeric map failed finite/clip check")
            minimum, maximum = min(minimum, float(block.min())), max(maximum, float(block.max()))
        sha = digest(path)
        verification[mode] = {"sha256": sha, "hash_rule": "SHA256 of complete NPY bytes including header",
                              "finite_all_rows": True, "minimum": minimum, "maximum": maximum,
                              "source_mapping_probes": len(probes), "all_500_probe_columns_exact": True}
        emit("numeric_verified", mode=mode, **verification[mode])
    for mode in staging:
        staging[mode].replace(finals[mode])
    alignment_checks = {"all_original_log_id_user_id_date_equal_full_cache": True,
                        "all_log_id_user_id_date_equal_existing_full_cache": True,
                        "original_row_count_verified": True, "oof_final_ids_and_indices_exact": True,
                        "history_identity_sha256_equal": True, "multiscale_identity_sha256_equal": True,
                        "source_original_row_sections_equal": True, "raw_training_prefix_probes": probe_results,
                        "oof_past_first39_equal_all_rows": True, "final_dates_equal_20191117": True}
    provenance = {"base_oof": str(bases["oof"].path), "base_past": str(bases["past"].path),
                  "base_columns": [0, 75], "base_already_standardized_rescaled_again": True,
                  "raw_blocks": [{"path": str(matrix.path), "columns": len(fields), "offset": offset}
                                 for matrix, fields, offset in blocks],
                  "oof_target_statistics": "Existing duplicate-group five-fold OOF on all training dates; final encoded from all training labels",
                  "past_target_statistics": "Existing full cache per-row strictly earlier dates; final uses only training dates",
                  "architecture_prepare_numeric": str(HERE / "train_architecture.py"),
                  "architecture_prepare_numeric_sha256": digest(HERE / "train_architecture.py"),
                  "builder_sha256": digest(Path(__file__)),
                  "source_metadata_sha256": {str(path): digest(path) for path in (
                      CACHE / "manifest.json", OOF / "manifest.json", MULTISCALE / "multiscale_raw.json",
                      history.path.with_suffix(".json"), source.path.with_suffix(".json"))}}
    info = {"status": "complete_and_verified", "rows": NROWS, "train_rows": NTRAIN, "final_rows": NFINAL,
            "shape": [NROWS, 500], "dtype": "float32", "storage_order": "C", "numeric_fields": names,
            "numeric_oof_path": str(finals["oof"]), "numeric_past_path": str(finals["past"]),
            "numeric_paths": {mode: str(path) for mode, path in finals.items()},
            "source_scalers_oof": scalers["oof"], "source_scalers_past": scalers["past"],
            "scalers": scalers, "clip": [-8, 8], "std_floor": 1e-5, "scaler_rule": SCALER_RULE,
            "scaler_fit_scope": "Original train.csv rows [0,2526320), never final rows",
            "scaler_fit_rows": NTRAIN, "scaler_fit_indices": [0, NTRAIN],
            "final_labels_read": False, "final_labels_used": False, "gpu_used": False, "cpu_threads": 2,
            "row_order": "Original train.csv rows followed by original final_test.csv rows, no sorting/deduplication",
            "row_identity_sha256": identities, "identity_hash_rule": IDENTITY_RULE,
            "input_sections": sections, "dates": days, "alignment_checks": alignment_checks,
            "per_row_strictly_historical": {"past": True, "oof": False},
            "availability": "Same-day complete auxiliary news snapshot and unlabeled exposures, offline; history metadata strictly prior dates",
            "input_provenance": provenance, "numeric_sha256": {mode: result["sha256"] for mode, result in verification.items()},
            "verification": verification, "free_disk_bytes_at_start": free,
            "output_bytes": sum(path.stat().st_size for path in finals.values()),
            "seconds": round(time.perf_counter() - started, 3)}
    write(OUT / "manifest.json", info)
    progress.update(status=info["status"], manifest_path=str(OUT / "manifest.json"))
    write(OUT / "progress.json", progress)
    emit("complete_and_verified", manifest=str(OUT / "manifest.json"), seconds=info["seconds"],
         numeric_paths=info["numeric_paths"], output_bytes=info["output_bytes"])
    return info


def verify_only(chunksize):
    info = read(OUT / "manifest.json")
    if info["status"] != "complete_and_verified" or info["final_labels_read"] is not False:
        raise AssertionError("Existing manifest is not verified")
    if info["numeric_fields"] != read(HERE / "cache/day16_oof/manifest.json")["numeric_fields"]:
        raise AssertionError("Feature order changed")
    for mode, path in info["numeric_paths"].items():
        matrix = Matrix(path, 500)
        for start in range(0, NROWS, chunksize):
            values = matrix.rows(start, min(start + chunksize, NROWS))
            if not np.isfinite(values).all() or np.any(np.abs(values) > 8):
                raise AssertionError("Numeric finite/clip recheck failed")
        if digest(path) != info["numeric_sha256"][mode]:
            raise AssertionError("Completed NPY bytes changed")
        emit("verified_existing", mode=mode, path=path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunksize", type=int, default=16384)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.chunksize <= 32768:
        parser.error("chunksize must be in 1..32768")
    if args.verify_only:
        verify_only(args.chunksize)
    else:
        prepare(args.chunksize)
