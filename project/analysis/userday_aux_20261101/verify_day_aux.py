"""Independent input aggregation, target isolation, and fresh checkpoint replay."""
from pathlib import Path
import sys
import hashlib
import gc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = ROOT / "analysis/learned_audience_20261101"
sys.path.insert(0, str(ROOT / "analysis/strong_interest_control_20261031"))
from features import prepare, read, write, digest, emit, ARCH
from experiment import configure, scores, state_digest
from evaluate import paired_interval
sys.path.insert(0, str(REFERENCE))
from freeze_reference import freeze, V7_HASH
from day_context import HISTOGRAM_FIELDS, HISTORY_FIELDS
from day_model import UserDayCTR
from run_day_aux import predict
import numpy as np
import pandas as pd
import torch


def independent_history(raw):
    result = np.zeros((len(raw), 5), dtype="float32")
    ordinal = pd.to_datetime(raw.pt_d.astype(str), format="%Y%m%d").to_numpy(dtype="datetime64[D]").astype("int64")
    for day in sorted(raw.pt_d.unique()):
        positions = np.flatnonzero(raw.pt_d.to_numpy() == day)
        current = ordinal[positions[0]]
        past = raw.loc[(ordinal < current) & (ordinal >= current - 3), ["pt_d", "user_id", "task_id"]]
        query = raw.iloc[positions]
        if not len(past):
            continue
        pair = past.groupby(["user_id", "task_id"]).size().reindex(
            pd.MultiIndex.from_frame(query[["user_id", "task_id"]])).fillna(0).to_numpy()
        user = past.groupby("user_id").size().reindex(query.user_id).fillna(0).to_numpy()
        task = past.groupby("task_id").size().reindex(query.task_id).fillna(0).to_numpy()
        coverage = max(int(past.pt_d.nunique()), 1)
        result[positions] = np.column_stack([
            np.log1p(pair), pair > 0, task > 0, np.log1p(user / coverage), user > 0])
    return result


def verify_context(root, raw, data, day):
    predictors = raw.iloc[data["global_rows"]][["pt_d", "user_id", "task_id"]].reset_index(drop=True)
    keys = pd.MultiIndex.from_frame(predictors[["pt_d", "user_id"]]).unique().sort_values()
    groups = keys.get_indexer(pd.MultiIndex.from_frame(predictors[["pt_d", "user_id"]]))
    sizes = np.bincount(groups, minlength=len(keys))
    manifest = read(root / "context_manifest.json")
    if hashlib.sha256(groups.astype("int32").tobytes()).hexdigest() != manifest["query_groups_sha256"]:
        raise AssertionError("User-day query mapping differs")
    fields = data["feature_manifest"]["numeric_fields"]
    allowed = [index for index, name in enumerate(fields) if not name.startswith("target_stat__")]
    excluded = [name for name in fields if name.startswith("target_stat__")]
    if len(excluded) != 36 or excluded != manifest["excluded_target_stat_columns"]:
        raise AssertionError("Target statistics entered pooled inputs")
    means, maxima = [], []
    for start in range(0, len(allowed), 16):
        columns = allowed[start:start + 16]
        block = pd.DataFrame(data["numeric"][:, columns].astype("float64"))
        block["group"] = groups
        grouped = block.groupby("group", sort=True)[list(range(len(columns)))]
        means.append(grouped.mean().to_numpy(dtype="float32"))
        maxima.append(grouped.max().to_numpy(dtype="float32"))
    pieces = [np.column_stack(means), np.column_stack(maxima), np.log1p(sizes).astype("float32")[:, None]]
    for field in HISTOGRAM_FIELDS:
        column = data["meta"]["categorical_fields"].index(field)
        width = data["meta"]["vocabulary_sizes"][data["meta"]["group_for_field"][field]]
        histogram = pd.crosstab(pd.Series(groups), pd.Series(data["categories"][:, column])).reindex(
            index=np.arange(len(keys)), columns=np.arange(width), fill_value=0).to_numpy()
        pieces.append((histogram / sizes[:, None]).astype("float32"))
    history = pd.DataFrame(independent_history(predictors))
    history["group"] = groups
    grouped = history.groupby("group", sort=True)[list(range(len(HISTORY_FIELDS)))]
    history_mean = grouped.mean().to_numpy(dtype="float32")
    pieces += [history_mean, grouped.max().to_numpy(dtype="float32")]
    pieces.append((np.log1p(sizes) - history_mean[:, 3]).astype("float32")[:, None])
    features = np.column_stack(pieces).astype("float32")
    dates = np.array([value[0] for value in keys])
    train_groups = np.flatnonzero(dates < day)
    center = features[train_groups].mean(0, dtype="float64")
    scale = features[train_groups].std(0, dtype="float64").clip(.05)
    np.testing.assert_allclose(center, manifest["scaler_mean"], rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(scale, manifest["scaler_std"], rtol=3e-6, atol=3e-6)
    features = np.clip((features - center) / scale, -8, 8).astype("float32")
    with np.load(root / "context.npz") as saved:
        if set(saved.files) != {"keys", "sizes", "values"}:
            raise AssertionError("Unexpected context asset fields")
        np.testing.assert_array_equal(saved["keys"], np.asarray(keys.tolist(), dtype="int64"))
        np.testing.assert_array_equal(saved["sizes"], sizes)
        error = float(np.max(np.abs(saved["values"] - features)))
        np.testing.assert_allclose(saved["values"], features, rtol=1e-5, atol=1e-5)
        contexts = saved["values"].copy()
        if hashlib.sha256(contexts.tobytes()).hexdigest() != manifest["context_values_sha256"]:
            raise AssertionError("Context asset changed")
    targets = np.full(len(keys), -1, dtype="int8")
    counts = np.bincount(groups[data["train"]], weights=data["label"][data["train"]], minlength=len(keys))
    targets[train_groups] = (counts[train_groups] > 0).astype("int8")
    np.testing.assert_array_equal(np.load(root / "training_any_targets.npy"), targets)
    auxiliary = read(root / "auxiliary_manifest.json")
    if abs(float(targets[train_groups].mean()) - auxiliary["prior"]) > 1e-12:
        raise AssertionError("Training-only auxiliary prior differs")
    if not np.all(targets[groups[data["valid"]]] == -1):
        raise AssertionError("Validation labels entered auxiliary training targets")
    return groups.astype("int32"), contexts, {
        "status": "passed", "groups": len(keys), "width": contexts.shape[1],
        "max_independent_feature_error": error, "excluded_target_columns": 36,
        "validation_auxiliary_targets_missing": True}


def run():
    freeze()
    raw = pd.read_csv(ROOT / "2026ml/train.csv",
                      usecols=["pt_d", "user_id", "task_id", "label"], dtype="int32")
    records, differences, checks, auxiliary_diagnostics = [], [], [], []
    verified = 0
    unique_inference_replays = 0
    for day in [20191115, 20191116]:
        valid = np.flatnonzero(raw.pt_d.to_numpy() == day)
        y, users = raw.label.to_numpy()[valid], raw.user_id.to_numpy()[valid]
        reference = np.load(REFERENCE / f"reference/day{day % 100}_prediction.npy")
        records.append({"day": day, "model": "frozen_candidate", **scores(y, reference)})
        v7 = np.load(ARCH / f"fixed_gated20_allocated10_core30_day{day % 100}.npy").astype("float64")
        old_core = np.load(ARCH / f"results/gated166_day{day % 100}/validation_prediction.npy")
        day_data = None
        inference_cache = {}
        for root in sorted((HERE / "runs").glob(f"*_day{day % 100}_seed*")):
            if not (root / "comparison.json").exists():
                raise RuntimeError("Training must finish before full independent context replay")
            protocol = read(root / "protocol.json")
            for path, expected in protocol["sources"].items():
                if digest(path) != expected:
                    raise AssertionError("User-day training source changed")
            feature = read(root / "feature_manifest.json")
            for name, expected in feature["input_sha256"].items():
                if digest(ROOT / "2026ml" / name) != expected:
                    raise AssertionError("Original data changed")
            for path, expected in feature["metadata_sha256"].items():
                if digest(path) != expected:
                    raise AssertionError("Original preprocessing source changed")
            configure(protocol["seed"])
            if day_data is None:
                day_data = prepare(day, HERE / "verification_inputs" / str(day))
            data = day_data
            np.testing.assert_array_equal(data["global_rows"], np.flatnonzero(raw.pt_d.to_numpy() <= day))
            np.testing.assert_array_equal(data["label"], raw.label.to_numpy()[data["global_rows"]])
            groups, contexts, context_check = verify_context(root, raw, data, day)
            checks.append({"run": root.name, **context_check})
            data["day_rows"] = groups
            context_tensor = torch.as_tensor(contexts, device="cuda")
            predictions, hashes = {}, []
            for mode in protocol["modes"]:
                directory = root / mode
                metric = read(directory / "metrics.json")
                if metric["status"] != "complete_and_verified" or max(metric["reload_errors"].values()) > 2e-7:
                    raise AssertionError("User-day arm not verified")
                np.testing.assert_array_equal(np.load(directory / "valid_indices.npy"), valid)
                for name, expected in metric["files_sha256"].items():
                    if digest(directory / name) != expected:
                        raise AssertionError("User-day output changed")
                hashes.append(tuple(metric[name] for name in [
                    "initial_backbone_sha256", "initial_full_weights_sha256", "row_order_sha256"]))
                for endpoint in ["final", "best"]:
                    saved = np.load(directory / f"{endpoint}_prediction.npy")
                    actual = scores(y, saved)
                    expected = metric["fixed_endpoint" if endpoint == "final" else "best_checkpoint"]
                    for name, value in actual.items():
                        if abs(value - expected[name]) > 1e-12:
                            raise AssertionError("Raw-label score differs")
                    checkpoint = torch.load(directory / f"{endpoint}_model.pt", map_location="cpu", weights_only=False)
                    if checkpoint["context_manifest_sha256"] != digest(root / "context_manifest.json"):
                        raise AssertionError("Checkpoint context identity differs")
                    if checkpoint["meta"] != data["meta"]:
                        raise AssertionError("Checkpoint feature contract differs")
                    context_manifest = read(root / "context_manifest.json")
                    replay_key = (state_digest(checkpoint["state_dict"]), mode in ["context", "both"],
                                  context_manifest["context_values_sha256"], context_manifest["query_groups_sha256"])
                    # Equal states and inputs define the same eval function, including protected training routes.
                    if replay_key not in inference_cache:
                        model = UserDayCTR(checkpoint["meta"], .0084, context_tensor, checkpoint["any_prior"],
                                           use_context=mode in ["context", "both"]).cuda()
                        model.load_state_dict(checkpoint["state_dict"], strict=True)
                        inference_cache[replay_key] = predict(model, data, data["valid"])
                        unique_inference_replays += 1
                        del model
                    replay, any_replay = inference_cache[replay_key]
                    replay_error = float(np.max(np.abs(replay - saved)))
                    aux_saved = np.load(directory / f"{endpoint}_any_prediction.npy")
                    aux_error = float(np.max(np.abs(any_replay - aux_saved)))
                    if max(replay_error, aux_error) > 2e-7:
                        raise AssertionError("Independent user-day checkpoint replay differs")
                    predictions[(mode, endpoint)] = saved
                    descriptor = root.name.replace(f"_day{day % 100}", "") + "/" + mode + "/" + endpoint
                    records.append({"day": day, "model": descriptor, **actual,
                                    "independent_replay_error": replay_error, "aux_replay_error": aux_error})
                    work_prediction = saved.astype("float64")
                    replacement = v7 + .3 * (work_prediction - old_core)
                    for suffix, blend in [("v7_replace30", replacement),
                                           ("candidate_blend10", .9 * reference + .1 * work_prediction),
                                           ("candidate_equal_replacement", .5 * reference + .5 * replacement)]:
                        records.append({"day": day, "model": descriptor + "/" + suffix, **scores(y, blend)})
                        differences.append({"day": day, "run": root.name, "mode": mode, "endpoint": endpoint,
                                            "comparison": suffix + "_minus_candidate",
                                            **paired_interval(y, blend, reference, users)})
                    base_prediction = predictions.get(("base", endpoint))
                    if mode != "base" and base_prediction is not None:
                        ratio = float(work_prediction.mean())
                        ratio /= max(float(base_prediction.astype("float64").mean()), 1e-12)
                        scaled = np.clip(base_prediction.astype("float64") * ratio, 0., 1.)
                        for suffix, control, actual_blend in [
                                ("mean_control_v7_replace30", v7 + .3 * (scaled - old_core),
                                 replacement),
                                ("mean_control_candidate_blend10", .9 * reference + .1 * scaled,
                                 .9 * reference + .1 * work_prediction),
                                ("mean_control_equal_replacement", .5 * reference + .5 * (v7 + .3 * (scaled - old_core)),
                                 .5 * reference + .5 * replacement)]:
                            records.append({"day": day, "model": descriptor + "/" + suffix,
                                            "unlabeled_mean_scale": ratio, **scores(y, control)})
                            differences.append({"day": day, "run": root.name, "mode": mode, "endpoint": endpoint,
                                                "comparison": "new_minus_" + suffix,
                                                **paired_interval(y, actual_blend, control, users)})
                    day_keys, inverse = np.unique(users, return_inverse=True)
                    group_count = np.bincount(inverse)
                    truth = np.bincount(inverse, weights=y) > 0
                    auxiliary = np.bincount(inverse, weights=aux_saved) / group_count
                    auxiliary_diagnostics.append({"day": day, "mode": mode, "endpoint": endpoint,
                                                  "groups": len(day_keys), "not_goal_metric": True,
                                                  **scores(truth.astype("int8"), auxiliary)})
                    del checkpoint
                    torch.cuda.empty_cache()
                verified += 1
            if len(set(hashes)) != 1:
                raise AssertionError("User-day matched control differs")
            if protocol.get("auxiliary_gradient_policy"):
                for endpoint in ["final", "best"]:
                    if ("aux", endpoint) in predictions and ("base", endpoint) in predictions:
                        error = float(np.max(np.abs(predictions[("aux", endpoint)] - predictions[("base", endpoint)])))
                        if error > 2e-7:
                            raise AssertionError("Protected auxiliary-only arm changed the backbone prediction")
                        checks.append({"run": root.name, "protected_auxiliary_endpoint": endpoint,
                                       "maximum_prediction_error": error})
            for endpoint in ["final", "best"]:
                for first, second in [("context", "base"), ("aux", "base"),
                                      ("both", "context"), ("both", "aux")]:
                    if (first, endpoint) in predictions and (second, endpoint) in predictions:
                        differences.append({"day": day, "run": root.name, "mode": first, "endpoint": endpoint,
                                            "comparison": first + "_minus_" + second,
                                            **paired_interval(y, predictions[(first, endpoint)], predictions[(second, endpoint)], users)})
                previous = ROOT / f"analysis/strong_interest_control_20261031/runs/day{day % 100}_seed{protocol['seed']}/control/{endpoint}_prediction.npy"
                if ("base", endpoint) in predictions and previous.exists():
                    error = float(np.max(np.abs(predictions[("base", endpoint)] - np.load(previous))))
                    if error > 2e-7:
                        raise AssertionError("Original one-pass strong baseline did not reproduce")
                    checks.append({"day": day, "baseline_endpoint": endpoint, "maximum_error": error})
            del context_tensor, contexts
            gc.collect()
            torch.cuda.empty_cache()
        if day_data is not None:
            del data, day_data
            gc.collect()
            torch.cuda.empty_cache()
    if digest(ROOT / "output/best_nextday_arch_v7/submission.csv") != V7_HASH:
        raise AssertionError("Original v7 submission changed")
    frame = pd.DataFrame(records)
    frame.to_csv(HERE / "comparison.csv", index=False)
    pd.DataFrame(differences).to_csv(HERE / "paired_changes.csv", index=False)
    write(HERE / "auxiliary_diagnostics.json", auxiliary_diagnostics)
    common = frame.pivot(index="model", columns="day", values="auc").dropna().min(axis=1).sort_values(ascending=False)
    result = {"status": "passed", "verified_model_count": verified, "independent_context_checks": checks,
              "confirmation_policy_sha256": digest(HERE / "confirmation_protocol.json"),
              "protected_confirmation_policy_sha256": digest(HERE / "protected_confirmation_protocol.json"),
              "mean_control_policy_sha256": digest(HERE / "mean_control_protocol.json"),
              "equal_ensemble_policy_sha256": digest(HERE / "equal_ensemble_protocol.json"),
              "unique_inference_states_replayed": unique_inference_replays,
              "goal_auc_both_days": .85, "goal_met": bool((common >= .85).any()),
              "best_common_model": str(common.index[0]), "best_common_min_auc": float(common.iloc[0]),
              "highest_each_day": {str(day): float(group.auc.max()) for day, group in frame.groupby("day")},
              "goal_uses_advertisement_row_auc_only": True, "final_test_auc": None,
              "evaluator_sha256": digest(Path(__file__)),
              "limitations": "Repeated development dates; checkpoint selection and paired intervals are exploratory"}
    write(HERE / "verification.json", result)
    print(frame.sort_values(["day", "auc"], ascending=[True, False]).groupby("day").head(8)[
        ["day", "model", "auc"]].to_string(index=False), flush=True)
    emit("userday_verified", **result)


if __name__ == "__main__":
    run()
