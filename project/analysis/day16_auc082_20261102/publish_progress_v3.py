"""Publish only independently verified real16 predictions; preserve raw inputs."""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from fit_stacker import HERE, ROOT, REFERENCE, digest, write


@contextmanager
def publication_lock():
    """Serialize publishers from the independent CPU/GPU completion queues."""
    import msvcrt
    path = HERE / "runs" / "publication.lock"
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        started = time.monotonic()
        while True:
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic()-started > 180:
                    raise TimeoutError("Another publication did not finish within180seconds")
                time.sleep(.25)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def auc(y, p):
    positives = int(y.sum())
    return float((rankdata(p,method="average")[y == 1].sum()-positives*(positives+1)/2)
                 /(positives*(len(y)-positives)))


def main():
    runs = HERE / "runs"
    candidates = []
    definitions = [
        ("score_stacker_v1", "independent_verification.json", "regularized_stack_prediction.npy", "stack"),
        ("group_score_calibration_v1", "verification.json", "prediction.npy", "plain"),
        ("group_tree_v2", "verification.json", "prediction.npy", "plain"),
        ("weighted_v1_day16", "verification.json", "best_verified_prediction.npy", "weighted"),
        ("duplicate_mean_v1_day16", "verification.json", "best_verified_prediction.npy", "weighted"),
        ("weighted_calibration_transfer_v1", "verification.json", "best_prediction.npy", "plain"),
        ("duplicate_calibration_transfer_v1", "verification.json", "best_prediction.npy", "plain"),
        ("rank_calibration_v1", "verification.json", "prediction.npy", "plain"),
        ("feature_residual_tree_v1", "verification.json", "prediction.npy", "plain"),
        ("recent_finetune_v1_day16", "verification.json", "prediction.npy", "plain"),
        ("wide_score_calibration_v1", "verification.json", "prediction.npy", "plain"),
        ("wide_score_candidates_v1", "verification.json", "prediction.npy", "plain"),
        ("wide_rank_calibration_v1", "verification.json", "prediction.npy", "plain"),
        ("rich_userday_activity_v1", "verification.json", "prediction.npy", "plain"),
        ("rich_group_residual_v1", "verification.json", "prediction.npy", "plain"),
        ("pairwise_group_boost_v1", "verification.json", "prediction.npy", "plain"),
        ("subgroup_residual_v1", "verification.json", "prediction.npy", "plain"),
        ("extended_history_residual_v1", "verification.json", "prediction.npy", "plain"),
        ("group_meta_calibration_v1", "verification.json", "prediction.npy", "plain"),
        ("catboost_userday_residual_v1", "verification.json", "prediction.npy", "plain"),
        ("hurdle_userday_v1", "verification.json", "prediction.npy", "plain"),
        ("personal_context_residual_v1", "verification.json", "prediction.npy", "plain"),
        ("feed_behavior_userday_v1", "verification.json", "prediction.npy", "plain"),
        ("neural_userday_v1", "verification.json", "prediction.npy", "plain"),
        ("context_rank_boost_v1", "verification.json", "prediction.npy", "plain"),
        ("catboost_daily_history_v1", "verification.json", "prediction.npy", "plain"),
        ("topic_context_residual_v1", "verification.json", "prediction.npy", "plain"),
        ("context_scale_rank_v1", "verification.json", "prediction.npy", "plain"),
        ("previous_day_residual_v1", "verification.json", "prediction.npy", "plain"),
        ("row_context_residual_v1", "verification.json", "prediction.npy", "plain"),
        ("rank_finetune_v1_day16", "verification.json", "prediction.npy", "plain"),
        ("global_row_context_v1", "verification.json", "prediction.npy", "plain"),
        ("rank_matched_v1_day16", "verification.json", "prediction.npy", "plain"),
        ("compose_verified_v1", "verification.json", "prediction.npy", "plain"),
        ("daily_pair_rank_v1", "verification.json", "prediction.npy", "plain"),
        ("recurrent_userday_v1", "verification.json", "prediction.npy", "plain"),
        ("catboost_global_rows_v1", "verification.json", "prediction.npy", "plain"),
        ("rolling_previous_app_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("rolling_rich_daily_catboost_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("rolling_rich_count_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("rolling_news_latent_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("conditional_position_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("compose_rich_response_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("augmented_rows_v1_day16", "verification.json", "prediction.npy", "plain"),
        ("bayesian_daily_offset_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("news_quality_daily_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("rich_augmented_rows_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("dense_context_rows_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("group_temperature_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("conditional_daily_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("conditional_daily_trees_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("group_temperature_joint_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("rich_row_catboost_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("full_quality_row_catboost_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("catboost_group_temperature_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("grouped_catboost_residual_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("quality_daily_catboost_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("compose_full_quality_transfer_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("compose_quality_daily_transfer_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("compact_quality_daily_catboost_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("quality_daily_neural_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("compose_dense_context_transfer_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("compose_compact_dense_transfer_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("within_user_ranker_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("catboost_within_ranker_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("catboost_within_ranker_v2", "verification.json", "day16_prediction.npy", "rolling"),
        ("daily_auc_pair_ranker_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("daily_binomial_rate_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("compose_daily_objectives_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("cross_user_pair_ranker_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("snapshot_pair_ranker_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("cold_task_ranker_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("firstday_task_ranker_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("compose_snapshot_cold_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("task_audience_rate_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("task_audience_row_ranker_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("audience_graph_ranker_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("profile_response_ranker_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("compact_full_sample_ranker_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("unordered_snapshot_ranker_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("compose_unordered_masks_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("sparse_cross_ranker_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("pointwise_compact_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("categorical_interactions_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("compose_probability_quantiles_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("compose_transition_priors_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("transition_posterior_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("row_deepfm_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("row_deepfm_v2", "verification.json", "day16_prediction.npy", "rolling"),
        ("compact_lgb_rows_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("compose_family_quantiles_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("group_event_posterior_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("full_negative_context_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("full_negative_context_v2", "verification.json", "day16_prediction.npy", "rolling"),
        ("full_rich_negative_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("full_rich_pair_ranker_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("full_rich_negative_v2", "verification.json", "day16_prediction.npy", "rolling"),
        ("full_rich_pair_ranker_v2", "verification.json", "day16_prediction.npy", "rolling"),
        ("compose_full_negative_masks_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("full_negative_prior_offset_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("full_negative_class_balance_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("full_negative_neural_embeddings_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("full_negative_neural_embeddings_v2", "verification.json", "day16_prediction.npy", "rolling"),
        ("full_negative_embedding_lgb_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("compose_class_balance_masks_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("compose_neural_embedding_masks_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("full_negative_semantic_embeddings_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("refresh_family_quantiles_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("compose_semantic_embedding_masks_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("full_negative_history_dot_embeddings_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("full_negative_embedding_lgb_v2", "verification.json", "day16_prediction.npy", "rolling"),
        ("full_negative_embedding_lgb_v3", "verification.json", "day16_prediction.npy", "rolling"),
        ("full_negative_dense_context_lgb_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("compose_history_dot_embedding_masks_v1", "verification.json", "day16_prediction.npy", "rolling"),
        ("augmented_blend_v1_day16", "verification.json", "prediction.npy", "plain"),
        ("rich_binary16_cpu_v1", "verification.json", "prediction.npy", "plain"),
        ("joint_group_calibration_v1", "verification.json", "prediction.npy", "plain"),
        ("recent_calibration_transfer_v1", "verification.json", "best_prediction.npy", "plain"),
        ("sam_calibration_transfer_v1", "verification.json", "best_prediction.npy", "plain"),
        ("sam_continuation_v1_day16", "verification.json", "prediction.npy", "plain"),
    ]
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["pt_d","label","log_id","user_id"],dtype="int32")
    rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191116)
    y = raw.label.to_numpy()[rows]
    if len(rows) != 421983 or int(y.sum()) != 3724:
        raise AssertionError("Real full16 cohort changed")
    expected_raw = json.loads((runs / "weighted_v1_day16/protocol.json").read_text(encoding="utf8"))["raw_datasets_sha256"]
    hashes = {name: digest(ROOT / "2026ml" / name) for name in expected_raw}
    if hashes != expected_raw:
        raise AssertionError("An original dataset changed")
    for name, verification_name, prediction_name, kind in definitions:
        directory = runs / name
        verification_path = directory / verification_name
        if not verification_path.is_file():
            continue
        verified = json.loads(verification_path.read_text(encoding="utf8"))
        if verified["status"] != "passed":
            continue
        if kind == "weighted" and not verified.get("evaluation_labels_masked_in_training_array", False):
            audit_path = directory / "label_boundary_audit.json"
            if not audit_path.is_file():
                continue
            audit = json.loads(audit_path.read_text(encoding="utf8"))
            if audit["status"] != "passed" or not audit["all_training_indices_strictly_before16"]:
                raise AssertionError("Unresolved unsigned label-mask audit")
        if kind == "rolling":
            np.testing.assert_array_equal(np.load(directory / "day16_indices.npy"), rows)
        expected = (verified["dates"]["20191116"]["auc"] if kind == "rolling" else
                    verified["results"]["regularized_stack"]["auc"] if kind == "stack" else
                    verified["best_candidate"]["auc"] if kind == "weighted" else verified["auc"])
        p = np.load(directory / prediction_name)
        if p.shape != y.shape or not np.isfinite(p).all() or not np.all((p>=0)&(p<=1)):
            raise AssertionError("Invalid candidate probability")
        score = auc(y,p)
        if abs(score-expected)>1e-12:
            raise AssertionError("Saved candidate AUC differs")
        candidates.append({"name":name,"auc":score,"prediction":str(directory / prediction_name),
                           "verification":str(verification_path), "verification_sha256":digest(verification_path)})
    if not candidates:
        raise AssertionError("No verified candidate")
    best = max(candidates,key=lambda c:c["auc"])
    p = np.load(best["prediction"])
    out = ROOT / "output/day16_auc082_progress_20261102"
    out.mkdir(exist_ok=True)
    previous = [p for p in out.iterdir() if p.is_file()]
    if previous:
        snapshot = out / "snapshots" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        snapshot.mkdir(parents=True)
        for path in previous:
            shutil.copy2(path, snapshot / path.name)
    np.save(out / "prediction.npy",p)
    frame = pd.DataFrame({"row_index": rows,"log_id":raw.log_id.to_numpy()[rows],
                          "user_id":raw.user_id.to_numpy()[rows],"label":y,"pctr":p})
    path = out / "day16_predictions.csv"
    frame.to_csv(path,index=False,float_format="%.17g")
    replay = pd.read_csv(path,float_precision="round_trip")
    np.testing.assert_array_equal(replay.row_index.to_numpy(),rows)
    np.testing.assert_array_equal(replay.label.to_numpy(),y)
    error = float(np.max(np.abs(replay.pctr.to_numpy()-p)))
    csv_auc = auc(y,replay.pctr.to_numpy())
    if error>1e-15 or abs(csv_auc-best["auc"])>1e-12:
        raise AssertionError("CSV export failed full-cohort replay")
    result = {"status":"verified_progress_target_not_reached" if best["auc"]<.82 else "target_verified",
              "objective":"Original datasets unchanged; full real20191116 ROC AUC>=0.82",
              "date":20191116,"rows":len(rows),"clicks":int(y.sum()),"target_auc":.82,
              "best":best,"candidates":candidates,"csv_auc":csv_auc,"csv_maximum_error":error,
              "original_datasets_sha256":hashes,"original_datasets_unchanged":True,
              "files_sha256":{name:digest(out/name) for name in ["prediction.npy","day16_predictions.csv"]},
              "day16_used_repeatedly_for_development":True,"final17_auc_claim":False,
              "training_and_verification_jobs_may_still_be_running":True}
    write(out / "verification.json",result)
    report = ("# 16日AUC优化进展\n\n"
              f"当前独立复算的完整2019-11-16 AUC为 **{csv_auc:.10f}**，目标为0.82。"
              f"验证包括全部{len(rows)}行和{int(y.sum())}个正例。\n\n"
              f"当前方案：`{best['name']}`。模型与核验文件保留在analysis对应目录。"
              "所有四个原始CSV的SHA256与本轮开始时一致。\n\n"
              "day16_predictions.csv包含原始行号、log_id、用户ID、真实label和预测pctr，"
              "可直接复算ROC AUC；它是本地16日验证输出。\n\n"
              "16日被反复用于模型开发；该结果不是新的盲测分数。"
              "尚未达到0.82时，任务保持进行中。\n")
    (out / "结果说明.md").write_text(report,encoding="utf8")
    state_path = ROOT / "analysis/auc85_goal_state.json"
    state = json.loads(state_path.read_text(encoding="utf8"))
    state["latest_real16_verified_progress"] = {"auc":csv_auc,"target":.82,"target_met":csv_auc>=.82,
                                               "output":str(out),"candidate":best["name"],"raw_unchanged":True}
    write(state_path,state)
    status_path = HERE / "STATUS.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf8"))
        status.update(snapshot_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                      current_verified_best_auc=csv_auc,
                      current_verified_best=str(Path(best["verification"]).relative_to(HERE)).replace("\\", "/"),
                      target_verified=csv_auc>=.82,
                      independently_verified_results={c["name"]:c["auc"] for c in candidates})
        write(status_path,status)
    print(json.dumps({"best":best,"csv_auc":csv_auc,"rows":len(rows),"export":str(out)},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    with publication_lock():
        main()
