"""Refresh current process and verified-cohort evidence without changing goal status."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--turn-classification', choices=['progress', 'verified_wait', 'no_progress'], default='progress')
classification = parser.parse_args().turn_classification

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
status_path = HERE/'STATUS.json'
state = json.loads(status_path.read_text(encoding='utf8'))
audit_path = ROOT/'output/two_day_auc_targets_20261102/independent_target_audit.json'
audit = json.loads(audit_path.read_text(encoding='utf8'))
assert audit['audit_status'] == 'passed' and audit['original_datasets_unchanged']
command = "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.Name -match 'python|powershell|pwsh' -and $_.CommandLine -match 'news_quality_daily|rich_augmented_rows|dense_context_rows|group_temperature|conditional_daily|rich_row_catboost|full_quality_row_catboost|catboost_group_temperature|grouped_catboost_residual|quality_daily_catboost|quality_daily_neural|compose_full_quality_transfer|compose_quality_daily_transfer|compose_dense_context_transfer|compose_compact_dense_transfer|within_user_ranker|catboost_within_ranker|final_best_recipe_compact_dense|daily_auc_pair|daily_binomial_rate|verify_catboost_ranker_precision' } | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Depth 3 -Compress"
command = command.replace('verify_catboost_ranker_precision', 'verify_catboost_ranker_precision|cross_user_pair|compose_daily_objectives|snapshot_pair|cold_task|firstday_task|compose_snapshot_cold|finalize_optimization_round|task_audience|audience_graph|profile_response|compact_full_sample|unordered_snapshot|compose_unordered_masks|sparse_cross|pointwise_compact|categorical_interactions|probability_quantiles')
observed = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command], capture_output=True,
                          text=True, encoding='utf8', errors='replace', check=True).stdout.strip()
processes = json.loads(observed) if observed else []
if isinstance(processes, dict):
    processes = [processes]
if classification == 'verified_wait':
    assert processes, 'A verified wait requires currently observed live processes'
experiments = {}
for name in ['news_quality_daily_v1', 'rich_augmented_rows_v1', 'dense_context_rows_v1',
             'group_temperature_v1', 'conditional_daily_v1', 'conditional_daily_trees_v1',
             'group_temperature_joint_v1', 'rich_row_catboost_v1',
             'full_quality_row_catboost_v1', 'catboost_group_temperature_v1',
             'grouped_catboost_residual_v1', 'quality_daily_catboost_v1', 'compose_full_quality_transfer_v1',
             'compose_quality_daily_transfer_v1', 'compact_quality_daily_catboost_v1', 'quality_daily_neural_v1',
             'compose_dense_context_transfer_v1','compose_compact_dense_transfer_v1','within_user_ranker_v1',
             'catboost_within_ranker_v1','catboost_within_ranker_v2','daily_auc_pair_ranker_v1','daily_binomial_rate_v1',
             'compose_daily_objectives_v1','cross_user_pair_ranker_v1','snapshot_pair_ranker_v1','cold_task_ranker_v1',
             'firstday_task_ranker_v1','compose_snapshot_cold_v1','task_audience_rate_v1','task_audience_row_ranker_v1','audience_graph_ranker_v1','profile_response_ranker_v1','compact_full_sample_ranker_v1','unordered_snapshot_ranker_v1','compose_unordered_masks_v1','sparse_cross_ranker_v1','pointwise_compact_v1','categorical_interactions_v1','compose_probability_quantiles_v1']:
    directory = HERE/'runs'/name
    record = {'exists': directory.exists()}
    for filename in ['protocol.json', 'models_locked.json', 'result.json', 'verification.json']:
        path = directory/filename
        record[filename] = path.is_file()
    verification = directory/'verification.json'
    if verification.is_file():
        report = json.loads(verification.read_text(encoding='utf8'))
        record['verification_status'] = report['status']
        record['aucs'] = {date: value['auc'] for date, value in report['dates'].items()}
    elif (directory/'result.json').is_file():
        provisional = json.loads((directory/'result.json').read_text(encoding='utf8'))
        record['unverified_result_aucs'] = {date:value['auc'] for date,value in provisional['dates'].items()}
    if name in ['rich_row_catboost_v1', 'full_quality_row_catboost_v1']:
        preflight_path = HERE/'runs'/name.replace('_v1', '_preflight_v1')/'verification.json'
        if preflight_path.is_file():
            preflight = json.loads(preflight_path.read_text(encoding='utf8'))
            record['preflight_status'] = preflight['status']
            record['preflight_shape'] = preflight['audit']['shape']
            record['preflight_maximum_serialization_error'] = preflight['serialization_maximum_error']
        record['saved_checkpoints'] = len(list(directory.glob('*.cbm')))
        record['source_parent_verified'] = (HERE/'runs/rich_augmented_rows_v1/verification.json').is_file()
        record['source_verification_binding'] = (directory/'parent_verification_binding.json').is_file()
    if name in ['quality_daily_catboost_v1', 'compact_quality_daily_catboost_v1','daily_auc_pair_ranker_v1','daily_binomial_rate_v1']:
        record['saved_checkpoints'] = len(list(directory.glob('*.cbm')))
    if name in ['audience_graph_ranker_v1', 'profile_response_ranker_v1', 'compact_full_sample_ranker_v1','unordered_snapshot_ranker_v1']:
        record['saved_checkpoints'] = len(list(directory.glob('*.cbm')))
        prefix = name.removesuffix('_ranker_v1')
        preflight_path = HERE/'runs'/f'{prefix}_preflight_v1'/'verification.json'
        record['preflight_verification_exists'] = preflight_path.is_file()
        if preflight_path.is_file():
            preflight = json.loads(preflight_path.read_text(encoding='utf8'))
            record['preflight_status'] = preflight['status']
            record['preflight_shape'] = preflight['audit']['shape']
        record['training_contracts'] = {}
        for day in [15,16]:
            contract_path = directory/f'input_contract{day}.json'
            if contract_path.is_file():
                contract = json.loads(contract_path.read_text(encoding='utf8'))
                record['training_contracts'][str(day)] = dict(shape=contract['shape'],
                    max_training_date=contract['max_training_date'])
    if name in ['catboost_within_ranker_v1','catboost_within_ranker_v2']:
        record['saved_checkpoints'] = len(list(directory.glob('*.cbm')))
        record['training_contracts'] = {}
        for day in [15,16]:
            contract_path = directory/f'input_contract{day}.json'
            if contract_path.is_file():
                contract = json.loads(contract_path.read_text(encoding='utf8'))
                record['training_contracts'][str(day)] = dict(shape=contract['shape'],
                    max_training_date=contract['max_training_date'],all337_quality_columns=contract['all337_quality_columns'])
        preflight_path = HERE/'runs'/name.replace('catboost_within_ranker_','catboost_within_ranker_preflight_')/'verification.json'
        record['preflight_verification_exists'] = preflight_path.is_file()
        if preflight_path.is_file():
            preflight = json.loads(preflight_path.read_text(encoding='utf8'))
            record['preflight_status'] = preflight['status']
            record['preflight_shape'] = preflight['audit']['shape']
            record['largest_full_prior_training_cohort_used'] = preflight['largest_full_prior_training_cohort_used']
            record['maximum_serialization_errors'] = [item['serialization_maximum_error'] for item in preflight['results']]
    if name == 'quality_daily_neural_v1':
        record['saved_checkpoints'] = len(list(directory.glob('*.pt')))
        record['normalization_contracts'] = {}
        for day in [15,16]:
            contract_path = directory/f'input_day{day}/contract.json'
            if contract_path.is_file():
                contract = json.loads(contract_path.read_text(encoding='utf8'))
                record['normalization_contracts'][str(day)] = {
                    'shape': contract['shape'],
                    'max_normalization_training_date': contract['max_normalization_training_date'],
                    'held_counts_masked': contract['held_counts_masked']}
    if name == 'within_user_ranker_v1':
        record['saved_checkpoints'] = len(list(directory.glob('*.txt')))
        record['training_contracts'] = {}
        for day in [15,16]:
            contract_path = directory/f'input_contract{day}.json'
            if contract_path.is_file():
                contract = json.loads(contract_path.read_text(encoding='utf8'))
                record['training_contracts'][str(day)] = dict(
                    shape=contract['training_shape'],queries=contract['prior']['queries'],
                    max_training_date=contract['prior']['max_training_date'])
    if name == 'compact_quality_daily_catboost_v1':
        compact_preflight_path = HERE/'runs/compact_quality_daily_catboost_preflight_v1/verification.json'
        if compact_preflight_path.is_file():
            compact_preflight = json.loads(compact_preflight_path.read_text(encoding='utf8'))
            record['preflight_status'] = compact_preflight['status']
            record['preflight_shape'] = compact_preflight['shape']
            record['preflight_depth'] = compact_preflight['depth']
            record['preflight_maximum_serialization_error'] = compact_preflight['serialization_maximum_error']
            record['full_prior_cohort_memory_preflight'] = compact_preflight['full_cohort_memory_preflight']
    experiments[name] = record
runtime = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'goal_status': 'active',
    'previous_goal_turn_classification': state.get('latest_goal_runtime', {}).get('last_goal_turn_classification',
        state.get('last_goal_turn_classification', 'unknown')),
    'last_goal_turn_classification': classification,
    'verified15_auc': audit['dates']['20191115']['auc'],
    'verified16_auc': audit['dates']['20191116']['auc'],
    'targets_met': bool(audit['goal_achieved']), 'raw_datasets_unchanged': True,
    'sessions': {'news_quality_pipeline': 76010, 'rich_augmented_rows_pipeline': 47453,
                 'dense_context_rows_pipeline': 25369, 'group_temperature_pipeline': 8948,
                 'conditional_daily_pipeline': 12372, 'conditional_daily_trees_pipeline': 93552,
                 'group_temperature_joint_pipeline': 40786, 'rich_row_catboost_pipeline': 63301,
                 'full_quality_row_catboost_pipeline': 32999, 'catboost_group_temperature_pipeline': 74385,
                 'grouped_catboost_residual_pipeline': 21323, 'quality_daily_catboost_pipeline': 30215,
                 'compose_full_quality_transfer_pipeline': 57853,
                 'compose_quality_daily_transfer_pipeline': 6572,
                 'compact_quality_daily_catboost_pipeline': 51730,
                 'quality_daily_neural_pipeline': 68568,
                 'compose_dense_context_transfer_pipeline': 50777,
                 'compose_compact_dense_transfer_pipeline': 14055,
                 'within_user_ranker_pipeline': 74453,
                 'day17_compact_dense_export': 55933,
                 'day17_ranked_compact_dense_export': 84728,
                 'catboost_within_ranker_failed_preflight': 56436,
                 'catboost_within_ranker_pipeline': 16660,
                 'catboost_ranker_precision_verification_v2':95590,
                 'daily_auc_pair_ranker_pipeline':46629,
                 'daily_binomial_rate_pipeline':5948,
                 'day17_daily_rate_export':16806,
                 'day17_daily_pair_refit_export':64048,
                 'day17_daily_binomial_refit_export':51829,
                 'daily_objectives_composition':4310,
                 'cross_user_pair_pipeline':68895,
                 'cross_user_final_refit_export':16430,
                 'cross_user_final_prediction_recovery':89934,
                 'snapshot_pair_pipeline':40996,
                 'snapshot_pair_export':32214,
                 'cold_task_pipeline':41341,
                 'cold_task_export':25299,
                 'firstday_task_pipeline':65699,
                 'firstday_task_export':34079,
                 'snapshot_cold_composition':39278,
                 'optimization_round_finalization':53762,
                 'task_audience_pipeline':89225,
                 'task_audience_row_pipeline':47460,
                 'task_audience_day17_export':91066,
                 'task_audience_row_day17_export':80965,
                 'audience_graph_pipeline':4867,
                 'profile_response_pipeline':11305,
                 'day17_catboost_ranked_export':6681},
    'processes': processes, 'experiments': experiments,
    'next_actions': ['Compact, dense, both new compositions, and full-quality neural pipelines all completed; inspect actual processes before starting new work',
        'CatBoost within-user ranking v2 session16660 runs full282904-row/1017-column GPU preflight, then strictly-prior PairLogit/YetiRank fits for both dates',
        'Version1 session56436 ended before fitting because the Ranker constructor received a params dictionary as its positional iterations argument; preserve it; v2 uses keyword arguments and both synthetic GPU API checks passed',
        'Preserve frozen catboost_within_ranker source, test, wrapper, preflight and all failed runs; do not restart while actual process is live',
        'CatBoost ranking retains all337 quality inputs and native high-cardinality categorical fields; all12 both-date checkpoints lock before held scoring',
        'All12 full-quality daily models and all1017-column row models passed independent raw-feature replay',
        'Daily-plus-row quality composition passed full-cohort verification and improved both dates',
        'Dense/context completed independent verification below the published winner on both dates',
        'Compact deeper Poisson completed all fits and independent both-date verification; its parent pipeline exited0',
        'Dense/context transfer improved both dates and passed full CSV/raw-alignment/AUC audits',
        'Compact/dense composition session14055 completed and passed full CSV/raw-alignment/AUC audits',
        'Full-quality neural session68568 completed16 checkpoints,97 candidates/date, and exact raw-feature/model replay; no improvement over its older frozen baseline',
        'Ranking core and4 boundary/math tests passed; real-prior preflight passed400 complete sampled queries,9884 rows,694 columns for both lambdarank and rank_xendcg',
        'The ranking core/test are now frozen by within_user_ranking_preflight_v1/protocol.json; preserve them',
        'Full within-user ranking runner and wrapper are frozen; inspect session74453 model locks and full-cohort independent verification',
        'Use compose_compact_dense_transfer_v1 as the frozen ranking baseline; all four prior-date fits and12 checkpoints lock before held-cohort scoring',
        'The neural wrapper runs under PowerShell7 after preserving a failed PowerShell5 UTF8 queue-read attempt; no fit started in that attempt',
        'All neural checkpoints for both dates must lock before either held cohort is scored',
        'Verify and publish completed full-cohort experiments; preserve source hashes and all raw datasets',
        'Keep both actual targets15>=.815 and16>=.82; no inferred17 outcomes for fitting or model selection']}
ranking_preflight_path = HERE/'runs/within_user_ranking_preflight_v1/verification.json'
if ranking_preflight_path.is_file():
    ranking_preflight = json.loads(ranking_preflight_path.read_text(encoding='utf8'))
    runtime['within_user_ranking_preflight'] = {
        'status': ranking_preflight['status'],
        'rows': ranking_preflight['rows'], 'queries': ranking_preflight['queries'],
        'features': ranking_preflight['features'],
        'max_training_date': ranking_preflight['max_training_date'],
        'lightgbm_version': ranking_preflight['lightgbm_version'],
        'maximum_serialization_errors': [r['maximum_serialization_error'] for r in ranking_preflight['results']],
        'full_date_models_fitted': ranking_preflight['full_date_models_fitted'],
        'held_auc_evaluated': ranking_preflight['held_auc_evaluated'],
        'protocol': str(ranking_preflight_path.parent/'protocol.json')}
neural_queue_path = HERE/'quality_daily_neural_queued_v1.json'
if neural_queue_path.is_file():
    neural_queue = json.loads(neural_queue_path.read_text(encoding='utf8'))
    runtime['quality_daily_neural_queue'] = {
        'frozen_protocol': str(neural_queue_path),
        'wait_processes': neural_queue['wait_processes'],
        'boundary_tests_passed': neural_queue['boundary_tests_passed'],
        'config': neural_queue['config'],
        'fit_started': experiments['quality_daily_neural_v1']['protocol.json'],
        'normalization_and_vocab_fit_only_before_prediction_date': True,
        'inferred17_labels_read': False}
day17_candidates = [ROOT/f'output/{folder}/verification.json' for folder in [
    'best_recipe_daily_quality_transfer_20191117_20261102',
    'best_recipe_compact_dense_20191117_20261102',
    'best_recipe_ranked_compact_dense_20191117_20261102',
    'best_recipe_catboost_ranked_20191117_20261102',
    'best_recipe_daily_rate_20191117_20261102',
    'best_recipe_daily_rate_refit17_20191117_20261102',
    'best_recipe_daily_rate_refit17_v2_20191117_20261102',
    'best_recipe_cross_user_pair_20191117_20261102',
    'best_recipe_cross_user_refit17_20191117_20261102',
    'best_recipe_snapshot_pair_20191117_20261102',
    'best_recipe_cold_task_20191117_20261103',
    'best_recipe_firstday_task_20191117_20261103',
    'best_recipe_snapshot_cold_20191117_20261103']]
verified17 = [(path,json.loads(path.read_text(encoding='utf8')))
              for path in day17_candidates if path.is_file()]
verified17 = [(path,report) for path,report in verified17 if report['status']=='passed']
runtime['day17_compact_dense_export'] = dict(
    session_id=55933,
    source=str(ROOT/'analysis/final_best_recipe_compact_dense_20261102/export_best_v2.py'),
    run=str(ROOT/'analysis/final_best_recipe_compact_dense_20261102/run_v2'),
    verification_exists=day17_candidates[1].is_file(),
    preserved_failed_run=str(ROOT/'analysis/final_best_recipe_compact_dense_20261102/run_v1'),
    recovery=str(ROOT/'analysis/final_best_recipe_compact_dense_20261102/recovery_v2.json'))
runtime['day17_ranked_compact_dense_export'] = dict(
    session_id=84728,
    source=str(ROOT/'analysis/final_best_recipe_compact_dense_20261102/export_ranked.py'),
    run=str(ROOT/'analysis/final_best_recipe_compact_dense_20261102/ranked_run_v1'),
    verification_exists=day17_candidates[2].is_file(),
    queue='Wait for the actual ranking wrapper113440 to exit; export only a verified real16 improvement',
    inferred17_labels_used_for_selection=False)
catboost_recovery=HERE/'catboost_ranker_recovery_v2.json'
if catboost_recovery.is_file():
    runtime['catboost_ranker_recovery_v2']=json.loads(catboost_recovery.read_text(encoding='utf8'))
catboost_api=HERE/'catboost_ranker_api_smoke_v2.json'
if catboost_api.is_file():
    runtime['catboost_ranker_api_smoke_v2']=json.loads(catboost_api.read_text(encoding='utf8'))
if experiments['within_user_ranker_v1']['verification.json']:
    runtime['next_actions'].insert(0,'Within-user ranking completed all12 checkpoints and independent both-date feature/model/candidate replay; its verified scores are recorded above')
elif experiments['within_user_ranker_v1']['models_locked.json']:
    runtime['next_actions'].insert(0,'All four within-user ranking fits completed and12 checkpoints are locked; full-cohort scoring/verification is in progress')
runtime['next_actions'].insert(0,'PairLogit/YetiRank frozen v2 has passed supplemental independent raw-feature/model replay; numpy/pandas float64 AUC tie effects are bounded and the winning recipes are unchanged')
runtime['next_actions'].insert(0,'Daily explicit-pair runner46629 uses the verified CatBoost ranking baseline,337 quality columns, prior-only feature selectors, and per-row CTR offsets; all both-date checkpoints lock before scoring')
runtime['next_actions'].insert(0,'Exposure-weighted CrossEntropy runner5948 waits for actual pairwise GPU fit105880 to exit, then compares all-history d6 with recent2-day d7, using12 passed boundary/math tests and the same frozen prior-only selectors')
runtime['next_actions'].insert(0,'CatBoost ranked day17 export6681 applies the independently verified real16 improvement; inferred17 labels remain evaluation-only')
runtime['next_actions'].insert(0,'Daily pair pipeline46629 passed raw-data feature replay, exact model margins, all121 candidates/date, full CSV and raw-hash audits; both real-date scores improved')
runtime['next_actions'].insert(0,'Daily pair day17 export16806 passed raw-feature and prediction replay with404182 rows; only real16 AUC selected its recipe')
runtime['prepared_final_day17_refit']=dict(
    source=str(ROOT/'analysis/final_best_recipe_compact_dense_20261102/export_daily_rate_refit17_v1.py'),
    run=str(ROOT/'analysis/final_best_recipe_compact_dense_20261102/daily_rate_refit17_run_v1'),
    boundary_tests_passed=4,training_user_days=51336,training_clicks=21354,
    training_advertisement_exposures=2526320,explicit_pairs=1024820,
    latest_advertisement_training_date=20191116,inferred17_labels_for_training_or_selection=False,
    next_action='Pair refit v1 completed; separately frozen v2 refits the verified real16 binomial winner through genuine day16 labels; select exports solely by real16 validation and latest genuine training date')
runtime['next_actions'].insert(0,'Daily binomial passed both-date independent feature/model/candidate replay; real16 improved to0.812728292273607. Both-day composition passed; real15 improved to0.8010205555738956 and real16 selects the binomial endpoint')
runtime['next_actions'].insert(0,'Cross-user row PairLogit pipeline68895 retains zero-click users, all337 quality inputs and native categories; three boundary/pair/reproducibility tests passed; preflight and all both-date checkpoints must lock before scoring')
runtime['cross_user_final_refit_export']=dict(session_id=16430,wait_for_actual_wrapper=125132,
    source=str(ROOT/'analysis/final_best_recipe_compact_dense_20261102/export_cross_user_refit17_v1.py'),
    run=str(ROOT/'analysis/final_best_recipe_compact_dense_20261102/cross_user_refit17_run_v1'),
    output=str(ROOT/'output/best_recipe_cross_user_refit17_20191117_20261102'),
    full_input_preflight_passed=True,training_rows=230100,clicks=21354,explicit_pairs=1024992,
    maximum_ad_training_date=20191116,inferred17_labels_for_training_or_selection=False,
    next_action='Wait for verified row ranking and publication/audit; skip unless real16 improves; otherwise freeze recipe, refit through genuine16, lock predictions, independently replay, then evaluate inferred17')
runtime['cross_user_final_refit_export'].update(
    prediction_resume_session=89934, original_session_missing_after_explicit_pause=True,
    refit_checkpoint_preserved=True,
    next_action='Original process was authoritatively absent after pause; resumed only predict/verify with saved refit17 model. Inspect89934 and output verification before further recovery.')
runtime['snapshot_quality_row_extension']=dict(session_id=40996,
    source=str(HERE/'snapshot_pair_ranker_v1.py'),run=str(HERE/'runs/snapshot_pair_ranker_v1'),
    new_features=240,boundary_and_alignment_tests_passed=3,original_columns=1017,
    feature_scope='Content-adjusted news response at exact interest/refresh/lifecycle contexts; other-user content expectation; only same or earlier news dates',
    inferred17_labels_read=False,
    next_action='Complete capacity preflight, lock both-date models/candidates, independently rebuild feature inputs and score full cohorts')
runtime['snapshot_quality_export']=dict(session_id=32214,wait_for_actual_wrapper=68632,
    source=str(ROOT/'analysis/final_best_recipe_compact_dense_20261102/export_snapshot_pair_v1.py'),
    output=str(ROOT/'output/best_recipe_snapshot_pair_20191117_20261102'),
    gate='Export only if independently verified real16 AUC improves current0.8127546559689401',
    inferred17_labels_used_for_model_or_feature_selection=False)
if experiments['snapshot_pair_ranker_v1']['verification.json']:
    snapshot_result=json.loads((HERE/'runs/snapshot_pair_ranker_v1/verification.json').read_text(encoding='utf8'))
    runtime['snapshot_quality_row_extension'].update(status=snapshot_result['status'],
        auc15=snapshot_result['dates']['20191115']['auc'],auc16=snapshot_result['dates']['20191116']['auc'],
        next_action='Independent1257-column feature/model/candidate replay and full-cohort audit passed; finish latest real16-selected day17 export')
runtime['cold_task_transfer']=dict(session_id=41341,export_session=25299,wait_for_wrapper=67580,
    source=str(HERE/'cold_task_ranker_v1.py'),run=str(HERE/'runs/cold_task_ranker_v1'),
    new_boundary_tests_passed=3,masked_task_and_creative_columns=93,
    motivation='Known/new task diagnostic: real15 new-task AUC0.7457389118033416, real16 new-task AUC0.7647526665094069',
    diagnosis=str(HERE/'remaining_auc_gap_v1.json'),
    inference='Apply fixed correction only to task IDs absent from genuinely earlier dates; preserve shared advertiser/app/user inputs',
    inferred17_labels_used_for_training_or_model_selection=False,
    next_action='Complete capacity preflight and all both-date checkpoints; independent replay and full-cohort publication; export only a latest audited real16 improvement')
runtime['firstday_task_transfer']=dict(session_id=65699,export_session=34079,wait_for_wrapper=68800,
    source=str(HERE/'firstday_task_ranker_v1.py'),run=str(HERE/'runs/firstday_task_ranker_v1'),
    boundary_tests_passed=3,full_prior_launch_preflight_rows=12792,
    training_scope='Only each task first observed date, Nov12 onward, strictly before prediction date; mask task/creative identity features',
    left_censored_initial_dates_excluded=[20191110,20191111],
    inferred17_labels_used_for_training_or_model_selection=False,
    next_action='Finish all both-date models, lock candidates, rebuild independent features, audit, and export only a real16 improvement')
runtime['task_audience_rate']=dict(session_id=89225,wrapper_process=61608,
    source=str(HERE/'task_audience_rate_v1.py'),run=str(HERE/'runs/task_audience_rate_v1'),
    boundary_and_aggregation_tests_passed=3,full_day16_task_date_groups=29512,features=1193,
    objective='Exposure-weighted binomial task/day response from the whole audience, using strictly earlier ad labels',
    input_scope='Unlabeled same-day advertisements; chronological user/task/advertiser/app history; all337 independent news quality inputs aggregated per task',
    inferred17_labels_read=False,
    next_action='Complete full-capacity and independent sparse feature preflight, fit both days, lock all candidates, verify full cohorts')
audience_preflight=HERE/'runs/task_audience_preflight_v1/verification.json'
if audience_preflight.is_file():
    audience_check=json.loads(audience_preflight.read_text(encoding='utf8'))
    runtime['task_audience_rate'].update(preflight_status=audience_check['status'],
        independent_sparse_features_replayed=audience_check['sparse_independent_feature_replay_exact'],
        full_population_audit=str(HERE/'task_audience_full_population_audit_v1.json'))
if experiments['task_audience_rate_v1']['verification.json']:
    runtime['task_audience_rate'].update(status='independently_verified',
        aucs=experiments['task_audience_rate_v1']['aucs'],
        next_action='Full cohort publication and raw-data audit passed; continue task audience row extension')
runtime['task_audience_row_extension']=dict(session_id=47460,wait_for_wrapper=61608,
    source=str(HERE/'task_audience_row_ranker_v1.py'),run=str(HERE/'runs/task_audience_row_ranker_v1'),
    selected_features=192,selector='Fixed depth6/600-tree task model trained only before each prediction date',
    boundary_and_alignment_tests_passed=3,selector_api_checked_on_real15_model=True,
    baseline='Verified task_audience_rate_v1 predictions for both dates',
    inferred17_labels_read=False,
    next_action='Wait for verified task model; full1209-column preflight, both-date PairLogit fits and independent replay')
for key,name in [('cold_task_transfer','cold_task_ranker_v1'),('firstday_task_transfer','firstday_task_ranker_v1')]:
    if experiments[name]['verification.json']:
        runtime[key].update(status='independently_verified',aucs=experiments[name]['aucs'],
            next_action='Both-date model/input/candidate verification complete; preserve checkpoints and use the highest real16 verified export')
export_recovery=ROOT/'analysis/final_best_recipe_compact_dense_20261102/firstday_task_export_recovery_v2.json'
if export_recovery.is_file():
    runtime['firstday_task_export_recovery']=json.loads(export_recovery.read_text(encoding='utf8'))
    runtime['firstday_task_export_recovery']['corrected_export_result']='No export needed: real16 score0.8127664985737552 below best0.8127918767763859'
if experiments['cross_user_pair_ranker_v1']['verification.json']:
    cross_report=json.loads((HERE/'runs/cross_user_pair_ranker_v1/verification.json').read_text(encoding='utf8'))
    runtime['cross_user_development_result']=dict(status=cross_report['status'],
        auc15=cross_report['dates']['20191115']['auc'],auc16=cross_report['dates']['20191116']['auc'],
        selected16=cross_report['dates']['20191116']['best'],
        three_boundary_pair_reproducibility_tests_passed=True,
        all12_checkpoints_and_all254_candidate_predictions_replayed=True,
        independently_verified_both_date_csv_and_raw_hashes=True,
        inferred17_labels_for_training_or_selection=False)
    runtime['next_actions'].insert(0,'Cross-user row ranking completed independent both-date feature/model/candidate replay and publication/audit; final17 refit session16430 uses the real16-selected depth8/1600-tree/add_global/.05 recipe through genuine16 labels')
platform_feedback=HERE/'platform_feedback_0814_ranked_v1.json'
if platform_feedback.is_file():
    runtime['user_reported_platform_feedback']=json.loads(platform_feedback.read_text(encoding='utf8'))
for key,folder in [('daily_pair_api_v2','daily_auc_pair_api_preflight_v2'),
                   ('daily_pair_full_preflight','daily_auc_pair_ranker_preflight_v1'),
                   ('daily_binomial_full_preflight','daily_binomial_rate_preflight_v1'),
                   ('catboost_precision_replay','catboost_ranker_precision_verification_v2')]:
    report_path=HERE/'runs'/folder/'verification.json'
    runtime[key]=dict(verification_exists=report_path.is_file(),verification=str(report_path))
    if report_path.is_file():
        report=json.loads(report_path.read_text(encoding='utf8'))
        runtime[key]['status']=report['status']
runtime['day17_catboost_ranked_export']=dict(session_id=6681,run=str(ROOT/'analysis/final_best_recipe_compact_dense_20261102/catboost_ranked_run_v1'),
    verification_exists=day17_candidates[3].is_file(),inferred17_labels_used_for_selection=False)
if verified17:
    # Select an export solely by its real16 development result, never by inferred17 AUC.
    day17_verification,day17_report=max(verified17,key=lambda item:(item[1]['development16_auc'],item[1].get('correction_training_max_date',0)))
    runtime['latest_day17_export'] = {
        'verification': str(day17_verification),
        'submission': str(day17_verification.parent/'submission_best_20191117.csv'),
        'development16_auc': day17_report['development16_auc'],
        'inferred_label_auc_evaluation_only': day17_report['inferred_label_auc'],
        'inferred17_labels_used_for_selection': False,
        'rows': day17_report['rows']}
queue_path = HERE/'dense_context_rows_queued_v1.json'
if queue_path.is_file():
    queue = json.loads(queue_path.read_text(encoding='utf8'))
    preflight = json.loads((HERE/'runs/dense_context_rows_preflight_v1/verification.json').read_text(encoding='utf8'))
    runtime['dense_context_queue'] = {'frozen_protocol': str(queue_path),
        'wait_for_parent_process_id': queue['wait_for_process_id'],
        'preflight_status': preflight['status'], 'preflight_shape': preflight['shape'],
        'preflight_maximum_serialization_error': preflight['serialization_maximum_error'],
        'boundary_and_relabeling_tests_passed': 4,
        'fit_started': experiments['dense_context_rows_v1']['protocol.json'],
        'source_bound_at_queue_freeze': True}
quality_queue_path = HERE/'full_quality_row_catboost_queued_v1.json'
if quality_queue_path.is_file():
    quality_queue = json.loads(quality_queue_path.read_text(encoding='utf8'))
    runtime['full_quality_queue'] = {
        'frozen_protocol': str(quality_queue_path),
        'boundary_tests_passed': quality_queue['boundary_tests_passed'],
        'preflight_shape': quality_queue['preflight_shape'],
        'preflight_serialization_error': quality_queue['preflight_serialization_error'],
        'all_quality_features': 337, 'total_row_features': 1017,
        'arms': quality_queue['arms'], 'checkpoints': quality_queue['checkpoints'],
        'fit_started': experiments['full_quality_row_catboost_v1']['protocol.json'],
        'inferred17_labels_read': False}
    runtime['full_quality_queue']['training_contracts'] = {}
    for day in [15, 16]:
        contract_path = HERE/f'runs/full_quality_row_catboost_v1/training_contract{day}.json'
        if contract_path.is_file():
            contract = json.loads(contract_path.read_text(encoding='utf8'))
            runtime['full_quality_queue']['training_contracts'][str(day)] = {
                'shape': contract['shape'], 'max_training_date': contract['max_training_date'],
                'frame_sha256': contract['frame_sha256']}
    runtime['full_quality_queue']['completed_training_arms'] = []
    for day in [15, 16]:
        for arm in quality_queue['arms']:
            files = [HERE/f'runs/full_quality_row_catboost_v1/day{day}_{arm["name"]}_trees{trees}.cbm'
                     for trees in quality_queue['checkpoints']]
            if all(path.is_file() for path in files):
                runtime['full_quality_queue']['completed_training_arms'].append({'day': day, 'arm': arm['name']})
daily_queue_path = HERE/'quality_daily_catboost_queued_v1.json'
if daily_queue_path.is_file():
    daily_queue = json.loads(daily_queue_path.read_text(encoding='utf8'))
    runtime['quality_daily_queue'] = {
        'frozen_protocol': str(daily_queue_path),
        'wait_for_process_id': daily_queue['wait_for_process_id'],
        'boundary_tests_passed': daily_queue['boundary_tests_passed'],
        'baseline_pool_semantics_test_passed': daily_queue['baseline_pool_semantics_test_passed'],
        'source_frame_shapes': daily_queue['source_frame_shapes'],
        'arms': daily_queue['arms'], 'checkpoints': daily_queue['checkpoints'],
        'fit_started': experiments['quality_daily_catboost_v1']['protocol.json'],
        'inferred17_labels_read': False}
transfer_queue_path = HERE/'compose_full_quality_transfer_queued_v1.json'
if transfer_queue_path.is_file():
    transfer_queue = json.loads(transfer_queue_path.read_text(encoding='utf8'))
    runtime['quality_transfer_queue'] = {
        'frozen_protocol': str(transfer_queue_path),
        'wait_for_process_id': transfer_queue['wait_for_process_id'],
        'weights': transfer_queue['weights'], 'locked_parent_models': transfer_queue['locked_models'],
        'syntax_and_zero_increment_identity_checks_passed': transfer_queue['syntax_and_zero_increment_identity_checks_passed'],
        'parent_verification_required': transfer_queue['parent_verification_required_before_prepare'],
        'prepare_started': experiments['compose_full_quality_transfer_v1']['protocol.json']}
runtime['user_focus'] = {
    'instruction': 'Focus on genuine day15/day16 AUC; stop day17 inferred-label work',
    'day17_export_session91066': 'stopped by user direction; partial files preserved; not a verified delivery',
    'day17_row_export_session80965': 'stopped before execution',
    'inferred17_labels_permitted_for_current_work': False,
    'targets': {'20191115': .815, '20191116': .82}}
runtime['audience_graph_extension'] = {
    'session_id': 4867, 'wait_for_wrapper': 70752,
    'source': str(HERE/'audience_graph_ranker_v1.py'),
    'core': str(HERE/'audience_graph_core_v1.py'),
    'run': str(HERE/'runs/audience_graph_ranker_v1'),
    'real_data_preflight': str(HERE/'audience_graph_core_preflight_v1.json'),
    'features': 24, 'expected_total_row_features': 1233,
    'ranks': [8,16,32,64], 'boundary_tests_passed': 3,
    'graph_label_dates_strictly_before_each_query': True,
    'inferred17_labels_read': False}
runtime['profile_response_extension'] = {
    'session_id': 11305, 'wait_for_wrapper': 65652,
    'source': str(HERE/'profile_response_ranker_v1.py'),
    'core': str(HERE/'profile_response_core_v1.py'),
    'run': str(HERE/'runs/profile_response_ranker_v1'),
    'real_data_preflight': str(HERE/'profile_response_core_preflight_v1.json'),
    'features': 168, 'expected_total_row_features': 1377,
    'boundary_and_hand_calculation_tests_passed': 4,
    'labels_strictly_before_each_feature_date': True,
    'inferred17_labels_read': False}
compact_queue_path = HERE/'compact_full_sample_queued_v1.json'
if compact_queue_path.is_file():
    runtime['compact_full_sample_extension'] = dict(
        json.loads(compact_queue_path.read_text(encoding='utf8')),
        session_id=81614, queue_wrapper=79716,
        source=str(HERE/'compact_full_sample_ranker_v1.py'),
        core=str(HERE/'compact_full_sample_core_v1.py'),
        run=str(HERE/'runs/compact_full_sample_ranker_v1'),
        boundary_and_alignment_tests_passed=3)
runtime['next_actions'].insert(0, 'Profile-response experiment completed independent full-cohort verification without improvement; continue compact full-sample session81614 on genuine day15/day16 only')
runtime['unordered_snapshot_extension'] = {
    'source': str(HERE/'unordered_snapshot_ranker_v1.py'),
    'core': str(HERE/'unordered_snapshot_core_v1.py'),
    'feature_preflight': str(HERE/'unordered_snapshot_core_preflight_v1.json'),
    'features': 72, 'expected_total_row_features': 495,
    'boundary_permutation_and_replay_tests_passed': 3,
    'full_prior14_replay_shape': [382961,72],
    'advertisement_labels_read_by_features': False,
    'inferred17_labels_read': False,
    'training_not_started_at_preparation': True}
unordered_queue_path = HERE/'unordered_snapshot_queued_v1.json'
if unordered_queue_path.is_file():
    runtime['unordered_snapshot_extension'].update(json.loads(unordered_queue_path.read_text(encoding='utf8')))
    runtime['unordered_snapshot_extension']['session_id'] = 81406
    if experiments['unordered_snapshot_ranker_v1']['verification.json']:
        runtime['unordered_snapshot_extension']['phase'] = 'verified'
    elif experiments['unordered_snapshot_ranker_v1']['models_locked.json']:
        runtime['unordered_snapshot_extension']['phase'] = 'prediction_or_verification'
    elif experiments['unordered_snapshot_ranker_v1']['protocol.json']:
        runtime['unordered_snapshot_extension']['phase'] = 'fit'
    elif (HERE/'runs/unordered_snapshot_preflight_v1/protocol.json').exists():
        runtime['unordered_snapshot_extension']['phase'] = 'preflight'
    if experiments['compact_full_sample_ranker_v1']['verification.json']:
        runtime['next_actions'].insert(0,'Compact full-sample verified; continue unordered-snapshot session81406 on genuine dates15/16 only')
mask_queue_path = HERE/'compose_unordered_masks_queued_v1.json'
if mask_queue_path.is_file():
    runtime['unordered_mask_composition'] = dict(
        json.loads(mask_queue_path.read_text(encoding='utf8')), session_id=85503,
        mask_math_and_outside_identity_test_passed=True)
sparse_queue_path = HERE/'sparse_cross_queued_v1.json'
if sparse_queue_path.is_file():
    runtime['sparse_cross_extension'] = dict(
        json.loads(sparse_queue_path.read_text(encoding='utf8')),session_id=69815,
        hash_replay_and_sparse_margin_tests_passed=2,
        real_prior14_capacity_and_independent_probe_passed=True)
    if experiments['compose_unordered_masks_v1']['verification.json']:
        runtime['next_actions'].insert(0,'Continue sparse-cross session69815; both target dates remain genuine reused development data and inferred17 remains forbidden')
runtime['pointwise_compact_extension'] = dict(session_id=86885,
    source=str(HERE/'pointwise_compact_v1.py'),
    run=str(HERE/'runs/pointwise_compact_v1'),
    preflight=str(HERE/'runs/pointwise_compact_preflight_v1/verification.json'),
    loss='Logloss',positive_class_weight=3.,depths=[5,7],checkpoints=[600,1800,3600],
    features=495,train_rows={'15':569257,'16':713199},
    pointwise_pool_semantics_test_passed=True,generator_exact_replay_passed=True,
    inferred17_labels_read=False)
if experiments['sparse_cross_ranker_v1']['verification.json']:
    runtime['next_actions'].insert(0,'Sparse cross completed without gain; continue pointwise compact experiment, using genuine dates15/16 only')
interaction_queue_path = HERE/'categorical_interactions_queued_v1.json'
if interaction_queue_path.is_file():
    runtime['categorical_interactions_extension'] = json.loads(interaction_queue_path.read_text(encoding='utf8'))
    runtime['next_actions'].insert(0,'Finish pointwise session86885 and its independent verification; then run queued categorical interactions on genuine dates15/16 only')
quantile_queue_path = HERE/'probability_quantiles_queued_v1.json'
if quantile_queue_path.is_file():
    runtime['probability_quantile_extension'] = json.loads(quantile_queue_path.read_text(encoding='utf8'))
    runtime['next_actions'].insert(0,'Finish probability and categorical-interaction models and verify them; then compare label-free quantile matched ensembles on genuine dates15/16')
for experiment_name, extension_name in [('pointwise_compact_v1','pointwise_compact_extension'),
        ('categorical_interactions_v1','categorical_interactions_extension'),
        ('compose_probability_quantiles_v1','probability_quantile_extension')]:
    if extension_name in runtime:
        current = experiments[experiment_name]
        phase = ('verified' if current['verification.json'] else
                 'prediction_or_verification' if current['models_locked.json'] else
                 'fitting' if current['protocol.json'] else 'queued')
        runtime[extension_name]['phase'] = phase
        runtime[extension_name]['saved_checkpoints'] = len(list((HERE/'runs'/experiment_name).glob('*.cbm')))
previous = state.get('latest_goal_runtime')
if previous:
    state.setdefault('goal_runtime_history', []).append(previous)
state.update(latest_goal_runtime=runtime, snapshot_utc=runtime['timestamp_utc'], goal_status='active',
             current_verified15_auc=runtime['verified15_auc'], current_verified_best_auc=runtime['verified16_auc'],
             last_goal_turn_classification=classification, target_verified=runtime['targets_met'],
             live_session_ids_to_revalidate=(
                 ([51730] if not experiments['compact_quality_daily_catboost_v1']['verification.json'] else [])+
                 ([68568] if not experiments['quality_daily_neural_v1']['verification.json'] else [])+
                 ([14055] if not experiments['compose_compact_dense_transfer_v1']['verification.json'] else [])+
                 ([74453] if not experiments['within_user_ranker_v1']['verification.json'] else [])+
                 ([55933] if not day17_candidates[1].is_file() else [])+
                 ([84728] if not day17_candidates[2].is_file() else [])+
                 ([16660] if not experiments['catboost_within_ranker_v2']['verification.json'] else [])+
                 ([46629] if not experiments['daily_auc_pair_ranker_v1']['verification.json'] else [])+
                 ([5948] if not experiments['daily_binomial_rate_v1']['verification.json'] else [])+
                 ([68895] if not experiments['cross_user_pair_ranker_v1']['verification.json'] else [])+
                 ([40996] if not experiments['snapshot_pair_ranker_v1']['verification.json'] else [])+
                 ([41341] if not experiments['cold_task_ranker_v1']['verification.json'] else [])+
                 ([65699] if not experiments['firstday_task_ranker_v1']['verification.json'] else [])+
                 ([89225] if not experiments['task_audience_rate_v1']['verification.json'] else [])+
                 ([47460] if not experiments['task_audience_row_ranker_v1']['verification.json'] else [])+
                 ([4867] if not experiments['audience_graph_ranker_v1']['verification.json'] else [])+
                 ([11305] if not experiments['profile_response_ranker_v1']['verification.json'] else [])+
                 ([81614] if not experiments['compact_full_sample_ranker_v1']['verification.json'] else [])+
                 ([81406] if unordered_queue_path.is_file() and not experiments['unordered_snapshot_ranker_v1']['verification.json'] else [])+
                 ([85503] if mask_queue_path.is_file() and not experiments['compose_unordered_masks_v1']['verification.json'] else [])+
                 ([69815] if sparse_queue_path.is_file() and not experiments['sparse_cross_ranker_v1']['verification.json'] else [])+
                 ([86885] if not experiments['pointwise_compact_v1']['verification.json'] else [])+
                 ([16640] if interaction_queue_path.is_file() and not experiments['categorical_interactions_v1']['verification.json'] else [])+
                 ([91790] if quantile_queue_path.is_file() and not experiments['compose_probability_quantiles_v1']['verification.json'] else [])+
                 ([34079] if not day17_candidates[11].is_file() and any('export_firstday_task_v1.py' in p.get('CommandLine','') for p in processes) else [])+
                 ([25299] if not day17_candidates[10].is_file() and any('export_cold_task_v1.py' in p.get('CommandLine','') for p in processes) else [])+
                 ([32214] if not day17_candidates[9].is_file() and any('export_snapshot_pair_v1.py' in p.get('CommandLine','') for p in processes) else [])+
                 ([89934] if not day17_candidates[8].is_file() else [])+
                 ([64048] if not day17_candidates[5].is_file() else [])+
                 ([51829] if not day17_candidates[6].is_file() else [])+
                 ([16430] if not day17_candidates[8].is_file() and any('export_cross_user_refit17_v1.py' in p.get('CommandLine','') for p in processes) else [])+
                 ([16806] if not day17_candidates[4].is_file() else [])+
                 ([6681] if not day17_candidates[3].is_file() else [])))
status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(runtime, ensure_ascii=False, indent=2))
