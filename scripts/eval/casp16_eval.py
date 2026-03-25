# ==============================================================================
# casp16_eval.py — CASP16 local evaluation orchestrator
# ==============================================================================
# Flow (top to bottom; log banners EVAL_LOG_SEC_* in casp16_eval_constants.py):
#   - Optional full per-target pass if eval/local_eval.csv is missing: write local_eval.csv,
#     raw_data_for_pooling.pkl, and run pooled analysis once.
#   - ARC ensemble: residue cache, variance/mean CSV, all-combinations + summary + highlights,
#     Wilcoxon; load arc_ensemble_summary_all_comb.csv when present.
#   - Reload local_eval.csv; run pooled analysis from raw_data_for_pooling.pkl when present
#     (regenerates plots/tables/prints). Set ARC_SKIP_POOLED_RECOMPUTE=1 to reuse existing
#     pooled tables and only print the S2-style log when the stamp CSV exists.
#   - Z-scores, RS-local, CASP-style rank CSVs; stdout Table S1 layout from same CSVs as LaTeX S1.
# Shared implementations: casp16_eval_*.py in this directory.
#
# Run:  python /path/to/ARC/scripts/eval/casp16_eval.py
#   or: cd ARC && python scripts/eval/casp16_eval.py
# ==============================================================================

import os
import pickle
import sys
import warnings
from multiprocessing import Pool, cpu_count

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
_scripts_root = os.path.dirname(_script_dir)
_manuscript_dir = os.path.join(_scripts_root, "manuscript")
if _manuscript_dir not in sys.path:
    sys.path.insert(0, _manuscript_dir)

import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm

from casp16_eval_constants import (
    EVAL_LOG_SEC_1,
    EVAL_LOG_SEC_2,
    EVAL_LOG_SEC_2_7,
    EVAL_LOG_SEC_3,
    EVAL_LOG_SEC_4,
    EXCLUDED_LOCAL_EVAL_METHODS,
    USE_PARALLEL,
    targets,
)
from casp16_eval_data import df_local_stoch
from casp16_eval_io import target_chains_for_targets_safe
from casp16_eval_ensemble import (
    _run_all_comb_wilcoxon_tests,
    build_arc_residue_data_for_target,
    default_arc_ensemble_dir,
    ensemble_summary_all_comb_csv,
    run_arc_all_combinations,
    run_arc_variance_mean_analysis,
)
from casp16_eval_paths import (
    ARC_PREDICTIONS_CASP16,
    ARC_RESIDUE_TRUTH_PRED_PKL,
    FULL_PIPELINE_LOG,
    LOCAL_EVAL_CSV,
    LOCAL_RESULTS_DIR,
    PER_TARGET_ANALYSIS_DIR,
    POOLED_ANALYSIS_STAMP_CSV,
    POOLED_TABLES_DIR,
    RAW_POOLING_PKL,
    TRUE_IFACE_STATS_CSV,
    ensure_eval_output_layout,
    local_eval_csv_for_read,
    raw_pooling_pkl_for_read,
)
from casp16_eval_pooled import _print_pooled_analysis_results, run_pooled_analysis
from casp16_eval_target import process_target
from plots_manuscript_supplementary_tables import print_s1_style_log_summary

sys.stdout.reconfigure(line_buffering=True)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*concatenation with empty or all-NA entries.*",
)
# Pandas groupby/std/quantile on nearly constant series (e.g. ensemble ≈ individual).
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*Precision loss occurred in moment calculation due to catastrophic cancellation.*",
)

ensure_eval_output_layout()


def _assert_apollo_predictions_per_target() -> None:
    """Eval compares against packaged APOLLO local scores; require APOLLO/ for every CASP16 target."""
    assert os.path.isdir(ARC_PREDICTIONS_CASP16), (
        f"Missing predictions layout: expected directory {ARC_PREDICTIONS_CASP16}"
    )
    missing = [
        t
        for t in targets
        if not os.path.isdir(os.path.join(ARC_PREDICTIONS_CASP16, t, "APOLLO"))
    ]
    assert not missing, (
        "Eval requires outputs/predictions/CASP16/<target>/APOLLO/ for each target "
        f"(fetch predictions_apollo.tar.gz or run inference). Missing APOLLO for "
        f"{len(missing)} target(s): {', '.join(missing[:20])}"
        + (" ..." if len(missing) > 20 else "")
    )


_assert_apollo_predictions_per_target()

try:
    _log_file = open(FULL_PIPELINE_LOG, "w", encoding="utf-8")
    _log_file.write("ARC casp16_eval.py — full pipeline log\n")
    _log_file.write("Run started: {}\n\n".format(pd.Timestamp.now().isoformat()))
    _log_file.flush()
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr

    class _TeeOut:
        def __init__(self, stream, f):
            self._stream, self._f = stream, f

        def write(self, data):
            self._stream.write(data)
            self._stream.flush()
            if data:
                self._f.write(data)
                self._f.flush()

        def flush(self):
            self._stream.flush()
            self._f.flush()

        def __getattr__(self, name):
            return getattr(self._stream, name)

    sys.stdout = _TeeOut(_orig_stdout, _log_file)
    sys.stderr = _TeeOut(_orig_stderr, _log_file)
except Exception:
    _log_file = None


def _merge_target_run_into_aggregate(
    used_targets,
    loc_results,
    all_models,
    filtered_models,
    kicked_out_models,
    all_true_iface_stats,
    all_raw_data,
    target_result,
    stats,
    raw_data,
):
    if len(target_result["method"]) > 0:
        used_targets.update(target_result["target"])
    for key in loc_results:
        loc_results[key].extend(target_result[key])
    all_models.update(stats["all_models"])
    filtered_models.update(stats["filtered_models"])
    kicked_out_models.extend(stats["kicked_out_models"])
    all_true_iface_stats.extend(stats.get("true_iface_stats", []))
    for method, score_data in raw_data.items():
        if method not in all_raw_data:
            all_raw_data[method] = {}
        for score_type, df in score_data.items():
            if score_type not in all_raw_data[method]:
                all_raw_data[method][score_type] = []
            all_raw_data[method][score_type].append(df)


# ==============================================================================
# Per-target evaluation (skipped when eval/local_eval.csv already exists)
# ==============================================================================

if not os.path.exists(local_eval_csv_for_read()):
    all_models = set()
    filtered_models = set()
    kicked_out_models = []

    loc_results = {
        "method": [],
        "target": [],
        "score": [],
        "metric": [],
        "value": [],
    }

    all_true_iface_stats = []
    used_targets = set()
    all_raw_data = {}

    if USE_PARALLEL:
        num_workers = cpu_count()
        print(f"\n{'=' * 72}", flush=True)
        print(
            f"{EVAL_LOG_SEC_1}: {len(targets)} targets, {num_workers} workers "
            f"(logs below are ordered; no interleaved stdout).",
            flush=True,
        )
        print(f"{'=' * 72}\n", flush=True)

        with Pool(processes=num_workers) as pool:
            results_list = pool.map(process_target, targets)

        for tname, (target_result, stats, raw_data) in zip(targets, results_list):
            print(f"  [{tname}]", flush=True)
            for line in stats.get("_target_log_lines", []):
                print(f"      {line}", flush=True)
            print(flush=True)

        for target_result, stats, raw_data in results_list:
            _merge_target_run_into_aggregate(
                used_targets,
                loc_results,
                all_models,
                filtered_models,
                kicked_out_models,
                all_true_iface_stats,
                all_raw_data,
                target_result,
                stats,
                raw_data,
            )

        print(f"{'=' * 72}\n", flush=True)
    else:
        print(f"\n{'=' * 72}", flush=True)
        print(f"{EVAL_LOG_SEC_1}: {len(targets)} targets (sequential)", flush=True)
        print(f"{'=' * 72}\n", flush=True)
        for target in tqdm(list(targets)):
            target_result, stats, raw_data = process_target(target)
            print(f"  [{target}]", flush=True)
            for line in stats.get("_target_log_lines", []):
                print(f"      {line}", flush=True)
            print(flush=True)
            _merge_target_run_into_aggregate(
                used_targets,
                loc_results,
                all_models,
                filtered_models,
                kicked_out_models,
                all_true_iface_stats,
                all_raw_data,
                target_result,
                stats,
                raw_data,
            )

        print(f"{'=' * 72}\n", flush=True)

    if all_true_iface_stats:
        stats_df = pd.DataFrame(all_true_iface_stats)
        stats_df.to_csv(TRUE_IFACE_STATS_CSV, index=False)

    loc_results = pd.DataFrame(loc_results)
    loc_results.to_csv(LOCAL_EVAL_CSV, index=False)

    if len(all_raw_data) > 0:
        with open(RAW_POOLING_PKL, "wb") as f:
            pickle.dump(all_raw_data, f)
        print(f"Saved raw data for pooled analysis: {RAW_POOLING_PKL}")

    target_chains = target_chains_for_targets_safe(df_local_stoch, used_targets)

    if len(all_raw_data) > 0:
        run_pooled_analysis(all_raw_data, target_chains, LOCAL_RESULTS_DIR)
    else:
        print("Warning: No raw data collected for pooled analysis")


# ==============================================================================
# ARC ensemble: residue cache, all-combinations, Wilcoxon, summary CSV
# ==============================================================================

print("\n" + "=" * 80)
print(EVAL_LOG_SEC_2)
print("=" * 80)
ARC_ENSEMBLE_DIR = default_arc_ensemble_dir()
os.makedirs(ARC_ENSEMBLE_DIR, exist_ok=True)
arc_residue_data = {}
if os.path.exists(ARC_RESIDUE_TRUTH_PRED_PKL):
    try:
        with open(ARC_RESIDUE_TRUTH_PRED_PKL, "rb") as f:
            arc_residue_data = pickle.load(f)
        print(f"Loaded ARC residue data from {ARC_RESIDUE_TRUTH_PRED_PKL} ({len(arc_residue_data)} targets)")
    except Exception as e:
        print(f"Warning: Could not load ARC residue data: {e}")
if not arc_residue_data:
    for target in tqdm(targets, desc="Building ARC residue data"):
        data = build_arc_residue_data_for_target(target)
        if data:
            arc_residue_data[target] = data
    if arc_residue_data:
        with open(ARC_RESIDUE_TRUTH_PRED_PKL, "wb") as f:
            pickle.dump(arc_residue_data, f)
        print(f"Saved ARC residue data: {ARC_RESIDUE_TRUTH_PRED_PKL} ({len(arc_residue_data)} targets)")

if arc_residue_data:
    run_arc_variance_mean_analysis(arc_residue_data, output_dir=ARC_ENSEMBLE_DIR)
    _skip_all_comb = os.environ.get("ARC_SKIP_ALL_COMBINATIONS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if _skip_all_comb:
        print(
            f"\nSkipping ARC all-combinations (ARC_SKIP_ALL_COMBINATIONS=1). "
            "Using existing arc_all_combinations/*.csv and arc_ensemble_summary_all_comb.csv if present.",
            flush=True,
        )
    else:
        print(f"\n\nRunning ARC all-combinations (per target)...", flush=True)
        run_arc_all_combinations(
            arc_residue_data,
            output_dir=ARC_ENSEMBLE_DIR,
            force_run=False,
            log_file_path=None,
        )
    _run_all_comb_wilcoxon_tests(ARC_ENSEMBLE_DIR)

_SUMMARY_ALL_COMB_CSV = ensemble_summary_all_comb_csv(ARC_ENSEMBLE_DIR)

if os.path.exists(_SUMMARY_ALL_COMB_CSV):
    print("\n" + "-" * 80)
    print(EVAL_LOG_SEC_2_7)
    print("-" * 80)
    df_ensemble_summary = pd.read_csv(_SUMMARY_ALL_COMB_CSV)
    print(f"Loaded: {_SUMMARY_ALL_COMB_CSV}  ({len(df_ensemble_summary)} rows, {len(df_ensemble_summary.columns)} columns)")
else:
    df_ensemble_summary = None
    print(
        f"\nNo arc_ensemble_summary_all_comb.csv found "
        "(skip ensemble summary load; run all-combinations when needed)."
    )

# ==============================================================================
# Local results: reload eval CSV, pooling gate, rankings & stratification
# ==============================================================================

# ------------------------------------------------------------------------------
# Load local_eval.csv
# ------------------------------------------------------------------------------

try:
    loc_results = pd.read_csv(local_eval_csv_for_read())
    if len(loc_results) == 0:
        print("WARNING: CSV file is empty! No results were generated.")
        print("This likely means no models had valid local_truth files or all were filtered out.")
        exit(1)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

# ------------------------------------------------------------------------------
# Pooled analysis (ROC plots, tables, log): from pickle by default; optional skip
# ------------------------------------------------------------------------------

_force_pooled = os.environ.get("ARC_FORCE_POOLED_ANALYSIS", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
_skip_pooled_recompute = os.environ.get("ARC_SKIP_POOLED_RECOMPUTE", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
_stamp_pooled = os.path.exists(POOLED_ANALYSIS_STAMP_CSV)

if _skip_pooled_recompute and _stamp_pooled and not _force_pooled:
    print(
        f"Skipping pooled recomputation (ARC_SKIP_POOLED_RECOMPUTE=1); "
        "printing from existing pooled tables.",
        flush=True,
    )
    _print_pooled_analysis_results(POOLED_TABLES_DIR)
else:
    if _force_pooled and _stamp_pooled:
        print(
            f"ARC_FORCE_POOLED_ANALYSIS=1: re-running {EVAL_LOG_SEC_3}...",
            flush=True,
        )

    all_raw_data = {}
    if os.path.exists(raw_pooling_pkl_for_read()):
        try:
            with open(raw_pooling_pkl_for_read(), "rb") as f:
                all_raw_data = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load raw data: {e}")

    target_chains = target_chains_for_targets_safe(df_local_stoch, set(loc_results["target"].unique()))

    if len(all_raw_data) > 0:
        run_pooled_analysis(all_raw_data, target_chains, LOCAL_RESULTS_DIR)
    elif _stamp_pooled:
        print(
            f"Warning: raw pooling pickle missing or empty; "
            "printing from existing pooled tables if present.",
            flush=True,
        )
        _print_pooled_analysis_results(POOLED_TABLES_DIR)

print(f"Selected targets (rows in local_eval): {len(loc_results['target'].unique())}")

loc_results = loc_results[~loc_results["method"].isin(EXCLUDED_LOCAL_EVAL_METHODS)]

# ------------------------------------------------------------------------------
# Z-scores (per target × metric × score)
# ------------------------------------------------------------------------------
print("#" * 15)
loc_results["z-score"] = loc_results.groupby(["target", "metric", "score"])["value"].transform(zscore).copy()
loc_results.loc[loc_results["metric"] == "RL", "z-score"] = -loc_results.loc[loc_results["metric"] == "RL", "z-score"]
loc_results.loc[loc_results["z-score"] < 0, "z-score"] = 0

# ------------------------------------------------------------------------------
# RS-local composite from z-scores
# ------------------------------------------------------------------------------
rs_rows = []

for group, df in loc_results.groupby(["target", "method"]):
    df["cat"] = df["score"] + ":" + df["metric"]
    zscores = df.pivot(index="target", columns="cat", values="z-score").loc[group[0]].to_dict()

    rs_tot = 0
    # RS-local as in main.tex: sum over local CAD, local lDDT, Patch QS, Patch DockQ only.
    for score in ["local_cad", "local_lddt", "patch_qs", "patch_dockq"]:
        rs = (
            0.5 * zscores[f"{score}:pearson"]
            + 0.5 * zscores[f"{score}:spearman"]
            + zscores[f"{score}:adaptive_rocauc"]
        )
        rs_tot += rs
        rs_rows += [
            {
                "method": group[1],
                "target": group[0],
                "score": score,
                "metric": "RS-local",
                "value": None,
                "z-score": rs,
            }
        ]

    rs_rows += [
        {
            "method": group[1],
            "target": group[0],
            "score": "all",
            "metric": "RS-local",
            "value": None,
            "z-score": rs_tot,
        }
    ]

    if "true_interface_residue:aucroc" in zscores and "true_interface_residue:aucprc" in zscores:
        rs_rows += [
            {
                "method": group[1],
                "target": group[0],
                "score": "true_interface_residue",
                "metric": "AUCroc-z + AUCprc-z",
                "value": None,
                "z-score": zscores["true_interface_residue:aucroc"] + zscores["true_interface_residue:aucprc"],
            }
        ]

loc_results = pd.concat([loc_results, pd.DataFrame(rs_rows)], ignore_index=True)

# ------------------------------------------------------------------------------
# Rank percentiles (per target × metric × score)
# ------------------------------------------------------------------------------
loc_results["rank_pct"] = loc_results.groupby(["target", "metric", "score"])["value"].rank(
    ascending=False, method="max", pct=True
)
loc_results.loc[loc_results["metric"] == "RL", "rank_pct"] = 1 - loc_results.loc[loc_results["metric"] == "RL", "rank_pct"]

# ------------------------------------------------------------------------------
# All-targets aggregates and CASP-style rank CSVs (per_target_analysis/)
# ------------------------------------------------------------------------------

print("\n" + "=" * 80)
print(EVAL_LOG_SEC_4)
print("=" * 80)

loc_results_full = loc_results.copy()
strat_data = loc_results_full.copy()
strat_agg = strat_data.groupby(["method", "metric", "score"]).mean(numeric_only=True).reset_index()

for cat, df in strat_agg.groupby(["metric", "score"]):
    df["rank-zscore"] = df["z-score"].rank(ascending=False, method="dense")
    df["rank-rank_pct"] = df["rank_pct"].rank(ascending=True, method="dense")
    df["rank-value"] = df["value"].rank(ascending=cat[0] == "RL", method="dense")
    mask = (strat_agg["metric"] == cat[0]) & (strat_agg["score"] == cat[1])
    strat_agg.loc[mask, "rank-zscore"] = df["rank-zscore"].values
    strat_agg.loc[mask, "rank-rank_pct"] = df["rank-rank_pct"].values
    strat_agg.loc[mask, "rank-value"] = df["rank-value"].values

all_t = strat_agg

os.makedirs(PER_TARGET_ANALYSIS_DIR, exist_ok=True)

if len(all_t) > 0:
    rs_local = all_t[(all_t["metric"] == "RS-local") & (all_t["score"] == "all")][
        ["method", "z-score", "rank-zscore"]
    ].copy()
    tir = all_t[(all_t["metric"] == "AUCroc-z + AUCprc-z") & (all_t["score"] == "true_interface_residue")][
        ["method", "z-score", "rank-zscore"]
    ].copy()
    if len(rs_local) == 0 or len(tir) == 0:
        print(f"  Warning: Missing RS-local or true-interface rank data; skipping casp_style_RS_local_vs_true_interface_rank.csv")
    else:
        rs_local = rs_local.rename(columns={"z-score": "RS_local_zscore", "rank-zscore": "RS_local_rank"})
        tir = tir.rename(columns={"z-score": "TIR_zscore", "rank-zscore": "TIR_rank"})
        casp_rank_merge = rs_local.merge(tir, on="method", how="outer")
        casp_rank_merge = casp_rank_merge.sort_values("RS_local_rank", na_position="last")
        for c in ["RS_local_rank", "TIR_rank"]:
            if c in casp_rank_merge.columns:
                casp_rank_merge[c] = casp_rank_merge[c].apply(lambda x: int(x) if pd.notna(x) else np.nan)
        casp_rank_path = os.path.join(PER_TARGET_ANALYSIS_DIR, "casp_style_RS_local_vs_true_interface_rank.csv")
        casp_rank_merge.to_csv(casp_rank_path, index=False)
        print(f"  Saved CASP-style RS-local vs true-interface rank table: {casp_rank_path}")

    tir_roc = all_t[(all_t["metric"] == "aucroc") & (all_t["score"] == "true_interface_residue")][
        ["method", "z-score", "rank-zscore"]
    ].copy()
    tir_pr = all_t[(all_t["metric"] == "aucprc") & (all_t["score"] == "true_interface_residue")][
        ["method", "z-score", "rank-zscore"]
    ].copy()
    if len(tir_roc) > 0 and len(tir_pr) > 0:
        tir_roc = tir_roc.rename(columns={"z-score": "TIR_ROC_AUC_zscore", "rank-zscore": "TIR_ROC_AUC_rank"})
        tir_pr = tir_pr.rename(columns={"z-score": "TIR_PR_AUC_zscore", "rank-zscore": "TIR_PR_AUC_rank"})
        tir_per_metric = tir_roc.merge(tir_pr[["method", "TIR_PR_AUC_zscore", "TIR_PR_AUC_rank"]], on="method", how="outer")
        for c in ["TIR_ROC_AUC_rank", "TIR_PR_AUC_rank"]:
            tir_per_metric[c] = tir_per_metric[c].apply(lambda x: int(x) if pd.notna(x) else np.nan)
        tir_per_metric_path = os.path.join(PER_TARGET_ANALYSIS_DIR, "casp_style_TIR_ROC_vs_PR_rank.csv")
        tir_per_metric.to_csv(tir_per_metric_path, index=False)
        print(f"  Saved CASP-style TIR ROC vs PR rank table: {tir_per_metric_path}")

print("\n" + "=" * 80)
print("CASP-style rank CSVs complete (per_target_analysis/).")
print("=" * 80 + "\n")

# ------------------------------------------------------------------------------
# Final aggregate over targets (mean) — used by downstream tools; log mirrors Table S1.
# ------------------------------------------------------------------------------
_n_targets_for_s1_log = int(loc_results["target"].nunique())
loc_results = loc_results.groupby(["method", "metric", "score"]).mean(numeric_only=True).reset_index()

print_s1_style_log_summary(PER_TARGET_ANALYSIS_DIR, n_targets=_n_targets_for_s1_log)
