"""ARC repository paths for `casp16_eval.py` and manuscript figure scripts.

Eval and manuscript artifacts from this repository's scripts default to a single tree under
``ARC/outputs/results/`` (see layout below). Optional ``CASP16_EVAL_OUTPUT_ROOT`` must
resolve to a directory **inside** ``ARC_ROOT`` (otherwise the default is used).

Environment overrides (optional):
  CASP16_EVAL_OUTPUT_ROOT - root for eval outputs; must lie under the ARC repo
                            (default: ``ARC/outputs/results``).
  CASP16_RAW_ROOT - CASP16 multimer QA tree: info.json, QA_1/, local_preds/, ...
                    (default: ARC/data/raw_16).
  ARC_SKIP_ALL_COMBINATIONS - if 1/true/yes, skip recomputing
                    ``arc_all_combinations/<target>.csv`` (use existing cache).
  ARC_SKIP_POOLED_RECOMPUTE - if 1/true/yes, skip re-pooling from
                    ``raw_data_for_pooling.pkl`` when pooled tables already exist;
                    only prints the pooled log from on-disk CSVs (old fast path).
  ARC_FORCE_POOLED_ANALYSIS - if 1/true/yes, re-run pooled analysis even when
                    ``ARC_SKIP_POOLED_RECOMPUTE=1`` (override). Otherwise the default
                    is to regenerate pooled plots/tables whenever the raw pooling
                    pickle is present.
  ARC_POOLED_DATA_CACHE_BYPASS - if 1/true/yes, do not load/save
                    ``pooled_analysis/pooled_concat_cache.pkl`` (always concat from
                    the in-memory raw dict). Does not skip regenerating pooled PNG/CSVs.
  ARC_STRATIFIED_METRICS_CACHE_BYPASS - if 1/true/yes, do not load/save
                    ``pooled_analysis/stratified_metrics_full_cache.pkl`` (always run
                    ``compute_all_stratified_metrics`` for the full pooled pass).
                    Does not skip regenerating pooled PNG/CSVs.

Output layout (default ``EVAL_OUTPUT_ROOT``)
--------------------------------------------
::

  outputs/results/                      <- EVAL_OUTPUT_ROOT
    eval/                               <- EVAL_RUN_DIR
      local_eval.csv
      true_iface_stats_detailed.csv
      raw_data_for_pooling.pkl
    logs/
      casp16_eval.log
    local_results/
      cache/
        local_residue_quantile_stats.pkl
      per_target_analysis/
      pooled_analysis/
      arc_ensemble/
      manuscript_figures/
      manuscript_tables/

**Mandatory inputs** (see module docstring; casp16_eval_data loads CSVs + raw root):

  * ``EMA_LOCAL_STOCH_CSV`` - stoichiometry-filtered assessor local scores.
  * ``CASP_MODEL_SCORES_CSV`` - model-level CASP table (QSBEST filter, correlations).
  * ``raw`` / ``CASP16_RAW_ROOT`` - per-target QA and cached ``local_preds/*.json``.
  * ``ARC_PREDICTIONS_CASP16`` - packaged GNN ``LOCAL.json`` under ``ARC/`` (and ``APOLLO/``) per target;
    ``casp16_eval.py`` asserts ``<target>/APOLLO`` exists for every id in ``targets``.
  * ``casp16_eval_constants.CASP16_EMA_RESULTS`` - ``{model}_{target}.json`` assessor outputs.
  * ``CASP16_APPROX_TARGET_SIZES_JSON`` - ``approx_target_size`` (used by ensemble summaries).

**Intermediate artifacts** (under ``LOCAL_RESULTS_DIR`` and ``EVAL_RUN_DIR``):

  * ``LOCAL_EVAL_CSV``, ``TRUE_IFACE_STATS_CSV`` - long-form metrics + residue stats.
  * ``RAW_POOLING_PKL`` - cached dict for pooled analysis (skip re-aggregation if present).
  * ``LOCAL_RESULTS_DIR`` - pooled tables, per-target analysis, ensemble cache, manuscript.
  * ``PER_TARGET_ANALYSIS_DIR`` - CASP-style rank CSVs for supplementary S1.
  * ``POOLED_ANALYSIS_STAMP_CSV`` - sentinel for a completed pooled pass; stratified tables under ``pooled_analysis/tables/``.
  * ``pooled_analysis/pooled_concat_cache.pkl`` - optional reuse of target-concatenated frames (invalidated when ``RAW_POOLING_PKL`` mtime changes).
  * ``pooled_analysis/stratified_metrics_full_cache.pkl`` - optional reuse of full stratified metric dict (mtime + target-chain fingerprint; not used for ``figure2_roc_only`` slim passes).
  * ``ARC_ENSEMBLE_DIR`` - residue PKL/CSVs, variance plots, all-combinations CSVs.

**Manuscript deliverables** (under ``LOCAL_RESULTS_DIR``):

  * ``MANUSCRIPT_FIGURES_DIR`` - PDF/PNG + CSV/JSON from ``plots_manuscript_figures.py``.
  * ``MANUSCRIPT_TABLES_DIR`` - LaTeX ``table_supplementary_S1.tex`` ... ``S4.tex``
    (S3/S4 written when all-comb summary has L=6 rows).

**Logs:**

  * ``FULL_PIPELINE_LOG`` - ``casp16_eval.py`` tees stdout/stderr here (truncate on each orchestrator run).
    ``plots_manuscript_figures.py`` / ``plots_manuscript_supplementary_tables.py`` **append** the same file
    when run as main (see ``eval_log_tee.py``); set ``ARC_NO_EVAL_LOG_TEE=1`` to skip manuscript tee.
"""
from __future__ import annotations

import os
import warnings

_script_dir = os.path.dirname(os.path.abspath(__file__))
_scripts_dir = os.path.dirname(_script_dir)
ARC_ROOT = os.path.dirname(_scripts_dir)
ARC_DATA_DIR = os.path.join(ARC_ROOT, "data")
ARC_OUTPUTS_DIR = os.path.join(ARC_ROOT, "outputs")
ARC_RESULTS_ROOT = os.path.join(ARC_OUTPUTS_DIR, "results")

# -----------------------------------------------------------------------------
# Mandatory inputs (see module docstring; casp16_eval_data loads CSVs + raw root)
# -----------------------------------------------------------------------------
EMA_LOCAL_STOCH_CSV = os.path.join(ARC_DATA_DIR, "ema_local_scores_with_lddt_added_mdl_contacts.csv")
CASP_MODEL_SCORES_CSV = os.path.join(ARC_DATA_DIR, "casp_model_scores.csv")
RAW_16_DEFAULT = os.path.join(ARC_DATA_DIR, "raw_16")

ARC_PREDICTIONS_CASP16 = os.path.join(ARC_OUTPUTS_DIR, "predictions", "CASP16")
CASP16_APPROX_TARGET_SIZES_JSON = os.path.join(ARC_DATA_DIR, "casp16_approx_target_sizes.json")

CASP16_EMA_REFERENCE_RESULTS_DIR = os.path.join(ARC_DATA_DIR, "casp16_ema_reference_results")


def _resolve_eval_output_root() -> str:
    """Default ``ARC/outputs/results``; env override must resolve under ``ARC_ROOT``."""
    arc_rp = os.path.realpath(ARC_ROOT)
    default = os.path.realpath(ARC_RESULTS_ROOT)
    env = os.environ.get("CASP16_EVAL_OUTPUT_ROOT", "").strip()
    if not env:
        return default
    candidate = os.path.abspath(env if os.path.isabs(env) else os.path.join(ARC_ROOT, env))
    cand_rp = os.path.realpath(candidate)
    try:
        common = os.path.commonpath([arc_rp, cand_rp])
    except ValueError:
        warnings.warn(
            f"CASP16_EVAL_OUTPUT_ROOT {env!r} is not under the ARC repository; "
            f"using default {default!r}.",
            UserWarning,
            stacklevel=2,
        )
        return default
    if common != arc_rp:
        warnings.warn(
            f"CASP16_EVAL_OUTPUT_ROOT {env!r} is outside ARC_ROOT; using default {default!r}.",
            UserWarning,
            stacklevel=2,
        )
        return default
    return cand_rp


EVAL_OUTPUT_ROOT = _resolve_eval_output_root()

# -----------------------------------------------------------------------------
# Run-level artifacts (grouped under eval/ + logs/)
# -----------------------------------------------------------------------------
EVAL_RUN_DIR = os.path.join(EVAL_OUTPUT_ROOT, "eval")
EVAL_LOGS_DIR = os.path.join(EVAL_OUTPUT_ROOT, "logs")

LOCAL_EVAL_CSV = os.path.join(EVAL_RUN_DIR, "local_eval.csv")
TRUE_IFACE_STATS_CSV = os.path.join(EVAL_RUN_DIR, "true_iface_stats_detailed.csv")
RAW_POOLING_PKL = os.path.join(EVAL_RUN_DIR, "raw_data_for_pooling.pkl")
FULL_PIPELINE_LOG = os.path.join(EVAL_LOGS_DIR, "casp16_eval.log")

# -----------------------------------------------------------------------------
# local_results/ tree
# -----------------------------------------------------------------------------
LOCAL_RESULTS_DIR = os.path.join(EVAL_OUTPUT_ROOT, "local_results")
LOCAL_RESULTS_CACHE_DIR = os.path.join(LOCAL_RESULTS_DIR, "cache")
LOCAL_RESIDUE_QUANTILE_STATS_PKL = os.path.join(LOCAL_RESULTS_CACHE_DIR, "local_residue_quantile_stats.pkl")

# Presence of this file means a full pooled pass completed (tables + fig.~2 ROC PNGs).
POOLED_ANALYSIS_STAMP_CSV = os.path.join(
    LOCAL_RESULTS_DIR,
    "pooled_analysis",
    "tables",
    "all_targets",
    "table_true_interface_residue_performance.csv",
)
POOLED_TABLES_DIR = os.path.join(LOCAL_RESULTS_DIR, "pooled_analysis", "tables")
PER_TARGET_ANALYSIS_DIR = os.path.join(LOCAL_RESULTS_DIR, "per_target_analysis")
ARC_ENSEMBLE_DIR = os.path.join(LOCAL_RESULTS_DIR, "arc_ensemble")
ARC_RESIDUE_TRUTH_PRED_PKL = os.path.join(ARC_ENSEMBLE_DIR, "arc_residue_truth_pred.pkl")

MANUSCRIPT_FIGURES_DIR = os.path.join(LOCAL_RESULTS_DIR, "manuscript_figures")
MANUSCRIPT_TABLES_DIR = os.path.join(LOCAL_RESULTS_DIR, "manuscript_tables")


def ensure_eval_output_layout() -> None:
    """Create output directories before writing logs, CSVs, or pickles."""
    os.makedirs(EVAL_RUN_DIR, exist_ok=True)
    os.makedirs(EVAL_LOGS_DIR, exist_ok=True)
    os.makedirs(LOCAL_RESULTS_CACHE_DIR, exist_ok=True)
    os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
    os.makedirs(ARC_ENSEMBLE_DIR, exist_ok=True)
    os.makedirs(PER_TARGET_ANALYSIS_DIR, exist_ok=True)
    os.makedirs(MANUSCRIPT_FIGURES_DIR, exist_ok=True)
    os.makedirs(MANUSCRIPT_TABLES_DIR, exist_ok=True)


def _resolve_read_path(new_path: str, legacy_flat_name: str) -> str:
    """Prefer new layout; else same file name directly under ``EVAL_OUTPUT_ROOT`` (pre-reorg)."""
    legacy = os.path.join(EVAL_OUTPUT_ROOT, legacy_flat_name)
    if os.path.isfile(new_path):
        return new_path
    if os.path.isfile(legacy):
        return legacy
    return new_path


def local_eval_csv_for_read() -> str:
    return _resolve_read_path(LOCAL_EVAL_CSV, "local_eval.csv")


def true_iface_stats_csv_for_read() -> str:
    return _resolve_read_path(TRUE_IFACE_STATS_CSV, "true_iface_stats_detailed.csv")


def raw_pooling_pkl_for_read() -> str:
    return _resolve_read_path(RAW_POOLING_PKL, "raw_data_for_pooling.pkl")


def local_residue_quantile_stats_pkl_for_read() -> str:
    legacy = os.path.join(EVAL_OUTPUT_ROOT, "local_residue_quantile_stats.pkl")
    if os.path.isfile(LOCAL_RESIDUE_QUANTILE_STATS_PKL):
        return LOCAL_RESIDUE_QUANTILE_STATS_PKL
    if os.path.isfile(legacy):
        return legacy
    return LOCAL_RESIDUE_QUANTILE_STATS_PKL
