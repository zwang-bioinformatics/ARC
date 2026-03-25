"""Pooled analysis across targets: target-balanced metrics, six main-text ROC PNGs (fig.~2), CSV tables.

Optional pickles reuse **numeric** prep (cross-target concat, stratified metrics) when inputs are unchanged.
``run_pooled_analysis`` always **overwrites** fig.~2 PNGs, performance CSVs, and rankings—nothing skips matplotlib
or table writers.
"""
import json
import os
import pickle as _pickle
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

from casp16_eval_constants import EVAL_LOG_SEC_3, EVAL_LOG_SEC_3C, STRAT_DISPLAY
from sklearn.utils.class_weight import compute_sample_weight

_script_dir = os.path.dirname(os.path.abspath(__file__))
_scripts_root = os.path.dirname(_script_dir)
_manuscript_dir = os.path.join(_scripts_root, "manuscript")
if _manuscript_dir not in sys.path:
    sys.path.insert(0, _manuscript_dir)

# ==============================================================================
# Pooled analysis - pool targets, metrics, fig.~2 ROC PNGs, CSV tables
# ==============================================================================

# ----------------------------------------------------------------------------
# Pool raw per-target frames into stratified pooled sets
# ----------------------------------------------------------------------------

def _common_target_model_pairs(all_raw_data, methods):
    """
    Compute (target, model) pairs that appear in every one of the given methods.
    Used so pooled analysis only includes targets and models predicted by all methods.
    Returns None if any method's data lacks a "model" column (e.g. old pickle).
    """
    if not methods:
        return None
    pairs_per_method = []
    for method in methods:
        if method not in all_raw_data:
            return None
        pairs = set()
        for score_type, df_list in all_raw_data[method].items():
            for df in df_list:
                if df is None or len(df) == 0:
                    continue
                if "target" not in df.columns or "model" not in df.columns:
                    return None
                pairs.update(zip(df["target"], df["model"]))
        pairs_per_method.append(pairs)
    if not pairs_per_method:
        return None
    common = set.intersection(*pairs_per_method)
    return common


def pool_data_across_targets(all_raw_data, common_target_model_pairs=None):
    """
    Pool data across targets for each method.
    If common_target_model_pairs is provided, only rows with (target, model) in that set
    are kept, so pooled metrics are comparable across methods (same targets and models).
    
    Args:
        all_raw_data: Dictionary {method: {score_type: list of DataFrames}}
        common_target_model_pairs: Optional set of (target, model) tuples to keep; if None, no filter.
    
    Returns:
        Dictionary {method: {score_type: pooled DataFrame}}
    """
    pooled_data = {}
    for method, score_data in all_raw_data.items():
        pooled_data[method] = {}
        for score_type, df_list in score_data.items():
            if len(df_list) > 0:
                # Concatenate all DataFrames across targets
                pooled_df = pd.concat(df_list, ignore_index=True)
                # Remove rows with NaN predictions
                pooled_df = pooled_df[pooled_df["pred"].notna()]
                # Restrict to (target, model) pairs predicted by all methods (after default method filter)
                if common_target_model_pairs is not None and "target" in pooled_df.columns and "model" in pooled_df.columns:
                    tm_tuples = list(zip(pooled_df["target"], pooled_df["model"]))
                    mask = [tm in common_target_model_pairs for tm in tm_tuples]
                    pooled_df = pooled_df[mask].copy()
                if len(pooled_df) > 0:
                    pooled_data[method][score_type] = pooled_df
    return pooled_data


# ----------------------------------------------------------------------------
# Weighted metrics and ROC/PR curve points per stratum
# ----------------------------------------------------------------------------

def _pooled_sample_weight_and_metrics(clean_df, truth_col, pred_col):
    """
    For pooled analysis: compute sample weights so each target contributes equally,
    and weighted PCC, SCC, MAE. We always use weighted metrics; asserts if "target"
    is missing or fewer than 2 targets (required for target-balanced weighting).
    clean_df must have truth_col, pred_col, and "target" with at least 2 unique targets.
    Returns: (sample_weight, pearson, spearman, mae).
    """
    # Pooled analysis always uses target-balanced weights; require valid target info.
    assert "target" in clean_df.columns, (
        "Pooled metrics: 'target' column missing; weighted metrics required."
    )
    assert clean_df["target"].nunique() >= 2, (
        "Pooled metrics: need at least 2 targets for target-balanced weighting; got "
        f"{clean_df['target'].nunique()}."
    )
    y_target = pd.Categorical(clean_df["target"]).codes
    sample_weight = compute_sample_weight("balanced", y_target)

    # Weighted Pearson
    x = np.asarray(clean_df[truth_col], dtype=float)
    y = np.asarray(clean_df[pred_col], dtype=float)
    w = np.asarray(sample_weight, dtype=float)
    w = w / w.sum()
    xbar = np.sum(w * x)
    ybar = np.sum(w * y)
    cov = np.sum(w * (x - xbar) * (y - ybar))
    var_x = np.sum(w * (x - xbar) ** 2)
    var_y = np.sum(w * (y - ybar) ** 2)
    if var_x > 0 and var_y > 0:
        pearson = cov / np.sqrt(var_x * var_y)
    else:
        pearson = np.nan

    # Weighted Spearman: rank then weighted Pearson on ranks
    rx = pd.Series(x).rank(method="average").values
    ry = pd.Series(y).rank(method="average").values
    rxbar = np.sum(w * rx)
    rybar = np.sum(w * ry)
    cov_r = np.sum(w * (rx - rxbar) * (ry - rybar))
    var_rx = np.sum(w * (rx - rxbar) ** 2)
    var_ry = np.sum(w * (ry - rybar) ** 2)
    if var_rx > 0 and var_ry > 0:
        spearman = cov_r / np.sqrt(var_rx * var_ry)
    else:
        spearman = np.nan

    # Weighted MAE
    mae = np.average(np.abs(x - y), weights=w)

    return sample_weight, pearson, spearman, mae


def get_binary_threshold(score_type):
    """
    Get the threshold for converting a continuous metric to binary.
    
    Args:
        score_type: Name of the score type
    
    Returns:
        Threshold value or None if using adaptive quartile
    """
    thresholds = {
        "local_lddt": 0.236,  # iLDDT >= 0.236
        "patch_dockq": 0.23,  # DockQ >= 0.23
    }
    return thresholds.get(score_type, None)


def convert_to_binary(df, truth_col, threshold=None):
    """
    Convert continuous truth values to binary using threshold or adaptive quartile.
    
    Args:
        df: DataFrame with truth column
        truth_col: Name of truth column
        threshold: Fixed threshold value, or None for adaptive quartile
    
    Returns:
        Series of binary values (0/1 or False/True)
    """
    if threshold is not None:
        # Use fixed threshold
        return (df[truth_col] >= threshold).astype(int)
    else:
        # Use adaptive quartile (75th percentile as threshold)
        threshold_value = df[truth_col].quantile(0.75)
        return (df[truth_col] >= threshold_value).astype(int)


def compute_pooled_metrics_stratified_scalars(df, truth_col, pred_col="pred"):
    """PCC, SCC, MAE, ROC AUC, PR AUC for stratified pooled tables (no curve coordinate arrays)."""
    cols = [truth_col, pred_col]
    if "target" in df.columns:
        cols = [truth_col, pred_col, "target"]
    clean_df = df[cols].dropna()

    if len(clean_df) == 0:
        return None

    sample_weight, pearson, spearman, mae = _pooled_sample_weight_and_metrics(
        clean_df, truth_col, pred_col
    )

    roc_auc = None
    pr_auc = None

    if truth_col == "true_interface_residue":
        binary_truth = clean_df[truth_col].astype(int)
    else:
        threshold = get_binary_threshold(truth_col)
        binary_truth = convert_to_binary(clean_df, truth_col, threshold)

    kw = {}
    if sample_weight is not None:
        kw["sample_weight"] = sample_weight

    if binary_truth.nunique() >= 2:
        try:
            roc_auc = roc_auc_score(binary_truth, clean_df[pred_col], **kw)
            pr_prec, pr_rec, _ = precision_recall_curve(
                binary_truth, clean_df[pred_col], **kw
            )
            pr_auc = auc(pr_rec, pr_prec)
        except Exception:
            pass

    return {
        "pcc": pearson,
        "scc": spearman,
        "MAE": mae,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "n_samples": len(clean_df),
    }


def compute_pooled_metrics_with_roc_curves(df, truth_col, pred_col="pred"):
    """Same as stratified scalars plus ``fpr``/``tpr`` for main-text fig.~2 ROC line plots."""
    cols = [truth_col, pred_col]
    if "target" in df.columns:
        cols = [truth_col, pred_col, "target"]
    clean_df = df[cols].dropna()

    if len(clean_df) == 0:
        return None

    sample_weight, pearson, spearman, mae = _pooled_sample_weight_and_metrics(
        clean_df, truth_col, pred_col
    )

    roc_auc = None
    pr_auc = None
    fpr = None
    tpr = None

    if truth_col == "true_interface_residue":
        binary_truth = clean_df[truth_col].astype(int)
    else:
        threshold = get_binary_threshold(truth_col)
        binary_truth = convert_to_binary(clean_df, truth_col, threshold)

    kw = {}
    if sample_weight is not None:
        kw["sample_weight"] = sample_weight

    if binary_truth.nunique() >= 2:
        try:
            fpr, tpr, _ = roc_curve(binary_truth, clean_df[pred_col], **kw)
            roc_auc = roc_auc_score(binary_truth, clean_df[pred_col], **kw)
            pr_prec, pr_rec, _ = precision_recall_curve(
                binary_truth, clean_df[pred_col], **kw
            )
            pr_auc = auc(pr_rec, pr_prec)
        except Exception:
            pass

    return {
        "pcc": pearson,
        "scc": spearman,
        "MAE": mae,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "fpr": fpr,
        "tpr": tpr,
        "n_samples": len(clean_df),
    }


_FIG2_ROC_STRATA = ("All Targets", "Dimer only")
_FIG2_ROC_SCORES = ("patch_qs", "patch_dockq", "true_interface_residue")
FIG2_ROC_METRICS_CACHE_FILENAME = "stratified_metrics_fig2_roc.pkl"

# Output of ``pool_data_across_targets`` (large). Bump schema when concat/filter rules change.
POOLED_DATA_CACHE_FILENAME = "pooled_concat_cache.pkl"
POOLED_DATA_CACHE_SCHEMA = 1

# Full ``compute_all_stratified_metrics`` result (figure2_roc_only=False). Bump schema when
# stratifications, score_types, or metric functions change.
STRATIFIED_METRICS_FULL_CACHE_FILENAME = "stratified_metrics_full_cache.pkl"
STRATIFIED_METRICS_CACHE_SCHEMA = 1


def _target_chains_fingerprint(target_chains: dict) -> tuple:
    """Stable tuple so cache invalidates if dimer/multimer strata inputs change."""
    return tuple(sorted((str(k), int(v)) for k, v in target_chains.items()))


def compute_all_stratified_metrics(
    pooled_data,
    stratifications,
    score_types,
    filtered_methods=None,
    *,
    figure2_roc_only: bool = False,
):
    """
    Compute all metrics (PCC, SCC, MAE, ROC AUC, PR AUC) for all stratifications.
    This avoids duplicate computation when creating tables and plots.

    Weighting is the same as unstratified pooled analysis: within each stratum we use
    target-balanced weights (each target in that stratum contributes equally). We only
    filter rows by the stratum. Per cell: ``compute_pooled_metrics_stratified_scalars``
    for tables; ``compute_pooled_metrics_with_roc_curves`` only for main-text fig.~2
    (patch_qs, patch_dockq, true_interface_residue x All Targets and Dimer only).

    If ``figure2_roc_only`` is True, only those two strata and three score types are
    processed and every cell uses ``compute_pooled_metrics_with_roc_curves`` (fast path
    for ``replot_figure2_roc_curves_only``).

    Args:
        pooled_data: Dictionary {method: {score_type: DataFrame}}
        stratifications: List of (name, target_filter) tuples
        score_types: List of score types to compute metrics for
        filtered_methods: Set of method names to include (None = all methods)

    Returns:
        Nested dict ``strat_name -> method -> score_type -> metrics``. Every cell has
        ``pcc``, ``scc``, ``MAE``, ``ROC_AUC``, ``PR_AUC``, ``n_samples``. Cells for
        fig.~2 (``patch_qs``, ``patch_dockq``, ``true_interface_residue`` under
        ``All Targets`` or ``Dimer only``) additionally include ``fpr`` and ``tpr``.
    """
    if figure2_roc_only:
        stratifications = [s for s in stratifications if s[0] in _FIG2_ROC_STRATA]
        score_types = [st for st in score_types if st in _FIG2_ROC_SCORES]

    stratified_results = {}

    for strat_name, target_filter in stratifications:
        stratified_results[strat_name] = {}

        for method in pooled_data.keys():
            if filtered_methods is not None and method not in filtered_methods:
                continue

            stratified_results[strat_name][method] = {}

            for score_type in score_types:
                if score_type not in pooled_data[method]:
                    continue

                df = pooled_data[method][score_type].copy()
                if "target" in df.columns:
                    df = df[df["target"].apply(target_filter)]

                if len(df) == 0:
                    continue

                if figure2_roc_only:
                    metrics = compute_pooled_metrics_with_roc_curves(df, score_type)
                elif strat_name in _FIG2_ROC_STRATA and score_type in _FIG2_ROC_SCORES:
                    metrics = compute_pooled_metrics_with_roc_curves(df, score_type)
                else:
                    metrics = compute_pooled_metrics_stratified_scalars(df, score_type)
                if metrics:
                    stratified_results[strat_name][method][score_type] = metrics

    return stratified_results


def _figure2_roc_cache_path(output_base_dir: str) -> str:
    return os.path.join(output_base_dir, "pooled_analysis", FIG2_ROC_METRICS_CACHE_FILENAME)


def _extract_figure2_roc_plot_inputs(stratified_metrics: dict) -> dict:
    """Narrow nested dict to only what ``plot_figure2_roc_curves`` reads (fpr/tpr/ROC_AUC)."""
    out = {}
    for strat in _FIG2_ROC_STRATA:
        src = stratified_metrics.get(strat)
        if not src:
            continue
        out[strat] = {}
        for method, md in src.items():
            out[strat][method] = {}
            for st in _FIG2_ROC_SCORES:
                if st not in md:
                    continue
                m = md[st]
                if m.get("fpr") is None or m.get("tpr") is None:
                    continue
                out[strat][method][st] = {
                    "ROC_AUC": m.get("ROC_AUC"),
                    "fpr": m["fpr"],
                    "tpr": m["tpr"],
                }
    return out


def save_figure2_roc_metrics_cache(
    stratified_metrics: dict,
    output_base_dir: str,
    raw_pooling_pkl_path: str | None = None,
) -> str | None:
    """
    Write a small pickle (fpr/tpr/AUC only) for fast ``replot_figure2_roc_curves_only``.
    Invalidated automatically when ``raw_data_for_pooling.pkl`` mtime changes.
    """
    from casp16_eval_paths import raw_pooling_pkl_for_read

    path = _figure2_roc_cache_path(output_base_dir)
    pkl_path = raw_pooling_pkl_path or raw_pooling_pkl_for_read()
    mtime = os.path.getmtime(pkl_path) if os.path.isfile(pkl_path) else None
    payload = {
        "meta": {
            "raw_pooling_pkl": os.path.abspath(pkl_path),
            "raw_pooling_pkl_mtime": mtime,
        },
        "plot_inputs": _extract_figure2_roc_plot_inputs(stratified_metrics),
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            _pickle.dump(payload, f, protocol=_pickle.HIGHEST_PROTOCOL)
        return path
    except OSError:
        return None


def load_figure2_roc_metrics_cache(
    output_base_dir: str,
    raw_pooling_pkl_path: str | None = None,
) -> dict | None:
    """Return plot_inputs dict if cache exists and matches current raw pooling pickle mtime."""
    from casp16_eval_paths import raw_pooling_pkl_for_read

    path = _figure2_roc_cache_path(output_base_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = _pickle.load(f)
    except (OSError, EOFError, _pickle.UnpicklingError, ValueError):
        return None
    meta = payload.get("meta") or {}
    expected_pkl = raw_pooling_pkl_path or raw_pooling_pkl_for_read()
    current_mtime = os.path.getmtime(expected_pkl) if os.path.isfile(expected_pkl) else None
    if meta.get("raw_pooling_pkl_mtime") != current_mtime:
        return None
    plot_inputs = payload.get("plot_inputs")
    if not isinstance(plot_inputs, dict) or not plot_inputs:
        return None
    return plot_inputs


def _pooled_data_cache_path(output_base_dir: str) -> str:
    return os.path.join(output_base_dir, "pooled_analysis", POOLED_DATA_CACHE_FILENAME)


def load_pooled_data_cache(
    output_base_dir: str,
    raw_pooling_pkl_path: str | None = None,
) -> dict | None:
    """Return cached ``pooled_data`` if file exists, schema matches, and raw pickle mtime matches."""
    from casp16_eval_paths import raw_pooling_pkl_for_read

    path = _pooled_data_cache_path(output_base_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = _pickle.load(f)
    except (OSError, EOFError, _pickle.UnpicklingError, ValueError):
        return None
    meta = payload.get("meta") or {}
    if meta.get("schema") != POOLED_DATA_CACHE_SCHEMA:
        return None
    expected_pkl = raw_pooling_pkl_path or raw_pooling_pkl_for_read()
    current_mtime = os.path.getmtime(expected_pkl) if os.path.isfile(expected_pkl) else None
    if meta.get("raw_pooling_pkl_mtime") != current_mtime:
        return None
    pooled_data = payload.get("pooled_data")
    if not isinstance(pooled_data, dict):
        return None
    return pooled_data


def save_pooled_data_cache(
    pooled_data: dict,
    output_base_dir: str,
    raw_pooling_pkl_path: str | None = None,
) -> str | None:
    """Persist concat-pooled frames; invalidated when ``raw_data_for_pooling.pkl`` mtime changes."""
    from casp16_eval_paths import raw_pooling_pkl_for_read

    path = _pooled_data_cache_path(output_base_dir)
    pkl_path = raw_pooling_pkl_path or raw_pooling_pkl_for_read()
    mtime = os.path.getmtime(pkl_path) if os.path.isfile(pkl_path) else None
    payload = {
        "meta": {
            "schema": POOLED_DATA_CACHE_SCHEMA,
            "raw_pooling_pkl": os.path.abspath(pkl_path),
            "raw_pooling_pkl_mtime": mtime,
        },
        "pooled_data": pooled_data,
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            _pickle.dump(payload, f, protocol=_pickle.HIGHEST_PROTOCOL)
        return path
    except OSError:
        return None


def _stratified_metrics_full_cache_path(output_base_dir: str) -> str:
    return os.path.join(output_base_dir, "pooled_analysis", STRATIFIED_METRICS_FULL_CACHE_FILENAME)


def load_stratified_metrics_full_cache(
    output_base_dir: str,
    target_chains: dict,
    raw_pooling_pkl_path: str | None = None,
) -> dict | None:
    """Return cached stratified metrics if raw pickle mtime, schema, and chain map match."""
    from casp16_eval_paths import raw_pooling_pkl_for_read

    path = _stratified_metrics_full_cache_path(output_base_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = _pickle.load(f)
    except (OSError, EOFError, _pickle.UnpicklingError, ValueError):
        return None
    meta = payload.get("meta") or {}
    if meta.get("schema") != STRATIFIED_METRICS_CACHE_SCHEMA:
        return None
    expected_pkl = raw_pooling_pkl_path or raw_pooling_pkl_for_read()
    current_mtime = os.path.getmtime(expected_pkl) if os.path.isfile(expected_pkl) else None
    if meta.get("raw_pooling_pkl_mtime") != current_mtime:
        return None
    if meta.get("pooled_cache_schema") != POOLED_DATA_CACHE_SCHEMA:
        return None
    if meta.get("chains_fp") != _target_chains_fingerprint(target_chains):
        return None
    sm = payload.get("stratified_metrics")
    if not isinstance(sm, dict) or not sm:
        return None
    return sm


def save_stratified_metrics_full_cache(
    stratified_metrics: dict,
    output_base_dir: str,
    target_chains: dict,
    raw_pooling_pkl_path: str | None = None,
) -> str | None:
    """Persist full stratified metrics; same invalidation keys as load."""
    from casp16_eval_paths import raw_pooling_pkl_for_read

    path = _stratified_metrics_full_cache_path(output_base_dir)
    pkl_path = raw_pooling_pkl_path or raw_pooling_pkl_for_read()
    mtime = os.path.getmtime(pkl_path) if os.path.isfile(pkl_path) else None
    payload = {
        "meta": {
            "schema": STRATIFIED_METRICS_CACHE_SCHEMA,
            "raw_pooling_pkl": os.path.abspath(pkl_path),
            "raw_pooling_pkl_mtime": mtime,
            "pooled_cache_schema": POOLED_DATA_CACHE_SCHEMA,
            "chains_fp": _target_chains_fingerprint(target_chains),
        },
        "stratified_metrics": stratified_metrics,
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            _pickle.dump(payload, f, protocol=_pickle.HIGHEST_PROTOCOL)
        return path
    except OSError:
        return None


# ----------------------------------------------------------------------------
# Fig.~2 ROC PNGs, performance tables, ARC rankings summary
# ----------------------------------------------------------------------------

_SCORE_TYPE_DISPLAY = {
    "local_lddt": "iLDDT",
    "patch_dockq": "Patch DockQ",
    "local_cad": "CAD",
    "patch_qs": "Patch QS",
    "true_interface_residue": "Interface Residue",
    "QSGLOB": "QSGLOB",
    "QSBEST": "QSBEST",
    "DOCKQ_AVG": "DOCKQ_AVG",
}

_METHOD_STYLE = {
    "ARC":                   {"color": "#D62728", "linestyle": "-",  "linewidth": 3.0, "zorder": 10},
    "APOLLO":                {"color": "#FF7F0E", "linestyle": "--", "linewidth": 1.8, "zorder": 2},
    "ModFOLDdock2S":         {"color": "#1F77B4", "linestyle": "-",  "linewidth": 1.8, "zorder": 3},
    "GuijunLab-PAthreader":  {"color": "#2CA02C", "linestyle": "-.", "linewidth": 1.8, "zorder": 3},
    "GuijunLab-Assembly":    {"color": "#9467BD", "linestyle": "--", "linewidth": 1.8, "zorder": 2},
    "Guijunlab-Complex":     {"color": "#8C564B", "linestyle": ":",  "linewidth": 1.8, "zorder": 2},
    "MQA_server":            {"color": "#E377C2", "linestyle": "-.", "linewidth": 1.8, "zorder": 2},
    "MQA_base":              {"color": "#7F7F7F", "linestyle": ":",  "linewidth": 1.8, "zorder": 2},
    "VifChartreuse":         {"color": "#BCBD22", "linestyle": "--", "linewidth": 1.8, "zorder": 2},
    "VifChartreuseJaune":    {"color": "#17BECF", "linestyle": ":",  "linewidth": 1.8, "zorder": 2},
}

_METHOD_STYLE_DEFAULT = {"color": "#333333", "linestyle": "--", "linewidth": 1.5, "zorder": 1}


def _get_method_style(method):
    return _METHOD_STYLE.get(method, _METHOD_STYLE_DEFAULT)


def _plot_curves(ax, method_curves):
    """Plot ROC curves; method_curves: list of (method, fpr, tpr, auc_score)."""
    sorted_methods = sorted(method_curves, key=lambda x: x[3], reverse=True)
    for method, xv, yv, auc_score in sorted_methods:
        s = _get_method_style(method)
        # Mathtext bold (works with text.usetex=False); plain "ARC" lost emphasis under some styles.
        if method == "ARC":
            label = fr"$\mathbf{{ARC}}$ (AUC={auc_score:.3f})"
        else:
            label = f"{method} (AUC={auc_score:.3f})"
        ax.plot(xv, yv, label=label,
                color=s["color"], linestyle=s["linestyle"],
                linewidth=s["linewidth"], zorder=s["zorder"])


# Main-text fig.~2 panel order (must match nested loops in ``plot_figure2_roc_curves``).
_FIG2_MANUSCRIPT_NAME = {
    ("patch_qs", "All Targets"): "fig2a",
    ("patch_dockq", "All Targets"): "fig2b",
    ("true_interface_residue", "All Targets"): "fig2c",
    ("patch_qs", "Dimer only"): "fig2d",
    ("patch_dockq", "Dimer only"): "fig2e",
    ("true_interface_residue", "Dimer only"): "fig2f",
}


def plot_figure2_roc_curves(stratified_metrics, output_dir):
    """Write the six main-text fig.~2 pooled ROC PNGs as ``fig2a.png``-``fig2f.png`` (Wiley / ``main.tex``)."""
    display = dict(_SCORE_TYPE_DISPLAY)
    for score_type in ("patch_qs", "patch_dockq", "true_interface_residue"):
        display_name = display.get(score_type, score_type)
        for strat_name in ("All Targets", "Dimer only"):
            strat_data = stratified_metrics.get(strat_name)
            if not strat_data:
                continue
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            method_curves = []
            for method, method_data in strat_data.items():
                if score_type not in method_data:
                    continue
                m = method_data[score_type]
                if m.get("ROC_AUC") is not None and m.get("fpr") is not None and m.get("tpr") is not None:
                    method_curves.append((method, m["fpr"], m["tpr"], m["ROC_AUC"]))

            if not method_curves:
                plt.close()
                continue

            _plot_curves(ax, method_curves)
            ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)", linewidth=1, alpha=0.5)

            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title(
                f"ROC Curves - {display_name} - {STRAT_DISPLAY.get(strat_name, strat_name)}",
                fontsize=14,
                fontweight="bold",
            )
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

            ms = _FIG2_MANUSCRIPT_NAME.get((score_type, strat_name))
            if not ms:
                plt.close()
                continue
            ms_path = os.path.join(output_dir, f"{ms}.png")
            plt.savefig(ms_path, format="png", dpi=350, bbox_inches="tight")
            print(f"  Saved ROC curve: {ms_path}")
            plt.close()


def create_performance_tables_from_metrics(stratified_metrics, filtered_methods, output_dir, table_metrics=None):
    """
    Create performance tables using pre-computed metrics (manuscript S2 / fig.~3 inputs).
    
    Args:
        stratified_metrics: Dictionary from compute_all_stratified_metrics
        filtered_methods: Set of method names to include
        output_dir: Directory to save tables
        table_metrics: List of (score_type, display_name) tuples
    """
    if table_metrics is None:
        table_metrics = [
            ("true_interface_residue", "True Interface Residue"),
            ("patch_qs", "Patch QS"),
            ("patch_dockq", "Patch DockQ"),
            ("QSBEST", "QSBEST"),
            ("QSGLOB", "QSGLOB"),
            ("DOCKQ_AVG", "DOCKQ_AVG"),
        ]
    
    # Create tables for each stratification
    for strat_name, strat_data in stratified_metrics.items():
        # Sanitize directory name
        safe_name = strat_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("<", "lt").replace(">=", "ge").replace("=", "eq")
        strat_output_dir = os.path.join(output_dir, safe_name)
        os.makedirs(strat_output_dir, exist_ok=True)
        
        # Create tables for each metric type
        for score_type, display_name in table_metrics:
            table_data = []
            for method in sorted(filtered_methods):
                if method in strat_data and score_type in strat_data[method]:
                    metrics = strat_data[method][score_type]
                    row_dict = {
                        "Method": method,
                        f"{display_name} PCC ^": metrics['pcc'],
                        f"{display_name} SCC ^": metrics['scc'],
                        f"{display_name} MAE v": metrics['MAE']
                    }
                    # Add AUC columns if available
                    if metrics.get("ROC_AUC") is not None:
                        row_dict[f"{display_name} ROC AUC ^"] = metrics["ROC_AUC"]
                    if metrics.get("PR_AUC") is not None:
                        row_dict[f"{display_name} PR AUC ^"] = metrics["PR_AUC"]
                    table_data.append(row_dict)
            
            if len(table_data) > 0:
                table_df = pd.DataFrame(table_data)
                
                # Add ranking columns
                # For PCC and SCC: higher is better (rank 1 = highest)
                pcc_col = f"{display_name} PCC ^"
                scc_col = f"{display_name} SCC ^"
                mae_col = f"{display_name} MAE v"
                roc_auc_col = f"{display_name} ROC AUC ^"
                pr_auc_col = f"{display_name} PR AUC ^"
                
                table_df[f"{display_name} PCC Rank"] = table_df[pcc_col].rank(ascending=False, method='min').astype(int)
                table_df[f"{display_name} SCC Rank"] = table_df[scc_col].rank(ascending=False, method='min').astype(int)
                # For MAE: lower is better (rank 1 = lowest)
                table_df[f"{display_name} MAE Rank"] = table_df[mae_col].rank(ascending=True, method='min').astype(int)
                
                # Add AUC ranking columns if AUC columns exist
                if roc_auc_col in table_df.columns:
                    table_df[f"{display_name} ROC AUC Rank"] = table_df[roc_auc_col].rank(ascending=False, method='min', na_option='bottom').astype(int)
                if pr_auc_col in table_df.columns:
                    table_df[f"{display_name} PR AUC Rank"] = table_df[pr_auc_col].rank(ascending=False, method='min', na_option='bottom').astype(int)
                
                # Reorder columns: Method, then metrics with their ranks
                cols_order = ["Method", pcc_col, f"{display_name} PCC Rank", 
                             scc_col, f"{display_name} SCC Rank",
                             mae_col, f"{display_name} MAE Rank"]
                # Add AUC columns if they exist
                if roc_auc_col in table_df.columns:
                    cols_order.extend([roc_auc_col, f"{display_name} ROC AUC Rank"])
                if pr_auc_col in table_df.columns:
                    cols_order.extend([pr_auc_col, f"{display_name} PR AUC Rank"])
                table_df = table_df[cols_order]
                
                # Find best and second best for formatting
                for col in [pcc_col, scc_col]:
                    if col in table_df.columns:
                        values = table_df[col].values
                        best_idx = np.argmax(values)
                        second_best_idx = np.argmax([v if i != best_idx else -np.inf for i, v in enumerate(values)])
                        table_df.loc[best_idx, col] = values[best_idx]
                        table_df.loc[second_best_idx, col] = values[second_best_idx]
                
                for col in [mae_col]:
                    if col in table_df.columns:
                        values = table_df[col].values
                        best_idx = np.argmin(values)
                        second_best_idx = np.argmin([v if i != best_idx else np.inf for i, v in enumerate(values)])
                        table_df.loc[best_idx, col] = values[best_idx]
                        table_df.loc[second_best_idx, col] = values[second_best_idx]
                
                # Format AUC columns if they exist
                for col in [roc_auc_col, pr_auc_col]:
                    if col in table_df.columns:
                        values = table_df[col].values
                        valid_values = [v for v in values if v is not None and not pd.isna(v)]
                        if len(valid_values) > 0:
                            best_idx = np.argmax([v if v is not None and not pd.isna(v) else -np.inf for v in values])
                            second_best_idx = np.argmax([v if i != best_idx and v is not None and not pd.isna(v) else -np.inf for i, v in enumerate(values)])
                            table_df.loc[best_idx, col] = values[best_idx]
                            table_df.loc[second_best_idx, col] = values[second_best_idx]
                
                # Save CSV
                csv_path = os.path.join(strat_output_dir, f"table_{score_type}_performance.csv")
                table_df.to_csv(csv_path, index=False)
                print(f"  Saved performance table: {csv_path} ({strat_name})")


def create_arc_rankings_summary(stratified_metrics, filtered_methods, output_dir, table_metrics=None):
    """
    Create a summary table showing ARC's rankings across all metrics and stratifications.
    Reads the already-saved CSV tables and extracts ARC's rankings from them.
    
    Args:
        stratified_metrics: Dictionary from compute_all_stratified_metrics (used to get strat names)
        filtered_methods: Set of method names to include (not used, but kept for API consistency)
        output_dir: Directory where tables are saved (same as pooled_tables_dir)
        table_metrics: List of (score_type, display_name) tuples
    """
    if table_metrics is None:
        table_metrics = [
            ("true_interface_residue", "True Interface Residue"),
            ("patch_qs", "Patch QS"),
            ("patch_dockq", "Patch DockQ"),
            ("QSBEST", "QSBEST"),
            ("QSGLOB", "QSGLOB"),
            ("DOCKQ_AVG", "DOCKQ_AVG"),
        ]
    
    summary_data = []
    
    # Process each stratification by reading the saved CSV files
    for strat_name in stratified_metrics.keys():
        row = {"Stratification": strat_name}
        
        # Sanitize directory name (same logic as in create_performance_tables_from_metrics)
        safe_name = strat_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("<", "lt").replace(">=", "ge").replace("=", "eq")
        strat_output_dir = os.path.join(output_dir, safe_name)
        
        # Extract rankings for regular metrics (PCC, SCC, MAE) from saved CSV files
        for score_type, display_name in table_metrics:
            csv_path = os.path.join(strat_output_dir, f"table_{score_type}_performance.csv")
            
            if os.path.exists(csv_path):
                try:
                    table_df = pd.read_csv(csv_path)
                    
                    # Extract ARC's rankings from the loaded table
                    arc_row = table_df[table_df["Method"] == "ARC"]
                    if len(arc_row) > 0:
                        pcc_rank_col = f"{display_name} PCC Rank"
                        scc_rank_col = f"{display_name} SCC Rank"
                        mae_rank_col = f"{display_name} MAE Rank"
                        
                        if pcc_rank_col in arc_row.columns:
                            row[pcc_rank_col] = arc_row[pcc_rank_col].iloc[0]
                        if scc_rank_col in arc_row.columns:
                            row[scc_rank_col] = arc_row[scc_rank_col].iloc[0]
                        if mae_rank_col in arc_row.columns:
                            row[mae_rank_col] = arc_row[mae_rank_col].iloc[0]
                        
                        # Also extract AUC rankings from the same table
                        roc_auc_rank_col = f"{display_name} ROC AUC Rank"
                        pr_auc_rank_col = f"{display_name} PR AUC Rank"
                        
                        if roc_auc_rank_col in arc_row.columns:
                            row[roc_auc_rank_col] = arc_row[roc_auc_rank_col].iloc[0]
                        if pr_auc_rank_col in arc_row.columns:
                            row[pr_auc_rank_col] = arc_row[pr_auc_rank_col].iloc[0]
                except Exception as e:
                    print(f"  Warning: Could not read {csv_path}: {e}")
        
        if len(row) > 1:  # More than just "Stratification"
            summary_data.append(row)
    
    if len(summary_data) == 0:
        print("  Warning: No ARC rankings found in saved tables")
        return
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Reorder columns: Stratification first, then all metric ranks
    metric_cols = [col for col in summary_df.columns if col != "Stratification"]
    metric_cols.sort()  # Sort alphabetically for consistency
    cols_order = ["Stratification"] + metric_cols
    summary_df = summary_df[cols_order]
    
    # Save summary table
    summary_path = os.path.join(output_dir, "arc_rankings_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved ARC rankings summary: {summary_path}")


# ----------------------------------------------------------------------------
# run_pooled_analysis - entry point from casp16_eval.py
# ----------------------------------------------------------------------------

def prepare_stratified_metrics_for_pooled_analysis(
    all_raw_data,
    target_chains,
    output_base_dir,
    *,
    figure2_roc_only: bool = False,
):
    """
    Pool across targets, apply the same method filter as full pooled analysis, compute stratified metrics
    (including ROC curve coordinates for fig.~2).

    Set ``figure2_roc_only=True`` to compute only the two fig.~2 strata x three ROC score types (faster).

    After the first run, ``pool_data_across_targets`` can be skipped via
    ``pooled_analysis/pooled_concat_cache.pkl`` when ``eval/raw_data_for_pooling.pkl`` is unchanged
    (set ``ARC_POOLED_DATA_CACHE_BYPASS=1`` to disable).     For the full pooled pass (not ``figure2_roc_only``), ``compute_all_stratified_metrics`` can be skipped via
    ``pooled_analysis/stratified_metrics_full_cache.pkl`` when the raw pickle mtime and
    target-chain map match (set ``ARC_STRATIFIED_METRICS_CACHE_BYPASS=1`` to force recomputation).
    Callers such as ``run_pooled_analysis`` always rewrite fig.~2 PNGs and pooled CSVs afterward; these
    caches never skip matplotlib or table I/O.

    Returns:
        stratified_metrics, pooled_plots_dir, pooled_tables_dir, filtered_methods, pooled_data
    """
    pooled_output_dir = os.path.join(output_base_dir, "pooled_analysis")
    pooled_plots_dir = os.path.join(pooled_output_dir, "plots")
    pooled_tables_dir = os.path.join(pooled_output_dir, "tables")
    os.makedirs(pooled_plots_dir, exist_ok=True)
    os.makedirs(pooled_tables_dir, exist_ok=True)

    filtered_methods = set(all_raw_data.keys())
    filtered_methods.discard("ChaePred")
    filtered_methods.discard("AF_unmasked")
    filtered_methods.discard("MQA")
    filtered_methods.discard("ModFOLDdock2")
    filtered_methods.discard("ModFOLDdock2R")
    filtered_methods.discard("GuijunLab-QA")
    filtered_methods.discard("GuijunLab-Human")
    for ARC_methods in [
        "ARC_GLFP",
        "ARC_TransConv",
        "ARC_ResGatedGraphConv",
        "ARC_GINEConv",
        "ARC_GENConv",
        "ARC_GeneralConv",
        "ARC_PDNConv",
    ]:
        filtered_methods.discard(ARC_methods)

    common_target_model_pairs = _common_target_model_pairs(all_raw_data, list(filtered_methods))
    if common_target_model_pairs is None:
        warnings.warn(
            "Pooled analysis: 'model' column missing in raw data (e.g. old pickle); not filtering by (target, model).",
            UserWarning,
            stacklevel=2,
        )
        common_target_model_pairs = None
    else:
        print(
            f"  Restricting to {len(common_target_model_pairs)} (target, model) pairs "
            f"predicted by all {len(filtered_methods)} methods"
        )
        if len(common_target_model_pairs) == 0:
            warnings.warn(
                "Pooled analysis: no (target, model) pair is predicted by all methods; pooled data may be empty.",
                UserWarning,
                stacklevel=2,
            )

    bypass_pool_cache = os.environ.get("ARC_POOLED_DATA_CACHE_BYPASS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    pooled_data = None
    if not bypass_pool_cache:
        pooled_data = load_pooled_data_cache(output_base_dir)

    if pooled_data is not None:
        print(
            f"\nPooling data across targets... "
            f"(using {POOLED_DATA_CACHE_FILENAME}; raw pickle mtime unchanged)",
            flush=True,
        )
    else:
        print(f"\nPooling data across targets...", flush=True)
        pooled_data = pool_data_across_targets(all_raw_data, common_target_model_pairs=common_target_model_pairs)
        if not bypass_pool_cache:
            saved_pd = save_pooled_data_cache(pooled_data, output_base_dir)
            if saved_pd:
                print(f"  Wrote pooled concat cache for next run: {saved_pd}", flush=True)
    print(f"  Pooled data for {len(pooled_data)} methods")

    stratifications = [
        ("All Targets", lambda t: True),
        ("Dimer only", lambda t: target_chains.get(t, 0) == 2),
        ("Multimer only", lambda t: target_chains.get(t, 0) > 2),
    ]

    bypass_strat_cache = os.environ.get("ARC_STRATIFIED_METRICS_CACHE_BYPASS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    stratified_metrics = None
    if not figure2_roc_only and not bypass_strat_cache:
        stratified_metrics = load_stratified_metrics_full_cache(output_base_dir, target_chains)

    if stratified_metrics is not None:
        print(
            f"\nComputing metrics for all stratifications... "
            f"(using {STRATIFIED_METRICS_FULL_CACHE_FILENAME}; raw pickle + chain map unchanged)",
            flush=True,
        )
    else:
        print(f"\nComputing metrics for all stratifications...", flush=True)
        score_types = [
            "true_interface_residue",
            "patch_qs",
            "patch_dockq",
            "QSGLOB",
            "QSBEST",
            "DOCKQ_AVG",
        ]
        stratified_metrics = compute_all_stratified_metrics(
            pooled_data,
            stratifications,
            score_types,
            filtered_methods,
            figure2_roc_only=figure2_roc_only,
        )
        if not figure2_roc_only and not bypass_strat_cache:
            saved_sm = save_stratified_metrics_full_cache(stratified_metrics, output_base_dir, target_chains)
            if saved_sm:
                print(f"  Wrote stratified metrics cache for next run: {saved_sm}", flush=True)
    print(f"  Computed metrics for {len(stratified_metrics)} stratifications")

    return stratified_metrics, pooled_plots_dir, pooled_tables_dir, filtered_methods, pooled_data


def replot_figure2_roc_curves_only(output_base_dir=None, raw_pickle_path=None):
    """
    Regenerate only the six main-text pooled ROC PNGs under ``pooled_analysis/plots/``.

    **Fast path:** if ``pooled_analysis/stratified_metrics_fig2_roc.pkl`` exists and matches the
    mtime of ``raw_data_for_pooling.pkl``, loads cached fpr/tpr/AUC and only runs matplotlib (instant).

    **Otherwise:** pools from the raw pickle and runs a **slim** metric pass (All Targets + Dimer only,
    three ROC score types only)-much faster than the full stratified table pass.

    Set ``ARC_FIG2_ROC_RECOMPUTE=1`` to ignore the cache and force recomputation from the raw pickle.

    Does **not** rewrite pooled CSV tables or rankings summary.

    Uses ``LOCAL_RESULTS_DIR`` / ``raw_pooling_pkl_for_read()`` / ``local_eval_csv_for_read()`` from
    ``casp16_eval_paths`` unless paths are passed explicitly.
    """
    from casp16_eval_data import df_local_stoch
    from casp16_eval_io import target_chains_for_targets_safe
    from casp16_eval_paths import (
        LOCAL_RESULTS_DIR,
        local_eval_csv_for_read,
        raw_pooling_pkl_for_read,
    )

    base = output_base_dir if output_base_dir is not None else LOCAL_RESULTS_DIR
    pooled_plots_dir = os.path.join(base, "pooled_analysis", "plots")
    os.makedirs(pooled_plots_dir, exist_ok=True)

    pkl_path = raw_pickle_path if raw_pickle_path is not None else raw_pooling_pkl_for_read()
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"raw pooling pickle not found: {pkl_path}")

    force = os.environ.get("ARC_FIG2_ROC_RECOMPUTE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not force:
        cached_inputs = load_figure2_roc_metrics_cache(base, raw_pooling_pkl_path=pkl_path)
        if cached_inputs is not None:
            print(
                f"Fig.~2 ROC replot: using {FIG2_ROC_METRICS_CACHE_FILENAME} (matched raw pickle mtime).",
                flush=True,
            )
            print(
                f"\nWriting fig2a.png-fig2f.png -> {pooled_plots_dir}",
                flush=True,
            )
            plot_figure2_roc_curves(cached_inputs, pooled_plots_dir)
            print(f"Fig.~2 ROC replot complete.", flush=True)
            return

    with open(pkl_path, "rb") as f:
        all_raw_data = _pickle.load(f)
    if not all_raw_data:
        raise ValueError(f"raw pooling pickle is empty: {pkl_path}")

    loc_path = local_eval_csv_for_read()
    if not os.path.isfile(loc_path):
        raise FileNotFoundError(f"local_eval.csv not found: {loc_path}")
    loc = pd.read_csv(loc_path)
    target_chains = target_chains_for_targets_safe(df_local_stoch, set(loc["target"].unique()))

    print(
        f"Fig.~2 ROC replot: slim metric pass (2 strata x 3 scores) from raw pickle; "
        f"skipping pooled CSV tables.",
        flush=True,
    )
    stratified_metrics, _, _, _, _ = prepare_stratified_metrics_for_pooled_analysis(
        all_raw_data, target_chains, base, figure2_roc_only=True
    )
    saved = save_figure2_roc_metrics_cache(stratified_metrics, base, raw_pooling_pkl_path=pkl_path)
    if saved:
        print(f"  Wrote fig.~2 ROC cache for next replot: {saved}", flush=True)
    print(
        f"\nWriting fig2a.png-fig2f.png -> {pooled_plots_dir}",
        flush=True,
    )
    plot_figure2_roc_curves(stratified_metrics, pooled_plots_dir)
    print(f"Fig.~2 ROC replot complete.", flush=True)


def run_pooled_analysis(all_raw_data, target_chains, output_base_dir):
    """
    Main function to run pooled analysis: pool data, compute metrics, create plots and tables.

    Stratifications are All Targets, Dimer only, and Multimer only (manuscript S2 / fig2 / fig8).

    ``prepare_stratified_metrics_for_pooled_analysis`` may load ``pooled_concat_cache.pkl`` /
    ``stratified_metrics_full_cache.pkl`` by default (see env bypass vars). This function **always**
    regenerates fig.~2 ROC PNGs, performance tables, rankings CSV, and log printout—caches never skip I/O.

    Args:
        all_raw_data: Dictionary {method: {score_type: list of DataFrames}}
        target_chains: Dictionary mapping target names to chain counts
        output_base_dir: Base directory for outputs
    """
    print("\n" + "=" * 80)
    print(EVAL_LOG_SEC_3)
    print("=" * 80)

    stratified_metrics, pooled_plots_dir, pooled_tables_dir, filtered_methods, pooled_data = (
        prepare_stratified_metrics_for_pooled_analysis(all_raw_data, target_chains, output_base_dir)
    )
    pooled_output_dir = os.path.join(output_base_dir, "pooled_analysis")

    print(
        f"\nGenerating fig.~2 ROC curve PNGs (always overwrite; metrics may be cached)...",
        flush=True,
    )
    plot_figure2_roc_curves(stratified_metrics, pooled_plots_dir)
    cache_path = save_figure2_roc_metrics_cache(stratified_metrics, output_base_dir)
    if cache_path:
        print(f"  Wrote fig.~2 ROC cache (fast replot): {cache_path}", flush=True)

    # Create performance tables using pre-computed metrics
    print(
        f"\nGenerating performance tables (always overwrite from metrics above)...",
        flush=True,
    )
    create_performance_tables_from_metrics(stratified_metrics, filtered_methods, pooled_tables_dir)
    
    # Create ARC rankings summary table
    print(f"\nGenerating ARC rankings summary...")
    create_arc_rankings_summary(stratified_metrics, filtered_methods, pooled_tables_dir)
    
    # Print organized results to log
    _print_pooled_analysis_results(pooled_tables_dir)
    
    print("\n" + "=" * 80)
    print("Pooled analysis complete (tables, plots, rankings summary).")
    print(f"  Output directory: {pooled_output_dir}")
    print("=" * 80 + "\n")
    
    return pooled_data


def _print_pooled_analysis_results(tables_dir):
    """Print a structured, informative summary of pooled analysis results to the log."""
    W = 100

    from plots_manuscript_supplementary_tables import print_pooled_s2_style_log_summary

    # Same data and column semantics as supplementary Table S2 (all stratifications).
    if not print_pooled_s2_style_log_summary(tables_dir, width=W):
        # Fallback: flat per-column dump for All Targets only if S2-style tables are incomplete.
        all_dir = os.path.join(tables_dir, "all_targets")
        if not os.path.isdir(all_dir):
            return

        metric_display = {
            "true_interface_residue": "True Interface Residue",
            "patch_qs": "Patch QS",
            "patch_dockq": "Patch DockQ",
            "QSBEST": "QSBEST",
            "QSGLOB": "QSGLOB",
            "DOCKQ_AVG": "DOCKQ_AVG",
        }

        print(f"\n{'=' * W}")
        print("  POOLED ANALYSIS RESULTS - All Targets (ARC Performance, fallback)")
        print(f"{'=' * W}")

        for score_type, display_name in metric_display.items():
            csv_path = os.path.join(all_dir, f"table_{score_type}_performance.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            arc_row = df[df["Method"] == "ARC"]
            if arc_row.empty:
                continue

            n_methods = len(df)
            print(f"\n  {display_name} ({n_methods} methods)")
            print(f"  {'-' * (W - 4)}")

            rank_cols = [c for c in df.columns if "Rank" in c]
            value_cols = [c for c in df.columns if c != "Method" and "Rank" not in c]

            for vc in value_cols:
                rc = vc.replace(" ^", " Rank").replace(" v", " Rank")
                if rc not in rank_cols:
                    rc_alt = (
                        vc.split("^")[0].strip() + " Rank"
                        if "^" in vc
                        else vc.split(" v")[0].strip() + " Rank"
                    )
                    rc = rc_alt if rc_alt in rank_cols else None

                val = arc_row[vc].values[0]
                if pd.isna(val):
                    continue

                rank = int(arc_row[rc].values[0]) if rc and rc in arc_row.columns else None

                star = ""
                if rank is not None and rank <= 3:
                    star = " *" if rank == 1 else " *" if rank <= 3 else ""

                metric_label = vc.replace(f"{display_name} ", "")
                rank_str = f"Rank {rank}/{n_methods}" if rank else ""
                print(f"    {metric_label:<20} {val:>10.4f}   {rank_str:<15}{star}")
    
    # ---- Part 2: ARC Rankings Summary across stratifications ----
    summary_path = os.path.join(tables_dir, "arc_rankings_summary.csv")
    if not os.path.exists(summary_path):
        return
    
    summary_df = pd.read_csv(summary_path)
    
    print(f"\n\n{'=' * W}")
    print(f"  {EVAL_LOG_SEC_3C}")
    print("  * = Rank 1  |  * = Rank 2-3  |  Values shown are ranks out of 10 methods")
    print(f"{'=' * W}")
    
    key_metrics = [
        "True Interface Residue ROC AUC Rank",
        "True Interface Residue PR AUC Rank",
        "True Interface Residue PCC Rank",
        "Patch QS PCC Rank",
        "Patch DockQ ROC AUC Rank",
        "QSBEST PCC Rank",
        "QSBEST ROC AUC Rank",
        "QSGLOB MAE Rank",
        "DOCKQ_AVG MAE Rank",
    ]
    available = [m for m in key_metrics if m in summary_df.columns]
    
    short_labels = {
        "True Interface Residue ROC AUC Rank": "TIR ROC",
        "True Interface Residue PR AUC Rank": "TIR PR",
        "True Interface Residue PCC Rank": "TIR PCC",
        "Patch QS PCC Rank": "PQS PCC",
        "Patch DockQ ROC AUC Rank": "PDQ ROC",
        "QSBEST PCC Rank": "QSB PCC",
        "QSBEST ROC AUC Rank": "QSB ROC",
        "QSGLOB MAE Rank": "QSG MAE",
        "DOCKQ_AVG MAE Rank": "DQ MAE",
    }
    
    header = f"  {'Stratification':<25}"
    for m in available:
        header += f" {short_labels.get(m, m[:7]):>8}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    
    for _, row in summary_df.iterrows():
        line = f"  {row['Stratification']:<25}"
        for m in available:
            rank = int(row[m]) if not pd.isna(row[m]) else 0
            marker = "*" if rank == 1 else " *" if rank <= 3 else "  "
            line += f" {rank:>5}{marker}"
        print(line)
    
    print(f"  {'-' * (len(header) - 2)}")
    print(f"  Legend: TIR=true_interface_residue, PQS=patch_qs, PDQ=patch_dockq,")
    print(f"          QSB=QSBEST, QSG=QSGLOB, DQ=DOCKQ_AVG")
    print()

