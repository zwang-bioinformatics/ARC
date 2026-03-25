"""ARC multi-GNN ensemble: residue tables, variance/mean CSV, exhaustive combinations, summaries."""
import itertools
import json
import os
from math import comb
from collections import defaultdict
from multiprocessing import cpu_count
import multiprocessing as mp

import numpy as np
import pandas as pd
from MLstatkit import Delong_test
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None
from scipy.stats import wilcoxon

from casp16_eval_paths import ARC_ENSEMBLE_DIR
from casp16_eval_constants import (
    APPROX_TARGET_SIZE,
    ARC_GNNS,
    CASP16_EMA_RESULTS,
    EVAL_LOG_SEC_2_2,
    EVAL_LOG_SEC_2_3,
    EVAL_LOG_SEC_2_4,
    EVAL_LOG_SEC_2_5,
    EVAL_LOG_SEC_2_6,
    PACKAGED_PREDICTIONS_BASE,
)
from casp16_eval_data import CASP_MODEL_SCORES_CSV, df_local_stoch, raw, truth
from casp16_eval_io import packaged_local_json_path


def default_arc_ensemble_dir():
    return ARC_ENSEMBLE_DIR


def ensemble_summary_all_comb_csv(arc_ensemble_dir=None):
    d = arc_ensemble_dir if arc_ensemble_dir is not None else default_arc_ensemble_dir()
    return os.path.join(d, "arc_ensemble_summary_all_comb.csv")


# All metric pairs: (display_name, ensemble_col, individual_col)
_METRIC_PAIRS = [
    ("iface_ROC_AUC",    "true_interface_residue_AUC_ROC_ensemble",  "true_interface_residue_AUC_ROC_individual"),
    ("iface_PR_AUC",     "true_interface_residue_AUC_PRC_ensemble",  "true_interface_residue_AUC_PRC_individual"),
    ("patch_qs_PCC",     "patch_qs_pearson_ensemble",                "patch_qs_pearson_individual"),
    ("patch_qs_SCC",     "patch_qs_spearman_ensemble",               "patch_qs_spearman_individual"),
    ("patch_qs_adaAUC",  "patch_qs_adaptive_auc_ensemble",           "patch_qs_adaptive_auc_individual"),
    ("patch_dockq_PCC",  "patch_dockq_pearson_ensemble",             "patch_dockq_pearson_individual"),
    ("patch_dockq_SCC",  "patch_dockq_spearman_ensemble",            "patch_dockq_spearman_individual"),
    ("patch_dockq_adaAUC","patch_dockq_adaptive_auc_ensemble",       "patch_dockq_adaptive_auc_individual"),
    ("local_lddt_PCC",   "local_lddt_pearson_ensemble",              "local_lddt_pearson_individual"),
    ("local_lddt_SCC",   "local_lddt_spearman_ensemble",             "local_lddt_spearman_individual"),
    ("local_lddt_adaAUC","local_lddt_adaptive_auc_ensemble",         "local_lddt_adaptive_auc_individual"),
    ("local_cad_PCC",    "local_cad_pearson_ensemble",               "local_cad_pearson_individual"),
    ("local_cad_SCC",    "local_cad_spearman_ensemble",              "local_cad_spearman_individual"),
    ("local_cad_adaAUC", "local_cad_adaptive_auc_ensemble",          "local_cad_adaptive_auc_individual"),
]
_DELONG_COL = "true_interface_residue_DeLong_ROC_pvalue"

# ==============================================================================
# ARC ensemble - residue tables, variance/mean, all-combinations, summaries
# ==============================================================================
# Per-(target, model) residue data: truth + seven GNN preds (+ ensemble stats).

def build_arc_residue_data_for_target(target):
    """
    Build ARC residue-level data for one target: for each model, a DataFrame with
    rows = residues (model_interface_residues), columns = true_interface_residue, pred_0..pred_6,
    patch_qs, patch_dockq, local_lddt, local_cad. Used for variance/mean and all-combinations analysis.
    Returns: dict {model: DataFrame} or empty dict if target skipped.
    """
    if not os.path.exists(f"{raw}/{target}/QA_1/"):
        return {}
    df = df_local_stoch[df_local_stoch["trg"].str.contains(target)]
    df_filtered = df[df["n_mdl_chains"] == df["n_trg_chains"]]
    filtered_mdls = set(df_filtered["mdl"])
    targ_truth = truth[truth["TARGET"] == target]
    targ_truth = targ_truth[targ_truth["MODEL"].isin(filtered_mdls)]
    if targ_truth["QSBEST"].max() < 0.6 or not len(targ_truth):
        return {}
    targ_truth = targ_truth.set_index('MODEL')
    targ_truth = targ_truth.to_dict(orient='index')

    # Load 7 ARC GNN predictors
    predictors = {}
    for group in ARC_GNNS:
        path = packaged_local_json_path(PACKAGED_PREDICTIONS_BASE, target, group)
        if path:
            try:
                predictors[group] = json.load(open(path, "r"))
            except Exception:
                pass
    if len(predictors) == 0:
        return {}

    out = {}
    for model in targ_truth:
        if model not in filtered_mdls:
            continue
        json_path = os.path.join(CASP16_EMA_RESULTS, f"{model}_{target}.json")
        if not os.path.exists(json_path):
            continue
        local_truth = json.load(open(json_path, 'r'))
        if "model_interface_residues" not in local_truth:
            continue
        mapping = {v: k for k, v in local_truth["chain_mapping"].items()}

        true_iface = []
        pred_cols = [[] for _ in range(7)]
        patch_qs_list = []
        patch_dockq_list = []
        local_lddt_list = []
        local_cad_list = []
        residue_list = []

        for r, residue in enumerate(local_truth["model_interface_residues"]):
            residue_list.append(residue)
            rkey = "".join(residue.split(".")[:-1])
            # True interface
            int_res_data = residue.split('.')
            cname = int_res_data[0]
            rnum = int(int_res_data[1])
            trg_cname = mapping.get(cname)
            if trg_cname is not None:
                true_iface.append(int(f"{trg_cname}.{rnum}." in local_truth["reference_interface_residues"]))
            else:
                true_iface.append(0)
            # 7 GNN predictions (NaN if: GNN not loaded for target, model missing in that GNN's JSON,
            # or residue key rkey not in predictors[gnn][model] - i.e. that GNN did not output a score for this residue)
            for i, gnn in enumerate(ARC_GNNS):
                if gnn in predictors and model in predictors[gnn]:
                    val = predictors[gnn][model].get(rkey)
                    pred_cols[i].append(float(val) if val is not None else np.nan)
                else:
                    pred_cols[i].append(np.nan)
            patch_qs_list.append(local_truth["patch_qs"][r])
            patch_dockq_list.append(local_truth["patch_dockq"][r])
            local_lddt_list.append(local_truth.get("local_lddt", {}).get(residue, np.nan))
            local_cad_list.append(local_truth.get("local_cad_score", {}).get(residue, np.nan))

        df_model = pd.DataFrame({
            "residue": residue_list,
            "true_interface_residue": true_iface,
            **{f"pred_{i}": pred_cols[i] for i in range(7)},
            "patch_qs": patch_qs_list,
            "patch_dockq": patch_dockq_list,
            "local_lddt": local_lddt_list,
            "local_cad": local_cad_list,
        })
        out[model] = df_model
    return out


def _adaptive_rocauc_from_arrays(truth_arr, pred_arr, quantile=0.75):
    """Adaptive ROC AUC with top quantile of truth as positive class (CASP 75th)."""
    truth_arr = np.asarray(truth_arr)
    pred_arr = np.asarray(pred_arr)
    mask = ~(np.isnan(truth_arr) | np.isnan(pred_arr))
    if mask.sum() == 0:
        return np.nan
    t = truth_arr[mask]
    p = pred_arr[mask]
    thresh = np.nanquantile(t, quantile)
    binary = (t > thresh).astype(int)
    if np.unique(binary).size < 2:
        return 0.5
    return max(0.5, roc_auc_score(binary, p))


def _compute_arc_ensemble_row(arc_residue_data_T, target, K, other_indices, N_GNNS=7):
    """Compute one row of metrics for ARC ensemble vs individual for a fixed (target, K, other_indices).
    Returns one row dict or None if skipped. Used by the all-combinations pipeline."""
    L = len(other_indices)
    ensemble_indices = [K] + other_indices

    all_true_iface = []
    all_pred_ind = []
    all_pred_ens = []
    all_patch_qs = []
    all_patch_dockq = []
    all_local_lddt = []
    all_local_cad = []
    for model, df in arc_residue_data_T.items():
        pred_cols = [df[f"pred_{i}"].values for i in range(N_GNNS)]
        pred_ind = pred_cols[K]
        stack_ens = np.stack([pred_cols[i] for i in ensemble_indices], axis=1)
        row_has_any = np.any(~np.isnan(stack_ens), axis=1)
        pred_ens = np.full(stack_ens.shape[0], np.nan, dtype=float)
        if row_has_any.any():
            pred_ens[row_has_any] = np.nanmean(stack_ens[row_has_any], axis=1)
        valid = ~(np.isnan(pred_ind) | np.any(np.isnan(stack_ens), axis=1))
        if valid.sum() == 0:
            continue
        all_true_iface.extend(df["true_interface_residue"].values[valid].tolist())
        all_pred_ind.extend(pred_ind[valid].tolist())
        all_pred_ens.extend(pred_ens[valid].tolist())
        all_patch_qs.extend(df["patch_qs"].values[valid].tolist())
        all_patch_dockq.extend(df["patch_dockq"].values[valid].tolist())
        all_local_lddt.extend(df["local_lddt"].values[valid].tolist())
        all_local_cad.extend(df["local_cad"].values[valid].tolist())

    if len(all_true_iface) < 10 or len(set(all_true_iface)) < 2:
        return None

    row = {
        "Target": target, "K": K, "L": L,
        "other_indices": ",".join(str(i) for i in sorted(other_indices)),
        "ensemble_indices": ",".join(str(i) for i in sorted(ensemble_indices)),
    }
    try:
        row["true_interface_residue_AUC_ROC_individual"] = roc_auc_score(all_true_iface, all_pred_ind)
    except Exception:
        row["true_interface_residue_AUC_ROC_individual"] = np.nan
    try:
        row["true_interface_residue_AUC_ROC_ensemble"] = roc_auc_score(all_true_iface, all_pred_ens)
    except Exception:
        row["true_interface_residue_AUC_ROC_ensemble"] = np.nan
    try:
        precision_ind, recall_ind, _ = precision_recall_curve(all_true_iface, all_pred_ind)
        row["true_interface_residue_AUC_PRC_individual"] = auc(recall_ind, precision_ind)
    except Exception:
        row["true_interface_residue_AUC_PRC_individual"] = np.nan
    try:
        precision_ens, recall_ens, _ = precision_recall_curve(all_true_iface, all_pred_ens)
        row["true_interface_residue_AUC_PRC_ensemble"] = auc(recall_ens, precision_ens)
    except Exception:
        row["true_interface_residue_AUC_PRC_ensemble"] = np.nan

    y = np.asarray(all_true_iface, dtype=float)
    p1 = np.asarray(all_pred_ind, dtype=float)
    p2 = np.asarray(all_pred_ens, dtype=float)
    _, p = Delong_test(y, p1, p2, return_ci=False, return_auc=False, verbose=0)
    row["true_interface_residue_DeLong_ROC_pvalue"] = float(p)

    for name, truth_vals, pred_ind_vals, pred_ens_vals in [
        ("patch_qs", all_patch_qs, all_pred_ind, all_pred_ens),
        ("patch_dockq", all_patch_dockq, all_pred_ind, all_pred_ens),
        ("local_lddt", all_local_lddt, all_pred_ind, all_pred_ens),
        ("local_cad", all_local_cad, all_pred_ind, all_pred_ens),
    ]:
        truth_vals = np.asarray(truth_vals, dtype=float)
        pred_ind_vals = np.asarray(pred_ind_vals, dtype=float)
        pred_ens_vals = np.asarray(pred_ens_vals, dtype=float)
        mask = ~(np.isnan(truth_vals) | np.isnan(pred_ind_vals) | np.isnan(pred_ens_vals))
        if mask.sum() < 5:
            for s in ["pearson", "spearman", "adaptive_auc"]:
                row[f"{name}_{s}_individual"] = np.nan
                row[f"{name}_{s}_ensemble"] = np.nan
            continue
        t, pi, pe = truth_vals[mask], pred_ind_vals[mask], pred_ens_vals[mask]
        row[f"{name}_pearson_individual"] = np.corrcoef(t, pi)[0, 1] if np.std(pi) > 0 else np.nan
        row[f"{name}_pearson_ensemble"] = np.corrcoef(t, pe)[0, 1] if np.std(pe) > 0 else np.nan
        row[f"{name}_spearman_individual"] = pd.Series(t).corr(pd.Series(pi), method="spearman")
        row[f"{name}_spearman_ensemble"] = pd.Series(t).corr(pd.Series(pe), method="spearman")
        row[f"{name}_adaptive_auc_individual"] = _adaptive_rocauc_from_arrays(t, pi)
        row[f"{name}_adaptive_auc_ensemble"] = _adaptive_rocauc_from_arrays(t, pe)
    return row


def run_arc_variance_mean_analysis(arc_residue_data, output_dir=None, force_run=False):
    """
    Build variance/mean CSV (L=7, ddof=1) per residue for optional offline analysis.
    Not used by manuscript figures; skips if CSV exists unless force_run=True.
    """
    if output_dir is None:
        output_dir = ARC_ENSEMBLE_DIR
    csv_path = os.path.join(output_dir, "arc_residue_variance_mean.csv")

    if not force_run and os.path.exists(csv_path):
        print(
            f"{EVAL_LOG_SEC_2_2}: file exists; skipping. "
            "Use force_run=True to regenerate."
        )
        return

    rows = []
    for target, models in arc_residue_data.items():
        for model, df in models.items():
            pred_cols = [df[f"pred_{i}"].values for i in range(7)]
            stack = np.stack(pred_cols, axis=1)
            valid = ~np.any(np.isnan(stack), axis=1)
            if valid.sum() == 0:
                continue
            mean_vals = np.nanmean(stack[valid], axis=1)
            var_vals = np.nanvar(stack[valid], axis=1, ddof=1)
            true_iface = df["true_interface_residue"].values[valid]
            residue_indices = np.where(valid)[0]
            for i in range(valid.sum()):
                rows.append({
                    "target": target, "model": model, "residue_index": int(residue_indices[i]),
                    "L": 7, "variance": float(var_vals[i]), "mean": float(mean_vals[i]),
                    "is_interface_residue": int(true_iface[i])
                })
    vdf = pd.DataFrame(rows)
    vdf.to_csv(csv_path, index=False)
    print(f"{EVAL_LOG_SEC_2_2}: saved {csv_path}")


def _build_arc_ensemble_summary_from_combinations(df_all, output_dir=None, log_fn=None):
    """Compute per-target, per-L, per-metric stats from all-combinations CSVs. Applies BH to DeLong p-values within each (target, L) group. Returns DataFrame."""
    _out = log_fn if log_fn is not None else lambda s: print(s, flush=True)
    if output_dir is None:
        output_dir = default_arc_ensemble_dir()
    _out("")
    _out("-" * 80)
    _out(EVAL_LOG_SEC_2_4)
    _out("-" * 80)
    _out("  For each (target, L) we have multiple combinations (different base K and subset of others).")
    _out("  Metrics: mean(ensemble - individual), %% ensemble better, and for ROC AUC: BH-adjusted DeLong p, %% sig., %% sig. favor ensemble.")
    _out("")
    rows = []
    for (target, L), grp in df_all.groupby(["Target", "L"]):
        base = {"target": target, "L": L, "ensemble_size": 1 + L, "n_combinations": len(grp)}
        for mname, ecol, icol in _METRIC_PAIRS:
            if ecol not in grp.columns or icol not in grp.columns:
                continue
            diff = (grp[ecol] - grp[icol]).dropna()
            if len(diff) == 0:
                continue
            base[f"{mname}_ind_mean"] = grp[icol].mean()
            base[f"{mname}_ens_mean"] = grp[ecol].mean()
            base[f"{mname}_diff_mean"] = diff.mean()
            base[f"{mname}_diff_std"] = diff.std()
            base[f"{mname}_diff_ci_lo"] = diff.quantile(0.025)
            base[f"{mname}_diff_ci_hi"] = diff.quantile(0.975)
            base[f"{mname}_pct_ens_better"] = (diff > 0).mean()
            if mname == "iface_ROC_AUC" and _DELONG_COL in grp.columns and multipletests is not None:
                valid = grp[ecol].notna() & grp[icol].notna() & grp[_DELONG_COL].notna()
                if valid.sum() > 0:
                    diff_vals = (grp.loc[valid, ecol] - grp.loc[valid, icol]).values
                    delong_vals = grp.loc[valid, _DELONG_COL].values
                    _, pvals_adjusted, _, _ = multipletests(delong_vals, alpha=0.05, method="fdr_bh")
                    pvals_adjusted = np.asarray(pvals_adjusted)
                    base[f"{mname}_delong_median_p"] = np.median(pvals_adjusted)
                    base[f"{mname}_delong_pct_sig"] = (pvals_adjusted < 0.05).mean()
                    base[f"{mname}_delong_pct_sig_ens_better"] = ((pvals_adjusted < 0.05) & (diff_vals > 0)).mean()
                    base[f"{mname}_delong_pct_sig_ind_better"] = ((pvals_adjusted < 0.05) & (diff_vals < 0)).mean()
                else:
                    base[f"{mname}_delong_median_p"] = np.nan
                    base[f"{mname}_delong_pct_sig"] = np.nan
                    base[f"{mname}_delong_pct_sig_ens_better"] = np.nan
                    base[f"{mname}_delong_pct_sig_ind_better"] = np.nan
        rows.append(base)
    df_summary = pd.DataFrame(rows)
    casp_scores_path = CASP_MODEL_SCORES_CSV
    target_sizes = APPROX_TARGET_SIZE
    qsbest_stats = {}
    if os.path.exists(casp_scores_path):
        df_casp = pd.read_csv(casp_scores_path)
        target_col = "TARGET" if "TARGET" in df_casp.columns else "target"
        for trg, grp in df_casp.groupby(target_col):
            trg_clean = trg.replace(".pdb", "")
            qsbest_stats[trg_clean] = {"qsbest_max": grp["QSBEST"].max(), "qsbest_mean": grp["QSBEST"].mean()}
    df_filt = df_local_stoch[df_local_stoch["n_mdl_chains"] == df_local_stoch["n_trg_chains"]]
    chain_df = df_filt[["trg", "n_trg_chains"]].drop_duplicates()
    chain_df["trg"] = chain_df["trg"].str.replace(".pdb", "", regex=False)
    target_chains = dict(zip(chain_df["trg"], chain_df["n_trg_chains"]))
    df_summary["target_type"] = df_summary["target"].apply(lambda t: "H" if t.startswith("H") else "T")
    df_summary["n_chains"] = df_summary["target"].map(target_chains)
    df_summary["oligomeric_state"] = df_summary["n_chains"].apply(lambda x: "Dimer" if x == 2 else ("Multimer (>2)" if x > 2 else "Unknown"))
    df_summary["target_size"] = df_summary["target"].map(target_sizes)
    df_summary["size_category"] = pd.cut(df_summary["target_size"], bins=[0, 1000, 1500, 3000, 10000],
                                          labels=["Small (<1000)", "Medium (1000-1500)", "Large (1500-3000)", "Huge (>3000)"])
    df_summary["qsbest_max"] = df_summary["target"].map(lambda t: qsbest_stats.get(t, {}).get("qsbest_max", np.nan))
    df_summary["qsbest_mean"] = df_summary["target"].map(lambda t: qsbest_stats.get(t, {}).get("qsbest_mean", np.nan))
    df_summary["difficulty"] = df_summary["qsbest_max"].apply(
        lambda q: "Hard (<0.7)" if q < 0.7 else ("Medium (0.7-0.85)" if q < 0.85 else "Easy (>0.85)") if not pd.isna(q) else "Unknown"
    )
    id_cols = ["target", "L", "ensemble_size", "n_combinations"]
    strat_cols = ["target_type", "n_chains", "oligomeric_state", "target_size", "size_category", "qsbest_max", "qsbest_mean", "difficulty"]
    metric_cols = [c for c in df_summary.columns if c not in id_cols + strat_cols]
    df_summary = df_summary[id_cols + strat_cols + metric_cols]
    out_path = os.path.join(output_dir, "arc_ensemble_summary_all_comb.csv")
    df_summary.to_csv(out_path, index=False)
    _out(f"Saved: {out_path}  ({len(df_summary)} rows, {len(df_summary.columns)} columns)")
    return df_summary


# Populated only while ``run_arc_all_combinations`` holds a worker pool (fork: inherited copy-on-write;
# spawn: set via ``_init_all_comb_registry`` in each worker).
_ALL_COMB_REGISTRY: dict | None = None


def _init_all_comb_registry(data: dict) -> None:
    global _ALL_COMB_REGISTRY
    _ALL_COMB_REGISTRY = data


def _compute_arc_ensemble_row_registry(args):
    """Worker: args are ``(target, K, tuple(other_indices))``; residue tables from ``_ALL_COMB_REGISTRY``."""
    target, K, other_indices = args
    reg = _ALL_COMB_REGISTRY
    if reg is None:
        return None
    data_t = reg.get(target)
    if not data_t:
        return None
    return _compute_arc_ensemble_row(data_t, target, K, list(other_indices))


def _try_read_all_comb_cache(path: str) -> pd.DataFrame | None:
    """Return a non-empty DataFrame if ``path`` is a valid cached per-target CSV; else None."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return None
        df = pd.read_csv(path)
        if df.empty or len(df.columns) == 0:
            return None
        return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return None


def run_arc_all_combinations(arc_residue_data, output_dir=None, n_jobs=None, force_run=False, log_file_path=None):
    """Generate all (K, other_indices) combinations per target; write per-target CSVs and summary with BH-corrected DeLong.

    Uses one process pool over *all* pending (target, combination) jobs so workers are kept busy across targets.
    On Unix ``fork``, tasks only pickle small tuples; residue tables are read from a process-global registry.
    Set ``ARC_ALL_COMB_NJOBS`` (integer) to override the default ``max(1, min(cpu_count() - 4, 64))`` pool size.
    If log_file_path is set, the same lines are also appended there.
    """
    def _log(msg):
        print(msg, flush=True)
        if log_file_path:
            try:
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass

    def _log_ms(msg):
        m = msg
        print(m, flush=True)
        if log_file_path:
            try:
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(m + "\n")
            except Exception:
                pass
    if output_dir is None:
        output_dir = default_arc_ensemble_dir()
    N_GNNS = 7
    comb_dir = os.path.join(output_dir, "arc_all_combinations")
    os.makedirs(comb_dir, exist_ok=True)
    targets_with_data = [t for t in arc_residue_data if arc_residue_data[t] and len(arc_residue_data[t]) > 0]
    if not targets_with_data:
        _log("ARC all-combinations: no targets with residue data, skipping.")
        return
    if n_jobs is None:
        env_nj = os.environ.get("ARC_ALL_COMB_NJOBS", "").strip()
        if env_nj.isdigit():
            n_jobs = max(1, int(env_nj))
        else:
            n_jobs = max(1, min(cpu_count() - 4, 64))
    _log_ms("")
    _log_ms("=" * 80)
    _log_ms(f"  {EVAL_LOG_SEC_2_3}")
    _log_ms("=" * 80)
    _log_ms("  Setup: For each target we enumerate all ways to pick one GNN as 'individual' (K) and L others for the ensemble.")
    _log_ms("  Per target: 7 choices for K x C(6,L) subsets for L=1..6 -> 7 x (C(6,1)+...+C(6,6)) = 7 x 63 = 441 combinations at L=6.")
    _log_ms("  For each combination we compute ensemble (mean of K + L predictors) vs individual (GNN K only).")
    _log_ms("  DeLong p-values are computed per combination; Benjamini-Hochberg correction is applied within each (target, L) group.")
    if os.environ.get("ARC_ALL_COMB_NJOBS", "").strip().isdigit():
        _log_ms(f"  Pool size ARC_ALL_COMB_NJOBS={os.environ.get('ARC_ALL_COMB_NJOBS')!r} (override).")
    _log_ms("")
    all_dfs = []
    n_targets = len(targets_with_data)
    tasks: list[tuple] = []
    targets_to_compute: list[str] = []
    for ti, target in enumerate(targets_with_data):
        _log(f"  All-combinations target {ti + 1}/{n_targets}: {target}")
        target_csv = os.path.join(comb_dir, f"{target}.csv")
        if not force_run:
            cached = _try_read_all_comb_cache(target_csv)
            if cached is not None:
                all_dfs.append(cached)
                _log(f"    Loaded cached {target}.csv")
                continue
            if os.path.exists(target_csv):
                _log(f"    Ignoring invalid/empty cache {target}.csv; recomputing...")
        targets_to_compute.append(target)
        for K in range(N_GNNS):
            others = [i for i in range(N_GNNS) if i != K]
            for L in range(1, N_GNNS):
                for other_indices in itertools.combinations(others, L):
                    tasks.append((target, K, tuple(other_indices)))
        n_combo = N_GNNS * sum(comb(N_GNNS - 1, L) for L in range(1, N_GNNS))
        _log(
            f"    Queued {n_combo} combinations "
            f"({N_GNNS} bases x sum over L=1..6 of C(6,L))..."
        )

    global _ALL_COMB_REGISTRY
    if tasks:
        n_pending = len(set(t[0] for t in tasks))
        _log_ms(
            f"  Running {len(tasks)} combination jobs with {n_jobs} workers "
            f"across {n_pending} target(s) (one pool; small task payloads on fork)..."
        )
        ctx = mp.get_context()
        start_method = ctx.get_start_method()
        pool_kw: dict = {}
        if start_method == "fork":
            _ALL_COMB_REGISTRY = arc_residue_data
        else:
            pool_kw["initializer"] = _init_all_comb_registry
            pool_kw["initargs"] = (arc_residue_data,)
        chunksize = max(8, len(tasks) // (n_jobs * 8)) if n_jobs else 8
        try:
            if n_jobs <= 1:
                if start_method != "fork":
                    _init_all_comb_registry(arc_residue_data)
                flat_results = [_compute_arc_ensemble_row_registry(t) for t in tasks]
                if start_method != "fork":
                    _ALL_COMB_REGISTRY = None
            else:
                with ctx.Pool(n_jobs, **pool_kw) as pool:
                    flat_results = list(
                        pool.imap(_compute_arc_ensemble_row_registry, tasks, chunksize=chunksize)
                    )
        finally:
            _ALL_COMB_REGISTRY = None

        by_target: dict[str, list] = defaultdict(list)
        for row in flat_results:
            if row is not None:
                by_target[row["Target"]].append(row)
        for target in targets_to_compute:
            rows = by_target.get(target, [])
            df_t = pd.DataFrame(rows)
            target_csv = os.path.join(comb_dir, f"{target}.csv")
            df_t.to_csv(target_csv, index=False)
            all_dfs.append(df_t)
            _log(f"    Wrote {target}.csv ({len(rows)} rows)")
    if not all_dfs:
        return
    _log_ms("")
    _log_ms("  Building summary: aggregating per (target, L), applying BH to DeLong within each group...")
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_summary = _build_arc_ensemble_summary_from_combinations(df_all, output_dir, log_fn=_log_ms)
    _print_arc_all_comb_highlights(df_summary, log_fn=_log_ms, log_file_path=log_file_path)


def _run_all_comb_wilcoxon_tests(ensemble_dir):
    """Load all-combination CSVs; print Wilcoxon tests (ensemble vs individual) to log."""
    comb_dir = os.path.join(ensemble_dir, "arc_all_combinations")
    if not os.path.isdir(comb_dir):
        return
    bdf = []
    for f in sorted(os.listdir(comb_dir)):
        if f.endswith(".csv"):
            p = os.path.join(comb_dir, f)
            chunk = _try_read_all_comb_cache(p)
            if chunk is not None:
                bdf.append(chunk)
    if not bdf:
        return
    df = pd.concat(bdf, ignore_index=True)
    if "L" not in df.columns:
        return
    pairs = [
        ("patch_qs_pearson_ensemble", "patch_qs_pearson_individual"),
        ("patch_qs_spearman_ensemble", "patch_qs_spearman_individual"),
        ("patch_dockq_pearson_ensemble", "patch_dockq_pearson_individual"),
        ("patch_dockq_spearman_ensemble", "patch_dockq_spearman_individual"),
    ]
    for col in [p[0] for p in pairs] + [p[1] for p in pairs]:
        if col not in df.columns:
            return
    print("\n" + "=" * 80)
    print(f"  {EVAL_LOG_SEC_2_6}")
    print("=" * 80)
    print("ensemble vs random base predictor - all samples wilcoxon signed rank test ->")
    for A, B in pairs:
        stat, p_value = wilcoxon(df[A], df[B], alternative="greater")
        print(f"\t {A} > {B} ~ p-value: {p_value}")
    print("=" * 30)
    if multipletests is None:
        return
    for A, B in pairs:
        ps = []
        L_vals = []
        for l, ldf in df.groupby("L"):
            L_vals.append(l)
            ps.append(wilcoxon(ldf[A], ldf[B], alternative="greater")[1])
        ps_adj = multipletests(ps, method="fdr_bh")[1]
        for l, p_value in zip(L_vals, ps_adj):
            print(f"random base predictor - wilcoxon signed rank test, {A} > {B} ->")
            print(f"\t L={l} ~ BH corrected p-value: {p_value}")
        print("=" * 30)

    # Main-text Section 3.3 explicit p/q printout for L=6.
    # q-values are BH-adjusted across the four primary L=6 tests.
    if multipletests is not None:
        l6 = df[df["L"] == 6]
        primary = [
            ("true_interface_residue_AUC_ROC_ensemble", "true_interface_residue_AUC_ROC_individual", "TIR ROC AUC"),
            ("true_interface_residue_AUC_PRC_ensemble", "true_interface_residue_AUC_PRC_individual", "TIR PR AUC"),
            ("patch_qs_pearson_ensemble", "patch_qs_pearson_individual", "Patch QS PCC"),
            ("patch_dockq_pearson_ensemble", "patch_dockq_pearson_individual", "Patch DockQ PCC"),
        ]
        if len(l6) > 0 and all((a in l6.columns and b in l6.columns) for a, b, _ in primary):
            pvals = []
            labels = []
            for a, b, label in primary:
                pvals.append(wilcoxon(l6[a], l6[b], alternative="greater")[1])
                labels.append(label)
            qvals = multipletests(pvals, method="fdr_bh")[1]
            print("Main-text L=6 Wilcoxon + BH(q across 4 primary tests) ->")
            for label, p_value, q_value in zip(labels, pvals, qvals):
                print(f"\t{label}: p-value = {p_value}; q-value = {q_value}")
            print("=" * 30)
    print()


def _print_arc_all_comb_highlights(
    df_summary_all_comb, log_fn=None, log_file_path=None, df_all_comb=None
):
    """Log L=6 headline stats from the all-combinations summary (for main text section 3.3)."""
    _out = log_fn if log_fn is not None else lambda s: print(s, flush=True)
    _out("")
    _out("=" * 80)
    _out(f"  {EVAL_LOG_SEC_2_5}")
    _out("=" * 80)
    _out("")
    _out("  [main.tex §3.3 ``Ensemble aggregation improves robustness'']")
    _out(
        "    Lines tagged [main.tex] match statistics quoted in main.tex (L=6 all-combinations)."
    )
    _out(
        f"    Wilcoxon p- and q-values for the same four headline metrics are under "
        f"``{EVAL_LOG_SEC_2_6}'' in the block ``[main.tex §3.3 ensemble] L=6 Wilcoxon...''."
    )
    _out("")
    df_L6 = df_summary_all_comb[df_summary_all_comb["L"] == 6] if "L" in df_summary_all_comb.columns else df_summary_all_comb
    if len(df_L6) == 0:
        _out("  No L=6 all-combinations summary rows; run all-combinations first.")
        return
    n_t = len(df_L6)
    _out("  L=6 highlights ({} targets, mean over targets):".format(n_t))
    for col_pct, label in [
        ("iface_ROC_AUC_pct_ens_better", "True interface residue ROC AUC"),
        ("iface_PR_AUC_pct_ens_better", "True interface residue PR AUC"),
        ("patch_qs_PCC_pct_ens_better", "Patch QS PCC"),
        ("patch_dockq_PCC_pct_ens_better", "Patch DockQ PCC"),
    ]:
        if col_pct in df_L6.columns:
            pct = df_L6[col_pct].mean() * 100.0
            _out(
                "    [main.tex] {:>32}  % comparisons with ensemble > individual (mean over targets): {}".format(
                    label + ":", f"{pct:.1f}%" if pd.notna(pct) else "N/A"
                )
            )
    delong_raw_col = "true_interface_residue_DeLong_ROC_pvalue"
    if df_all_comb is not None and "L" in df_all_comb.columns and delong_raw_col in df_all_comb.columns:
        pool = df_all_comb[df_all_comb["L"] == 6]
        if len(pool) > 0:
            raw_med = pool[delong_raw_col].median()
            _out(
                "    [main.tex] Pooled median *raw* DeLong p (TIR ROC, all L=6 comparisons, n={}): {:.3e}".format(
                    len(pool), raw_med
                )
            )
    if "iface_ROC_AUC_delong_pct_sig_ens_better" in df_L6.columns and "iface_ROC_AUC_delong_pct_sig" in df_L6.columns:
        sig = df_L6["iface_ROC_AUC_delong_pct_sig"].mean() * 100.0
        sig_ens = df_L6["iface_ROC_AUC_delong_pct_sig_ens_better"].mean() * 100.0
        if pd.notna(sig) and pd.notna(sig_ens):
            # sig_ens is P(sig AND ens better) per target, averaged — not P(ens better | sig).
            cond = (100.0 * sig_ens / sig) if sig > 0 else float("nan")
            _out(
                "    [main.tex] DeLong (BH within each target, L=6): mean % comparisons with q<0.05: {:.1f}% "
                "(manuscript ``p values below 0.05''; BH-adjusted).".format(sig)
            )
            _out(
                "    [main.tex -- joint rate only] Mean % (BH sig and ensemble better): {:.1f}% "
                "(not the ``among significant'' proportion).".format(sig_ens)
            )
            _out(
                "    [main.tex] % favor ensemble *among* BH-significant comparisons "
                "(ratio joint/sig): {:.1f}%.".format(cond)
            )
        else:
            _out("    DeLong (BH within target,L): N/A (no valid DeLong stats).")
    if "iface_ROC_AUC_diff_mean" in df_L6.columns and "oligomeric_state" in df_L6.columns:
        for state in ["Dimer", "Multimer (>2)"]:
            sub = df_L6[df_L6["oligomeric_state"] == state]
            if len(sub) > 0:
                margin = sub["iface_ROC_AUC_diff_mean"].mean()
                mt = (
                    "other higher-order assemblies"
                    if state == "Multimer (>2)"
                    else "dimers"
                )
                _out(
                    "    [main.tex] Mean target-wise TIR ROC AUC margin (ensemble-individual) for {}: {} (n={} targets).".format(
                        mt, f"{margin:+.3f}" if pd.notna(margin) else "N/A", len(sub)
                    )
                )
    _out("")
    _out("  Lines tagged [main.tex] are the copy-check list for main.tex ensemble subsection (L=6).")
    _out("=" * 80)
    _out("")
    _out("End of all-combinations block (run, summary, highlights).")
