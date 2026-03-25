"""Load cached CSVs for manuscript figures and optional output directory override."""
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_scripts_root = os.path.dirname(_script_dir)
_eval_dir = os.path.join(_scripts_root, "eval")
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

import numpy as np
import pandas as pd

try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None

from casp16_eval_paths import (
    ARC_ENSEMBLE_DIR,
    MANUSCRIPT_FIGURES_DIR,
    POOLED_TABLES_DIR,
)

from plots_manuscript_constants import targets

# Optional override for tests or custom export path (default: MANUSCRIPT_FIGURES_DIR)
FIG_OUT_DIR = None


def manuscript_out_dir():
    return FIG_OUT_DIR if FIG_OUT_DIR is not None else MANUSCRIPT_FIGURES_DIR


def set_manuscript_figures_out_dir(out_dir=None):
    """Set where figure PNG/PDF are written; pass None to use MANUSCRIPT_FIGURES_DIR."""
    global FIG_OUT_DIR
    FIG_OUT_DIR = out_dir


def load_rankings():
    return pd.read_csv(os.path.join(POOLED_TABLES_DIR, "arc_rankings_summary.csv"))


def load_perf_table(strat_dir, metric):
    p = os.path.join(POOLED_TABLES_DIR, strat_dir, f"table_{metric}_performance.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None


def load_ensemble():
    """Summary table from exhaustive all-combinations ensemble analysis."""
    return pd.read_csv(os.path.join(ARC_ENSEMBLE_DIR, "arc_ensemble_summary_all_comb.csv"))


def _try_read_all_comb_csv(path: str) -> pd.DataFrame | None:
    """Same validity rules as ensemble ``_try_read_all_comb_cache``: skip missing/empty/unparseable CSVs."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return None
        df = pd.read_csv(path)
        if df.empty or len(df.columns) == 0:
            return None
        return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return None


def load_all_combination_rows(L):
    """Rows for ensemble size L stacked across targets (from arc_all_combinations/<target>.csv)."""
    bdf = []
    comb_dir = os.path.join(ARC_ENSEMBLE_DIR, "arc_all_combinations")
    for target in targets:
        p = os.path.join(comb_dir, f"{target}.csv")
        bdf_ = _try_read_all_comb_csv(p)
        if bdf_ is None or "L" not in bdf_.columns:
            continue
        bdf_ = bdf_[bdf_["L"] == L]
        if len(bdf_) == 0:
            continue
        bdf.append(bdf_)
    if not bdf:
        return pd.DataFrame()
    return pd.concat(bdf, ignore_index=True)


def bh_adjust_delong_within_target_l(bdf: pd.DataFrame) -> pd.Series:
    """Benjamini-Hochberg adjust DeLong p-values within each (Target, L) group."""
    pcol = "true_interface_residue_DeLong_ROC_pvalue"
    if bdf is None or len(bdf) == 0 or pcol not in bdf.columns:
        return pd.Series([], dtype=float, index=bdf.index if bdf is not None else None)
    if multipletests is None:
        return bdf[pcol].astype(float)
    if "Target" not in bdf.columns:
        return bdf[pcol].astype(float)

    group_cols = ["Target"]
    if "L" in bdf.columns:
        group_cols.append("L")

    out = pd.Series(index=bdf.index, dtype=float)
    for _, idx in bdf.groupby(group_cols, dropna=False).groups.items():
        idx = list(idx)
        p = bdf.loc[idx, pcol].astype(float)
        valid = p.notna()
        if valid.sum() == 0:
            out.loc[idx] = np.nan
            continue
        _, p_adj, _, _ = multipletests(p[valid].values, alpha=0.05, method="fdr_bh")
        p_out = p.copy()
        p_out.loc[valid] = p_adj
        out.loc[idx] = p_out.values
    return out
