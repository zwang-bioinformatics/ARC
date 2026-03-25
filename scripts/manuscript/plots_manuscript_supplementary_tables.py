"""LaTeX S1-S4 from casp16_eval outputs; run after eval or ``pixi run manuscript-tables``."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
_scripts_root = os.path.dirname(_script_dir)
_eval_dir = os.path.join(_scripts_root, "eval")
_common_dir = os.path.join(_scripts_root, "common")
for _p in (_eval_dir, _common_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from casp16_eval_constants import EVAL_LOG_SEC_3B
from casp16_eval_ensemble import ensemble_summary_all_comb_csv
from casp16_eval_paths import (
    MANUSCRIPT_TABLES_DIR,
    PER_TARGET_ANALYSIS_DIR,
    POOLED_TABLES_DIR,
    ensure_eval_output_layout,
)
from plots_manuscript_constants import targets

# S2: (score_type, display_name) for each metric block
S2_METRICS = [
    ("true_interface_residue", "Interface residue"),
    ("patch_qs", "Patch QS"),
    ("patch_dockq", "Patch DockQ"),
    ("QSBEST", "QSBEST"),
    ("QSGLOB", "QSGLOB"),
    ("DOCKQ_AVG", "DockQ-avg"),
]

# S2: metrics in display order (value col suffix, rank col suffix)
S2_COLS = [
    ("ROC AUC", "ROC AUC"),
    ("PR AUC", "PR AUC"),
    ("PCC", "PCC"),
    ("SCC", "SCC"),
    ("MAE", "MAE"),
]

# --- S3/S4 (L=6 all-comb summary; columns match arc_ensemble_summary_all_comb.csv) ---
_S34_COL_DIFF = "iface_ROC_AUC_diff_mean"
_S34_COL_CI_LO = "iface_ROC_AUC_diff_ci_lo"
_S34_COL_CI_HI = "iface_ROC_AUC_diff_ci_hi"
_S34_COL_PCT_BETTER = "iface_ROC_AUC_pct_ens_better"
_S34_COL_DELONG_SIG_ENS = "iface_ROC_AUC_delong_pct_sig_ens_better"
_S34_COL_DELONG_MEDIAN_P = "iface_ROC_AUC_delong_median_p"
_S34_STRAT_ORDER = [
    ("Target Type", "target_type", ["H", "T"]),
    ("Oligomeric State", "oligomeric_state", ["Dimer", "Multimer (>2)", "Unknown"]),
    ("Target Size", "size_category", ["Small (<1000)", "Medium (1000-1500)", "Large (1500-3000)", "Huge (>3000)"]),
    ("Difficulty", "difficulty", ["Easy (>0.85)", "Medium (0.7-0.85)", "Hard (<0.7)", "Unknown"]),
]


def _s34_latexify_gtlt(s: str) -> str:
    if s is None:
        return s
    s = s.replace("Multimer (>2)", "Multimer ($>2$)")
    s = s.replace("Small (<1000)", "Small ($<1000$)")
    s = s.replace("Huge (>3000)", "Huge ($>3000$)")
    s = s.replace("Easy (>0.85)", "Easy ($>0.85$)")
    s = s.replace("Hard (<0.7)", "Hard ($<0.7$)")
    return s


def _s34_format_delong_p_tex(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return r"\texttt{NaN}"
    if isinstance(p, (int, float)) and (p <= 0 or p < 1e-320):
        return r"$\approx 0$"
    if p >= 0.01:
        return f"${p:.2g}$"
    exp = int(np.floor(np.log10(p)))
    mant = p / (10**exp)
    return rf"${mant:.2g} \times 10^{{{exp}}}$"


def _s34_load_summary_L6_all_comb(summary_csv: str) -> pd.DataFrame:
    if not os.path.isfile(summary_csv):
        return pd.DataFrame()
    df = pd.read_csv(summary_csv)
    if "L" not in df.columns:
        return df
    return df[df["L"] == 6].copy()


def _s34_write_S3(df_summary: pd.DataFrame, out_dir: str) -> str | None:
    if len(df_summary) == 0 or _S34_COL_DIFF not in df_summary.columns:
        return None
    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    cap = (
        r"  \caption{Stratified ensemble benefit at $L=6$ (true interface residue ROC AUC) from all-combinations "
        r"analysis. ROC AUC Diff = mean(ensemble $-$ individual) over targets in category; "
        r"\% Ens Better = \% of combinations where ensemble better; DeLong \% sig.\ ens.\ = mean \% where Benjamini--Hochberg-adjusted "
        r"DeLong $p < 0.05$ and ensemble has higher ROC AUC.}"
    )
    lines.append(cap)
    lines.append(r"  \label{tab:S3allcomb}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{@{}llrrrr@{}}")
    lines.append(r"    \toprule")
    lines.append(r"    Stratification & Category & $N$ & ROC AUC Diff & \% Ens Better & DeLong \% sig.\ ens. \\")
    lines.append(r"    \midrule")
    for strat_name, col, order_vals in _S34_STRAT_ORDER:
        if col not in df_summary.columns:
            continue
        uniq = [str(x) for x in df_summary[col].dropna().unique()]
        seen: set[str] = set()
        for val in list(order_vals) + [u for u in uniq if u not in order_vals]:
            if val in seen:
                continue
            sub = df_summary[df_summary[col].astype(str) == str(val)]
            if len(sub) == 0:
                continue
            seen.add(val)
            roc_diff = sub[_S34_COL_DIFF].mean()
            pct = sub[_S34_COL_PCT_BETTER].mean() * 100.0
            sig = sub[_S34_COL_DELONG_SIG_ENS].mean() * 100.0 if _S34_COL_DELONG_SIG_ENS in sub.columns else np.nan
            diff_str = f"{roc_diff:+.3f}" if pd.notna(roc_diff) and not np.isnan(roc_diff) else "--"
            pct_str = f"{pct:.0f}\\%" if pd.notna(pct) and not np.isnan(pct) else "--"
            sig_str = f"{sig:.0f}\\%" if pd.notna(sig) and not np.isnan(sig) else r"\texttt{NaN}"
            val_tex = _s34_latexify_gtlt(str(val))
            lines.append(f"    {strat_name}     & {val_tex}   & {len(sub)} & {diff_str} & {pct_str} & {sig_str} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    out = os.path.join(out_dir, "table_supplementary_S3.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out


def _s34_write_S4(df_summary: pd.DataFrame, out_dir: str) -> str | None:
    if len(df_summary) == 0 or _S34_COL_DIFF not in df_summary.columns:
        return None
    col_target = "target"
    col_oly = "oligomeric_state"
    if col_target not in df_summary.columns or col_oly not in df_summary.columns:
        return None
    df_summary = df_summary.sort_values(col_target).reset_index(drop=True)
    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    cap = (
        r"  \caption{Per-target ensemble benefit at $L=6$ from all-combinations analysis. "
        r"ROC AUC margin = mean(ensemble $-$ individual); 95\% CI from combination distribution; "
        r"\% ens.\ better = \% of combinations where ensemble better; Median DeLong $p$ = median Benjamini--Hochberg-adjusted "
        r"DeLong $p$-value per target.}"
    )
    lines.append(cap)
    lines.append(r"  \label{tab:S4allcomb}")
    lines.append(r"  \scriptsize")
    lines.append(r"  \begin{tabular}{@{}ll r@{\,}l rr@{}}")
    lines.append(r"    \toprule")
    lines.append(r"    Target & Oligomeric state & \multicolumn{2}{c}{ROC AUC margin [95\% CI]} & \% ens.\ better & Median DeLong $p$ \\")
    lines.append(r"    \midrule")
    for _, row in df_summary.iterrows():
        t = row[col_target]
        oly = row[col_oly] if pd.notna(row[col_oly]) else "?"
        oly = _s34_latexify_gtlt(str(oly))
        diff = row[_S34_COL_DIFF]
        ci_lo = row.get(_S34_COL_CI_LO, np.nan)
        ci_hi = row.get(_S34_COL_CI_HI, np.nan)
        pct = row[_S34_COL_PCT_BETTER] * 100.0 if pd.notna(row[_S34_COL_PCT_BETTER]) and not np.isnan(row[_S34_COL_PCT_BETTER]) else np.nan
        med_p = row.get(_S34_COL_DELONG_MEDIAN_P, np.nan)
        diff_str = f"{diff:+.3f}" if pd.notna(diff) and not np.isnan(diff) else "--"
        ci_str = (
            f"[${ci_lo:+.3f}$, ${ci_hi:+.3f}$]"
            if pd.notna(ci_lo) and pd.notna(ci_hi) and not np.isnan(ci_lo) and not np.isnan(ci_hi)
            else "[--, --]"
        )
        pct_str = f"{pct:.0f}" if pd.notna(pct) and not np.isnan(pct) else "--"
        p_str = _s34_format_delong_p_tex(med_p)
        lines.append(f"    {t}    & {oly} & ${diff_str}$ & {ci_str} & {pct_str} & {p_str} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    out = os.path.join(out_dir, "table_supplementary_S4.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out


def write_supplementary_s3_s4_all_comb_tables(
    out_dir: str | None = None,
    arc_ensemble_dir: str | None = None,
) -> tuple[str | None, str | None]:
    """Write S3/S4 from ``arc_ensemble_summary_all_comb.csv`` (L=6 rows). Skip if missing or empty."""
    ensure_eval_output_layout()
    out = out_dir if out_dir is not None else MANUSCRIPT_TABLES_DIR
    os.makedirs(out, exist_ok=True)
    summary_path = ensemble_summary_all_comb_csv(arc_ensemble_dir)
    df_sum = _s34_load_summary_L6_all_comb(summary_path)
    if len(df_sum) == 0:
        print(
            f"    Skipped S3/S4 (no L=6 rows in {summary_path!r}; run full eval with all-combinations).",
            flush=True,
        )
        return None, None
    p3 = _s34_write_S3(df_sum, out)
    p4 = _s34_write_S4(df_sum, out)
    if p3:
        print(f"    Wrote {p3}", flush=True)
    if p4:
        print(f"    Wrote {p4}", flush=True)
    return p3, p4


def _s2_stratifications() -> list[tuple[str, str, int]]:
    """Folder key, short label, target count (from ``targets`` + stoichiometry table)."""
    from casp16_eval_data import df_local_stoch

    tdf = df_local_stoch[["trg", "n_trg_chains"]].drop_duplicates()
    tdf["trg"] = tdf["trg"].str.replace(".pdb", "", regex=False)
    chains = dict(zip(tdf["trg"], tdf["n_trg_chains"]))
    n_all = len(targets)
    n_dimer = sum(1 for t in targets if chains.get(t, 0) == 2)
    n_multimer = sum(1 for t in targets if chains.get(t, 0) > 2)
    return [
        ("all_targets", "All", n_all),
        ("dimer_only", "Dimer", n_dimer),
        ("multimer_only", "Multimer", n_multimer),
    ]


def _method_to_latex(name: str) -> str:
    return name.replace("_", r"\_")


def _find_cols(df: pd.DataFrame, prefix: str, metric: str) -> tuple[str | None, str | None]:
    val_col = None
    rank_col = None
    for c in df.columns:
        if c == "Method":
            continue
        if prefix in c and metric in c:
            if "Rank" in c:
                rank_col = c
            else:
                val_col = c
    return val_col, rank_col


def load_s1_rank_merge(per_target_dir: str | None = None) -> pd.DataFrame:
    """Merge CASP-style CSVs the same way as ``generate_S1`` / ``table_supplementary_S1.tex``."""
    base = per_target_dir if per_target_dir is not None else PER_TARGET_ANALYSIS_DIR
    tir_path = os.path.join(base, "casp_style_TIR_ROC_vs_PR_rank.csv")
    rs_path = os.path.join(base, "casp_style_RS_local_vs_true_interface_rank.csv")
    if not os.path.isfile(tir_path) or not os.path.isfile(rs_path):
        raise FileNotFoundError(f"missing CASP-style CSVs ({tir_path!r} or {rs_path!r})")

    tir = pd.read_csv(tir_path)
    rs = pd.read_csv(rs_path)
    tir = tir.rename(columns={"method": "Method"}) if "method" in tir.columns else tir
    rs = rs.rename(columns={"method": "Method"}) if "method" in rs.columns else rs
    merged = tir.merge(
        rs[["Method", "TIR_zscore", "TIR_rank", "RS_local_zscore", "RS_local_rank"]],
        on="Method",
        how="inner",
    )
    return merged.sort_values("TIR_rank", na_position="last")


def print_s1_style_log_summary(
    per_target_dir: str | None = None,
    n_targets: int | None = None,
    *,
    width: int = 100,
) -> bool:
    """Print Table S1 layout to stdout (eval log); same numbers as ``table_supplementary_S1.tex``."""
    try:
        merged = load_s1_rank_merge(per_target_dir)
    except FileNotFoundError as e:
        print(f"Table S1 log: skip ({e})", flush=True)
        return False
    if len(merged) == 0:
        print(f"Table S1 log: skip (merged table is empty)", flush=True)
        return False

    nt = n_targets if n_targets is not None else len(targets)
    print(f"\n{'=' * width}", flush=True)
    print(
        f"Supplementary Table S1 layout (CASP-style Z-scores, N={nt} targets; "
        "order by TIR combined rank; same merge as manuscript table)",
        flush=True,
    )
    print(f"{'=' * width}", flush=True)
    print(
        f"{'Method':<22}"
        f"{'ROC Z':>8}{' Rk':>4}"
        f"{'PR Z':>9}{' Rk':>4}"
        f"{'Comb':>8}{' Rk':>4}"
        f"{'RS-loc':>8}{' Rk':>4}",
        flush=True,
    )
    print(f"{'-' * 71}", flush=True)
    for _, row in merged.iterrows():
        m = str(row["Method"])
        print(
            f"{m:<22}"
            f"{float(row['TIR_ROC_AUC_zscore']):8.3f}{int(row['TIR_ROC_AUC_rank']):4d}"
            f"{float(row['TIR_PR_AUC_zscore']):9.3f}{int(row['TIR_PR_AUC_rank']):4d}"
            f"{float(row['TIR_zscore']):8.2f}{int(row['TIR_rank']):4d}"
            f"{float(row['RS_local_zscore']):8.2f}{int(row['RS_local_rank']):4d}",
            flush=True,
        )
    print(f"{'=' * width}", flush=True)
    return True


def generate_S1(per_target_dir: str | None = None, n_targets: int | None = None) -> str:
    """Generate S1 LaTeX from CASP-style CSVs."""
    merged = load_s1_rank_merge(per_target_dir)
    nt = n_targets if n_targets is not None else len(targets)

    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        rf"  \caption{{CASP-style $Z$-score rankings for all 10 single-model EMA evaluators over {nt} targets. TIR is the true interface residue $Z$-score; the TIR combined $Z$-score is the sum of the TIR ROC AUC and TIR PR AUC $Z$-scores. RS-local is the composite ranking score combining correlation and adaptive ROC AUC across patch and global metrics. Higher $Z$-score is better. Evaluators are ordered by TIR combined rank.}}",
        r"  \label{tab:S1}",
        r"  \small",
        r"  \begin{tabular}{@{}l rr rr rr rr@{}}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{TIR ROC AUC} & \multicolumn{2}{c}{TIR PR AUC} & \multicolumn{2}{c}{TIR Combined} & \multicolumn{2}{c}{RS-local} \\",
        r"    \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r"    Method & $Z$-score & Rank & $Z$-score & Rank & $Z$-score & Rank & $Z$-score & Rank \\",
        r"    \midrule",
    ]

    for _, row in merged.iterrows():
        method = _method_to_latex(str(row["Method"]))
        tir_roc_z = f"{row['TIR_ROC_AUC_zscore']:.3f}"
        tir_roc_r = int(row["TIR_ROC_AUC_rank"])
        tir_pr_z = f"{row['TIR_PR_AUC_zscore']:.3f}"
        tir_pr_r = int(row["TIR_PR_AUC_rank"])
        tir_comb_z = f"{row['TIR_zscore']:.2f}"
        tir_comb_r = int(row["TIR_rank"])
        rs_z = f"{row['RS_local_zscore']:.2f}"
        rs_r = int(row["RS_local_rank"])
        lines.append(
            f"    {method:25} & {tir_roc_z} & {tir_roc_r} & {tir_pr_z} & {tir_pr_r} & {tir_comb_z} & {tir_comb_r}  & {rs_z} & {rs_r} \\\\"
        )

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _get_prefix(score_type: str, df: pd.DataFrame) -> str:
    for c in df.columns:
        if c == "Method":
            continue
        parts = c.split()
        for i, p in enumerate(parts):
            if p in ("ROC", "PR", "PCC", "SCC", "MAE"):
                return " ".join(parts[:i])
    return ""


def collect_s2_body_rows(pooled_tables_dir: str) -> list[tuple[str | None, str, list[str]]]:
    """S2 table body rows shared by LaTeX generation and log printing.

    Each tuple is ``(reference metric name or None for continuation row, strat label, cell strings)``.
    """
    strat_specs = _s2_stratifications()
    out: list[tuple[str | None, str, list[str]]] = []
    for score_type, display_name in S2_METRICS:
        table_path = os.path.join(pooled_tables_dir, "all_targets", f"table_{score_type}_performance.csv")
        if not os.path.isfile(table_path):
            raise FileNotFoundError(f"Missing table: {table_path}")

        df_sample = pd.read_csv(table_path, nrows=1)
        prefix = _get_prefix(score_type, df_sample)
        if not prefix:
            raise ValueError(f"Cannot infer prefix for {score_type}")

        for i, (folder, label, n) in enumerate(strat_specs):
            path = os.path.join(pooled_tables_dir, folder, f"table_{score_type}_performance.csv")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing table: {path}")

            df = pd.read_csv(path)
            arc = df[df["Method"] == "ARC"]
            if arc.empty:
                raise ValueError(f"ARC not found in {path}")

            arc = arc.iloc[0]
            cells: list[str] = []
            for metric, _ in S2_COLS:
                val_col, rank_col = _find_cols(df, prefix, metric)
                if val_col is None or rank_col is None:
                    raise ValueError(f"Columns not found for {metric} in {path}")
                v = float(arc[val_col])
                r = int(arc[rank_col])
                cells.append(f"{v:.3f} ({r})")

            strat_label = f"{label} ({n})"
            out.append((display_name if i == 0 else None, strat_label, cells))
    return out


def print_pooled_s2_style_log_summary(
    pooled_tables_dir: str | None = None, *, width: int = 100
) -> bool:
    """Print pooled ARC summary in the same layout as supplementary Table S2 (stdout / eval log).

    Returns True if printed, False if tables are missing or invalid (caller may fall back).
    """
    base = pooled_tables_dir if pooled_tables_dir is not None else POOLED_TABLES_DIR
    all_dir = os.path.join(base, "all_targets")
    if not os.path.isdir(all_dir):
        return False
    try:
        rows = collect_s2_body_rows(base)
    except (FileNotFoundError, ValueError):
        return False

    strat_specs = _s2_stratifications()
    n_all, n_dimer, n_mult = strat_specs[0][2], strat_specs[1][2], strat_specs[2][2]
    print(f"\n{'=' * width}", flush=True)
    print(
        "  " + EVAL_LOG_SEC_3B + " (ARC value and rank in parentheses; 10 methods)",
        flush=True,
    )
    print(
        f"  All = {n_all} targets; Dimer = {n_dimer}; Multimer = {n_mult} "
        "(n >= 3 chains). Higher is better for ROC/PR/PCC/SCC; lower is better for MAE.",
        flush=True,
    )
    print(f"{'=' * width}", flush=True)

    ref_w, strat_w, cell_w = 20, 18, 13
    hdr = f"  {'Reference metric':<{ref_w}} {'Stratification':<{strat_w}}"
    hdr += "".join(f"{m:>{cell_w}}" for m, _ in S2_COLS)
    print(hdr, flush=True)
    print(f"  {'-' * (len(hdr) - 2)}", flush=True)

    first_block = True
    for name_opt, strat_label, cells in rows:
        if name_opt is not None and not first_block:
            print(flush=True)
        if name_opt is not None:
            first_block = False
        ref_col = (name_opt or "").ljust(ref_w)
        strat_col = strat_label.ljust(strat_w)
        cell_str = "".join(f"{c:>{cell_w}}" for c in cells)
        print(f"  {ref_col} {strat_col}{cell_str}", flush=True)

    print(f"{'=' * width}", flush=True)
    return True


def generate_S2(pooled_tables_dir: str | None = None) -> str:
    """Generate S2 LaTeX from pooled performance tables (ARC row only)."""
    base = pooled_tables_dir if pooled_tables_dir is not None else POOLED_TABLES_DIR
    strat_specs = _s2_stratifications()
    n_all = strat_specs[0][2]
    n_dimer = strat_specs[1][2]
    n_mult = strat_specs[2][2]

    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        rf"  \caption{{ARC performance under pooled, target-balanced evaluation by stratification, with each cell giving ARC's value and rank (in parentheses) out of 10 single-model EMA evaluators. All = {n_all} targets; Dimer = {n_dimer} targets; Multimer = {n_mult} targets ($n \geq 3$ chains). Higher is better for ROC AUC, PR AUC, PCC, and SCC ($\uparrow$); lower MAE is better ($\downarrow$).}}",
        r"  \label{tab:S2}",
        r"  \small",
        r"  \begin{tabular}{@{}ll ccccc@{}}",
        r"    \toprule",
        r"    Reference metric & Stratification & ROC AUC $\uparrow$ & PR AUC $\uparrow$ & PCC $\uparrow$ & SCC $\uparrow$ & MAE $\downarrow$ \\",
        r"    \midrule",
    ]

    started_any = False
    for name_opt, strat_label, cells in collect_s2_body_rows(base):
        if name_opt is not None:
            if started_any:
                lines.append(r"    \addlinespace")
            started_any = True
            lines.append(f"    {name_opt} & {strat_label:15} & " + " & ".join(cells) + r" \\")
        else:
            lines.append(f"                      & {strat_label:15} & " + " & ".join(cells) + r" \\")

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def write_supplementary_s1_s2_tables(out_dir: str | None = None) -> tuple[str, str]:
    """Write ``table_supplementary_S1.tex`` and ``S2.tex``; return paths."""
    ensure_eval_output_layout()
    out = out_dir if out_dir is not None else MANUSCRIPT_TABLES_DIR
    os.makedirs(out, exist_ok=True)

    s1_tex = generate_S1()
    s1_path = os.path.join(out, "table_supplementary_S1.tex")
    with open(s1_path, "w", encoding="utf-8") as f:
        f.write(s1_tex)
    print(f"    Wrote {s1_path}", flush=True)

    s2_tex = generate_S2()
    s2_path = os.path.join(out, "table_supplementary_S2.tex")
    with open(s2_path, "w", encoding="utf-8") as f:
        f.write(s2_tex)
    print(f"    Wrote {s2_path}", flush=True)

    return s1_path, s2_path


if __name__ == "__main__":
    from eval_log_tee import append_eval_log_tee

    append_eval_log_tee("plots_manuscript_supplementary_tables")
    try:
        write_supplementary_s1_s2_tables()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", flush=True)
        raise SystemExit(1) from e
    write_supplementary_s3_s4_all_comb_tables()
