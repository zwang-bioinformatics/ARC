"""Fig5 quantile heatmaps from casp16_eval PKL + EMA JSON; writes fig5.png, CSVs, targ_diff_tir.json."""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
import pickle
import sys
import warnings

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
_scripts_root = os.path.dirname(_script_dir)
_eval_dir = os.path.join(_scripts_root, "eval")
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

from plots_science_style import science_style_context

try:
    import scienceplots  # noqa: F401

    _HAS_SCIENCE = True
except ImportError:
    _HAS_SCIENCE = False

from tqdm import tqdm

from casp16_eval_constants import CASP16_EMA_RESULTS
from casp16_eval_paths import (
    ARC_RESIDUE_TRUTH_PRED_PKL,
    LOCAL_RESIDUE_QUANTILE_STATS_PKL,
    local_residue_quantile_stats_pkl_for_read,
)

from plots_manuscript_constants import MS_FIG_FILENAME_QUANTILE_COMPOSITE
from plots_manuscript_io import manuscript_out_dir

# ---------------------------------------------------------------------------
# Quantile bin layout (10% steps; seven single-GNN preds)
# ---------------------------------------------------------------------------
BIN_SIZE = 10
STEPS = 100 / BIN_SIZE
PRED_COLS = [f"pred_{i}" for i in range(7)]

Q_CONSENSUS = r"$Q(\text{consensus})$"
Q_PRED = r"$Q(\text{pred})$"

CUSTOM_CMAP = sns.color_palette("blend:#4A90C2,#F7F7F7,#3D9B5A", as_cmap=True)

def _pivot_mean_tir_wide(df_pl: pl.DataFrame, q_col: str) -> pd.DataFrame:
    """Mean ``true_interface`` per (``target``, quantile bin); wide matrix for heatmaps."""
    df_pl = df_pl.filter(pl.col(q_col).is_not_null())
    agg = df_pl.group_by(["target", q_col]).agg(pl.col("true_interface").mean())
    if agg.is_empty():
        return pd.DataFrame(columns=pd.Index([], dtype=float, name=q_col))
    wide = agg.pivot(on=q_col, index="target", values="true_interface")
    pdf = wide.to_pandas().set_index("target")
    # Polars pivot can name null keys ``null``; only keep numeric bin columns (10..100).
    cols = _sorted_quantile_bin_columns(pdf.columns)
    if not cols:
        return pd.DataFrame(index=pdf.index, columns=pd.Index([], dtype=float, name=q_col))
    out = pdf[cols]
    out.columns = pd.Index([float(c) for c in out.columns], dtype=float, name=q_col)
    return out


def _sorted_quantile_bin_columns(columns) -> list:
    """Sort bin labels by numeric value; skip non-numeric / null pivot keys."""
    parsed: list[tuple[float, object]] = []
    for c in columns:
        if c is None:
            continue
        if isinstance(c, str) and c.lower() in ("null", "nan", "none"):
            continue
        try:
            fv = float(c)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(fv):
            continue
        parsed.append((fv, c))
    return [c for _, c in sorted(parsed, key=lambda t: t[0])]


def _load_arc_residue_data():
    if not os.path.isfile(ARC_RESIDUE_TRUTH_PRED_PKL):
        raise FileNotFoundError(
            f"Missing {ARC_RESIDUE_TRUTH_PRED_PKL}. Run casp16_eval to build arc residue cache."
        )
    with open(ARC_RESIDUE_TRUTH_PRED_PKL, "rb") as f:
        return pickle.load(f)


def _build_stats_rows(arc_residue_data: dict) -> pd.DataFrame:
    rows = {
        "target": [],
        "mean": [],
        "residue": [],
        "min": [],
        "max": [],
        "median": [],
        "full_preds": [],
        "frequency": [],
        "true_interface": [],
    }

    for target in tqdm(list(arc_residue_data), desc="local residue quantile stats"):
        tdata = []
        residue_bias: dict[str, float] = {}

        for model in arc_residue_data[target]:
            json_path = os.path.join(CASP16_EMA_RESULTS, f"{model}_{target}.json")
            if not os.path.isfile(json_path):
                continue
            with open(json_path, "r") as jf:
                local_truth = json.load(jf)
            arc_residue_data[target][model] = arc_residue_data[target][model].copy()
            mapping = {v: k for k, v in local_truth["chain_mapping"].items()}
            tres = arc_residue_data[target][model]["residue"].apply(
                lambda x: (
                    "UNK"
                    if x.split(".")[0] not in mapping
                    else (mapping[x.split(".")[0]] + "." + x.split(".")[1])
                )
            )
            arc_residue_data[target][model] = arc_residue_data[target][model].assign(tres=tres)

            for residue in tres:
                if residue == "UNK":
                    continue
                residue_bias[residue] = residue_bias.get(residue, 0) + 1

            tdata.append(arc_residue_data[target][model])

        if not tdata:
            continue

        n_models = len(arc_residue_data[target])
        for residue in residue_bias:
            residue_bias[residue] /= n_models

        # Suppress concat FutureWarning; keep full tdata union for stable groupby/melt/stats.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=r".*DataFrame concatenation with empty or all-NA entries.*",
            )
            tdata_df = pd.concat(tdata, ignore_index=True)

        for residue, df in tdata_df.groupby("tres"):
            if residue == "UNK":
                continue
            gt = df["true_interface_residue"].unique()
            if len(gt) != 1:
                continue
            gt = gt.item()

            preds = (
                df.melt(
                    id_vars=["true_interface_residue"],
                    value_vars=PRED_COLS,
                    var_name="prediction_type",
                    value_name="prediction_value",
                )["prediction_value"]
                .dropna()
                .to_numpy()
            )
            if preds.shape[0] == 0:
                continue

            rows["target"].append(target)
            rows["mean"].append(float(preds.mean()))
            rows["residue"].append(residue)
            rows["min"].append(float(preds.min()))
            rows["max"].append(float(preds.max()))
            rows["median"].append(float(np.median(preds)))
            rows["full_preds"].append(preds)
            rows["frequency"].append(residue_bias[residue])
            rows["true_interface"].append(gt)

    return pd.DataFrame(rows)


def _load_or_build_stats(cache_pkl: str, force_rebuild: bool) -> pd.DataFrame:
    read_pkl = (
        local_residue_quantile_stats_pkl_for_read()
        if cache_pkl == LOCAL_RESIDUE_QUANTILE_STATS_PKL
        else cache_pkl
    )
    if not force_rebuild and os.path.isfile(read_pkl):
        try:
            return pd.read_pickle(read_pkl)
        except Exception:
            pass
    arc_residue_data = _load_arc_residue_data()
    out = _build_stats_rows(arc_residue_data)
    os.makedirs(os.path.dirname(cache_pkl) or ".", exist_ok=True)
    out.to_pickle(cache_pkl)
    return out


def _consensus_heatmap_df(data: pd.DataFrame) -> pd.DataFrame:
    """Per-target ``mean`` rank -> ``Q_CONSENSUS`` bins; mean TIR per bin (Polars windows + pivot)."""
    df_pl = pl.from_pandas(data).with_columns(pl.col("target").cast(pl.Categorical))
    df_pl = df_pl.with_columns(
        (
            BIN_SIZE
            + BIN_SIZE
            * (
                (
                    (pl.col("mean").rank(method="average").over("target") - 0.5)
                    / pl.col("mean").count().over("target")
                    * STEPS
                )
                .floor()
                .clip(0, STEPS - 1)
            )
        ).alias(Q_CONSENSUS)
    )
    return _pivot_mean_tir_wide(df_pl, Q_CONSENSUS)


def _single_pred_quantile_heatmap_df(data: pd.DataFrame) -> pd.DataFrame:
    """Explode per-GNN preds, assign ``Q_PRED`` bins (window rank/count), mean TIR -> wide heatmap."""
    df_pl = pl.from_pandas(data).with_columns(pl.col("target").cast(pl.Categorical))
    df_pl = df_pl.explode("full_preds")
    df_pl = df_pl.with_columns(pl.col("full_preds").cast(pl.Float64))
    df_pl = df_pl.with_columns(
        (
            BIN_SIZE
            + BIN_SIZE
            * (
                (
                    (
                        (pl.col("full_preds").rank(method="average").over("target") - 0.5)
                        / pl.col("full_preds").count().over("target")
                    )
                    * STEPS
                )
                .floor()
                .clip(0, STEPS - 1)
            )
        ).alias(Q_PRED)
    )
    return _pivot_mean_tir_wide(df_pl, Q_PRED)


def _export_targ_diff_tir_json(data_full: pd.DataFrame, out_path: str) -> None:
    """Per-target mean TIR vs frequency threshold -> JSON."""
    difficulty: dict = {}
    for thresh in (0.0, 0.01, 0.025, 0.05, 0.1, 0.25):
        sub = data_full[data_full["frequency"] >= thresh].groupby("target")["true_interface"].mean()
        for target, val in sub.items():
            difficulty.setdefault(target, {})[thresh] = float(val)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(difficulty, f, indent=4)


def fig5_local_residue_quantile_heatmap(
    freq_threshold: float = 0.0,
    cache_stats_pkl: str | None = None,
    force_rebuild: bool = False,
):
    """Write PNG/PDF and tables (CSV + ``targ_diff_tir.json``) for fig5 to ``manuscript_out_dir()``."""
    cache_pkl = cache_stats_pkl if cache_stats_pkl is not None else LOCAL_RESIDUE_QUANTILE_STATS_PKL
    print(f"    Fig5 local residue quantile heatmap (consensus + single-GNN quantiles)...", flush=True)

    data_full = _load_or_build_stats(cache_pkl, force_rebuild=force_rebuild)
    if len(data_full) == 0:
        print(f"      No residue stats rows; skip fig5 heatmap.", flush=True)
        return

    out_dir = manuscript_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    targ_json = os.path.join(out_dir, "targ_diff_tir.json")
    _export_targ_diff_tir_json(data_full, targ_json)
    print(f"      Wrote {targ_json}", flush=True)

    data = data_full[data_full["frequency"] >= freq_threshold].copy()
    if len(data) == 0:
        print(f"      No rows after frequency filter; skip fig5 heatmap plots/CSVs.", flush=True)
        return
    ptir = data.groupby("target")["true_interface"].mean().to_dict()
    print(f"      Targets after frequency filter (>={freq_threshold}): {len(data['target'].unique())}", flush=True)

    consensus_df = _consensus_heatmap_df(data)

    side_df = data.groupby("target")["true_interface"].mean().to_frame()
    side_df[Q_PRED] = "All"
    side_df = side_df.pivot_table(
        index="target",
        columns=Q_PRED,
        values="true_interface",
        aggfunc="mean",
        observed=False,
    )

    heatmap_df = _single_pred_quantile_heatmap_df(data)

    sort_keys = list(sorted(ptir.keys(), key=lambda x: ptir[x], reverse=True))
    heatmap_df = heatmap_df.reindex(sort_keys)
    side_df = side_df.reindex(sort_keys)
    consensus_df = consensus_df.reindex(sort_keys)

    consensus_df.to_csv(os.path.join(out_dir, "quant_consensus_heat.csv"))
    heatmap_df.to_csv(os.path.join(out_dir, "quant_heat.csv"))

    norm = colors.TwoSlopeNorm(vcenter=0.5, vmin=0, vmax=1)
    base = os.path.join(out_dir, MS_FIG_FILENAME_QUANTILE_COMPOSITE)

    ctx = science_style_context if _HAS_SCIENCE else nullcontext
    with ctx():
        fig, (ax_side, ax_consensus, ax_single, ax_cbar) = plt.subplots(
            1,
            4,
            figsize=(25, 10),
            gridspec_kw={"width_ratios": [1.5, 11.5, 11.5, 0.5]},
        )

        sns.heatmap(
            side_df,
            fmt=".2f",
            ax=ax_side,
            cbar=False,
            cmap=CUSTOM_CMAP,
            norm=norm,
            annot=True,
            yticklabels=True,
            annot_kws={"size": 14.5},
        )

        ax_side.set_ylabel("Target")
        ax_side.set_xlabel("")
        ax_side.minorticks_off()
        ax_side.tick_params(axis="x", which="both", bottom=False, top=False)
        ax_side.set_ylabel(ax_side.get_ylabel(), fontsize=25, labelpad=10)

        sns.heatmap(
            consensus_df,
            annot=False,
            cmap=CUSTOM_CMAP,
            norm=norm,
            ax=ax_consensus,
            cbar=False,
            yticklabels=False,
        )

        sns.heatmap(
            heatmap_df,
            annot=False,
            cmap=CUSTOM_CMAP,
            norm=norm,
            ax=ax_single,
            cbar_ax=ax_cbar,
            yticklabels=False,
        )

        common_ylim = ax_side.get_ylim()
        ax_consensus.set_ylim(common_ylim)
        ax_single.set_ylim(common_ylim)

        ax_consensus.yaxis.set_ticks([])
        ax_single.yaxis.set_ticks([])
        ax_consensus.set_ylabel("")
        ax_single.set_ylabel("")

        ax_cbar.set_ylabel("True Interface Residue Rate", size=25, labelpad=18)
        ax_cbar.yaxis.set_ticks_position("left")
        ax_cbar.tick_params(labelsize=19)

        ax_single.set_xlabel(r"$Q(\text{single})$", fontsize=25, labelpad=10)
        ax_consensus.set_xlabel(ax_consensus.get_xlabel(), fontsize=25, labelpad=10)

        ax_consensus.tick_params(axis="both", which="major", labelsize=19)
        ax_single.tick_params(axis="both", which="major", labelsize=19)
        ax_side.tick_params(axis="both", which="major", labelsize=19)

        ax_single.set_xticklabels([f"{float(label.get_text()):g}" for label in ax_single.get_xticklabels()])
        ax_consensus.set_xticklabels([f"{float(label.get_text()):g}" for label in ax_consensus.get_xticklabels()])

        plt.subplots_adjust(wspace=0.05)

        pos = ax_cbar.get_position()
        ax_cbar.set_position([pos.x0 + 0.01, pos.y0, pos.width, pos.height])

        plt.savefig(f"{base}.png", format="png", dpi=350, bbox_inches="tight", transparent=False)
        # for ext in ("pdf",):
        #     plt.savefig(f"{base}.{ext}", dpi=350, bbox_inches="tight", transparent=False)

    plt.clf()
    plt.close()

    print(f"      Saved {base}.png", flush=True)


if __name__ == "__main__":
    from plots_manuscript_io import set_manuscript_figures_out_dir

    set_manuscript_figures_out_dir(None)
    fig5_local_residue_quantile_heatmap()
