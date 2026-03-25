"""Manuscript figures used in the Wiley submission (ARC pipeline outputs only).

Run via ``plots_manuscript_figures.py``. PNG basenames match ``main.tex`` / ``supplementary.tex``:

  - ``fig3`` - rank profile (function ``fig8_rank_profile_comparison``; historical internal name).
  - ``fig5`` - quantile heatmap composite (``plots_local_residue_quantile_heatmap``).
  - ``fig6a`` / ``fig6b`` - L=6 lollipop and patch 2x2 ensemble benefit.
  - ``figS1`` - supplementary ensemble scaling heatmap (``fig7_ensemble_scaling_heatmap``).

Pooled ROC panels ``fig2a``-``fig2f`` are written by ``casp16_eval_pooled.plot_figure2_roc_curves``.

  - ``plots_manuscript_supplementary_tables.py`` - LaTeX S1-S4 under ``MANUSCRIPT_TABLES_DIR``.

Fig1 and non-pooled fig2 sources are not in this module.
T1259o structure fig4 is in plots_fig4_t1259o_structure.py.
"""
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_scripts_root = os.path.dirname(_script_dir)
_eval_dir = os.path.join(_scripts_root, "eval")
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, MultipleLocator

from plots_manuscript_constants import (
    ARC_RED,
    BLUE,
    ENSEMBLE_ACCENT,
    ENSEMBLE_ALL_COMB_COL_MAP,
    ENSEMBLE_FONT_AXES,
    ENSEMBLE_FONT_LEGEND,
    ENSEMBLE_FONT_TICK,
    ENSEMBLE_FONT_TITLE,
    GREEN,
    GREY,
    HEATMAP_FONT_AXES,
    HEATMAP_FONT_TICK,
    MS_FIG_FILENAME_ENSEMBLE_PATCH_2X2,
    MS_FIG_FILENAME_LOLLIPOP_L6,
    MS_FIG_FILENAME_RANK_PROFILE,
    MS_FIG_FILENAME_SI_SCALING_HEATMAP,
    MULTIMER_LABEL_LATEX,
    MUTED_BLUE,
    ORANGE,
    STRAT_DISPLAY,
)
from plots_manuscript_io import (
    bh_adjust_delong_within_target_l,
    load_all_combination_rows,
    load_ensemble,
    load_rankings,
    manuscript_out_dir,
)


def fig3_ensemble_benefit_vs_L_patch_only_2x2():
    """2x2 grid: row 0 = PCC (Patch QS, Patch DockQ), row 1 = SCC (Patch QS, Patch DockQ)."""
    df_summary = load_ensemble()
    metrics_2x2 = [
        [("patch_qs_PCC", "Patch QS PCC"), ("patch_dockq_PCC", "Patch DockQ PCC")],
        [("patch_qs_SCC", "Patch QS SCC"), ("patch_dockq_SCC", "Patch DockQ SCC")],
    ]
    all_prefixes = [m[0] for row in metrics_2x2 for m in row]
    col_map = {k: ENSEMBLE_ALL_COMB_COL_MAP[k] for k in all_prefixes}

    Ls = sorted(df_summary["L"].unique())
    ens_sizes = [L + 1 for L in Ls]

    rng = np.random.default_rng(42)
    n_resamp = 10000
    per_l_delta_ci = {}
    for L in Ls:
        print(f"    Loading all-combination rows L={L} (patch only, 2x2)...", flush=True)
        bdf = load_all_combination_rows(L)
        for prefix, (ens_col, ind_col) in col_map.items():
            diffs = (bdf[ens_col] - bdf[ind_col]).values
            tgt = bdf["Target"].values
            per_target = {}
            for t, d in zip(tgt, diffs):
                per_target.setdefault(t, []).append(d)
            for t in per_target:
                per_target[t] = np.array(per_target[t])
            sampled = np.column_stack([
                rng.choice(per_target[t], size=n_resamp, replace=True)
                for t in per_target
            ])
            agg_means = sampled.mean(axis=1)
            per_l_delta_ci[(prefix, L)] = (
                agg_means.mean(),
                np.percentile(agg_means, 2.5),
                np.percentile(agg_means, 97.5),
            )
        del bdf

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            prefix, title = metrics_2x2[i][j]
            pct_col = f"{prefix}_pct_ens_better"
            means = []
            ci_los = []
            ci_his = []
            pcts = []
            for L in Ls:
                m, lo, hi = per_l_delta_ci[(prefix, L)]
                means.append(m)
                ci_los.append(lo)
                ci_his.append(hi)
                sub = df_summary[df_summary["L"] == L]
                pcts.append(sub[pct_col].mean() * 100)
            panel_lo = min(ci_los)
            panel_hi = max(ci_his)
            margin = (panel_hi - panel_lo) * 0.05 or 0.01
            ax.set_ylim(panel_lo - margin, panel_hi + margin)
            ax.plot(ens_sizes, means, "o-", color=ENSEMBLE_ACCENT, linewidth=2, markersize=5, zorder=5)
            ax.fill_between(ens_sizes, ci_los, ci_his, alpha=0.25, color=ENSEMBLE_ACCENT)
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
            ax.set_title(title, fontsize=ENSEMBLE_FONT_TITLE)
            if i == 1:
                ax.set_xlabel("Ensemble Size (1 + L)", fontsize=ENSEMBLE_FONT_AXES)
            ax.set_xticks(ens_sizes)
            if j == 0:
                ax.set_ylabel(r"$\Delta$ (Ensemble $-$ Individual)", fontsize=ENSEMBLE_FONT_AXES)
            ax.tick_params(axis="both", labelsize=ENSEMBLE_FONT_TICK)
            ax2 = ax.twinx()
            ax2.bar(ens_sizes, pcts, alpha=0.15, color=MUTED_BLUE, width=0.5, zorder=1)
            ax2.set_ylim(40, 70 if i == 1 else 80)
            ax2.tick_params(axis="y", labelcolor=MUTED_BLUE, labelsize=ENSEMBLE_FONT_TICK)
            if j == 1:
                ax2.set_ylabel(r"\% Ensemble Better", fontsize=ENSEMBLE_FONT_LEGEND, color=MUTED_BLUE)
            else:
                ax2.set_ylabel("")
    fig.subplots_adjust(hspace=0.35, wspace=0.30)
    out_png = os.path.join(manuscript_out_dir(), f"{MS_FIG_FILENAME_ENSEMBLE_PATCH_2X2}.png")
    plt.savefig(out_png, format="png", dpi=350, bbox_inches="tight")
    print(
        f"  Fig 6b ({MS_FIG_FILENAME_ENSEMBLE_PATCH_2X2}.png): patch-only 2x2 ensemble benefit vs L",
        flush=True,
    )
    plt.close()


def _fig4_base_lollipop(
    d6,
    color_by,
    color_map,
    legend_labels,
    suffix,
    subtitle,
    L_val=6,
    out_dir=None,
    target_order=None,
    figsize=(10, 8),
    ensemble_output_name=None,
    manuscript_basename=None,
    use_heatmap_fonts=False,
):
    """Shared lollipop core with right-side context panels for target size and chain count."""
    from matplotlib.lines import Line2D
    import matplotlib.gridspec as gridspec

    if out_dir is None:
        out_dir = manuscript_out_dir()
    use_ensemble_fonts = ensemble_output_name is not None or use_heatmap_fonts

    if target_order is not None:
        d6 = d6.set_index("target").reindex(target_order).dropna(how="all").reset_index()
    else:
        d6 = d6.sort_values("iface_ROC_AUC_diff_mean", ascending=True).reset_index(drop=True)
    targets = d6["target"].values
    diffs = d6["iface_ROC_AUC_diff_mean"].values
    ci_lo = d6["iface_ROC_AUC_diff_ci_lo"].values
    ci_hi = d6["iface_ROC_AUC_diff_ci_hi"].values
    sig_outline = None
    if "sig_delong_favor_ens" in d6.columns:
        sig_outline = d6["sig_delong_favor_ens"].astype(bool).values

    def _match_color(val, cmap):
        s = str(val)
        if s in cmap:
            return cmap[s]
        for k, v in cmap.items():
            if k in s:
                return v
        return GREY

    colors = [_match_color(c, color_map) for c in d6[color_by].values]
    y = np.arange(len(targets))

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, width_ratios=[5, 1.2, 1.2], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_size = fig.add_subplot(gs[1], sharey=ax)
    ax_chains = fig.add_subplot(gs[2], sharey=ax)

    ax.hlines(y, 0, diffs, colors=colors, linewidth=1.5)
    ax.scatter(diffs, y, c=colors, s=30, zorder=5)
    if sig_outline is not None and sig_outline.any():
        ax.scatter(
            diffs[sig_outline],
            y[sig_outline],
            s=100,
            facecolors="none",
            edgecolors=ENSEMBLE_ACCENT,
            linewidths=1.0,
            zorder=6,
        )
    for i in range(len(targets)):
        ax.plot([ci_lo[i], ci_hi[i]], [y[i], y[i]], color=colors[i], alpha=0.3, linewidth=3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    if use_ensemble_fonts:
        fs_ax = HEATMAP_FONT_AXES
        fs_tick = HEATMAP_FONT_TICK
        fs_small = HEATMAP_FONT_TICK
        fs_leg = HEATMAP_FONT_TICK
    else:
        fs_ax = fs_tick = fs_small = fs_leg = None

    ax.set_yticks(y)
    _kw_tick = {} if fs_tick is None else {"fontsize": fs_tick}
    ax.set_yticklabels(targets, **_kw_tick)
    _kw_ax = {} if fs_ax is None else {"fontsize": fs_ax}
    ax.set_ylabel("Target", **_kw_ax)
    ax.set_xlabel(r"$\Delta$ Interface ROC AUC (Ensemble $-$ Individual)", **_kw_ax)
    ax.grid(axis="x", alpha=0.2)

    sizes = d6["target_size"].values
    ax_size.barh(y, sizes, color=colors, height=0.6, edgecolor="none", alpha=0.5)
    ax_size.set_xscale("log")
    ax_size.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
    _kw_small = {} if fs_small is None else {"fontsize": fs_small}
    ax_size.set_xlabel(r"Residues ($\log_{10}$ scale)", **_kw_small)
    ax_size.set_title("Target\nSize", fontweight="bold", **_kw_small)
    ax_size.tick_params(axis="y", labelleft=False)
    if fs_tick is not None:
        ax_size.tick_params(axis="x", labelsize=fs_tick - 1)
    for spine in ["top", "right"]:
        ax_size.spines[spine].set_visible(False)

    chains = d6["n_chains"].values
    ax_chains.barh(y, chains, color=colors, height=0.6, edgecolor="none", alpha=0.5)
    ax_chains.set_xlabel("Chains", **_kw_small)
    ax_chains.set_title("No.\nChains", fontweight="bold", **_kw_small)
    ax_chains.xaxis.set_major_locator(MultipleLocator(2))
    ax_chains.tick_params(axis="y", labelleft=False)
    if fs_tick is not None:
        ax_chains.tick_params(axis="x", labelsize=fs_tick - 1)
    for spine in ["top", "right"]:
        ax_chains.spines[spine].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=v, markersize=7, linestyle="None", label=k)
        for k, v in legend_labels.items()
    ]
    if sig_outline is not None:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=ENSEMBLE_ACCENT,
                markerfacecolor="none",
                markersize=7,
                markeredgewidth=1.0,
                linestyle="None",
                label=r"DeLong $q<0.05$, ensemble better",
            )
        )
    _kw_leg = {} if fs_leg is None else {"fontsize": fs_leg}
    ax.legend(handles=legend_elements, loc="upper left", frameon=True, **_kw_leg)

    if ensemble_output_name:
        plt.savefig(os.path.join(out_dir, f"{ensemble_output_name}.png"), format="png", dpi=350, bbox_inches="tight")
        print(f"  Ensemble panel (a): {ensemble_output_name}.png", flush=True)
    elif manuscript_basename:
        out_png = os.path.join(out_dir, f"{manuscript_basename}.png")
        plt.savefig(out_png, format="png", dpi=350, bbox_inches="tight")
        print(f"  Fig 6a ({manuscript_basename}.png): per-target lollipop L={L_val}", flush=True)
    else:
        base = os.path.join(out_dir, f"fig4_lollipop_L{L_val}_{suffix}")
        plt.savefig(base + ".png", format="png", dpi=350, bbox_inches="tight")
        print(f"  Fig 4 (L={L_val}, {suffix}): Per-target lollipop", flush=True)
    plt.close()


def fig4_per_target_lollipop():
    """Per-target lollipop at L=6, oligomer coloring only (main text ``fig6a``)."""
    df = load_ensemble()
    Ls = [6] if 6 in df["L"].values else []
    color_by, cmap, legend, suffix, subtitle = (
        "oligomeric_state",
        {"Dimer": MUTED_BLUE, "Multimer (>2)": GREY},
        {"Dimers": MUTED_BLUE, MULTIMER_LABEL_LATEX: GREY},
        "by_oligomer",
        " - colored by oligomeric state",
    )

    for L in Ls:
        dL = df[df["L"] == L].copy()
        try:
            bdf = load_all_combination_rows(L)
            if len(bdf) > 0:
                bdf = bdf.copy()
                bdf["true_interface_residue_DeLong_ROC_pvalue_bh"] = bh_adjust_delong_within_target_l(bdf)
            sig = (
                (bdf["true_interface_residue_DeLong_ROC_pvalue_bh"].astype(float) < 0.05)
                & (
                    bdf["true_interface_residue_AUC_ROC_ensemble"].astype(float)
                    > bdf["true_interface_residue_AUC_ROC_individual"].astype(float)
                )
            )
            frac = sig.groupby(bdf["Target"]).mean()
            dL["sig_delong_favor_ens_frac"] = dL["target"].map(frac).fillna(0.0)
            dL["sig_delong_favor_ens"] = dL["sig_delong_favor_ens_frac"] >= 0.5
        except Exception as e:
            print(f"Warning: could not compute DeLong significance overlay for Fig 4: {e}", flush=True)

        target_order = dL.sort_values("iface_ROC_AUC_diff_mean", ascending=True)["target"].tolist()
        _fig4_base_lollipop(
            dL,
            color_by=color_by,
            color_map=cmap,
            legend_labels=legend,
            suffix=suffix,
            subtitle=subtitle,
            L_val=L,
            out_dir=manuscript_out_dir(),
            manuscript_basename=MS_FIG_FILENAME_LOLLIPOP_L6,
            target_order=target_order,
            use_heatmap_fonts=False,
        )


def fig7_ensemble_scaling_heatmap():
    df = load_ensemble()
    metrics = [
        ("iface_ROC_AUC_diff_mean", "Interface\nROC AUC"),
        ("iface_PR_AUC_diff_mean", "Interface\nPR AUC"),
        ("patch_qs_PCC_diff_mean", "Patch QS\nPCC"),
        ("patch_qs_SCC_diff_mean", "Patch QS\nSCC"),
        ("patch_dockq_PCC_diff_mean", "Patch DockQ\nPCC"),
        ("patch_dockq_SCC_diff_mean", "Patch DockQ\nSCC"),
        ("local_lddt_PCC_diff_mean", "iLDDT\nPCC"),
        ("local_cad_PCC_diff_mean", "CAD\nPCC"),
    ]

    Ls = sorted(df["L"].unique())
    data = np.zeros((len(metrics), len(Ls)))
    ylabels = []
    for i, (col, label) in enumerate(metrics):
        ylabels.append(label)
        for j, L in enumerate(Ls):
            data[i, j] = df[df["L"] == L][col].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    vmax = max(abs(data.min()), abs(data.max()))
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["#d73027", "#ffffff", "#1a9850"], N=256)
    im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(Ls)))
    ax.set_xticklabels([f"L={L}" for L in Ls], fontsize=8)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_title("Mean Ensemble Improvement Across Metrics and Ensemble Sizes", fontsize=11, fontweight="bold")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            sign = "+" if v > 0 else ""
            color = "black" if abs(v) < vmax * 0.6 else "white"
            ax.text(j, i, f"{sign}{v:.3f}", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r"$\Delta$ (Ensemble $-$ Individual)", fontsize=9)

    out_png = os.path.join(manuscript_out_dir(), f"{MS_FIG_FILENAME_SI_SCALING_HEATMAP}.png")
    plt.savefig(out_png, format="png", dpi=350, bbox_inches="tight")
    print(f"  Fig S1 ({MS_FIG_FILENAME_SI_SCALING_HEATMAP}.png): ensemble scaling heatmap", flush=True)
    plt.close()


def fig8_rank_profile_comparison():
    df = load_rankings().set_index("Stratification")
    key_metrics = [
        ("True Interface Residue ROC AUC Rank", "Interface\nROC AUC"),
        ("True Interface Residue PCC Rank", "Interface\nPCC"),
        ("True Interface Residue SCC Rank", "Interface\nSCC"),
        ("Patch QS PCC Rank", "Patch QS\nPCC"),
        ("Patch DockQ ROC AUC Rank", "Patch DockQ\nROC AUC"),
        ("QSBEST PCC Rank", "QS-best\nPCC"),
        ("QSGLOB MAE Rank", "QS-glob\nMAE"),
        ("DOCKQ_AVG MAE Rank", "DockQ-avg\nMAE"),
    ]

    strats_to_compare = [
        ("All Targets", ARC_RED, "-"),
        ("Dimer only", BLUE, "--"),
        ("Multimer only", ORANGE, ":"),
    ]
    strat_display_for_legend = {s[0]: STRAT_DISPLAY.get(s[0], s[0]) for s in strats_to_compare}

    cols = [c for c, _ in key_metrics if c in df.columns]
    labels = [l for c, l in key_metrics if c in df.columns]

    tab = df.loc[[s[0] for s in strats_to_compare if s[0] in df.index], cols]
    tab.index = [STRAT_DISPLAY.get(i, i) for i in tab.index]
    tab.columns = [l.replace("\n", " ") for l in labels]
    print(f"\n  Fig 3 table (ARC rank by metric):")
    for _ln in tab.to_string().split("\n"):
        print(f"{_ln}", flush=True)
    print()

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))

    ax.axhspan(0.5, 3.5, color=GREEN, alpha=0.10, zorder=0)
    ax.axhspan(3.5, 5.5, color="#FFF9C4", alpha=0.35, zorder=0)
    ax.axhline(3.5, color=GREEN, linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axhline(5.5, color="#C0A000", linewidth=0.8, linestyle=":", alpha=0.5)

    ax.text(
        len(labels) - 0.5,
        2.0,
        "Top 3",
        fontsize=8,
        color=GREEN,
        ha="right",
        va="center",
        fontstyle="italic",
        alpha=0.8,
    )
    ax.text(
        len(labels) - 0.5,
        4.5,
        "Top 5",
        fontsize=8,
        color="#9E8500",
        ha="right",
        va="center",
        fontstyle="italic",
        alpha=0.8,
    )

    for strat, color, ls in strats_to_compare:
        if strat not in df.index:
            continue
        ranks = df.loc[strat, cols].values.astype(float)
        ax.plot(
            x,
            ranks,
            marker="o",
            color=color,
            linestyle=ls,
            linewidth=2.5,
            markersize=8,
            label=strat_display_for_legend[strat],
            zorder=3,
        )
        for i, r in enumerate(ranks):
            if strat == "Dimer only":
                vert_off = -10
            elif strat == "Multimer only":
                vert_off = 10
            else:
                vert_off = -10 if ranks[i] <= 5 else 10
            ax.annotate(
                f"{int(r)}",
                (x[i], ranks[i]),
                textcoords="offset points",
                xytext=(0, vert_off),
                fontsize=7.5,
                ha="center",
                color=color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Rank (1 = Best)", fontsize=10)
    ax.set_ylim(0.5, 10.5)
    ax.invert_yaxis()
    ax.set_title(
        r"ARC Rank Profile: All Targets vs. Dimers vs. Multimers (n $\geq$ 3)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)

    out_png = os.path.join(manuscript_out_dir(), f"{MS_FIG_FILENAME_RANK_PROFILE}.png")
    plt.savefig(out_png, format="png", dpi=350, bbox_inches="tight")
    print(f"  Fig 3 ({MS_FIG_FILENAME_RANK_PROFILE}.png): rank profile (Top-3/Top-5 bands)", flush=True)
    plt.close()
