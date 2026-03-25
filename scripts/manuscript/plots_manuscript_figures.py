"""
Manuscript figures from cached eval/ensemble outputs.

Run casp16_eval.py first, then this script or: pixi run manuscript-figures

Reads pooled tables, arc_ensemble CSVs, arc_rankings_summary, etc. Fig5 needs
arc_residue_truth_pred.pkl and assessor JSONs (see plots_local_residue_quantile_heatmap).

Writes PNGs/PDFs to MANUSCRIPT_FIGURES_DIR (fig3, fig4, fig5, fig6a/b, figS1) and
supplementary LaTeX S1-S4 to MANUSCRIPT_TABLES_DIR. Which figures exist is defined
in plots_manuscript_figs / constants (Wiley-facing set).
"""

import os
import sys
import warnings

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
_scripts_root = os.path.dirname(_script_dir)
_eval_dir = os.path.join(_scripts_root, "eval")
_common_dir = os.path.join(_scripts_root, "common")
for _p in (_eval_dir, _common_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plots_science_style import try_apply_science_style

try_apply_science_style()

from casp16_eval_paths import MANUSCRIPT_FIGURES_DIR, MANUSCRIPT_TABLES_DIR, ensure_eval_output_layout
from plots_local_residue_quantile_heatmap import fig5_local_residue_quantile_heatmap
from plots_manuscript_supplementary_tables import (
    write_supplementary_s1_s2_tables,
    write_supplementary_s3_s4_all_comb_tables,
)
from plots_fig4_t1259o_structure import fig4_structural_comparison_t1259o
from plots_manuscript_figs import (
    fig3_ensemble_benefit_vs_L_patch_only_2x2,
    fig4_per_target_lollipop,
    fig7_ensemble_scaling_heatmap,
    fig8_rank_profile_comparison,
)
from plots_manuscript_io import manuscript_out_dir, set_manuscript_figures_out_dir

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*Precision loss occurred in moment calculation due to catastrophic cancellation.*",
)

ensure_eval_output_layout()
os.makedirs(MANUSCRIPT_FIGURES_DIR, exist_ok=True)
os.makedirs(MANUSCRIPT_TABLES_DIR, exist_ok=True)

if __name__ == "__main__":
    from eval_log_tee import append_eval_log_tee

    append_eval_log_tee("plots_manuscript_figures")
    set_manuscript_figures_out_dir(None)
    print(f"Generating Wiley manuscript figures (cached CSVs, all-combinations ensemble)...", flush=True)
    os.makedirs(manuscript_out_dir(), exist_ok=True)
    print(f"  Output: {manuscript_out_dir()}", flush=True)
    # Order: fig3 rank profile, fig4 T1259o structure, fig5 quantile heat, fig6a lollipop, fig6b patch 2x2, figS1 SI heatmap
    fig8_rank_profile_comparison()
    fig4_structural_comparison_t1259o()
    fig5_local_residue_quantile_heatmap()
    fig4_per_target_lollipop()
    fig3_ensemble_benefit_vs_L_patch_only_2x2()
    fig7_ensemble_scaling_heatmap()
    print(f"\nFigures saved to {manuscript_out_dir()}", flush=True)
    print(f"PNG + PDF figures (350 dpi, journal submission format) generated.", flush=True)
    print(f"Supplementary LaTeX tables S1/S2...", flush=True)
    try:
        write_supplementary_s1_s2_tables()
    except (FileNotFoundError, ValueError) as e:
        print(f"  Skipped S1/S2 tables ({e}). Run a full casp16_eval if inputs are missing.", flush=True)
    print(f"Supplementary LaTeX tables S3/S4 (L=6 all-comb)...", flush=True)
    write_supplementary_s3_s4_all_comb_tables()
