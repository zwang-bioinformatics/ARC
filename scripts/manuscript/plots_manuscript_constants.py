"""Manuscript figure constants (colors, fonts, labels, ensemble column map)."""
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_scripts_root = os.path.dirname(_script_dir)
_eval_dir = os.path.join(_scripts_root, "eval")
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

from casp16_eval_constants import targets

# Figure output basenames (LaTeX \\includegraphics).
MS_FIG_FILENAME_RANK_PROFILE = "fig3"
MS_FIG_FILENAME_QUANTILE_COMPOSITE = "fig5"
MS_FIG_FILENAME_LOLLIPOP_L6 = "fig6a"
MS_FIG_FILENAME_ENSEMBLE_PATCH_2X2 = "fig6b"
MS_FIG_FILENAME_SI_SCALING_HEATMAP = "figS1"

ARC_RED = "#D62728"
BLUE = "#1F77B4"
GREEN = "#2CA02C"
ORANGE = "#FF7F0E"
GREY = "#7F7F7F"
PURPLE = "#9467BD"
TEAL = "#17BECF"
MUTED_TEAL = "#6B8E9B"
MUTED_BLUE = "#6B8BA4"
ENSEMBLE_ACCENT = MUTED_TEAL

ENSEMBLE_FONT_TITLE = 10
ENSEMBLE_FONT_AXES = 9
ENSEMBLE_FONT_TICK = 8
ENSEMBLE_FONT_LEGEND = 8
HEATMAP_FONT_AXES = 18
HEATMAP_FONT_TICK = 15

METHOD_COLORS = {
    "ARC": ARC_RED,
    "ModFOLDdock2S": BLUE,
    "GuijunLab-PAthreader": GREEN,
    "MQA_server": "#E377C2",
    "MQA_base": GREY,
    "APOLLO": ORANGE,
    "GuijunLab-Assembly": PURPLE,
    "Guijunlab-Complex": "#8C564B",
    "VifChartreuse": "#BCBD22",
    "VifChartreuseJaune": TEAL,
}

MULTIMER_LABEL_LATEX = r"Multimers (n $\geq$ 3)"

STRAT_DISPLAY = {
    "All Targets": "All",
    "H only": "H-type",
    "T only": "T-type",
    "Small (<1000)": "Small",
    "Medium (1000-1500)": "Medium",
    "Large (>=1500)": "Large",
    "Huge (>=3000)": "Huge",
    "Dimer only": "Dimers",
    "Multimer only": MULTIMER_LABEL_LATEX,
}

# Ensemble vs individual columns in arc_all_combinations and arc_ensemble_summary_all_comb CSVs.
ENSEMBLE_ALL_COMB_COL_MAP = {
    "iface_ROC_AUC": (
        "true_interface_residue_AUC_ROC_ensemble",
        "true_interface_residue_AUC_ROC_individual",
    ),
    "iface_PR_AUC": (
        "true_interface_residue_AUC_PRC_ensemble",
        "true_interface_residue_AUC_PRC_individual",
    ),
    "patch_qs_PCC": ("patch_qs_pearson_ensemble", "patch_qs_pearson_individual"),
    "patch_dockq_PCC": ("patch_dockq_pearson_ensemble", "patch_dockq_pearson_individual"),
    "patch_qs_SCC": ("patch_qs_spearman_ensemble", "patch_qs_spearman_individual"),
    "patch_dockq_SCC": ("patch_dockq_spearman_ensemble", "patch_dockq_spearman_individual"),
}
