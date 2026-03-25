"""Small metric helpers used by per-target processing."""
from sklearn.metrics import roc_auc_score
import pandas as pd


def adaptive_rocauc(df, target, pred):
    # https://git.scicore.unibas.ch/schwede/casp16_ema/-/blob/main/analysis/local_analysis.ipynb?ref_type=heads
    """Returns ROC AUC with an adaptive class threshold (top quantile of target = positive)."""
    thresh = df[target].quantile(0.75)
    sub_df = df[(df[target].isnull() == False) & (df[pred].isnull() == False)]
    target_classes = [int(x > thresh) for x in sub_df[target]]
    return max(0.5, roc_auc_score(target_classes, sub_df[pred]))


def compute_rs(loc_df, score_type):
    """Compute correlation and AUC metrics for a given score type."""
    pearson = loc_df[score_type].corr(loc_df["pred"], method="pearson")
    spearman = loc_df[score_type].corr(loc_df["pred"], method="spearman")
    auc = adaptive_rocauc(loc_df, score_type, "pred")
    return pearson, spearman, auc
