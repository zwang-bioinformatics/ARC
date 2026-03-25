#!/usr/bin/env python3
"""Regenerate only the six pooled fig.~2 ROC PNGs as ``fig2a.png``-``fig2f.png`` (no pooled CSV table rewrite).

Uses ``pooled_analysis/stratified_metrics_fig2_roc.pkl`` when it matches the mtime of
``raw_data_for_pooling.pkl`` (matplotlib-only replot). Otherwise runs a **slim** metric
pass (All Targets + Dimer only, three ROC scores) and refreshes the cache.

Force recomputation from the raw pickle: ``ARC_FIG2_ROC_RECOMPUTE=1 pixi run replot-pooled-roc``.

Requires ``raw_data_for_pooling.pkl`` and ``local_eval.csv`` under the eval output root
(default ``outputs/results/local_results/``).

Usage::

    pixi run replot-pooled-roc

Or from repo root::

    python -u scripts/eval/replot_figure2_roc.py
"""
from __future__ import annotations

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from casp16_eval_paths import ensure_eval_output_layout
from casp16_eval_pooled import replot_figure2_roc_curves_only

if __name__ == "__main__":
    ensure_eval_output_layout()
    replot_figure2_roc_curves_only()
