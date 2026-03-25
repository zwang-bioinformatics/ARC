"""ARC manuscript figure 4 (structural comparison): paths to inputs and scratch output."""

from __future__ import annotations

import os

# Canonical scores table at repo root under data/ (alongside other eval CSVs).
FIG4_SCORES_BASENAME = "target_margin_scores.csv"


def fig4_scores_csv_path(repo_root: str) -> str:
    """Resolve the per-target model/scores CSV for the structural figure.

    Preferred: ``data/target_margin_scores.csv``.

    Legacy (older trees or tarballs): ``data/manuscript/figures/fig4/targets_scores.csv`` or
    ``data/manuscript/fig4_margin/targets_scores.csv``.
    """
    primary = os.path.join(repo_root, "data", FIG4_SCORES_BASENAME)
    if os.path.isfile(primary):
        return primary
    leg_a = os.path.join(repo_root, "data", "manuscript", "figures", "fig4", "targets_scores.csv")
    if os.path.isfile(leg_a):
        return leg_a
    leg_b = os.path.join(repo_root, "data", "manuscript", "fig4_margin", "targets_scores.csv")
    if os.path.isfile(leg_b):
        return leg_b
    return primary


def fig4_structure_work_dir(repo_root: str) -> str:
    """Scratch tree for PyMOL (PML, panel PNGs, ``pymol/pdb_models/``, ``best_target.txt``).

    Lives under ``outputs/`` (gitignored). Not part of downloadable asset tarballs.
    """
    return os.path.join(repo_root, "outputs", "fig4_structure")
