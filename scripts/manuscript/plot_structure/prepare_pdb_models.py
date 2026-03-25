"""Fig4: copy CASP16 assemblies from data/raw_16 to pymol/pdb_models.

``--target T`` -> per-target tree under outputs/fig4_structure; ``--all-targets`` -> all from target_margin_scores.csv;
else best_target.txt -> flat pymol/pdb_models/.
"""
import argparse
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
ARC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
REPO = ARC_ROOT
BASE = ARC_ROOT
from fig4_paths import fig4_scores_csv_path, fig4_structure_work_dir

MARGIN_DIR = fig4_structure_work_dir(ARC_ROOT)
BEST_TARGET_PATH = os.path.join(MARGIN_DIR, "best_target.txt")
TARGETS_SCORES_CSV = fig4_scores_csv_path(ARC_ROOT)
METHODS = ("ARC", "ModFOLDdock2S", "MQA_server")


def _oligo_dir(target: str) -> str:
    return os.path.join(ARC_ROOT, "data", "raw_16", target, "oligo")


def copy_models_for_target(target: str, out_dir: str) -> int:
    """Copy the three method PDBs for target into out_dir. Returns number of files written."""
    import pandas as pd

    assemblies_base = _oligo_dir(target)
    if not os.path.exists(TARGETS_SCORES_CSV):
        return 0
    df = pd.read_csv(TARGETS_SCORES_CSV)
    row = df[df["target"] == target]
    if row.empty:
        return 0
    models = []
    for m in METHODS:
        col = "%s_selected_model" % m
        if col in row.columns:
            models.append(str(row[col].iloc[0]).strip())
    if not models:
        return 0

    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for name in models:
        path = os.path.join(assemblies_base, name)
        if not os.path.exists(path):
            print("Skip (not found): %s" % path, file=sys.stderr)
            continue
        out_path = os.path.join(out_dir, name + ".pdb")
        with open(path) as f:
            lines = f.readlines()
        out_lines = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
        with open(out_path, "w") as f:
            f.writelines(out_lines)
            f.write("END\n")
        print("Wrote %s (%d atoms)" % (out_path, len(out_lines)))
        n += 1
    return n


def copy_models_by_basename(
    target: str, out_dir: str, model_basenames: list[str]
) -> int:
    """Copy named TS files from raw_16/<target>/oligo into out_dir as <name>.pdb (no CSV)."""
    assemblies_base = _oligo_dir(target)
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for raw_name in model_basenames:
        name = str(raw_name).strip()
        if not name:
            continue
        path = os.path.join(assemblies_base, name)
        if not os.path.exists(path):
            print("Skip (not found): %s" % path, file=sys.stderr)
            continue
        out_path = os.path.join(out_dir, name + ".pdb")
        with open(path) as f:
            lines = f.readlines()
        out_lines = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
        with open(out_path, "w") as f:
            f.writelines(out_lines)
            f.write("END\n")
        print("Wrote %s (%d atoms)" % (out_path, len(out_lines)))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="Prepare PDB models for fig. 4 PyMOL panels (ARC repo).")
    ap.add_argument(
        "--target",
        metavar="T",
        help="Copy only this target (to outputs/fig4_structure/pymol/pdb_models/<target>/).",
    )
    ap.add_argument("--all-targets", action="store_true", help="Copy PDBs for every target in target_margin_scores.csv.")
    args = ap.parse_args()

    if args.all_targets:
        if not os.path.exists(TARGETS_SCORES_CSV):
            raise SystemExit(
                "target_margin_scores.csv not found under data/. Run build_targets_scores_csv.py first."
            )
        import pandas as pd
        df = pd.read_csv(TARGETS_SCORES_CSV)
        targets = [str(t).strip() for t in df["target"].unique() if t and str(t).strip() != "(none)"]
        for target in targets:
            out_dir = os.path.join(MARGIN_DIR, "pymol", "pdb_models", target)
            copy_models_for_target(target, out_dir)
        print("Done: %d targets" % len(targets))
        return

    if args.target:
        target = args.target
        out_dir = os.path.join(MARGIN_DIR, "pymol", "pdb_models", target)
        copy_models_for_target(target, out_dir)
        return

    # Backward compatible: best target, flat pymol/pdb_models/
    if os.path.exists(BEST_TARGET_PATH):
        with open(BEST_TARGET_PATH) as f:
            target = f.read().strip()
        if target and target != "(none)":
            pass
        else:
            target = "H1222"
    else:
        target = "H1222"
    out_dir = os.path.join(MARGIN_DIR, "pymol", "pdb_models")
    copy_models_for_target(target, out_dir)


if __name__ == "__main__":
    main()
