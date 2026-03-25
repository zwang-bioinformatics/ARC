#!/usr/bin/env python3
"""
Generate a PyMOL .pml script and panel metadata for ARC manuscript figure 4 (structural comparison).

Model choices and metrics come from ``data/target_margin_scores.csv`` (per-target row).

Layout: 4 columns
  1. Native target structure (reference) — interface residues shown as sticks
  2. ARC's selected best model         — interface classification
  3. ModFOLDdock2S best model           — interface classification
  4. MQA_server best model              — interface classification

Two classification layers:

  Layer 1 — Structural truth (model geometry vs native):
    Compare each model's 3D interface against the native structure.
    - TP: residue at BOTH model interface AND native interface
    - FP: residue at model interface BUT NOT native interface
    - FN: residue NOT at model interface BUT at native interface
    - TN: residue at neither interface

  Layer 2 — Evaluator prediction overlay (on TP/FP only):
    Each evaluator (ARC, ModFOLDdock2S, MQA_server) produces per-residue
    quality scores for model_interface_residues (QMODE 2). These continuous
    scores are binarized at the pooled Youden threshold to predict "interface"
    or "not interface", then compared against native truth.
    - Evaluator agrees with native truth  → full opacity
    - Evaluator disagrees                 → semi-transparent (EVAL_DISAGREE_TRANSPARENCY)
    FN and TN have no evaluator scores (QMODE 2 only covers model interface).

Visual encoding (model panels):
  - TP (eval agrees):    sticks (chain colour), fully opaque
  - TP (eval disagrees): sticks (chain colour), semi-transparent
  - FP (eval disagrees): sticks (orange), semi-transparent
  - FP (eval agrees):    sticks (orange), fully opaque
  - FN:                  CA spheres (orange = wrong), opaque — no evaluator data
  - TN:                  uniform grey cartoon, 90% transparent

Inputs live under ``data/`` (CASP16 targets, EMA reference JSON, ``raw_16`` assemblies).
``configure_paths()`` can override the EMA directory, evaluator pooling pickle, and native
PDB folder. Written PML + JSON use the work directory passed to ``_run_one_target`` (under
``outputs/fig4_structure/`` by default).
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))

# three levels up from this file = ARC repo root
ARC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
REPO = ARC_ROOT
BASE = ARC_ROOT

from fig4_paths import fig4_scores_csv_path, fig4_structure_work_dir  # noqa: E402

MARGIN_DIR = fig4_structure_work_dir(ARC_ROOT)
PDB_MODELS_DIR = os.path.join(MARGIN_DIR, "pymol", "pdb_models")
EMA_RESULTS = os.path.join(ARC_ROOT, "data", "casp16_ema_reference_results")
TARGETS_PDB_DIR = os.path.join(ARC_ROOT, "data", "casp16_targets")

BEST_TARGET_PATH = os.path.join(MARGIN_DIR, "best_target.txt")
TARGETS_CSV = fig4_scores_csv_path(ARC_ROOT)

# default; plots_fig4_t1259o_structure.py passes resolved paths via configure_paths()
RAW_PICKLE = os.path.join(ARC_ROOT, "outputs", "results", "eval", "raw_data_for_pooling.pkl")


def configure_paths(
    ema_results: str | None = None,
    raw_pickle: str | None = None,
    targets_pdb_dir: str | None = None,
) -> None:
    """Replace default EMA dir, pooling pickle, or native PDB folder."""
    global EMA_RESULTS, RAW_PICKLE, TARGETS_PDB_DIR
    if ema_results is not None:
        EMA_RESULTS = os.path.abspath(ema_results)
    if raw_pickle is not None:
        RAW_PICKLE = os.path.abspath(raw_pickle)
    if targets_pdb_dir is not None:
        TARGETS_PDB_DIR = os.path.abspath(targets_pdb_dir)


OUT_DIR = os.path.join(MARGIN_DIR, "manuscript")
OUT_PML = os.path.join(OUT_DIR, "manuscript_four_panels.pml")
META_PATH = os.path.join(OUT_DIR, "manuscript_panels_metadata.json")

# ---------------------------------------------------------------------------
# Figure configuration — edit these to change layout / style
# ---------------------------------------------------------------------------
METHODS = ("ARC", "ModFOLDdock2S", "MQA_server")

OBJ_NAMES = {"ARC": "arc", "ModFOLDdock2S": "modfold", "MQA_server": "mqa"}

# Reference-chain → PyMOL colour (TP and FN use these; applied to interface)
CHAIN_COLORS = {"A": "red", "B": "marine", "C": "forest"}

FP_COLOR = "orange"

# Uniform colour for TN (non-interface) residues so they don't visually
# compete with classified interface residues. Cartoon for non-interface is grey, 90% transparent.
TN_COLOR = "grey80"
TN_CARTOON_TRANSPARENCY = 0.70  # non-interface cartoon: 70% transparent
ZOOM_BUFFER = 14  # zoom buffer (Angstroms) for model panels (B-D); large enough to avoid clipping
ZOOM_BUFFER_NATIVE = 28  # larger buffer for native (A) so it matches apparent zoom of model panels

SHOW_FOUR_CATEGORIES = True
SHOW_LABELS = False

# Transparency for sticks/cartoon where the evaluator disagrees with native
# truth (0.0 = opaque, 1.0 = invisible).
EVAL_DISAGREE_TRANSPARENCY = 0.7

DISPLAY_METRICS = [
    ("method_score",   "Interface ROC AUC"),
    ("gt_patch_qs",    "Patch QS"),
    ("gt_patch_dockq", "Patch DockQ"),
    ("gt_local_lddt",  "Local lDDT"),
]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def _load_best_target() -> str | None:
    """Read the single best target ID from best_target.txt."""
    if not os.path.exists(BEST_TARGET_PATH):
        return None
    with open(BEST_TARGET_PATH) as f:
        t = f.read().strip()
    return t if t and t != "(none)" else None


def _load_three_models(target: str) -> dict[str, str] | None:
    """Return {method: model_name} for the three methods on *target*."""
    df = pd.read_csv(TARGETS_CSV)
    row = df[df["target"] == target]
    if row.empty:
        return None
    row = row.iloc[0]
    out = {}
    for m in METHODS:
        col = f"{m}_selected_model"
        if col in row:
            out[m] = str(row[col]).strip()
    return out if len(out) == len(METHODS) else None


def _load_all_metrics(target: str) -> dict[str, dict[str, float]]:
    """Return {method: {csv_col: value, ...}} for DISPLAY_METRICS from the fig4 scores CSV."""
    df = pd.read_csv(TARGETS_CSV)
    row = df[df["target"] == target]
    if row.empty:
        return {}
    row = row.iloc[0]
    result: dict[str, dict[str, float]] = {}
    for method in METHODS:
        result[method] = {}
        for col, _ in DISPLAY_METRICS:
            if col == "method_score":
                key = "%s_interface_roc" % method
            else:
                key = "%s_%s" % (method, col)
            if key in row and pd.notna(row[key]):
                result[method][col] = float(row[key])
    return result


# ---------------------------------------------------------------------------
# EMA JSON helpers
# ---------------------------------------------------------------------------
def _load_ema_json(model: str, target: str) -> dict | None:
    """Load and return the EMA result JSON, or None if missing."""
    path = os.path.join(EMA_RESULTS, f"{model}_{target}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_ref_chains_and_mapping(
    model: str, target: str
) -> tuple[list[str] | None, dict[str, str] | None]:
    """
    From EMA JSON, return (reference_chains, ref_to_model mapping).
    chain_mapping in the JSON is ref_chain → model_chain.
    """
    data = _load_ema_json(model, target)
    if data is None:
        return None, None
    ref_chains = data.get("reference_chains")
    mapping = data.get("chain_mapping")
    if not ref_chains or not isinstance(mapping, dict):
        return None, None
    return ref_chains, mapping


def _reference_interface_selection(target: str, model_for_json: str) -> str:
    """
    Build a PyMOL selection string for the *reference* interface residues.
    Residue format in EMA: "A.42." → chain A, resi 42.
    """
    data = _load_ema_json(model_for_json, target)
    if data is None:
        return "none"
    by_chain = _ema_residues_to_chain_dict(
        data.get("reference_interface_residues") or []
    )
    return _chain_dict_to_selection(by_chain)


def _ema_residues_to_chain_dict(
    residue_list: list[str],
) -> dict[str, list[int]]:
    """Parse EMA residue strings ("A.42." or "A.42.ALA") into {chain: [resi, ...]}."""
    by_chain: dict[str, list[int]] = {}
    for entry in residue_list:
        parts = entry.strip().split(".")
        if len(parts) >= 2 and parts[0] and parts[1].isdigit():
            by_chain.setdefault(parts[0], []).append(int(parts[1]))
    return by_chain


def _chain_dict_to_selection(by_chain: dict[str, list[int]]) -> str:
    """Convert {chain: [resi, ...]} to a PyMOL selection string."""
    sel_parts = []
    for chain in sorted(by_chain):
        resis = "+".join(str(r) for r in sorted(set(by_chain[chain])))
        sel_parts.append(f"(chain {chain} and resi {resis})")
    return " or ".join(sel_parts) if sel_parts else "resi 999999"


def _classify_model_interface(
    model: str, target: str
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    """
    Classify residues into TP, FP, FN for a given model.

    chain_mapping in EMA is ref→model. We invert to model→ref for TP/FP,
    and use ref→model directly for FN.

    Returns:
      (tp_pairs, fp_pairs, fn_pairs)
      Each pair is (model_chain_id, residue_number) so the selection can be
      applied directly on the model object in PyMOL.
    """
    data = _load_ema_json(model, target)
    if data is None:
        return [], [], []
    model_iface = data.get("model_interface_residues", [])
    ref_iface = data.get("reference_interface_residues", [])
    if not ref_iface:
        return [], [], []

    ref_set = set(ref_iface)

    ref_to_model = data.get("chain_mapping", {}) or {}
    model_to_ref = {v: k for k, v in ref_to_model.items()} if ref_to_model else {}

    model_iface_set: set[tuple[str, str]] = set()
    tp, fp = [], []
    for residue in model_iface:
        parts = residue.split(".")
        if len(parts) != 3:
            continue
        model_chain, rnum_s, _ = parts
        try:
            rnum = int(rnum_s)
        except ValueError:
            continue
        model_iface_set.add((model_chain, rnum_s))
        ref_chain = model_to_ref.get(model_chain, model_chain)
        if f"{ref_chain}.{rnum}." in ref_set:
            tp.append((model_chain, rnum))
        else:
            fp.append((model_chain, rnum))

    fn = []
    for entry in ref_iface:
        parts = entry.strip().split(".")
        if len(parts) < 2 or not parts[0] or not parts[1].isdigit():
            continue
        ref_c, resi_s = parts[0], parts[1]
        model_c = ref_to_model.get(ref_c, ref_c)
        if (model_c, resi_s) not in model_iface_set:
            fn.append((model_c, int(resi_s)))

    return tp, fp, fn


def _count_model_residues(pdb_path: str) -> int:
    """Count standard (non-HETATM) residues in a PDB file by unique (chain, resi)."""
    seen: set[tuple[str, str]] = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21]
                resi = line[22:27].strip()
                seen.add((chain, resi))
    return len(seen)


def _pairs_to_pymol_selection(pairs: list[tuple[str, int]]) -> str:
    """Convert [(chain, resi), ...] to a PyMOL selection string."""
    by_chain: dict[str, list[int]] = {}
    for chain, resi in pairs:
        by_chain.setdefault(chain, []).append(resi)
    return _chain_dict_to_selection(by_chain)


# ---------------------------------------------------------------------------
# Evaluator prediction helpers (Layer 2)
# ---------------------------------------------------------------------------
def _load_pickle(pickle_path: str | None = None) -> dict:
    """Load pooling pickle (e.g. raw_data_for_pooling.pkl or ARC ``raw_arc_only.pkl``).

    Structure: {method: {"true_interface_residue": [DataFrames], ...}}
    Each DataFrame has columns [true_interface_residue, pred, target, model].
    Rows are 1:1 aligned with model_interface_residues from EMA JSON.
    """
    p = os.path.abspath(pickle_path) if pickle_path else RAW_PICKLE
    if not os.path.exists(p):
        print(f"  Warning: pickle not found at {p}")
        return {}
    with open(p, "rb") as f:
        return pickle.load(f)


def _youden_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Threshold maximising Youden's J (sensitivity + specificity - 1)."""
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_pred) & np.isfinite(y_true)
    if valid.sum() < 5:
        return None
    y_t, y_p = y_true[valid].astype(int), y_pred[valid]
    if np.unique(y_t).size < 2:
        return None
    try:
        fpr, tpr, thresh = roc_curve(y_t, y_p)
        idx = np.argmax(tpr - fpr)
        return float(thresh[idx]) if idx < len(thresh) else 0.5
    except Exception:
        return 0.5


def _compute_pooled_threshold(
    target: str, method: str, pickle_data: dict
) -> float | None:
    """Youden threshold pooled across ALL models for (target, method).

    Uses true_interface_residue (binary) as truth, evaluator pred as score.
    """
    if method not in pickle_data:
        return None
    ti_data = pickle_data[method].get("true_interface_residue")
    if not ti_data:
        return None
    pooled = pd.concat(ti_data, ignore_index=True)
    pooled = pooled[pooled["target"] == target]
    if len(pooled) < 10:
        return None
    y = pooled["true_interface_residue"].astype(float).values
    p = pooled["pred"].astype(float).values
    valid = np.isfinite(p) & np.isfinite(y) & ((y == 0) | (y == 1))
    if valid.sum() < 5:
        return None
    return _youden_threshold(y[valid].astype(int), p[valid])


def _evaluator_agreement_per_residue(
    model: str, target: str, method: str,
    pickle_data: dict, threshold: float,
) -> dict[tuple[str, int], bool]:
    """Determine whether the evaluator's binarized prediction agrees with
    native truth for each model_interface_residue of the selected model.

    Returns {(model_chain, resi_int): agrees_bool}.
    Residues with missing pred scores are treated as agreeing (benefit of doubt).
    """
    agreement: dict[tuple[str, int], bool] = {}

    data = _load_ema_json(model, target)
    if data is None:
        return agreement
    model_iface = data.get("model_interface_residues", [])
    if not model_iface:
        return agreement

    if method not in pickle_data:
        return agreement
    ti_data = pickle_data[method].get("true_interface_residue")
    if not ti_data:
        return agreement

    pooled = pd.concat(ti_data, ignore_index=True)
    model_rows = pooled[
        (pooled["target"] == target) & (pooled["model"] == model)
    ]
    if len(model_rows) == 0 or len(model_rows) != len(model_iface):
        print(
            f"  Warning: pickle rows ({len(model_rows)}) != "
            f"model_interface_residues ({len(model_iface)}) for {model}"
        )
        return agreement

    y = model_rows["true_interface_residue"].astype(float).values
    p = model_rows["pred"].astype(float).values

    for i, residue in enumerate(model_iface):
        parts = residue.split(".")
        if len(parts) != 3:
            continue
        model_chain, rnum_s, _ = parts
        try:
            rnum = int(rnum_s)
        except ValueError:
            continue
        if i >= len(y):
            break

        native_truth = bool(int(y[i])) if np.isfinite(y[i]) else False
        if np.isfinite(p[i]):
            eval_pred_positive = float(p[i]) >= threshold
            agrees = eval_pred_positive == native_truth
        else:
            agrees = True

        agreement[(model_chain, rnum)] = agrees

    return agreement


def _split_pairs_by_agreement(
    pairs: list[tuple[str, int]],
    agreement: dict[tuple[str, int], bool],
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Split (chain, resi) pairs into (agree, disagree) lists."""
    agree, disagree = [], []
    for pair in pairs:
        if agreement.get(pair, True):
            agree.append(pair)
        else:
            disagree.append(pair)
    return agree, disagree


# ---------------------------------------------------------------------------
# PML generation helpers
# ---------------------------------------------------------------------------
def _chain_color_commands(
    obj_name: str,
    ref_chains: list[str],
    ref_to_model: dict[str, str],
    chain_colors: dict[str, str],
) -> list[str]:
    """Return PyMOL 'color' commands mapping reference chain colours to model
    chain IDs for an entire object."""
    cmds = []
    for ref_c in ref_chains:
        model_c = ref_to_model.get(ref_c)
        color = chain_colors.get(ref_c, "forest")
        if model_c:
            cmds.append(f"color {color}, {obj_name} and chain {model_c}")
    return cmds


def _selection_chain_color_commands(
    selection: str,
    ref_chains: list[str],
    ref_to_model: dict[str, str],
    chain_colors: dict[str, str],
) -> list[str]:
    """Return colour commands for a named PyMOL selection (e.g. iface_arc_tp),
    mapping model chain IDs back to reference chain colours."""
    cmds = []
    for ref_c in ref_chains:
        model_c = ref_to_model.get(ref_c)
        color = chain_colors.get(ref_c, "forest")
        if model_c:
            cmds.append(f"color {color}, {selection} and chain {model_c}")
    return cmds


# ---------------------------------------------------------------------------
# Parallel worker (must be module-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------
def _worker_one_target(args: tuple[str, str, str]) -> None:
    target, out_dir, pdb_models_dir = args
    _run_one_target(target, out_dir, pdb_models_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _run_one_target(
    target: str,
    out_dir: str,
    pdb_models_dir: str,
    *,
    three_override: dict[str, str] | None = None,
    ref_pdb_override: str | None = None,
    skip_copy_models: bool = False,
    metrics_override: dict[str, dict[str, float]] | None = None,
    raw_pickle_path: str | None = None,
    no_eval_pickle: bool = False,
) -> None:
    """Generate PML and metadata for a single target. Uses module-level OUT_* and PDB_MODELS_DIR via globals."""
    global OUT_DIR, OUT_PML, META_PATH, PDB_MODELS_DIR
    OUT_DIR = out_dir
    OUT_PML = os.path.join(OUT_DIR, "manuscript_four_panels.pml")
    META_PATH = os.path.join(OUT_DIR, "manuscript_panels_metadata.json")
    PDB_MODELS_DIR = pdb_models_dir

    if three_override is not None:
        three = dict(three_override)
        for m in METHODS:
            if m not in three or not str(three[m]).strip():
                raise SystemExit(
                    f"three_override must include non-empty keys for all methods: {METHODS}"
                )
    else:
        three = _load_three_models(target)
    if not three:
        raise SystemExit(f"Could not load three models for target {target}")

    # Ensure PDB models exist (copy from casp16_ema/assemblies if needed)
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    if not skip_copy_models:
        try:
            from prepare_pdb_models import copy_models_by_basename, copy_models_for_target

            if three_override is not None:
                copy_models_by_basename(
                    target, PDB_MODELS_DIR, [three[m] for m in METHODS]
                )
            else:
                copy_models_for_target(target, PDB_MODELS_DIR)
        except Exception as e:
            print("Warning: could not copy PDB models: %s" % e, file=sys.stderr)

    metrics = (
        metrics_override
        if metrics_override is not None
        else _load_all_metrics(target)
    )
    ref_pdb = os.path.abspath(
        ref_pdb_override or os.path.join(TARGETS_PDB_DIR, f"{target}.pdb")
    )
    if not os.path.exists(ref_pdb):
        raise SystemExit(f"Reference PDB not found: {ref_pdb}")

    pdb_paths: dict[str, str] = {}
    for method in METHODS:
        p = os.path.abspath(os.path.join(PDB_MODELS_DIR, f"{three[method]}.pdb"))
        if not os.path.exists(p):
            raise SystemExit(f"Model PDB not found: {p}")
        pdb_paths[method] = p

    # ---- Layer 1: structural interface classification ---------------------
    ref_sel = _reference_interface_selection(target, three["ARC"])

    tp_pairs: dict[str, list[tuple[str, int]]] = {}
    fp_pairs: dict[str, list[tuple[str, int]]] = {}
    fn_pairs: dict[str, list[tuple[str, int]]] = {}
    cm_counts: dict[str, dict[str, int]] = {}

    for method, model in three.items():
        tp, fp, fn = _classify_model_interface(model, target)
        tp_pairs[method] = tp
        fp_pairs[method] = fp
        fn_pairs[method] = fn
        total_res = _count_model_residues(pdb_paths[method])
        n_tp, n_fp, n_fn = len(tp), len(fp), len(fn)
        n_tn = max(0, total_res - n_tp - n_fp - n_fn)
        cm_counts[method] = {"TP": n_tp, "FP": n_fp, "FN": n_fn, "TN": n_tn}

    # ---- Layer 2: evaluator prediction classification ---------------------
    pickle_data = (
        {} if no_eval_pickle else _load_pickle(raw_pickle_path)
    )

    eval_thresholds: dict[str, float | None] = {}
    tp_agree: dict[str, list[tuple[str, int]]] = {}
    tp_disagree: dict[str, list[tuple[str, int]]] = {}
    fp_agree: dict[str, list[tuple[str, int]]] = {}
    fp_disagree: dict[str, list[tuple[str, int]]] = {}
    eval_cm: dict[str, dict[str, int]] = {}

    for method, model in three.items():
        thresh = _compute_pooled_threshold(target, method, pickle_data)
        eval_thresholds[method] = thresh

        if thresh is not None:
            agreement = _evaluator_agreement_per_residue(
                model, target, method, pickle_data, thresh
            )
        else:
            agreement = {}

        ta, td = _split_pairs_by_agreement(tp_pairs[method], agreement)
        fa, fd = _split_pairs_by_agreement(fp_pairs[method], agreement)
        tp_agree[method] = ta
        tp_disagree[method] = td
        fp_agree[method] = fa
        fp_disagree[method] = fd

        eval_cm[method] = {
            "TP": len(ta), "FN": len(td),
            "FP": len(fd), "TN": len(fa),
        }

    # ---- Build PyMOL selection strings ------------------------------------
    sel_tp: dict[str, str] = {}
    sel_fp: dict[str, str] = {}
    sel_fn: dict[str, str] = {}
    sel_tp_disagree: dict[str, str] = {}
    sel_fp_disagree: dict[str, str] = {}

    for method in METHODS:
        sel_tp[method] = _pairs_to_pymol_selection(tp_pairs[method])
        sel_fp[method] = _pairs_to_pymol_selection(fp_pairs[method])
        sel_fn[method] = _pairs_to_pymol_selection(fn_pairs[method])
        sel_tp_disagree[method] = _pairs_to_pymol_selection(tp_disagree[method])
        sel_fp_disagree[method] = _pairs_to_pymol_selection(fp_disagree[method])

    # ---- Chain colour mapping per panel -----------------------------------
    ref_chains, _ = _load_ref_chains_and_mapping(three["ARC"], target)
    if not ref_chains:
        ref_chains = ["A", "B"]

    ref_to_model_per_method: dict[str, dict[str, str]] = {}
    for method in METHODS:
        _, mapping = _load_ref_chains_and_mapping(three[method], target)
        ref_to_model_per_method[method] = mapping or {c: c for c in ref_chains}
    ref_identity = {c: c for c in ref_chains}

    # ---- Build PML --------------------------------------------------------
    panel_names = ["native", "arc", "modfold", "mqa"]
    panel_pngs = [os.path.join(OUT_DIR, f"pymol_panel_{n}.png") for n in panel_names]

    L: list[str] = []
    L_global = [
        f"# Manuscript figure – target {target}",
        f"# SHOW_FOUR_CATEGORIES = {SHOW_FOUR_CATEGORIES}",
        f"# TN_COLOR = {TN_COLOR}",
        "",
        "set ray_opaque_background, off",
        "bg_color white",
        "set antialias, 2",
        "set orthoscopic, 1",
        "",
    ]
    L_load = [
        f"load {ref_pdb}, reference",
        f"load {pdb_paths['ARC']}, arc",
        f"load {pdb_paths['ModFOLDdock2S']}, modfold",
        f"load {pdb_paths['MQA_server']}, mqa",
        "",
        "remove solvent",
        "remove hydrogens",
        "",
    ]
    L_align = [
        "align modfold, arc",
        "align mqa, arc",
        "align reference, arc",
        "",
    ]
    L_native = [
        "# Native panel: non-interface grey 90 pct transparent, chain colours on interface only",
        f"color {TN_COLOR}, reference",
        "",
    ]
    L_model_base = []
    for method in METHODS:
        obj = OBJ_NAMES[method]
        mapping = ref_to_model_per_method[method]
        L_model_base.append(f"# {method}: uniform {TN_COLOR} base, chain colours on interface")
        L_model_base.append(f"color {TN_COLOR}, {obj}")
    L_model_base.append("")

    L_sel = [
        "# Interface selections (Layer 1: structural truth)",
        f"select iface_reference, reference and ({ref_sel})",
        "select non_iface_reference, reference and not (iface_reference)",
    ]
    for method in METHODS:
        obj = OBJ_NAMES[method]
        L_sel.append(f"select iface_{obj}_tp, {obj} and ({sel_tp[method]})")
        L_sel.append(f"select iface_{obj}_fp, {obj} and ({sel_fp[method]})")
        if SHOW_FOUR_CATEGORIES:
            L_sel.append(f"select iface_{obj}_fn, {obj} and ({sel_fn[method]})")
    L_sel.append("")

    has_eval_sels = False
    for method in METHODS:
        obj = OBJ_NAMES[method]
        if tp_disagree[method]:
            L_sel.append(
                f"select iface_{obj}_tp_dis, {obj} and ({sel_tp_disagree[method]})"
            )
            has_eval_sels = True
        if fp_disagree[method]:
            L_sel.append(
                f"select iface_{obj}_fp_dis, {obj} and ({sel_fp_disagree[method]})"
            )
            has_eval_sels = True
    if has_eval_sels:
        L_sel.append("")

    L = (
        L_global
        + L_load
        + L_align
        + L_native
        + L_model_base
        + L_sel
    )
    L += [
        "show cartoon, all",
        "set cartoon_fancy_helices, 1",
        f"set cartoon_transparency, {TN_CARTOON_TRANSPARENCY}",
        "",
    ]
    L += [
        "# Native: interface sticks chain colour opaque, non-interface explicitly grey 90 pct transparent (TN)",
        "show sticks, iface_reference",
    ] + _selection_chain_color_commands("iface_reference", ref_chains, ref_identity, CHAIN_COLORS) + [
        "set cartoon_transparency, 0.0, iface_reference",
        "color %s, non_iface_reference" % TN_COLOR,
        "set cartoon_transparency, %s, non_iface_reference" % TN_CARTOON_TRANSPARENCY,
        "",
    ]
    for method in METHODS:
        obj = OBJ_NAMES[method]
        mapping = ref_to_model_per_method[method]
        L.append(f"# {method}: structural classification + evaluator overlay")
        L += [
            f"show sticks, iface_{obj}_tp",
            f"set cartoon_transparency, 0.0, iface_{obj}_tp",
        ]
        L += _selection_chain_color_commands(
            f"iface_{obj}_tp", ref_chains, mapping, CHAIN_COLORS
        )
        L += [
            f"show sticks, iface_{obj}_fp",
            f"color {FP_COLOR}, iface_{obj}_fp",
            f"set cartoon_transparency, 0.0, iface_{obj}_fp",
        ]
        if SHOW_FOUR_CATEGORIES:
            L += [
                f"show spheres, iface_{obj}_fn and name CA",
                f"set sphere_scale, 0.5, iface_{obj}_fn",
                f"set cartoon_transparency, 0.0, iface_{obj}_fn",
                f"color {FP_COLOR}, iface_{obj}_fn",
            ]
        if tp_disagree[method]:
            L.append(
                f"set stick_transparency, {EVAL_DISAGREE_TRANSPARENCY}, "
                f"iface_{obj}_tp_dis"
            )
            L.append(
                f"set cartoon_transparency, {EVAL_DISAGREE_TRANSPARENCY}, "
                f"iface_{obj}_tp_dis"
            )
        if fp_disagree[method]:
            L.append(
                f"set stick_transparency, {EVAL_DISAGREE_TRANSPARENCY}, "
                f"iface_{obj}_fp_dis"
            )
            L.append(
                f"set cartoon_transparency, {EVAL_DISAGREE_TRANSPARENCY}, "
                f"iface_{obj}_fp_dis"
            )
        L.append("")
    if SHOW_LABELS:
        L += [
            "label iface_reference and name CA, resi",
            "set label_size, 18",
            "set label_color, black",
            "set label_bg_color, white",
            "set label_bg_transparency, 0.3",
            "",
        ]
    L += [
        "set depth_cue, 0",
        "set spec_reflect, 0.3",
        "",
    ]
    for i, (name, obj, png) in enumerate(
        zip(panel_names, ["reference", "arc", "modfold", "mqa"], panel_pngs)
    ):
        L.append(f"# Panel {i+1}: {name}")
        L.append("hide all")
        L.append(f"show cartoon, {obj}")
        if name == "native":
            L.append("show sticks, iface_reference")
            L.append(f"zoom iface_reference, {ZOOM_BUFFER_NATIVE}")
        else:
            L.append(f"show sticks, iface_{obj}_tp")
            L.append(f"show sticks, iface_{obj}_fp")
            zoom_sel = f"iface_{obj}_tp or iface_{obj}_fp"
            if SHOW_FOUR_CATEGORIES:
                L.append(f"show spheres, iface_{obj}_fn and name CA")
                zoom_sel += f" or iface_{obj}_fn"
            L.append(f"zoom {zoom_sel}, {ZOOM_BUFFER}")
        L += [
            "ray 2400, 2400",
            f"png {png}, dpi=600",
            "",
        ]
    L.append("quit")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PML, "w") as f:
        f.write("\n".join(L))

    panels_meta = [
        {
            "key": "native",
            "title": "(A) Native Structure",
            "target": target,
            "model": None,
            "pdb_path": ref_pdb,
            "metrics": {},
            "confusion_matrix": None,
            "eval_confusion_matrix": None,
            "eval_threshold": None,
            "eval_agreement_rate": None,
            "residue_classifications": None,
        },
    ]
    for i, method in enumerate(METHODS):
        label = chr(ord("B") + i)
        method_metrics = metrics.get(method, {})
        ecm = eval_cm.get(method, {})
        thresh = eval_thresholds.get(method)
        total_eval = sum(ecm.values()) if ecm else 0
        agree_count = ecm.get("TP", 0) + ecm.get("TN", 0)
        agreement_rate = (
            round(agree_count / total_eval, 3) if total_eval > 0 else None
        )
        panels_meta.append({
            "key": OBJ_NAMES[method],
            "title": f"({label}) Top-Model: {method}",
            "target": target,
            "model": three[method],
            "pdb_path": pdb_paths[method],
            "metrics": {
                display_label: round(method_metrics.get(col, 0.0), 3)
                for col, display_label in DISPLAY_METRICS
                if col in method_metrics
            },
            "confusion_matrix": cm_counts.get(method),
            "eval_confusion_matrix": ecm if ecm else None,
            "eval_threshold": round(thresh, 4) if thresh is not None else None,
            "eval_agreement_rate": agreement_rate,
            "residue_classifications": {
                "tp_agree": [[c, r] for c, r in tp_agree.get(method, [])],
                "tp_disagree": [[c, r] for c, r in tp_disagree.get(method, [])],
                "fp_agree": [[c, r] for c, r in fp_agree.get(method, [])],
                "fp_disagree": [[c, r] for c, r in fp_disagree.get(method, [])],
                "fn": [[c, r] for c, r in fn_pairs.get(method, [])],
                "tn_count": cm_counts.get(method, {}).get("TN", 0),
            },
        })

    with open(META_PATH, "w") as f:
        json.dump(panels_meta, f, indent=2)

    print(f"Target: {target}")
    print(f"Wrote PML:  {OUT_PML}")
    print(f"Wrote meta: {META_PATH}")


def main():
    # ---- Resolve target(s) and per-target output dirs ---------------------
    import argparse
    ap = argparse.ArgumentParser(description="Generate PyMOL script for fig. 4 structural panels (ARC repo).")
    ap.add_argument(
        "--target",
        metavar="T",
        help="Generate for this target only (default: outputs/fig4_structure/manuscript/<target>/).",
    )
    ap.add_argument("--all-targets", action="store_true", help="Generate for every target in target_margin_scores.csv.")
    ap.add_argument("--jobs", type=int, default=None, help="Parallel jobs for --all-targets (default: CPU count).")
    ap.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Write PML and metadata here (with --target); default: outputs/fig4_structure/manuscript/<target>/.",
    )
    ap.add_argument(
        "--pdb-models-dir",
        metavar="DIR",
        help="Directory for {model}.pdb (with --target); default: outputs/fig4_structure/pymol/pdb_models/<target>/.",
    )
    ap.add_argument(
        "--arc-model",
        metavar="NAME",
        help="Explicit CASP model basename for ARC (skip target_margin_scores.csv when all three --*-model are set).",
    )
    ap.add_argument("--modfold-model", metavar="NAME", help="Explicit model for ModFOLDdock2S panel.")
    ap.add_argument("--mqa-model", metavar="NAME", help="Explicit model for MQA_server panel.")
    ap.add_argument(
        "--raw-pickle",
        metavar="PATH",
        help="Evaluator pooling pickle (Youden thresholds / per-residue preds); default from configure_paths.",
    )
    ap.add_argument(
        "--ema-results",
        metavar="DIR",
        help="Directory of EMA JSON files {model}_{target}.json.",
    )
    ap.add_argument(
        "--targets-pdb-dir",
        metavar="DIR",
        help="Directory of native reference PDBs named {target}.pdb.",
    )
    ap.add_argument(
        "--native-pdb",
        metavar="PATH",
        help="Single reference PDB path (overrides --targets-pdb-dir/<target>.pdb).",
    )
    ap.add_argument(
        "--skip-copy-models",
        action="store_true",
        help="Do not copy assemblies into --pdb-models-dir (you must place {model}.pdb yourself).",
    )
    ap.add_argument(
        "--no-eval-pickle",
        action="store_true",
        help="Skip layer-2 evaluator overlay (do not load any pooling pickle; avoids huge default qmode2 pickle).",
    )
    args = ap.parse_args()

    if args.all_targets:
        if not os.path.exists(TARGETS_CSV):
            raise SystemExit(
                "target_margin_scores.csv not found under data/. Run build_targets_scores_csv.py first."
            )
        df = pd.read_csv(TARGETS_CSV)
        targets = [str(t).strip() for t in df["target"].unique() if t and str(t).strip() != "(none)"]
        n_workers = args.jobs if args.jobs is not None else min(len(targets), os.cpu_count() or 4)
        n_workers = max(1, min(n_workers, len(targets)))
        if n_workers == 1:
            for target in targets:
                out_dir = os.path.join(MARGIN_DIR, "manuscript", target)
                pdb_models_dir = os.path.join(MARGIN_DIR, "pymol", "pdb_models", target)
                _run_one_target(target, out_dir, pdb_models_dir)
        else:
            tasks = [
                (t, os.path.join(MARGIN_DIR, "manuscript", t), os.path.join(MARGIN_DIR, "pymol", "pdb_models", t))
                for t in targets
            ]
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = {ex.submit(_worker_one_target, task): task[0] for task in tasks}
                for fut in as_completed(futures):
                    t = futures[fut]
                    try:
                        fut.result()
                        print("Target: %s" % t)
                    except Exception as e:
                        print("Target %s failed: %s" % (t, e), file=sys.stderr)
        print("Done: %d targets" % len(targets))
        return

    if args.target:
        target = args.target
        if args.no_eval_pickle and args.raw_pickle:
            raise SystemExit("Use either --no-eval-pickle or --raw-pickle, not both.")
        if args.ema_results or args.raw_pickle or args.targets_pdb_dir:
            configure_paths(
                ema_results=args.ema_results,
                raw_pickle=args.raw_pickle,
                targets_pdb_dir=args.targets_pdb_dir,
            )
        out_dir = args.output_dir or os.path.join(MARGIN_DIR, "manuscript", target)
        pdb_models_dir = args.pdb_models_dir or os.path.join(
            MARGIN_DIR, "pymol", "pdb_models", target
        )
        explicit = (args.arc_model, args.modfold_model, args.mqa_model)
        if any(explicit) and not all(explicit):
            raise SystemExit(
                "Provide all three of --arc-model, --modfold-model, --mqa-model, or none for CSV-based selection."
            )
        three_override = None
        if all(explicit):
            three_override = {
                "ARC": args.arc_model.strip(),
                "ModFOLDdock2S": args.modfold_model.strip(),
                "MQA_server": args.mqa_model.strip(),
            }
        ref_pdb_override = args.native_pdb.strip() if args.native_pdb else None
        _run_one_target(
            target,
            out_dir,
            pdb_models_dir,
            three_override=three_override,
            ref_pdb_override=ref_pdb_override,
            skip_copy_models=args.skip_copy_models,
            raw_pickle_path=None if args.no_eval_pickle else args.raw_pickle,
            no_eval_pickle=args.no_eval_pickle,
        )
        return

    # Single target from best_target.txt (backward compatible)
    target = _load_best_target()
    if not target:
        raise SystemExit(f"No best target in {BEST_TARGET_PATH}")
    out_dir = os.path.join(MARGIN_DIR, "manuscript")
    pdb_models_dir = os.path.join(MARGIN_DIR, "pymol", "pdb_models")
    _run_one_target(target, out_dir, pdb_models_dir)


if __name__ == "__main__":
    main()
